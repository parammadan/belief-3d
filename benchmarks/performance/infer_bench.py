"""M7 inference benchmark: PyTorch vs ONNX Runtime vs TensorRT for the pose proposer.

Measured per backend and batch size: median / p95 / p99 latency (CUDA-synchronised), throughput,
peak GPU memory, and accuracy versus the PyTorch fp32 reference: max |Δ| of outputs and the
task metric, top-1 surface error (mm) on real val samples. Model-only timing; the full-pipeline
latency is reported separately by the M2/M4 runners. Backends that are unavailable are reported
as such, never assumed.

Usage (GPU node):  python -m benchmarks.performance.infer_bench --ckpt ../runs/pose_v1/best.pt \\
                       --val '../shards/val_*.npz' --out ../reports/m7/infer_bench.json
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from models.architectures.pointnet_pose import PointNetPose
from models.losses.sdf_loss import SurfaceBank, hypothesis_losses


def load_val(pattern: str, n: int):
    d = {k: np.concatenate([np.load(f)[k] for f in sorted(glob.glob(pattern))]) for k in ("points", "cls", "T_WO", "centroid")}
    idx = np.random.default_rng(0).choice(len(d["cls"]), n, replace=False)
    pts = d["points"][idx].astype(np.float32)
    cls = d["cls"][idx].astype(np.int64)
    T = d["T_WO"][idx].astype(np.float32)
    return pts, cls, T[:, :3, :3], T[:, :3, 3] - d["centroid"][idx].astype(np.float32)


def timeit(fn, iters: int, warmup: int, sync) -> np.ndarray:
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t0)
    return np.asarray(out)


def stats(lat: np.ndarray, batch: int) -> dict:
    return {"median_ms": 1e3 * float(np.median(lat)), "p95_ms": 1e3 * float(np.percentile(lat, 95)),
            "p99_ms": 1e3 * float(np.percentile(lat, 99)), "throughput_per_s": batch / float(np.median(lat))}


def top1_mm(bank, cls, R, t, logits, R_gt, t_gt) -> float:
    L = hypothesis_losses(bank, cls, R, t, R_gt, t_gt)
    return 1e3 * float(L.gather(1, logits.argmax(1, keepdim=True)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batches", default="1,32,256")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=True)
    model = PointNetPose(ck["n_classes"], k=ck["k"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    pts, cls, Rg, tg = load_val(a.val, a.n)
    P, C = torch.as_tensor(pts, device=dev), torch.as_tensor(cls, device=dev)
    Rg_t, tg_t = torch.as_tensor(Rg, device=dev), torch.as_tensor(tg, device=dev)
    bank = SurfaceBank(device=dev)
    sync = torch.cuda.synchronize
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "checkpoint": a.ckpt,
              "n_eval": a.n, "backends": {}}
    with torch.no_grad():
        ref = model(P, C)
    report["reference_top1_mm"] = top1_mm(bank, C, *ref, Rg_t, tg_t)

    def accuracy(outs) -> dict:
        R, t, lg = (torch.as_tensor(np.asarray(o), device=dev, dtype=torch.float32) for o in outs)
        return {"max_abs_diff_R": float((R - ref[0]).abs().max()), "max_abs_diff_t_m": float((t - ref[1]).abs().max()),
                "top1_mm": top1_mm(bank, C, R, t, lg, Rg_t, tg_t),
                "argmax_agreement": float((lg.argmax(1) == ref[2].argmax(1)).float().mean())}

    batches = [int(b) for b in a.batches.split(",")]
    # ---- PyTorch eager fp32 / fp16 autocast
    for name, ctx in (("torch_fp32", torch.autocast("cuda", enabled=False)),
                      ("torch_fp16_autocast", torch.autocast("cuda", dtype=torch.float16))):
        row = {}
        with torch.no_grad(), ctx:
            outs = model(P, C)
            row["accuracy"] = accuracy([o.float().cpu() for o in outs])
            for b in batches:
                torch.cuda.reset_peak_memory_stats()
                lat = timeit(lambda b=b: model(P[:b], C[:b]), a.iters, 20, sync)
                row[f"b{b}"] = stats(lat, b) | {"peak_mem_mb": torch.cuda.max_memory_allocated() / 1e6}
        report["backends"][name] = row
    # ---- ONNX export + ONNX Runtime (CUDA EP, TensorRT EP if present)
    onnx_path = Path(a.out).with_suffix(".onnx")
    try:
        torch.onnx.export(model, (P[:2], C[:2]), str(onnx_path), input_names=["points", "cls"],
                          output_names=["R", "t", "logits"], dynamic_axes={"points": {0: "B"}, "cls": {0: "B"},
                                                                          "R": {0: "B"}, "t": {0: "B"}, "logits": {0: "B"}},
                          opset_version=17)
        report["onnx_export"] = "ok"
    except Exception as e:  # noqa: BLE001 — report, don't assume
        report["onnx_export"] = f"failed: {type(e).__name__}: {e}"
    if report["onnx_export"] == "ok":
        import onnxruntime as ort
        report["onnxruntime"] = ort.__version__
        for name, provs in (("ort_cuda", ["CUDAExecutionProvider"]),
                            ("ort_tensorrt_fp16", [("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
                                                   "CUDAExecutionProvider"])):
            try:
                so = ort.SessionOptions()
                so.intra_op_num_threads = 4          # explicit: avoids affinity errors on shared nodes
                sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=provs)
                used = sess.get_providers()
                wanted = provs[0] if isinstance(provs[0], str) else provs[0][0]
                if wanted not in used:
                    # Never report a CPU fallback under a GPU label.
                    report["backends"][name] = {"unavailable": f"requested {wanted}, got {used} (fallback)"}
                    continue
                row = {"providers_used": used}
                full = {"points": pts, "cls": cls}
                row["accuracy"] = accuracy(sess.run(None, full))
                for b in batches:
                    feed = {"points": pts[:b], "cls": cls[:b]}
                    lat = timeit(lambda feed=feed, sess=sess: sess.run(None, feed), a.iters, 20, lambda: None)
                    row[f"b{b}"] = stats(lat, b)
                report["backends"][name] = row
            except Exception as e:  # noqa: BLE001
                report["backends"][name] = {"unavailable": f"{type(e).__name__}: {str(e)[:300]}"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
