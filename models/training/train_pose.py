"""Train the multi-hypothesis point-cloud pose proposer.

* data: shards from data.generators.pointcloud_samples; train and val come
  from disjoint seed splits (separate shard globs)
* augmentation: random yaw about world z, applied jointly to points and
  label. Gravity-aligned scenes make this an exact symmetry of the task.
* mixed precision chosen per GPU: bf16 where the hardware supports it (A100/H100), otherwise
  fp16 with a GradScaler. A V100 has no bf16 units: bf16 autocast ran ~6x slower than the A100
  and at ~0 % utilisation (measured, job 11348076).
* AdamW + one-cycle schedule, deterministic seeding
* logs: loss, best-hypothesis surface error (mm), throughput, peak VRAM,
  nvidia-smi utilisation; best-on-val checkpoint plus a metadata JSON with
  data hashes and git commit

Usage:
  python -m models.training.train_pose --train 'shards/train_*.npz' --val 'shards/val_*.npz' \\
      --out runs/pose_v1 --epochs 30 --batch 512
  (--max-steps N for a smoke test)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from data.schemas.objects import CATALOG
from models.architectures.pointnet_pose import PointNetPose
from models.losses.sdf_loss import SurfaceBank, hypothesis_losses, mhp_loss

AMP_DTYPE = torch.bfloat16


def load_shards(pattern: str) -> dict[str, np.ndarray]:
    """Load + concatenate shards. A packed cache (<dir>/packed_<split>.npz, uncompressed) is
    built once next to the shards and reused: compressed per-shard loading took ~5 min of
    GPU-allocated idle time (measured)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    tag = hashlib.sha256("".join(f"{f}:{Path(f).stat().st_size}" for f in files).encode()).hexdigest()[:12]
    cache = Path(files[0]).parent / f"packed_{Path(files[0]).name.split('_')[0]}_{tag}.npz"
    if cache.exists():
        out = dict(np.load(cache))
    else:
        parts = [dict(np.load(f)) for f in files]
        out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(cache, **out)
    out["_files"] = files
    return out


def to_tensors(d: dict, device) -> dict[str, torch.Tensor]:
    T = torch.as_tensor(d["T_WO"], dtype=torch.float32)
    cen = torch.as_tensor(d["centroid"], dtype=torch.float32)
    return {"points": torch.as_tensor(d["points"].astype(np.float32)).to(device),
            "cls": torch.as_tensor(d["cls"], dtype=torch.long).to(device),
            "R_gt": T[:, :3, :3].to(device), "t_gt": (T[:, :3, 3] - cen).to(device)}


def yaw_augment(b: dict, gen: torch.Generator) -> dict:
    n = b["cls"].shape[0]
    th = torch.rand(n, generator=gen, device=b["cls"].device) * 2 * math.pi
    c, s = torch.cos(th), torch.sin(th)
    Rz = torch.zeros(n, 3, 3, device=th.device)
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s, s, c, 1.0
    return {"points": b["points"] @ Rz.transpose(1, 2), "cls": b["cls"],
            "R_gt": Rz @ b["R_gt"], "t_gt": (Rz @ b["t_gt"][..., None])[..., 0]}


def gpu_util() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


@torch.no_grad()
def evaluate(model, bank, data, batch: int) -> dict:
    model.eval()
    errs, top1 = [], []
    n = data["cls"].shape[0]
    for i in range(0, n, batch):
        sl = slice(i, i + batch)
        with torch.autocast(device_type=data["cls"].device.type, dtype=AMP_DTYPE,
                            enabled=data["cls"].device.type == "cuda"):
            R, t, logits = model(data["points"][sl], data["cls"][sl])
        _, _, best = mhp_loss(bank, data["cls"][sl], R.float(), t.float(), logits.float(), data["R_gt"][sl],
                              data["t_gt"][sl])
        L = hypothesis_losses(bank, data["cls"][sl], R.float(), t.float(), data["R_gt"][sl], data["t_gt"][sl])
        errs.append(best.cpu())
        top1.append(L.gather(1, logits.argmax(1, keepdim=True))[:, 0].cpu())
    model.train()
    e, t1 = torch.cat(errs), torch.cat(top1)
    return {"best_of_k_mm_mean": 1e3 * float(e.mean()), "best_of_k_mm_median": 1e3 * float(e.median()),
            "top1_mm_mean": 1e3 * float(t1.mean()), "top1_mm_median": 1e3 * float(t1.median()),
            "top1_under_2mm": float((t1 < 0.002).float().mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global AMP_DTYPE
    AMP_DTYPE = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)) \
        else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda" and AMP_DTYPE == torch.float16))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_np, va_np = load_shards(a.train), load_shards(a.val)
    tr, va = to_tensors(tr_np, dev), to_tensors(va_np, dev)
    bank = SurfaceBank(device=dev)
    model = PointNetPose(len(CATALOG), k=a.k).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    n = tr["cls"].shape[0]
    steps_per_epoch = max(1, n // a.batch)
    total = a.max_steps or a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    meta = {"args": vars(a), "device": str(dev), "torch": torch.__version__, "amp_dtype": str(AMP_DTYPE),
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "n_train": int(n), "n_val": int(va["cls"].shape[0]),
            "train_files": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16] for f in tr_np["_files"]},
            "val_files": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16] for f in va_np["_files"]},
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                         check=False).stdout.strip()}
    log, best, step, t0 = [], math.inf, 0, time.perf_counter()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while step < total:
        perm = torch.randperm(n, generator=gen, device=dev)
        for i in range(0, n - a.batch + 1, a.batch):
            if step >= total:
                break
            idx = perm[i:i + a.batch]
            b = yaw_augment({k: v[idx] for k, v in tr.items()}, gen)
            with torch.autocast(device_type=dev.type, dtype=AMP_DTYPE, enabled=dev.type == "cuda"):
                R, t, logits = model(b["points"], b["cls"])
            pose, ce, bestk = mhp_loss(bank, b["cls"], R.float(), t.float(), logits.float(), b["R_gt"], b["t_gt"])
            loss = pose + 0.01 * ce
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % a.log_every == 0 or step == total:
                el = time.perf_counter() - t0
                row = {"step": step, "loss": float(loss), "pose_mm": 1e3 * float(pose), "ce": float(ce),
                       "best_k_mm": 1e3 * float(bestk.mean()), "lr": sched.get_last_lr()[0],
                       "samples_per_s": step * a.batch / el, "elapsed_s": el,
                       "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
                       "nvidia_smi": gpu_util() if dev.type == "cuda" else None}
                if step % (a.log_every * 4) == 0 or step == total:
                    row["val"] = evaluate(model, bank, va, 2048)
                    if row["val"]["top1_mm_mean"] < best:
                        best = row["val"]["top1_mm_mean"]
                        torch.save({"model": model.state_dict(), "k": a.k, "n_classes": len(CATALOG),
                                    "step": step, "val": row["val"]}, out / "best.pt")
                log.append(row)
                print(json.dumps(row), flush=True)
    torch.save({"model": model.state_dict(), "k": a.k, "n_classes": len(CATALOG), "step": step}, out / "last.pt")
    meta["final_val"] = evaluate(model, bank, va, 2048)
    meta["best_val_top1_mm_mean"] = best
    meta["wall_s"] = time.perf_counter() - t0
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "log.json").write_text(json.dumps(log))
    print(json.dumps({"final_val": meta["final_val"], "wall_s": meta["wall_s"]}))


if __name__ == "__main__":
    main()
