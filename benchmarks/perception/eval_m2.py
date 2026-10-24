"""Independent M2 evaluator: predictions (JSON) + gt_* files -> metrics.

Never imports inference code. Mask decoding is re-implemented here on purpose.

Detection: per (frame, camera), GT instances with >= 25 visible px are matched
to predicted masks one-to-one by IoU (Hungarian, IoU >= 0.5).
Pose: symmetry-aware errors for matched detections, using the *GT* object's
catalog spec.
Tracking (CLEAR MOT, class-agnostic, 3D): per frame all GT objects on the
table are matched to confirmed tracks by centre distance <= `match_dist`.
We report MOTA, MOTP, ID switches, fragmentations and occlusion recovery.
"Fully hidden" = visible fraction < 0.05 in every camera.

Usage:  PYTHONPATH=. python -m benchmarks.perception.eval_m2 <dataset_root> <pred_dir> [--out report.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from benchmarks.metrics import pose_errors
from data.schemas.objects import CATALOG

SPEC = {o.name: o for o in CATALOG}


def _rle_decode(runs: list[int], shape: tuple[int, int]) -> np.ndarray:
    vals = np.zeros(shape[0] * shape[1], bool)
    pos = 0
    for k, r in enumerate(runs):
        if k % 2:
            vals[pos:pos + r] = True
        pos += r
    return vals.reshape(shape)


def _bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> list[float]:
    v = np.asarray([x for x in values if np.isfinite(x)])
    if len(v) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def evaluate(dataset: Path, preds: Path, match_dist: float = 0.05) -> dict:
    manifest = json.loads((dataset / "manifest.json").read_text())
    cams = manifest["config"]["cameras"]
    det_counts = defaultdict(int)
    ious, cls_ok, pose_rows = [], [], []
    ep_mota, mot = [], defaultdict(float)
    recov = {"events": 0, "recovered_same_id": 0, "latency_frames": []}
    for ep in manifest["episodes"]:
        eid = ep["episode_id"]
        P = json.loads((preds / f"{eid}.json").read_text())
        gtj = {(k, c): json.loads((dataset / eid / f"gt_{k:04d}_{c}.json").read_text())
               for k in range(ep["n_frames"]) for c in cams}
        # ---------------- detection + pose ----------------
        for k in range(ep["n_frames"]):
            for c in cams:
                seg = np.load(dataset / eid / f"gt_{k:04d}_{c}.npz")["segmentation"]
                gobjs = [o for o in gtj[(k, c)]["objects"] if (seg == o["instance_id"]).sum() >= 25]
                pdet = [d for d in P["frames"][k]["detections"] if d["camera"] == c]
                pm = [_rle_decode(d["mask_rle"], seg.shape) for d in pdet]
                iou = np.zeros((len(gobjs), len(pdet)))
                for i, o in enumerate(gobjs):
                    g = seg == o["instance_id"]
                    for j, m in enumerate(pm):
                        u = np.logical_or(g, m).sum()
                        iou[i, j] = np.logical_and(g, m).sum() / u if u else 0.0
                r, cidx = linear_sum_assignment(-iou) if iou.size else ([], [])
                matched = [(i, j) for i, j in zip(r, cidx) if iou[i, j] >= 0.5]
                det_counts["tp"] += len(matched)
                det_counts["fn"] += len(gobjs) - len(matched)
                det_counts["fp"] += len(pdet) - len(matched)
                for i, j in matched:
                    o, d = gobjs[i], pdet[j]
                    spec = SPEC[o["class_name"]]
                    ious.append(iou[i, j])
                    cls_ok.append(d["color_class"] == spec.color_class)
                    e = pose_errors(spec, np.array(d["T_WO"]), np.array(o["T_WO"]))
                    e |= {"shape": spec.shape, "camera": c, "visible_fraction": o["visible_fraction"]}
                    pose_rows.append(e)
        # ---------------- tracking (CLEAR MOT) ----------------
        last_id: dict[int, int] = {}
        was_matched: dict[int, bool] = {}
        hidden_since: dict[int, tuple[int, int | None]] = {}
        g_tot = fn = fp = idsw = frag = 0
        dist_sum = n_match = 0
        for k in range(ep["n_frames"]):
            ref = gtj[(k, cams[0])]["objects"]
            vis = {o["instance_id"]: max(gtj[(k, c)]["objects"][n]["visible_fraction"] for c in cams)
                   for n, o in enumerate(ref)}
            tracks = P["frames"][k]["tracks"]
            G = np.array([np.array(o["T_WO"])[:3, 3] for o in ref])
            T = np.array([np.array(t["T_WO"])[:3, 3] for t in tracks]).reshape(-1, 3)
            D = np.linalg.norm(G[:, None] - T[None], axis=2) if len(T) else np.zeros((len(G), 0))
            r, cidx = linear_sum_assignment(np.where(D <= match_dist, D, 1e6)) if D.size else ([], [])
            pairs = {ref[i]["instance_id"]: tracks[j]["track_id"] for i, j in zip(r, cidx) if D[i, j] <= match_dist}
            dist_sum += sum(D[i, j] for i, j in zip(r, cidx) if D[i, j] <= match_dist)
            n_match += len(pairs)
            g_tot += len(ref)
            fn += len(ref) - len(pairs)
            fp += len(tracks) - len(pairs)
            for o in ref:
                gid = o["instance_id"]
                m = gid in pairs
                if m and gid in last_id and last_id[gid] != pairs[gid]:
                    idsw += 1
                if m and was_matched.get(gid) is False and gid in last_id:
                    frag += 1
                # occlusion recovery bookkeeping
                if vis[gid] < 0.05 and gid not in hidden_since:
                    hidden_since[gid] = (k, last_id.get(gid))
                elif vis[gid] >= 0.05 and gid in hidden_since:
                    start, id_before = hidden_since[gid]
                    if id_before is not None and k > start:
                        # find first frame from k on where gid is matched
                        recov["events"] += 1
                        hit = next((kk for kk in range(k, ep["n_frames"]) if gid in _pairs_at(P, gtj, kk, cams, match_dist)), None)
                        if hit is not None:
                            recov["latency_frames"].append(hit - k)
                            recov["recovered_same_id"] += int(_pairs_at(P, gtj, hit, cams, match_dist)[gid] == id_before)
                    del hidden_since[gid]
                if m:
                    last_id[gid] = pairs[gid]
                was_matched[gid] = m
        ep_mota.append(1.0 - (fn + fp + idsw) / max(g_tot, 1))
        for key, val in (("gt", g_tot), ("fn", fn), ("fp", fp), ("idsw", idsw), ("frag", frag),
                         ("dist_sum", dist_sum), ("n_match", n_match)):
            mot[key] += val

    def stats(key, rows, scale=1.0, shape=None):
        v = [r[key] * scale for r in rows if shape is None or r["shape"] == shape]
        v = [x for x in v if np.isfinite(x)]
        if not v:
            return None
        return {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "ci95_mean": [x for x in _bootstrap_ci(v)]}

    tp, fpd, fnd = det_counts["tp"], det_counts["fp"], det_counts["fn"]
    report = {
        "dataset": manifest["name"], "episodes": len(manifest["episodes"]), "dataset_commit": manifest["git_commit"],
        "detection": {"tp": tp, "fp": fpd, "fn": fnd,
                      "precision": tp / max(tp + fpd, 1), "recall": tp / max(tp + fnd, 1),
                      "mask_iou": stats("iou", [{"iou": x, "shape": None} for x in ious]),
                      "class_accuracy": float(np.mean(cls_ok)) if cls_ok else None},
        "pose": {shape: {"t_err_mm": stats("t_err", pose_rows, 1e3, shape),
                         "r_err_deg": stats("r_err_deg", pose_rows, 1.0, shape),
                         "add_s_mm": stats("add_s", pose_rows, 1e3, shape)}
                 for shape in ("box", "cylinder", "sphere")},
        "pose_success_rate_1cm_5deg": float(np.mean([r["t_err"] < 0.01 and not (r["r_err_deg"] > 5) for r in pose_rows]))
        if pose_rows else None,
        "tracking": {"gt_object_frames": int(mot["gt"]), "fn": int(mot["fn"]), "fp": int(mot["fp"]),
                     "id_switches": int(mot["idsw"]), "fragmentations": int(mot["frag"]),
                     "MOTA": 1.0 - (mot["fn"] + mot["fp"] + mot["idsw"]) / max(mot["gt"], 1),
                     "MOTA_ci95_over_episodes": _bootstrap_ci(ep_mota),
                     "MOTP_mm": 1e3 * mot["dist_sum"] / max(mot["n_match"], 1),
                     "match_dist_m": match_dist},
        "occlusion_recovery": {"events": recov["events"], "recovered_same_id": recov["recovered_same_id"],
                               "recovered_rate": recov["recovered_same_id"] / recov["events"] if recov["events"] else None,
                               "latency_frames_median": float(np.median(recov["latency_frames"]))
                               if recov["latency_frames"] else None},
    }
    return report


def _pairs_at(P, gtj, k, cams, match_dist) -> dict[int, int]:
    ref = gtj[(k, cams[0])]["objects"]
    tracks = P["frames"][k]["tracks"]
    if not tracks:
        return {}
    G = np.array([np.array(o["T_WO"])[:3, 3] for o in ref])
    T = np.array([np.array(t["T_WO"])[:3, 3] for t in tracks])
    D = np.linalg.norm(G[:, None] - T[None], axis=2)
    r, c = linear_sum_assignment(np.where(D <= match_dist, D, 1e6))
    return {ref[i]["instance_id"]: tracks[j]["track_id"] for i, j in zip(r, c) if D[i, j] <= match_dist}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("preds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = evaluate(Path(a.dataset), Path(a.preds))
    s = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(s)
    print(s)
