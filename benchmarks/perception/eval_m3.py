"""M3 pose comparison on identical held-out point clouds (test-split shards).

Methods, all on the same GT-mask segment points (this isolates pose estimation
from segmentation):
  classical       : SDF fit from geometric initialisations (M2 level 1)
  neural_top1     : most confident neural hypothesis, no refinement
  neural_refined  : SDF refinement started from all K neural hypotheses, best cost
  neural_best_of_k: ORACLE upper bound: the hypothesis closest to GT (uses labels; not a method)
Support plane: world z = 0.40 (the table) for the classical/refined fits, the
same for both. Reports symmetry-aware errors overall and by visibility bin.

Usage:  PYTHONPATH=. python -m benchmarks.perception.eval_m3 --shards '../shards/test_*.npz' \\
            --ckpt ../runs/pose_v1/best.pt --out reports/m3/eval_m3.json [--limit 3000]
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.metrics import pose_errors
from data.schemas.objects import CATALOG
from models.inference.proposer import NeuralProposer
from perception.pose.primitive_fit import fit_hypotheses

N, D = np.array([0.0, 0.0, 1.0]), -0.40
BINS = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.01)]


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    t = np.array([r["t_err"] for r in rows]) * 1e3
    rr = np.array([r["r_err_deg"] for r in rows])
    ad = np.array([r["add_s"] for r in rows]) * 1e3
    succ = (t < 10) & ~(rr > 5)
    rng = np.random.default_rng(0)
    boots = [succ[rng.integers(0, len(succ), len(succ))].mean() for _ in range(2000)]
    return {"n": len(rows), "t_err_mm_median": float(np.median(t)), "t_err_mm_p90": float(np.percentile(t, 90)),
            "r_err_deg_median": float(np.nanmedian(rr)) if np.isfinite(rr).any() else None,
            "add_s_mm_median": float(np.median(ad)), "add_s_mm_p90": float(np.percentile(ad, 90)),
            "success_1cm_5deg": float(succ.mean()),
            "success_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(a.shards))
    d = {k: np.concatenate([np.load(f)[k] for f in files]) for k in ("points", "centroid", "cls", "T_WO", "vis")}
    n = len(d["cls"]) if a.limit is None else min(a.limit, len(d["cls"]))
    prop = NeuralProposer(a.ckpt)
    rows = {m: [] for m in ("classical", "neural_top1", "neural_refined", "neural_best_of_k")}
    timing = {m: [] for m in ("classical", "neural", "refine")}
    for i in range(n):
        spec = CATALOG[int(d["cls"][i])]
        P = d["points"][i].astype(np.float64) + d["centroid"][i]
        T_gt = d["T_WO"][i].astype(np.float64)
        vis = float(d["vis"][i])
        t0 = time.perf_counter()
        cl = fit_hypotheses(spec, P, N, D, 0.0)[0].T_WO
        timing["classical"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        hyps, _ = prop.propose(P, spec.name)
        timing["neural"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ref = fit_hypotheses(spec, P, N, D, 0.0, extra_inits=hyps)[0].T_WO
        timing["refine"].append(time.perf_counter() - t0)
        errs = [pose_errors(spec, H, T_gt) for H in hyps]
        for m, T in (("classical", cl), ("neural_top1", hyps[0]), ("neural_refined", ref)):
            rows[m].append(pose_errors(spec, T, T_gt) | {"vis": vis, "shape": spec.shape})
        rows["neural_best_of_k"].append(min(errs, key=lambda e: e["add_s"]) | {"vis": vis, "shape": spec.shape})
    report = {"n": n, "shards": files, "checkpoint": a.ckpt,
              "timing_ms_median": {k: 1e3 * float(np.median(v)) for k, v in timing.items()},
              "methods": {m: {"all": summarise(r),
                              **{f"vis_{lo}-{hi}": summarise([x for x in r if lo <= x["vis"] < hi]) for lo, hi in BINS}}
                          for m, r in rows.items()},
              "note": "neural_best_of_k is an oracle upper bound (selects with GT), not a method"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    for m, v in report["methods"].items():
        print(m, {k: round(v["all"].get(k), 3) if isinstance(v["all"].get(k), float) else v["all"].get(k)
                  for k in ("n", "t_err_mm_median", "add_s_mm_median", "success_1cm_5deg")})


if __name__ == "__main__":
    main()
