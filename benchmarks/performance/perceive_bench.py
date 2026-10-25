"""PERF-M7-006: per-frame perceive() with scipy's finite-difference Jacobian vs the batched one.

    python benchmarks/performance/perceive_bench.py [--reps 5] [--out reports/m7/perceive_bench_m1.json]
"""
from __future__ import annotations

import argparse
import functools
import json
import platform
import time

import numpy as np

import perception.pose.primitive_fit as pf
from data.generators.episodes import EpisodeConfig, run_episode
from perception import pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    frames = list(run_episode(1_500_000, EpisodeConfig(n_frames=a.frames, scenario="varied", noise="harsh")))
    obs = [f.obs["overhead"] for f in frames]
    orig = pf.fit_hypotheses
    res, poses = {}, {}
    try:
        for name, fast in (("scipy_fd", False), ("batched_fd", True)):
            pipeline.fit_hypotheses = functools.partial(orig, fast_jac=fast)
            pipeline.perceive(obs[0])                                    # warm-up
            per = []
            for _ in range(a.reps):                                      # interleave-free, same frames each rep
                t0 = time.perf_counter()
                out = [pipeline.perceive(o) for o in obs]
                per.append((time.perf_counter() - t0) / len(obs) * 1e3)
            res[name] = {"median_ms_per_frame": float(np.median(per)), "all_ms": per}
            poses[name] = [[d.T_WO for d in fr.detections] for fr in out]
    finally:
        pipeline.fit_hypotheses = orig
    same = all(len(x) == len(y) and all(np.array_equal(p, q) for p, q in zip(x, y))
               for x, y in zip(poses["scipy_fd"], poses["batched_fd"]))
    sp = res["scipy_fd"]["median_ms_per_frame"] / res["batched_fd"]["median_ms_per_frame"]
    print(f"scipy FD {res['scipy_fd']['median_ms_per_frame']:.0f} ms/frame  batched "
          f"{res['batched_fd']['median_ms_per_frame']:.0f} ms/frame  x{sp:.2f}  identical poses: {same}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"machine": platform.platform(), "frames": a.frames, "reps": a.reps, "noise": "harsh",
                       "results": res, "speedup": sp, "identical_poses": same}, f, indent=1)


if __name__ == "__main__":
    main()
