"""One test episode, two systems, same seed: fixed camera vs the full system (D2 + self-calibration).

Same code path as scripts/run_m5.py (EXP-M8-001 regime); this only prints it readably.
Ground-truth numbers (centre error, what was lifted) come from the evaluator, after the grasp.

    .venv/bin/python scripts/demo_episode.py [--seed 2004192]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.run_m5 import run_episode

B, D, G, R, C, X = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[36m", "\033[0m"


def show(title: str, policy: str, selfcal: bool, seed: int) -> None:
    print(f"\n{B}{C}── {title} ──{X}")
    t0 = time.perf_counter()
    r = run_episode(seed, policy, 2, "mugs", "harsh", rng_seed=seed + 17, cameras=("mobile",),
                    elevations=(12.0, 25.0, 45.0), physical=True, selfcal=selfcal)
    v = r["views"]
    moved = "stayed put" if len(set(v)) == 1 else "moved twice"
    print(f"  target        {r['target_class']}")
    print(f"  camera views  {' → '.join(map(str, v))}   {D}({moved}){X}")
    if selfcal and r.get("selfcal_shift"):
        print(f"  self-cal      grasp shifted {1e3 * np.linalg.norm(r['selfcal_shift']):.1f} mm")
    print(f"  {D}evaluator:{X}    grasp centre error {1e3 * r['centre_err_m']:.1f} mm")
    ok = r["success"]
    res = f"{G}✓ {r['lifted'][0]} lifted{X}" if ok else f"{R}✗ fingers closed on air{X}"
    print(f"  grasp         {res}   {D}{time.perf_counter() - t0:.1f} s{X}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2004192)
    a = ap.parse_args()
    print(f"{D}test seed {a.seed} · mug scene · harsh noise (~1° calibration error) · 2 camera moves{X}")
    show("fixed camera", "fixed", False, a.seed)
    show("full system: D2 views + self-calibration", "task_risk_cal_l005", True, a.seed)


if __name__ == "__main__":
    main()
