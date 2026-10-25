"""EXP-M8-001 pre-registered analysis (final study).

Every arm runs on the same test seeds (paired). For each primary comparison: success rates, the
paired difference with a bootstrap 95 % CI, and an exact two-sided McNemar test. Holm corrects
across the primary family. Secondary comparisons are reported uncorrected and descriptive.
Only seeds present in every arm are used. Skips ("no confirmed tracks") happen before any
arm-specific step, so they are identical across arms.

Usage:  python experiments/analysis/m8_analysis.py "reports/m8/exp_m8_001_sc0+reports/m8/exp_m8_001_sc1" \\
            [--out reports/m8/exp_m8_001_analysis.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.analysis.m5_analysis import compare, holm, load

FULL, D2 = "task_risk_cal_l005+selfcal", "task_risk_cal_l005"
PRIMARY = {
    "P1 full system vs B1": (FULL, "fixed@b1"),
    "P2 D2 vs B3 entropy": (D2, "entropy"),
    "P3 selfcal on D2": (FULL, D2),
    "P4 B2 vs B1 (fixed views)": ("fixed", "fixed@b1"),
}
SECONDARY = {
    "B1 vs B0": ("fixed@b1", "fixed@b0"),
    "D2 vs random": (D2, "random"),
    "selfcal on fixed (control)": ("fixed+selfcal", "fixed"),
    "full system vs B2 fixed": (FULL, "fixed"),
    "full system vs B0": (FULL, "fixed@b0"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    by = load(a.dir)
    arms = sorted({x for pair in [*PRIMARY.values(), *SECONDARY.values()] for x in pair})
    missing = [x for x in arms if x not in by]
    if missing:
        raise SystemExit(f"missing arms: {missing}")
    seeds = sorted(set.intersection(*(set(by[x]) for x in arms)))
    rep = {"n_paired": len(seeds), "rates": {}, "primary": {}, "secondary": {}}
    for x in arms:
        s = np.array([bool(by[x][k].get("success")) for k in seeds])
        rep["rates"][x] = float(s.mean())
    for name, (t, b) in PRIMARY.items():
        rep["primary"][name] = compare(by, t, b, seeds)
    for name, p in holm({k: v["mcnemar_p"] for k, v in rep["primary"].items()}).items():
        rep["primary"][name]["holm_p"] = p
    for name, (t, b) in SECONDARY.items():
        rep["secondary"][name] = compare(by, t, b, seeds)
    amb = [k for k in seeds if by["fixed"][k].get("ambiguous_t0")]
    rep["ambiguous_t0"] = {"n": len(amb), "rates": {x: float(np.mean([bool(by[x][k].get("success")) for k in amb]))
                                                    for x in arms} if amb else {}}
    print(f"paired episodes: {len(seeds)} (ambiguous at t0: {len(amb)})")
    for x in arms:
        print(f"  {x:30s} {rep['rates'][x]:.3f}")
    for grp in ("primary", "secondary"):
        print(grp)
        for name, r in rep[grp].items():
            ks = list(r)
            print(f"  {name:30s} {r[ks[1]]:.3f} vs {r[ks[2]]:.3f}  diff {r['diff']:+.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  {r['a_only']}/{r['base_only']}  "
                  f"p={r['mcnemar_p']:.4f}" + (f"  holm={r['holm_p']:.4f}" if "holm_p" in r else ""))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
