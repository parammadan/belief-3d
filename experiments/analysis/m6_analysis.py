"""EXP-M6-001 pre-registered analysis: does recovery (re-observe + retry after a failed
proprioceptive check) raise pick-and-place success?

Inputs: two run directories with the same seeds: max_attempts = 1 and max_attempts = 2.
Primary: per policy, paired success (attempts 2 vs 1), exact McNemar, Holm across policies.
Descriptive: success rates, mean attempts, recovery rate (attempt-1 verification failed and the
episode still succeeded), verification error rates (verified "held" but not placed; not verified
but actually placed).

Usage:  python experiments/analysis/m6_analysis.py reports/m6/<tag_a1> reports/m6/<tag_a2>
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np


def load(d: Path) -> dict[str, dict[int, dict]]:
    by: dict[str, dict[int, dict]] = {}
    for f in sorted(d.glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "skipped" not in r:
                by.setdefault(r["policy"], {})[r["seed"]] = r
    return by


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def main() -> None:
    a1, a2 = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    out, ps = {"policies": {}}, {}
    for pol in sorted(set(a1) & set(a2)):
        seeds = sorted(set(a1[pol]) & set(a2[pol]))
        x1 = np.array([bool(a1[pol][s]["success"]) for s in seeds])
        x2 = np.array([bool(a2[pol][s]["success"]) for s in seeds])
        b, c = int((x2 & ~x1).sum()), int((~x2 & x1).sum())
        rec = [s for s in seeds if a2[pol][s]["attempts"] >= 2]
        held_not_placed = np.mean([a2[pol][s]["held"] and not a2[pol][s]["success"] for s in seeds])
        notheld_placed = np.mean([(not a2[pol][s]["held"]) and a2[pol][s]["success"] for s in seeds])
        ps[pol] = mcnemar(b, c)
        out["policies"][pol] = {
            "n": len(seeds), "success_1_attempt": float(x1.mean()), "success_2_attempts": float(x2.mean()),
            "gain": float((x2.astype(float) - x1).mean()), "recovered_only_with_retry": b, "lost_with_retry": c,
            "mcnemar_p": ps[pol], "episodes_needing_retry": len(rec),
            "retry_success_rate": float(np.mean([a2[pol][s]["success"] for s in rec])) if rec else None,
            "mean_attempts": float(np.mean([a2[pol][s]["attempts"] for s in seeds])),
            "verified_held_but_not_placed": float(held_not_placed),
            "not_verified_but_placed": float(notheld_placed)}
    order = sorted(ps, key=ps.get)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out["policies"][k]["p_holm"] = run
    Path(sys.argv[2], "analysis_m6.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
