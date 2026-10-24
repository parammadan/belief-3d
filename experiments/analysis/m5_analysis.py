"""EXP-M5-001 pre-registered analysis.

Primary (H3): grasp success of task_risk (D) vs entropy (C1) and vs covariance (C2), paired by
episode (same seed => identical t0 observations). Exact two-sided McNemar test on discordant
pairs; Holm correction across the two comparisons; report success rates, discordant counts,
paired difference with bootstrap 95 % CI.
Secondary (descriptive): D vs fixed and vs random (same test, no correction claimed); success on
the ambiguous-at-t0 subset; mean camera travel time; decision time.

Usage:  python experiments/analysis/m5_analysis.py reports/m5/<tag>/ [--treat X --primary a,b --secondary c,d]
(defaults reproduce EXP-M5-001's pre-registered analysis exactly)
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np


def load(dirpath: Path) -> dict[str, dict[int, dict]]:
    by: dict[str, dict[int, dict]] = {}
    for f in sorted(dirpath.glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "skipped" in r:
                continue
            by.setdefault(r["policy"], {})[r["seed"]] = r
    return by


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def compare(by, a: str, base: str, seeds: list[int]) -> dict:
    xa = np.array([bool(by[a][s].get("success")) for s in seeds])
    xb = np.array([bool(by[base][s].get("success")) for s in seeds])
    b, c = int((xa & ~xb).sum()), int((~xa & xb).sum())
    d = xa.astype(float) - xb.astype(float)
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1) if len(d) else np.array([0.0])
    return {"n": len(seeds), "rate_" + a: float(xa.mean()), "rate_" + base: float(xb.mean()),
            "diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "a_only": b, "base_only": c, "mcnemar_p": mcnemar_exact(b, c)}


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    m, out, run = len(ps), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--treat", default="task_risk")
    ap.add_argument("--primary", default="entropy,covariance")
    ap.add_argument("--secondary", default="fixed,random")
    a = ap.parse_args()
    d = Path(a.dir)
    by = load(d)
    pols = sorted(by)
    seeds = sorted(set.intersection(*(set(v) for v in by.values())))
    t = a.treat
    prim = {f"{t}_vs_{b}": compare(by, t, b, seeds) for b in a.primary.split(",")}
    adj = holm({k: v["mcnemar_p"] for k, v in prim.items()})
    for k, v in prim.items():
        v["p_holm"] = adj[k]
    sec = {f"{t}_vs_{b}": compare(by, t, b, seeds) for b in a.secondary.split(",") if b}
    amb = [s for s in seeds if by[t][s].get("ambiguous_t0")]
    amb_rates = {p: float(np.mean([bool(by[p][s].get("success")) for s in amb])) if amb else None for p in pols}
    travel = {p: float(np.mean([by[p][s].get("travel_s", 0.0) for s in seeds])) for p in pols}
    decide = {p: float(np.nanmean([by[p][s].get("decide_ms_mean") or np.nan for s in seeds])) for p in pols}
    overall = {p: float(np.mean([bool(by[p][s].get("success")) for s in seeds])) for p in pols}
    out = {"n_paired_episodes": len(seeds), "n_ambiguous_t0": len(amb), "success": overall, "treatment": t,
           "primary": prim, "secondary": sec, "success_ambiguous_subset": amb_rates,
           "mean_travel_s": travel, "mean_decide_ms": decide}
    (d / f"analysis_{t}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
