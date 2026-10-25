"""EXP-M8-002 pre-registered analysis: does B2's position covariance reach nominal coverage after
held-out calibration?

NEES = e^T Σ^-1 e for each matched track-frame (e = estimated - true position, Σ = reported 3x3
covariance). Scaling Σ by s divides NEES by s. Coverage at level α = P(NEES / s <= χ²₃(α)).

  primary    per noise preset p: s_p = q_0.8(NEES_calib,p) / χ²₃(0.8), one scalar fitted on the
             calibration split. H2 holds on test if coverage at 50/80/95 % is within ±5 pp of
             nominal in BOTH presets (6 checks; each with an episode-cluster bootstrap 95 % CI).
  secondary  (a) raw, uncalibrated; (b) one global s (both presets pooled); (c) per-level
             split-conformal: s_p,α = q_α(NEES_calib,p) / χ²₃(α) (calibrates each level separately);
             (d) the same numbers for B1 (Kalman), for reference.

Usage: python experiments/analysis/m8_h2.py reports/m8/h2 [--out reports/m8/exp_m8_002_analysis.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHI2 = {"50": 2.366, "80": 4.642, "95": 7.815}       # chi-square, 3 dof
PRESETS = ("nominal", "harsh")
SCEN = ("occlusion", "dynamic")


def rows(d: Path, split: str, noise: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    v, ep = [], []
    for sc in SCEN:
        for line in (d / f"{split}_{sc}_{noise}_{b}_rows.jsonl").read_text().splitlines():
            r = json.loads(line)
            v.append(r["nees"])
            ep.append(f"{sc}/{r['episode']}")
    return np.array(v), np.array(ep)


def coverage(v, ep, scale) -> dict:
    """Coverage per level; scale is a float or {level: float}. Bootstrap resamples episodes."""
    eps = np.unique(ep)
    idx = [np.flatnonzero(ep == e) for e in eps]
    rng = np.random.default_rng(0)
    out = {}
    for lv, q in CHI2.items():
        s = scale[lv] if isinstance(scale, dict) else scale
        hit = v / s <= q
        boots = []
        for _ in range(2000):
            pick = np.concatenate([idx[i] for i in rng.integers(0, len(eps), len(eps))])
            boots.append(hit[pick].mean())
        out[lv] = {"coverage": float(hit.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                           float(np.percentile(boots, 97.5))],
                   "within_5pp": bool(abs(hit.mean() - int(lv) / 100) <= 0.05)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    rep: dict = {}
    for b in ("b2", "b1"):
        cal = {p: rows(d, "calib", p, b) for p in PRESETS}
        tst = {p: rows(d, "test", p, b) for p in PRESETS}
        s_p = {p: float(np.quantile(cal[p][0], 0.8) / CHI2["80"]) for p in PRESETS}
        s_g = float(np.quantile(np.concatenate([cal[p][0] for p in PRESETS]), 0.8) / CHI2["80"])
        s_c = {p: {lv: float(np.quantile(cal[p][0], int(lv) / 100) / q) for lv, q in CHI2.items()} for p in PRESETS}
        r = {"scales": {"per_preset": s_p, "global": s_g, "conformal": s_c},
             "n_test": {p: len(tst[p][0]) for p in PRESETS}, "n_calib": {p: len(cal[p][0]) for p in PRESETS}}
        for name, sc in (("raw", {p: 1.0 for p in PRESETS}), ("per_preset", s_p),
                         ("global", {p: s_g for p in PRESETS}), ("conformal", s_c)):
            r[name] = {p: coverage(*tst[p], sc[p]) for p in PRESETS}
        r["H2_primary_holds"] = all(r["per_preset"][p][lv]["within_5pp"] for p in PRESETS for lv in CHI2)
        rep[b] = r
    for b, r in rep.items():
        print(f"{b}: scales per-preset {r['scales']['per_preset']}  global {r['scales']['global']:.4g}  "
              f"n_test {r['n_test']}")
        for name in ("raw", "per_preset", "global", "conformal"):
            cells = "  ".join(f"{p}: " + "/".join(f"{r[name][p][lv]['coverage']:.2f}" for lv in CHI2) for p in PRESETS)
            print(f"  {name:11s} {cells}")
        print(f"  H2 (primary, per-preset scale) holds: {r['H2_primary_holds']}")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
