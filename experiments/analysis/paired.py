"""Paired comparison of per-episode MOTA between two eval reports (same episodes).

Paired bootstrap 95 % CI of the mean difference (B - A) plus an exact two-sided sign test.
Usage:  python experiments/analysis/paired.py A.json B.json
"""
from __future__ import annotations

import json
import sys
from math import comb

import numpy as np


def paired(a: list[float], b: list[float], n_boot: int = 10000, seed: int = 0) -> dict:
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    pos, neg = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    n = pos + neg
    k = min(pos, neg)
    p_sign = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"n_episodes": len(d), "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "episodes_better": pos, "episodes_worse": neg, "sign_test_p": p_sign}


if __name__ == "__main__":
    from pathlib import Path
    A, B = (json.loads(Path(f).read_text())["tracking"]["MOTA_per_episode"] for f in sys.argv[1:3])
    print(json.dumps(paired(A, B)))
