"""Gated optimal one-to-one assignment (Hungarian / Jonker-Volgenant via scipy)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

INFEASIBLE = 1e12


def assign(cost: np.ndarray, gate: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Minimum-cost matching of rows (tracks) to columns (detections).

    Pairs with cost > gate (or non-finite) are never matched. Returns
    (matches, unmatched_rows, unmatched_cols).
    """
    n_r, n_c = cost.shape
    if n_r == 0 or n_c == 0:
        return [], list(range(n_r)), list(range(n_c))
    C = np.where(np.isfinite(cost) & (cost <= gate), cost, INFEASIBLE)
    rows, cols = linear_sum_assignment(C)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if C[r, c] < INFEASIBLE]
    mr = {r for r, _ in matches}
    mc = {c for _, c in matches}
    return matches, [r for r in range(n_r) if r not in mr], [c for c in range(n_c) if c not in mc]
