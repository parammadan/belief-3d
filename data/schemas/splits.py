"""Seed-range splits (docs/research_question.md). Disjoint by construction."""
from __future__ import annotations

SPLITS: dict[str, tuple[int, int]] = {
    "train": (0, 1_000_000),
    "calib": (1_000_000, 1_100_000),
    "val": (1_500_000, 1_600_000),
    "test": (2_000_000, 3_000_000),
}


def split_of(seed: int) -> str:
    for name, (lo, hi) in SPLITS.items():
        if lo <= seed < hi:
            return name
    raise ValueError(f"seed {seed} is not in any split range")


def seeds_for(split: str, n: int, offset: int = 0) -> list[int]:
    lo, hi = SPLITS[split]
    if lo + offset + n > hi:
        raise ValueError(f"split {split!r} has only {hi - lo} seeds")
    return list(range(lo + offset, lo + offset + n))
