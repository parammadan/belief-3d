"""Proper symmetry groups of catalog primitives (inference-side copy).

The evaluator keeps its own implementation (benchmarks/metrics.py), so that
inference and scoring stay independent. A contract test checks that they agree.

* box: proper rotations of the cube that map the extents onto themselves (D2 / D4 / O)
* cylinder: continuous about local z; represented by a discretised set plus the pi flip
* sphere: continuous SO(3); represented by the identity only (rotation is treated as
  unobservable downstream)
"""
from __future__ import annotations

import itertools

import numpy as np

from data.schemas.objects import ObjectSpec
from perception.geometry import se3


def _cube_rotations() -> list[np.ndarray]:
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                M[i, j] = s
            if np.linalg.det(M) > 0:
                out.append(M)
    return out


def symmetry_group(spec: ObjectSpec, n_cyl: int = 12) -> np.ndarray:
    """(G, 3, 3) array of rotations S with model(R S) == model(R)."""
    if spec.shape == "box":
        e = np.asarray(spec.size, float)
        return np.array([M for M in _cube_rotations() if np.allclose(np.abs(M) @ e, e, atol=1e-9)])
    if spec.shape == "cylinder":
        spins = [se3.so3_exp([0.0, 0.0, 2 * np.pi * k / n_cyl]) for k in range(n_cyl)]
        flip = se3.so3_exp([np.pi, 0.0, 0.0])
        return np.array(spins + [S @ flip for S in spins])
    return np.eye(3)[None]


def sym_distance(spec: ObjectSpec, R1: np.ndarray, R2: np.ndarray) -> float:
    """Symmetry-aware geodesic distance (rad). Cylinders use the exact axis angle."""
    if spec.shape == "sphere":
        return 0.0
    if spec.shape == "cylinder":
        return float(np.arccos(np.clip(abs(R1[:, 2] @ R2[:, 2]), -1.0, 1.0)))
    return min(se3.geodesic_distance(R1, R2 @ S) for S in symmetry_group(spec))


def align_to(spec: ObjectSpec, R_ref: np.ndarray, Rs: np.ndarray) -> np.ndarray:
    """For each R in Rs (N,3,3), the symmetric equivalent R S closest to R_ref."""
    G = symmetry_group(spec)
    cand = np.einsum("nij,gjk->ngik", Rs, G)                      # (N, G, 3, 3)
    tr = np.einsum("ij,ngij->ng", R_ref, cand)                    # tr(R_ref^T C)
    best = np.argmax(tr, axis=1)
    return cand[np.arange(len(Rs)), best]
