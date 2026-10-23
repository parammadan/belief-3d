"""Symmetry-aware pose metrics (evaluator side).

Rotation error for a symmetric object is the minimum over its proper symmetry
group G:  e_R = min_{S in G} angle(R_est, R_gt S).

* box, three distinct extents: D2 = {I, Rx(pi), Ry(pi), Rz(pi)}, 4 elements
  (the full 8-element cuboid group includes reflections, which are not rotations);
* box with two equal extents: D4 about the odd axis, 8 elements; all equal: 24;
* cylinder: continuous rotation about local z plus a pi flip about x, so
  e_R = angle between axes, minimised over the sign;
* sphere: rotation is unobservable, so e_R is undefined (NaN) and only
  translation is scored.

ADD-S is reported for all shapes and needs no group. It is computed *exactly*
for primitives as the mean |sdf_gt| of estimated-pose surface points, i.e. the
distance to the true surface rather than to a finite GT point sample. The
sampled version has a floor of ~2.3 mm at zero pose error (measured) whenever
the estimate is a symmetric flip, because the two point samples interleave.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree

from data.schemas.objects import ObjectSpec
from perception.geometry import se3
from perception.pose.primitive_fit import sdf


def box_symmetry_group(extents, tol: float = 1e-6) -> list[np.ndarray]:
    e = np.asarray(extents, float)
    # All 24 proper rotations of the cube; keep those that map the box onto itself.
    cube = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                M[i, j] = s
            if np.linalg.det(M) > 0:
                cube.append(M)
    return [M for M in cube if np.allclose(np.abs(M) @ e, e, atol=tol)]


def rotation_error(spec: ObjectSpec, R_est: np.ndarray, R_gt: np.ndarray) -> float:
    if spec.shape == "sphere":
        return float("nan")
    if spec.shape == "cylinder":
        c = abs(float(R_est[:, 2] @ R_gt[:, 2]))
        return float(np.arccos(np.clip(c, -1.0, 1.0)))
    return min(se3.geodesic_distance(R_est, R_gt @ S) for S in box_symmetry_group(spec.size))


def sample_surface(spec: ObjectSpec, n: int = 500, seed: int = 0) -> np.ndarray:
    """Roughly uniform points on the primitive surface (object frame)."""
    rng = np.random.default_rng(seed)
    if spec.shape == "sphere":
        v = rng.normal(size=(n, 3))
        return spec.size[0] * v / np.linalg.norm(v, axis=1, keepdims=True)
    if spec.shape == "box":
        e = np.asarray(spec.size)
        areas = np.array([e[1] * e[2], e[0] * e[2], e[0] * e[1]]) * 2
        face_axis = rng.choice(3, size=n, p=areas / areas.sum())
        P = rng.uniform(-e, e, size=(n, 3))
        P[np.arange(n), face_axis] = e[face_axis] * rng.choice([-1.0, 1.0], size=n)
        return P
    r, hh = spec.size
    side, cap = 2 * np.pi * r * 2 * hh, 2 * np.pi * r * r
    on_side = rng.random(n) < side / (side + cap)
    th = rng.uniform(0, 2 * np.pi, n)
    rad = np.where(on_side, r, r * np.sqrt(rng.random(n)))
    z = np.where(on_side, rng.uniform(-hh, hh, n), hh * rng.choice([-1.0, 1.0], n))
    return np.column_stack([rad * np.cos(th), rad * np.sin(th), z])


def add_s(spec: ObjectSpec, T_est: np.ndarray, T_gt: np.ndarray, n: int = 1000) -> float:
    """Mean distance (m) from estimated-pose surface points to the true surface."""
    A_W = se3.transform_points(T_est, sample_surface(spec, n))
    return float(np.abs(sdf(spec, se3.transform_points(se3.inv_T(T_gt), A_W))).mean())


def add_s_sampled(spec: ObjectSpec, T_est: np.ndarray, T_gt: np.ndarray, n: int = 500) -> float:
    """Classic point-sample ADD-S (kept for comparability with published numbers)."""
    M = sample_surface(spec, n)
    return float(cKDTree(se3.transform_points(T_gt, M)).query(se3.transform_points(T_est, M))[0].mean())


def pose_errors(spec: ObjectSpec, T_est: np.ndarray, T_gt: np.ndarray) -> dict[str, float]:
    return {
        "t_err": float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])),
        "r_err_deg": float(np.degrees(rotation_error(spec, T_est[:3, :3], T_gt[:3, :3]))),
        "add_s": add_s(spec, T_est, T_gt),
    }
