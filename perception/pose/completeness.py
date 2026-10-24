"""Model-based completeness of a detection: observed / expected pixels.

Given the known primitive and the estimated pose, sample its surface, keep
camera-facing points (outward normal from the analytic SDF gradient), project
them, and count distinct pixels. That is the area the object *would* cover if
unoccluded. completeness = observed pixels / expected pixels, clipped to [0, 1].
Low completeness flags a partial view (occlusion, frame edge, dropout), where a
single-view fit is unreliable.

Uses only the catalog model, the estimate and the camera model, so it is
valid at inference.
"""
from __future__ import annotations

import numpy as np

from data.schemas.objects import ObjectSpec
from perception.geometry import se3
from perception.geometry.camera import PinholeIntrinsics
from perception.pose.primitive_fit import sdf


def _sample_surface(spec: ObjectSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    # Area-weighted sampling, same scheme as the evaluator's (kept separate so
    # inference never imports benchmarks).
    if spec.shape == "sphere":
        v = rng.normal(size=(n, 3))
        return spec.size[0] * v / np.linalg.norm(v, axis=1, keepdims=True)
    if spec.shape == "box":
        e = np.asarray(spec.size)
        areas = np.array([e[1] * e[2], e[0] * e[2], e[0] * e[1]])
        ax = rng.choice(3, size=n, p=areas / areas.sum())
        P = rng.uniform(-e, e, size=(n, 3))
        P[np.arange(n), ax] = e[ax] * rng.choice([-1.0, 1.0], size=n)
        return P
    r, hh = spec.size
    side, cap = 2 * np.pi * r * 2 * hh, 2 * np.pi * r * r
    on_side = rng.random(n) < side / (side + cap)
    th = rng.uniform(0, 2 * np.pi, n)
    rad = np.where(on_side, r, r * np.sqrt(rng.random(n)))
    z = np.where(on_side, rng.uniform(-hh, hh, n), hh * rng.choice([-1.0, 1.0], n))
    return np.column_stack([rad * np.cos(th), rad * np.sin(th), z])


def _normals(spec: ObjectSpec, P: np.ndarray, h: float = 1e-5) -> np.ndarray:
    g = np.column_stack([(sdf(spec, P + h * e) - sdf(spec, P - h * e)) / (2 * h) for e in np.eye(3)])
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)


def expected_pixels(spec: ObjectSpec, T_WO: np.ndarray, T_WC: np.ndarray, K: PinholeIntrinsics,
                    n: int = 12000, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    P_O = _sample_surface(spec, n, rng)
    N_O = _normals(spec, P_O)
    T_CO = se3.inv_T(T_WC) @ T_WO
    P_C = se3.transform_points(T_CO, P_O)
    N_C = N_O @ T_CO[:3, :3].T
    facing = np.einsum("ij,ij->i", N_C, P_C) < 0.0          # normal points toward the camera
    uv, front = K.project(P_C[facing])
    uv = uv[front]
    uv = uv[K.in_image(uv)]
    # A rasteriser counts a pixel when its centre is covered. Rounding every
    # point to its nearest pixel also counts boundary pixels the silhouette
    # only grazes (+12-18 % measured), so only points near a pixel centre vote.
    idx = np.round(uv)
    near = np.all(np.abs(uv - idx) < 0.35, axis=1)
    if not near.any():
        return 0
    return len(np.unique(idx[near].astype(int), axis=0))


def completeness(observed_pixels: int, spec: ObjectSpec, T_WO: np.ndarray, T_WC: np.ndarray,
                 K: PinholeIntrinsics) -> float:
    exp = expected_pixels(spec, T_WO, T_WC, K)
    return float(np.clip(observed_pixels / exp, 0.0, 1.0)) if exp else 0.0
