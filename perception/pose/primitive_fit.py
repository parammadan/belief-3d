"""Level-1 geometric 6D pose: fit a known primitive to segmented depth points.

For object frame O (centred on the primitive), the analytic signed distance
sdf(p_O) is zero on the surface. Given world points p_i of one segment, solve

    T_WO* = argmin_T  sum_i rho( sdf( T^{-1} p_i ) )

with a robust (soft-L1) loss, parametrised as R = R0 exp(w), t = t0 + dt and
solved with scipy's trust-region least squares. Several initialisations
(which local axis points up, two yaw options) cover the discrete ambiguities.
The lowest robust cost wins.

Only visible surface points enter the cost. A partial view leaves some
directions weakly constrained, and that shows up in `rms` and `n_points`,
which the tracker uses for measurement noise.

Support prior (optional, on by default): objects at rest touch the support
plane, so one extra residual sqrt(N) * (f_scale / support_sigma) * gap
penalises the gap between the model's lowest point and the plane. Without
it, box_red (half-extents 30/20/25 mm) is ambiguous between "z up" and "y up"
under ~2 mm depth noise: both fit the visible points with RMS ~1.6 mm, and the
wrong one floats ~1 cm. This was measured on val seeds (8 of 381 detections).
**Assumption:** this is wrong for stacked, lifted or tipped-on-edge objects.
Disable it with support_sigma=None.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from data.schemas.objects import ObjectSpec
from perception.geometry import se3


def sdf(spec: ObjectSpec, P: np.ndarray) -> np.ndarray:
    """Signed distance (m) of object-frame points to the primitive surface."""
    if spec.shape == "sphere":
        return np.linalg.norm(P, axis=1) - spec.size[0]
    if spec.shape == "box":
        q = np.abs(P) - np.asarray(spec.size)
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
    if spec.shape == "cylinder":
        r, hh = spec.size
        d = np.column_stack([np.linalg.norm(P[:, :2], axis=1) - r, np.abs(P[:, 2]) - hh])
        return np.minimum(d.max(axis=1), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=1)
    if spec.shape == "mug":
        r, hh, hl, hw, hz = spec.size
        d = np.column_stack([np.linalg.norm(P[:, :2], axis=1) - r, np.abs(P[:, 2]) - hh])
        body = np.minimum(d.max(axis=1), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=1)
        q = np.abs(P - np.array([r + hl / 2, 0.0, 0.0])) - np.array([hl / 2, hw, hz])
        handle = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
        return np.minimum(body, handle)
    raise ValueError(spec.shape)


def lowest_point_height(spec: ObjectSpec, R: np.ndarray, t: np.ndarray, n: np.ndarray, d: float) -> float:
    """Height above plane (n.p + d = 0) of the primitive's lowest point."""
    centre = float(n @ t + d)
    a = R.T @ n                                  # plane normal in object frame
    if spec.shape == "sphere":
        return centre - spec.size[0]
    if spec.shape == "box":
        return centre - float(np.abs(a) @ np.asarray(spec.size))
    r, hh = spec.size[:2]                        # mug: the body bounds the lowest point when upright
    return centre - (r * float(np.sqrt(max(0.0, 1.0 - a[2] ** 2))) + hh * abs(float(a[2])))


def half_extents(spec: ObjectSpec) -> np.ndarray:
    if spec.shape == "sphere":
        return np.full(3, spec.size[0])
    if spec.shape == "box":
        return np.asarray(spec.size, float)
    return np.array([spec.size[0], spec.size[0], spec.size[1]])     # cylinder; mug body


@dataclass(frozen=True)
class PoseEstimate:
    spec_name: str
    T_WO: np.ndarray
    rms: float              # m, RMS of sdf residuals at the solution
    n_points: int
    cost: float


def _frame_with_up(up_local: int, n: np.ndarray, e1: np.ndarray, yaw_swap: bool) -> np.ndarray:
    """R_WO whose local axis `up_local` maps to world normal n, and whose next
    local axis maps to the in-plane direction e1 (or e1 rotated 90 deg)."""
    e1 = e1 - n * (e1 @ n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    if yaw_swap:
        e1, e2 = e2, -e1
    a, b = (up_local + 1) % 3, (up_local + 2) % 3
    R = np.zeros((3, 3))
    R[:, up_local], R[:, a], R[:, b] = n, e1, e2
    return R


def initialisations(spec: ObjectSpec, P: np.ndarray, plane_n: np.ndarray, plane_d: float) -> list[np.ndarray]:
    c = P.mean(axis=0)
    Q = P - c
    Q -= np.outer(Q @ plane_n, plane_n)
    e1 = np.linalg.svd(Q, full_matrices=False)[2][0] if len(P) >= 3 else np.array([1.0, 0.0, 0.0])
    if abs(e1 @ plane_n) > 0.99:
        e1 = np.array([1.0, 0.0, 0.0])
    c_on_plane = c - plane_n * (plane_n @ c + plane_d)
    he = half_extents(spec)
    if spec.shape == "mug":
        # Upright; the handle direction is unknown, so 8 yaw hypotheses about the up axis.
        inits = []
        for k in range(8):
            yaw = 2 * np.pi * k / 8
            e = np.cos(yaw) * e1 + np.sin(yaw) * np.cross(plane_n, e1)
            inits.append(se3.make_T(_frame_with_up(2, plane_n, e, False), c_on_plane + plane_n * he[2]))
        return inits
    ups = {"sphere": [2], "cylinder": [2, 0], "box": [0, 1, 2]}[spec.shape]
    inits = []
    for up in ups:
        for swap in ((False,) if spec.shape == "sphere" else (False, True)):
            R0 = _frame_with_up(up, plane_n, e1, swap)
            inits.append(se3.make_T(R0, c_on_plane + plane_n * he[up]))
    return inits


def fit_hypotheses(
    spec: ObjectSpec,
    P_W: np.ndarray,
    plane_n: np.ndarray,
    plane_d: float,
    noise_sigma: float = 0.0,
    max_points: int = 400,
    seed: int = 0,
    support_sigma: float | None = 0.005,
    extra_inits: list[np.ndarray] | None = None,
) -> list[PoseEstimate]:
    """All local optima from the initialisations, best (lowest robust cost) first.

    Near-ties are exactly the ambiguities a belief should keep. fit_primitive
    returns only the first of these. `extra_inits` *replaces* the geometric
    initialisations, e.g. with neural proposals (proposal + refinement)."""
    if len(P_W) < 10:
        raise ValueError("too few points to fit")
    rng = np.random.default_rng(seed)
    P = P_W[rng.choice(len(P_W), max_points, replace=False)] if len(P_W) > max_points else P_W
    f_scale = max(0.002, 2.0 * noise_sigma)
    out: list[PoseEstimate] = []
    inits = initialisations(spec, P, plane_n, plane_d) if extra_inits is None else list(extra_inits)
    for T0 in inits:
        R0, t0 = T0[:3, :3], T0[:3, 3]
        w_sup = np.sqrt(len(P)) * f_scale / support_sigma if support_sigma else 0.0

        def resid(x, R0=R0, t0=t0, w_sup=w_sup):
            R = R0 @ se3.so3_exp(x[:3])
            r = sdf(spec, (P - (t0 + x[3:])) @ R)             # rows: R^T (p - t)
            if not w_sup:
                return r
            return np.append(r, w_sup * lowest_point_height(spec, R, t0 + x[3:], plane_n, plane_d))

        sol = least_squares(resid, np.zeros(6), loss="soft_l1", f_scale=f_scale, x_scale=[0.1] * 3 + [0.01] * 3,
                            max_nfev=200)
        T = se3.make_T(R0 @ se3.so3_exp(sol.x[:3]), t0 + sol.x[3:])
        r = resid(sol.x)[: len(P)]
        out.append(PoseEstimate(spec.name, T, float(np.sqrt(np.mean(r**2))), len(P_W), float(sol.cost)))
    return sorted(out, key=lambda e: e.cost)


def fit_primitive(
    spec: ObjectSpec,
    P_W: np.ndarray,
    plane_n: np.ndarray,
    plane_d: float,
    noise_sigma: float = 0.0,
    max_points: int = 400,
    seed: int = 0,
    support_sigma: float | None = 0.005,
) -> PoseEstimate:
    return fit_hypotheses(spec, P_W, plane_n, plane_d, noise_sigma, max_points, seed, support_sigma)[0]
