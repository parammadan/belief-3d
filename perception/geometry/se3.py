"""SO(3)/SE(3) primitives — the CPU reference implementation.

Conventions (see docs/coordinate_frames.md):
  * T_AB is a 4x4 homogeneous matrix mapping coordinates expressed in frame B
    into frame A:  p_A = T_AB @ p_B.  Composition: T_AC = T_AB @ T_BC.
  * Quaternions are scalar-first [w, x, y, z] (MuJoCo's convention), unit norm,
    canonicalised to w >= 0 when returned.
  * Translations are in meters, angles in radians.
"""
from __future__ import annotations

import numpy as np

_EPS_SMALL = 1e-8      # below this angle use the first-order series
_EPS_NEAR_PI = 1e-4    # within this of pi use the symmetric-part branch


def hat(w: np.ndarray) -> np.ndarray:
    """so(3) hat operator: R^3 -> skew-symmetric 3x3."""
    x, y, z = np.asarray(w, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(W: np.ndarray) -> np.ndarray:
    """Inverse of hat (uses the antisymmetric part, so it tolerates noise)."""
    return 0.5 * np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]])


def is_rotation(R: np.ndarray, atol: float = 1e-6) -> bool:
    R = np.asarray(R, dtype=float)
    return (
        R.shape == (3, 3)
        and np.allclose(R.T @ R, np.eye(3), atol=atol)
        and abs(np.linalg.det(R) - 1.0) < atol
    )


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """Nearest rotation in Frobenius norm (SVD with det correction)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: axis-angle vector (radians) -> rotation matrix."""
    w = np.asarray(w, dtype=float)
    theta = np.linalg.norm(w)
    K = hat(w)
    if theta < _EPS_SMALL:
        return np.eye(3) + K + 0.5 * K @ K
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * K
        + ((1.0 - np.cos(theta)) / theta**2) * K @ K
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector with angle in [0, pi].

    Handles both degenerate regimes: theta ~ 0 (series) and theta ~ pi, where
    the antisymmetric part vanishes and the axis is recovered from the
    symmetric part (R + R^T)/2 - cos(theta) I = (1 - cos(theta)) a a^T.
    """
    R = np.asarray(R, dtype=float)
    v = vee(R)                      # = sin(theta) * a
    s = np.linalg.norm(v)
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arctan2(s, c)
    if theta < _EPS_SMALL:
        return v                    # first order: R ~ I + hat(w)
    if np.pi - theta > _EPS_NEAR_PI:
        return (theta / s) * v
    S = 0.5 * (R + R.T) - c * np.eye(3)
    S /= (1.0 - c)
    k = int(np.argmax(np.diag(S)))
    a = S[:, k] / np.sqrt(max(S[k, k], 1e-300))
    if np.dot(a, v) < 0.0:          # sin(theta) > 0 fixes the sign
        a = -a
    return theta * a / np.linalg.norm(a)


def geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angle (radians, [0, pi]) of the relative rotation R1^T R2.

    atan2 form is accurate near 0 and pi, unlike arccos((tr - 1)/2).
    """
    Rr = np.asarray(R1, dtype=float).T @ np.asarray(R2, dtype=float)
    s = np.linalg.norm(vee(Rr))
    c = np.clip((np.trace(Rr) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arctan2(s, c))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError(f"cannot normalise quaternion with norm {n}")
    q = q / n
    return q if q[0] >= 0.0 else -q


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] -> rotation matrix (input is normalised first)."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> [w, x, y, z] (Shepperd's method, w >= 0)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        s = 2.0 * np.sqrt(1.0 + tr)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return quat_normalize(np.array(q))


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse: T_BA from T_AB."""
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T_AB: np.ndarray, P_B: np.ndarray) -> np.ndarray:
    """Map (N, 3) points expressed in B into A."""
    P_B = np.asarray(P_B, dtype=float).reshape(-1, 3)
    return P_B @ T_AB[:3, :3].T + T_AB[:3, 3]


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = (0.0, 0.0, 1.0)) -> np.ndarray:
    """T_WC for an OpenCV-convention camera (x right, y down, z forward)."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-9:            # looking along up: pick any x
        x = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.column_stack([x, y, z]), eye)
