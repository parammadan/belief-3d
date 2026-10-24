"""Online extrinsic self-calibration of one camera from static, repeatedly observed objects.

Model. The camera's extrinsic error is a fixed dT = exp(xi), xi = (omega, tau), in the
camera frame: the reported pose is T_rep = T_true dT. A static point p seen from view v is
therefore reported at  p_hat = T_v dT T_v^-1 p,  which to first order in xi is

    p_hat_kv = p_k + R_v ( -[q_kv]x omega + tau ),     q_kv = R_v^T (p_hat_kv - c_v)

(q_kv: the point in the camera frame; c_v, R_v: the view's reported centre and rotation).
The bias R_v(...) changes with the view, so observations of the same objects from diverse views
identify xi. Unknowns xi (6) and the object positions p_k (3K) are solved jointly as a linear
Gaussian MAP problem, with prior xi ~ N(0, diag(sigma_rot^2 I, sigma_t^2 I)) (the calibration
spec) and isotropic measurement noise sigma_m. With low view diversity the prior dominates and
the correction shrinks towards 0 (fails safe).

Inputs are reported view poses and detection centres only (no ground truth).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from perception.geometry import se3


def _hat(w: np.ndarray) -> np.ndarray:
    return se3.hat(w)


@dataclass
class SelfCalibrator:
    sigma_rot: float = float(np.radians(1.0))   # rad per axis (prior)
    sigma_t: float = 0.005                      # m per axis (prior)
    sigma_m: float = 0.003                      # m, detection-centre noise
    obs: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)   # (key, T_WC_rep, p_hat)
    xi: np.ndarray = field(default_factory=lambda: np.zeros(6))
    cov: np.ndarray | None = None

    def add(self, key: int, T_WC_rep: np.ndarray, p_hat: np.ndarray) -> None:
        """One observation of static object `key` (e.g. a track id) from a reported camera pose."""
        self.obs.append((int(key), np.asarray(T_WC_rep, float), np.asarray(p_hat, float)))

    def solve(self) -> np.ndarray:
        keys = sorted({k for k, _, _ in self.obs})
        if not keys:
            return self.xi
        col = {k: 6 + 3 * i for i, k in enumerate(keys)}
        n = 6 + 3 * len(keys)
        rows, rhs = [], []
        for k, T, ph in self.obs:
            R, c = T[:3, :3], T[:3, 3]
            q = R.T @ (ph - c)
            A = np.zeros((3, n))
            A[:, 0:3] = -R @ _hat(q)
            A[:, 3:6] = R
            A[:, col[k]:col[k] + 3] = np.eye(3)
            rows.append(A / self.sigma_m)
            rhs.append(ph / self.sigma_m)
        A = np.vstack(rows)
        b = np.concatenate(rhs)
        # Prior on xi only; object positions get a very weak prior for conditioning.
        P = np.zeros((n, n))
        P[0:3, 0:3] = np.eye(3) / self.sigma_rot**2
        P[3:6, 3:6] = np.eye(3) / self.sigma_t**2
        P[6:, 6:] = np.eye(n - 6) * 1e-6
        H = A.T @ A + P
        x = np.linalg.solve(H, A.T @ b)
        self.xi = x[:6]
        self.cov = np.linalg.inv(H)[:6, :6]
        return self.xi

    def bias_at(self, p: np.ndarray, T_WC_rep: np.ndarray) -> np.ndarray:
        """Predicted reported-minus-true offset of point p from one view, under the current xi."""
        R, c = T_WC_rep[:3, :3], T_WC_rep[:3, 3]
        q = R.T @ (p - c)
        return R @ (-_hat(q) @ self.xi[:3] + self.xi[3:])

    def correct(self, p_hat_fused: np.ndarray, views: list[np.ndarray]) -> np.ndarray:
        """Remove the mean per-view bias from a position fused over `views` (reported poses)."""
        if not views:
            return np.asarray(p_hat_fused, float)
        b = np.mean([self.bias_at(p_hat_fused, T) for T in views], axis=0)
        return np.asarray(p_hat_fused, float) - b
