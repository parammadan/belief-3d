"""Constant-velocity Kalman filter over object position (unimodal reference, B1).

State x = [p (3), v (3)] in the world frame, meters and m/s. Process noise is
the discrete white-noise-acceleration model with spectral density q (m^2/s^3):

    F = [I  dt I; 0 I],   Q = q [dt^3/3 I, dt^2/2 I; dt^2/2 I, dt I]

Measurement z = p + noise, H = [I 0]. All updates use the Joseph form, which
keeps P symmetric positive definite.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

H = np.hstack([np.eye(3), np.zeros((3, 3))])


@dataclass
class CVKalman:
    x: np.ndarray
    P: np.ndarray
    q: float = 0.05
    t: float = 0.0
    nis_log: list[float] = field(default_factory=list)

    @staticmethod
    def init(p0: np.ndarray, R0: np.ndarray, t: float, v_sigma: float = 0.3, q: float = 0.05) -> CVKalman:
        P = np.zeros((6, 6))
        P[:3, :3] = R0
        P[3:, 3:] = np.eye(3) * v_sigma**2
        return CVKalman(np.concatenate([p0, np.zeros(3)]), P, q, t)

    def predicted(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        dt = t - self.t
        if dt < -1e-9:
            raise ValueError(f"non-causal predict: t={t} < filter time {self.t}")
        dt = max(dt, 0.0)
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        Q = self.q * np.block([[np.eye(3) * dt**3 / 3, np.eye(3) * dt**2 / 2],
                               [np.eye(3) * dt**2 / 2, np.eye(3) * dt]])
        return F @ self.x, F @ self.P @ F.T + Q

    def predict(self, t: float) -> None:
        self.x, self.P = self.predicted(t)
        self.t = max(self.t, t)

    def innovation(self, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return z - H @ self.x, H @ self.P @ H.T + R

    def mahalanobis2(self, z: np.ndarray, R: np.ndarray) -> float:
        y, S = self.innovation(z, R)
        return float(y @ np.linalg.solve(S, y))

    def update(self, z: np.ndarray, R: np.ndarray) -> float:
        y, S = self.innovation(z, R)
        K = self.P @ H.T @ np.linalg.inv(S)
        nis = float(y @ np.linalg.solve(S, y))
        self.x = self.x + K @ y
        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.nis_log.append(nis)
        return nis
