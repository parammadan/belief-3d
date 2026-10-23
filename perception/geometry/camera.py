"""Pinhole camera model in the OpenCV convention.

Camera frame C: x right, y down, z forward (optical axis). Pixel (u, v) has
its *centre* at integer coordinates, u along columns, v along rows, origin at
the top-left pixel centre. Depth images store z_C in meters (not ray range);
non-finite or <= 0 values mean "no return".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def project(self, P_C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) camera-frame points -> (N, 2) pixels and an in-front mask."""
        P_C = np.asarray(P_C, dtype=float).reshape(-1, 3)
        z = P_C[:, 2]
        front = z > 1e-9
        zs = np.where(front, z, np.nan)
        uv = np.column_stack([self.fx * P_C[:, 0] / zs + self.cx, self.fy * P_C[:, 1] / zs + self.cy])
        return uv, front

    def in_image(self, uv: np.ndarray) -> np.ndarray:
        u, v = uv[:, 0], uv[:, 1]
        return (u > -0.5) & (u < self.width - 0.5) & (v > -0.5) & (v < self.height - 0.5)

    def backproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Pixels + z-depth -> (N, 3) camera-frame points."""
        u, v, z = (np.asarray(a, dtype=float).ravel() for a in (u, v, z))
        return np.column_stack([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z])

    def depth_to_points(self, depth: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Depth image -> (N, 3) points in C and their (N, 2) integer pixels.

        Invalid depth (non-finite or <= 0) is dropped, as are pixels outside
        an optional boolean mask.
        """
        if depth.shape != (self.height, self.width):
            raise ValueError(f"depth shape {depth.shape} != {(self.height, self.width)}")
        valid = np.isfinite(depth) & (depth > 0.0)
        if mask is not None:
            valid &= mask
        v, u = np.nonzero(valid)
        return self.backproject(u, v, depth[v, u]), np.column_stack([u, v])
