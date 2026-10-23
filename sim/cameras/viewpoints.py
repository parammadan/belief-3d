"""Finite candidate viewpoint set for the mobile camera (camera B).

Motion model: camera B rides an abstract gantry/second arm (not simulated as a
body) with bounded linear and angular speed. Moving from view i to view j takes

    t_ij = max(||p_i - p_j|| / v_max, angle(R_i, R_j) / w_max) + t_settle

and is allowed only if the straight-line path is clear of all scene geometry
(checked with mj_ray). Viewpoints are kept only if their own position is clear
and they have line of sight to the workspace centre.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from perception.geometry import se3
from sim.environments.tabletop import SPAWN_BOUNDS, TABLE_TOP_Z, TabletopScene


@dataclass(frozen=True)
class CameraMotionModel:
    v_max: float = 0.5        # m/s
    w_max: float = 1.0        # rad/s
    t_settle: float = 0.1     # s, vibration settling before capture


DEFAULT_MOTION = CameraMotionModel()


@dataclass(frozen=True)
class Viewpoint:
    index: int
    T_WC: np.ndarray          # OpenCV-convention camera pose
    azimuth: float            # rad
    elevation: float          # rad


def workspace_centre() -> np.ndarray:
    lo, hi = np.array(SPAWN_BOUNDS[0]), np.array(SPAWN_BOUNDS[1])
    return np.array([*(lo + hi) / 2.0, TABLE_TOP_Z])


def _ray_hit(model: mujoco.MjModel, data: mujoco.MjData, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from a toward b to the first geom hit (inf if none before b)."""
    vec = b - a
    length = float(np.linalg.norm(vec))
    geomid = np.array([-1], dtype=np.int32)
    # geomgroup None = all groups; flg_static=1 includes the table; bodyexclude=-1.
    dist = mujoco.mj_ray(model, data, a, vec / length, None, 1, -1, geomid)
    return dist if 0 <= dist <= length else np.inf


class ViewpointSet:
    def __init__(
        self,
        scene: TabletopScene,
        radius: float = 0.75,
        elevations_deg: tuple[float, ...] = (35.0, 60.0),
        n_azimuth: int = 8,
        motion: CameraMotionModel = DEFAULT_MOTION,
    ):
        self.scene, self.motion = scene, motion
        c = workspace_centre()
        views, rejected = [], []
        for el in np.radians(elevations_deg):
            for az in np.linspace(-np.pi, np.pi, n_azimuth, endpoint=False):
                eye = c + radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
                T = se3.look_at(eye, c)
                if self._line_of_sight(eye, c):
                    views.append(Viewpoint(len(views), T, float(az), float(el)))
                else:
                    rejected.append((float(az), float(el)))
        self.views: tuple[Viewpoint, ...] = tuple(views)
        self.rejected = tuple(rejected)

    def _line_of_sight(self, eye: np.ndarray, target: np.ndarray) -> bool:
        # Stop 3 cm short of the table centre so the table itself does not count.
        d = target - eye
        stop = target - 0.03 * d / np.linalg.norm(d) + np.array([0.0, 0.0, 0.03])
        return _ray_hit(self.scene.model, self.scene.data, eye, stop) == np.inf

    def __len__(self) -> int:
        return len(self.views)

    def path_clear(self, i: int, j: int) -> bool:
        a, b = self.views[i].T_WC[:3, 3], self.views[j].T_WC[:3, 3]
        return i == j or (_ray_hit(self.scene.model, self.scene.data, a, b) == np.inf
                          and _ray_hit(self.scene.model, self.scene.data, b, a) == np.inf)

    def travel_time(self, i: int, j: int) -> float:
        """Seconds to move i -> j, or inf if the straight path is blocked."""
        if i == j:
            return 0.0
        if not self.path_clear(i, j):
            return np.inf
        Ti, Tj = self.views[i].T_WC, self.views[j].T_WC
        m = self.motion
        return max(
            float(np.linalg.norm(Ti[:3, 3] - Tj[:3, 3])) / m.v_max,
            se3.geodesic_distance(Ti[:3, :3], Tj[:3, :3]) / m.w_max,
        ) + m.t_settle

    def move_camera(self, j: int) -> None:
        self.scene.set_mobile_camera(self.views[j].T_WC)
