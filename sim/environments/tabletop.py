"""Minimal MuJoCo tabletop scene (M0): table, primitive objects, two RGB-D cameras.

Camera A ("overhead") is fixed. Camera B ("mobile") hangs from a mocap body so
its pose can be commanded. No robot yet — the arm arrives in M1.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from perception.geometry import se3

TABLE_TOP_Z = 0.40          # meters, world frame
TABLE_HALF = (0.45, 0.35)   # half-extents x, y


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    shape: str               # "box" | "cylinder" | "sphere"
    size: tuple[float, ...]  # MuJoCo geom size (half-extents / radius, half-height)
    rgba: tuple[float, float, float, float]
    symmetry: str


DEFAULT_OBJECTS: tuple[ObjectSpec, ...] = (
    ObjectSpec("box_red", "box", (0.030, 0.020, 0.025), (0.85, 0.15, 0.15, 1), "box"),
    ObjectSpec("cyl_green", "cylinder", (0.025, 0.035), (0.15, 0.75, 0.25, 1), "cylindrical_z"),
    ObjectSpec("sphere_blue", "sphere", (0.028,), (0.15, 0.30, 0.85, 1), "spherical"),
    ObjectSpec("box_yellow", "box", (0.040, 0.015, 0.015), (0.90, 0.80, 0.10, 1), "box"),
)


def _object_xml(o: ObjectSpec) -> str:
    size = " ".join(f"{s:.4f}" for s in o.size)
    rgba = " ".join(f"{c:.3f}" for c in o.rgba)
    return (
        f'<body name="{o.name}" pos="0 0 1">'
        f'<freejoint name="{o.name}_joint"/>'
        f'<geom name="{o.name}_geom" type="{o.shape}" size="{size}" rgba="{rgba}" '
        f'mass="0.1" friction="1 0.005 0.0001"/></body>'
    )


def build_xml(objects: tuple[ObjectSpec, ...] = DEFAULT_OBJECTS, width: int = 320, height: int = 240) -> str:
    hx, hy = TABLE_HALF
    return f"""
<mujoco model="belief3d_tabletop">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="{width}" offheight="{height}"/>
    <map znear="0.01" zfar="50"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
  </visual>
  <worldbody>
    <light name="key" pos="0.3 -0.3 2.0" dir="-0.15 0.15 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.35 0.35 0.38 1"/>
    <geom name="table" type="box" pos="0 0 {TABLE_TOP_Z / 2:.4f}" size="{hx} {hy} {TABLE_TOP_Z / 2:.4f}" rgba="0.62 0.55 0.45 1"/>
    <camera name="overhead" pos="0 0 {TABLE_TOP_Z + 0.85:.3f}" fovy="58"/>
    <body name="mobile_cam_mount" mocap="true" pos="0.5 -0.5 1.0">
      <camera name="mobile" fovy="58"/>
    </body>
    {''.join(_object_xml(o) for o in objects)}
  </worldbody>
</mujoco>
"""


# MuJoCo cameras look along -z with +y up (OpenGL). OpenCV frame = GL frame
# rotated pi about x: C_cv = C_gl @ diag(1, -1, -1).
R_GL_CV = np.diag([1.0, -1.0, -1.0])


class TabletopScene:
    def __init__(self, objects: tuple[ObjectSpec, ...] = DEFAULT_OBJECTS, width: int = 320, height: int = 240):
        self.objects = objects
        self.width, self.height = width, height
        self.model = mujoco.MjModel.from_xml_string(build_xml(objects, width, height))
        self.data = mujoco.MjData(self.model)
        self.body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o.name) for o in objects]
        self._qpos_adr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{o.name}_joint")]
            for o in objects
        ]
        self._mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mobile_cam_mount")]
        self._renderer: mujoco.Renderer | None = None

    @property
    def renderer(self) -> mujoco.Renderer:
        """One GL context per scene: a second Renderer on macOS CGL returned
        corrupt segmentation buffers (observed 2026-10-23), so sensors share it."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        return self._renderer

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def reset(self, seed: int, settle_steps: int = 400) -> None:
        """Deterministic reset: non-overlapping random placements, then settle."""
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        placed: list[np.ndarray] = []
        hx, hy = TABLE_HALF
        for adr in self._qpos_adr:
            for _ in range(200):
                xy = rng.uniform([-hx * 0.6, -hy * 0.6], [hx * 0.6, hy * 0.6])
                if all(np.linalg.norm(xy - p) > 0.11 for p in placed):
                    break
            placed.append(xy)
            yaw = rng.uniform(-np.pi, np.pi)
            self.data.qpos[adr:adr + 3] = [xy[0], xy[1], TABLE_TOP_Z + 0.06]
            self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.set_mobile_camera(se3.look_at([0.45, -0.45, TABLE_TOP_Z + 0.55], [0.0, 0.0, TABLE_TOP_Z]))
        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

    def set_mobile_camera(self, T_WC: np.ndarray) -> None:
        """Command camera B with an OpenCV-convention pose T_WC."""
        R_W_gl = T_WC[:3, :3] @ R_GL_CV.T
        self.data.mocap_pos[self._mocap_id] = T_WC[:3, 3]
        self.data.mocap_quat[self._mocap_id] = se3.rot_to_quat(R_W_gl)
        mujoco.mj_forward(self.model, self.data)
