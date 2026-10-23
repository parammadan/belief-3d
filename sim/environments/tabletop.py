"""Minimal MuJoCo tabletop scene (M0): table, primitive objects, two RGB-D cameras.

Camera A ("overhead") is fixed. Camera B ("mobile") hangs from a mocap body so
its pose can be commanded. With `with_robot=True` a Franka Panda (MuJoCo
Menagerie, Apache-2.0, fetched by scripts/fetch_assets.py) is attached on the
table's -x edge, with a `panda/tcp` site 0.1034 m past the flange.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from data.schemas.objects import CATALOG, DEFAULT_OBJECTS, ObjectSpec  # noqa: F401  (re-exported)
from perception.geometry import se3

TABLE_TOP_Z = 0.40          # meters, world frame
TABLE_HALF = (0.45, 0.35)   # half-extents x, y
ROBOT_BASE_POS = (-0.40, 0.0, TABLE_TOP_Z)
# Objects are spawned in this table region (world x, y bounds): inside the
# Panda's top-down workspace and away from its base.
SPAWN_BOUNDS = ((-0.12, -0.22), (0.28, 0.22))
PANDA_XML = Path(__file__).resolve().parents[1] / "assets/third_party/franka_emika_panda/panda.xml"
# Hovering, contact-free plate used to create controlled occlusion intervals.
OCCLUDER_PARK = (2.0, 2.0, 0.5)
# Geom groups: 0 static scene, 2/3 Panda visual/collision (Menagerie),
# 4 objects, 5 reserved for rendering one object in isolation.
OBJECT_GROUP, ISOLATE_GROUP = 4, 5






def _object_xml(o: ObjectSpec) -> str:
    size = " ".join(f"{s:.4f}" for s in o.size)
    rgba = " ".join(f"{c:.3f}" for c in o.rgba)
    return (
        f'<body name="{o.name}" pos="0 0 1">'
        f'<freejoint name="{o.name}_joint"/>'
        f'<geom name="{o.name}_geom" type="{o.shape}" size="{size}" rgba="{rgba}" group="{OBJECT_GROUP}" '
        f'mass="0.1" friction="1 0.005 0.0001"/></body>'
    )


def build_xml(objects: tuple[ObjectSpec, ...] = DEFAULT_OBJECTS, width: int = 320, height: int = 240) -> str:
    hx, hy = TABLE_HALF
    return f"""
<mujoco model="belief3d_tabletop">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
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
    <body name="occluder" mocap="true" pos="{OCCLUDER_PARK[0]} {OCCLUDER_PARK[1]} {OCCLUDER_PARK[2]}">
      <geom name="occluder_geom" type="box" size="0.07 0.07 0.004" rgba="0.45 0.45 0.5 1" contype="0" conaffinity="0"/>
    </body>
    {''.join(_object_xml(o) for o in objects)}
  </worldbody>
</mujoco>
"""


# MuJoCo cameras look along -z with +y up (OpenGL). OpenCV frame = GL frame
# rotated pi about x: C_cv = C_gl @ diag(1, -1, -1).
R_GL_CV = np.diag([1.0, -1.0, -1.0])


def build_model(objects: tuple[ObjectSpec, ...], width: int, height: int, with_robot: bool) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_string(build_xml(objects, width, height))
    if with_robot:
        if not PANDA_XML.exists():
            raise FileNotFoundError(f"{PANDA_XML} missing — run scripts/fetch_assets.py")
        panda = mujoco.MjSpec.from_file(str(PANDA_XML))
        panda.option.integrator = spec.option.integrator   # avoid attach-conflict warning
        panda.body("hand").add_site(name="tcp", pos=[0.0, 0.0, 0.1034], size=[0.005, 0, 0], group=5)
        spec.attach(panda, frame=spec.worldbody.add_frame(pos=list(ROBOT_BASE_POS)), prefix="panda/")
    return spec.compile()


class TabletopScene:
    def __init__(
        self,
        objects: tuple[ObjectSpec, ...] = DEFAULT_OBJECTS,
        width: int = 320,
        height: int = 240,
        with_robot: bool = False,
    ):
        self.objects = objects
        self.width, self.height = width, height
        self.with_robot = with_robot
        self.model = build_model(objects, width, height, with_robot)
        self.data = mujoco.MjData(self.model)
        self.body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o.name) for o in objects]
        self._qpos_adr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{o.name}_joint")]
            for o in objects
        ]
        self._mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mobile_cam_mount")]
        self._occluder_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "occluder")]
        self._renderer: mujoco.Renderer | None = None
        self.robot = None
        if with_robot:
            from sim.robot.panda import Panda  # local import: sim.robot depends on this module
            self.robot = Panda(self.model, self.data)

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
        lo, hi = SPAWN_BOUNDS
        if self.robot is not None:
            from sim.robot.panda import OBSERVE_Q
            self.robot.set_joint_state(OBSERVE_Q)
        for adr in self._qpos_adr:
            for _ in range(200):
                xy = rng.uniform(lo, hi)
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

    def set_occluder(self, pos_W: np.ndarray | None) -> None:
        """Hover the occluder plate at pos_W (None = park it out of view)."""
        self.data.mocap_pos[self._occluder_id] = OCCLUDER_PARK if pos_W is None else pos_W
        mujoco.mj_forward(self.model, self.data)

    def kick(self, instance: int, v_xy: np.ndarray) -> None:
        """Give an object a planar velocity (an 'unexpected' push)."""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.objects[instance].name}_joint")
        adr = self.model.jnt_dofadr[jid]
        self.data.qvel[adr:adr + 2] = v_xy

    def set_mobile_camera(self, T_WC: np.ndarray) -> None:
        """Command camera B with an OpenCV-convention pose T_WC."""
        R_W_gl = T_WC[:3, :3] @ R_GL_CV.T
        self.data.mocap_pos[self._mocap_id] = T_WC[:3, 3]
        self.data.mocap_quat[self._mocap_id] = se3.rot_to_quat(R_W_gl)
        mujoco.mj_forward(self.model, self.data)
