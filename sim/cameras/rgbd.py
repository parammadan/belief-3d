"""Synthetic RGB-D sensor over MuJoCo's offscreen renderer.

`capture()` returns an (Observation, GroundTruth) pair. Only the Observation
goes to inference; the GroundTruth goes to writers/evaluators.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from data.schemas.observation import GroundTruth, ObjectState, Observation
from perception.geometry import se3
from perception.geometry.camera import PinholeIntrinsics
from sim.environments.tabletop import ISOLATE_GROUP, R_GL_CV, TabletopScene


def intrinsics_from_model(model: mujoco.MjModel, cam_id: int, width: int, height: int) -> PinholeIntrinsics:
    """MuJoCo uses a symmetric frustum from fovy and square pixels.

    Continuous image coordinates put the principal point at (W/2, H/2) from
    the image edge; in the pixel-centre convention that is ((W-1)/2, (H-1)/2).
    Verified empirically in tests/simulation/test_render_geometry.py.
    """
    fovy = math.radians(float(model.cam_fovy[cam_id]))
    f = 0.5 * height / math.tan(0.5 * fovy)
    return PinholeIntrinsics(fx=f, fy=f, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0, width=width, height=height)


ROBOT_LABEL = -2   # segmentation id for robot pixels; -1 = static scene / background


class RGBDSensor:
    def __init__(self, scene: TabletopScene, camera_name: str, max_range: float = 5.0, compute_visibility: bool = True):
        self.scene = scene
        self.max_range = max_range
        self.compute_visibility = compute_visibility
        self.camera_name = camera_name
        self.cam_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id < 0:
            raise KeyError(f"no camera named {camera_name!r}")
        self.intrinsics = intrinsics_from_model(scene.model, self.cam_id, scene.width, scene.height)
        m = scene.model
        self._geom_to_instance = np.full(m.ngeom, -1, dtype=np.int32)
        for inst, bid in enumerate(scene.body_ids):
            self._geom_to_instance[m.geom_bodyid == bid] = inst
        if scene.robot is not None:
            self._geom_to_instance[scene.robot.is_robot_geom(np.arange(m.ngeom))] = ROBOT_LABEL
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = [1, 1, 1, 0, 1, 0]          # static, panda visual, objects
        self._iso = mujoco.MjvOption()
        self._iso.geomgroup[:] = [0, 0, 0, 0, 0, 1]

    def camera_pose(self) -> np.ndarray:
        d = self.scene.data
        R_W_gl = d.cam_xmat[self.cam_id].reshape(3, 3)
        return se3.make_T(R_W_gl @ R_GL_CV, d.cam_xpos[self.cam_id])

    def _render(self, kind: str, opt: mujoco.MjvOption) -> np.ndarray:
        r = self.scene.renderer
        if kind == "depth":
            r.enable_depth_rendering()
        elif kind == "seg":
            r.enable_segmentation_rendering()
        r.update_scene(self.scene.data, camera=self.camera_name, scene_option=opt)
        out = r.render().copy()
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        return out

    def _labels(self, seg_raw: np.ndarray) -> np.ndarray:
        geom_ids, obj_types = seg_raw[..., 0], seg_raw[..., 1]
        is_geom = (obj_types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geom_ids >= 0)
        seg = np.full(geom_ids.shape, -1, dtype=np.int32)
        seg[is_geom] = self._geom_to_instance[geom_ids[is_geom]]
        return seg

    def _visible_fractions(self, seg: np.ndarray) -> dict[int, float]:
        """visible pixels / pixels the object would cover if nothing occluded it."""
        m = self.scene.model
        out = {}
        for inst in range(len(self.scene.objects)):
            gids = np.nonzero(self._geom_to_instance == inst)[0]
            saved = m.geom_group[gids].copy()
            m.geom_group[gids] = ISOLATE_GROUP
            try:
                alone = int((self._labels(self._render("seg", self._iso)) == inst).sum())
            finally:
                m.geom_group[gids] = saved
            out[inst] = float((seg == inst).sum()) / alone if alone else 0.0
        return out

    def capture(self) -> tuple[Observation, GroundTruth]:
        d = self.scene.data
        rgb = self._render("rgb", self._opt)
        depth = self._render("depth", self._opt).astype(np.float32)
        # Beyond max_range (sky, far floor) is "no return": real RGB-D sensors
        # saturate, and CGL depth precision is poor there anyway.
        depth[~np.isfinite(depth) | (depth > self.max_range)] = 0.0
        seg = self._labels(self._render("seg", self._opt))
        vis = self._visible_fractions(seg) if self.compute_visibility else {}

        objects = tuple(
            ObjectState(
                instance_id=i,
                class_name=spec.name,
                T_WO=se3.make_T(se3.quat_to_rot(d.xquat[bid]), d.xpos[bid]),
                symmetry=spec.symmetry,
            )
            for i, (spec, bid) in enumerate(zip(self.scene.objects, self.scene.body_ids))
        )
        obs = Observation(self.camera_name, float(d.time), rgb, depth, self.intrinsics, self.camera_pose())
        gt = GroundTruth(self.camera_name, float(d.time), seg, objects, vis)
        return obs, gt
