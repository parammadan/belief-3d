"""Rendered RGB-D must agree with simulator geometry (M0 exit check).

Back-projecting depth through our intrinsics/extrinsics must land on the
known surfaces: sphere pixels on the sphere, table pixels on the table plane.
A half-pixel principal-point error or a frame flip breaks these bounds.
"""
from dataclasses import replace

import numpy as np
import pytest

from perception.geometry import se3
from sim.cameras.rgbd import RGBDSensor
from sim.environments.tabletop import TABLE_TOP_Z, TabletopScene

SPHERE = 2


@pytest.fixture(scope="module")
def scene():
    s = TabletopScene()
    s.reset(seed=0)
    yield s
    s.close()


def _sphere_residual(obs, gt, K=None):
    K = K or obs.intrinsics
    P, _ = K.depth_to_points(obs.depth, gt.segmentation == SPHERE)
    Pw = se3.transform_points(obs.T_WC, P)
    c = gt.objects[SPHERE].T_WO[:3, 3]
    return np.abs(np.linalg.norm(Pw - c, axis=1) - 0.028)


@pytest.mark.parametrize("cam", ["overhead", "mobile"])
def test_rgbd_shapes_and_validity(scene, cam):
    obs, gt = RGBDSensor(scene, cam).capture()
    assert obs.rgb.shape == (240, 320, 3) and obs.depth.dtype == np.float32
    assert np.isfinite(obs.depth).all() and (obs.depth >= 0).all()
    assert gt.segmentation.shape == obs.depth.shape
    assert se3.is_rotation(obs.T_WC[:3, :3])
    for i in range(len(gt.objects)):
        assert (gt.segmentation == i).sum() > 50, f"object {i} not visible from {cam}"


@pytest.mark.parametrize("cam", ["overhead", "mobile"])
def test_depth_backprojects_onto_sphere_surface(scene, cam):
    obs, gt = RGBDSensor(scene, cam).capture()
    r = _sphere_residual(obs, gt)
    assert r.mean() < 0.5e-3 and r.max() < 1.0e-3


@pytest.mark.parametrize("cam", ["overhead", "mobile"])
def test_table_pixels_lie_on_table_plane(scene, cam):
    obs, gt = RGBDSensor(scene, cam).capture()
    P, _ = obs.intrinsics.depth_to_points(obs.depth, gt.segmentation == -1)
    z = se3.transform_points(obs.T_WC, P)[:, 2]
    on_table = np.abs(z - TABLE_TOP_Z) < 0.02
    assert on_table.sum() > 10_000
    assert np.abs(z[on_table] - TABLE_TOP_Z).mean() < 0.5e-3


def test_principal_point_convention_is_the_measured_best(scene):
    obs, gt = RGBDSensor(scene, "overhead").capture()
    base = _sphere_residual(obs, gt).mean()
    for d in (0.5, -0.5):
        K = replace(obs.intrinsics, cx=obs.intrinsics.cx + d, cy=obs.intrinsics.cy + d)
        assert _sphere_residual(obs, gt, K).mean() > 3 * base


def test_projected_object_centres_fall_inside_masks(scene):
    obs, gt = RGBDSensor(scene, "overhead").capture()
    for o in gt.objects:
        P_C = se3.transform_points(se3.inv_T(obs.T_WC), o.T_WO[:3, 3][None])
        (u, v), = np.round(obs.intrinsics.project(P_C)[0]).astype(int)
        assert gt.segmentation[v, u] == o.instance_id


def test_commanded_mobile_pose_is_reported_back(scene):
    T_cmd = se3.look_at([0.3, 0.4, 1.0], [0.0, 0.0, TABLE_TOP_Z])
    scene.set_mobile_camera(T_cmd)
    sensor = RGBDSensor(scene, "mobile")
    assert np.allclose(sensor.camera_pose(), T_cmd, atol=1e-6)
    obs, gt = sensor.capture()
    assert _sphere_residual(obs, gt).mean() < 0.5e-3 or (gt.segmentation == SPHERE).sum() == 0


def test_reset_is_deterministic():
    a, b = TabletopScene(), TabletopScene()
    a.reset(seed=7)
    b.reset(seed=7)
    oa, ga = RGBDSensor(a, "overhead").capture()
    ob, gb = RGBDSensor(b, "overhead").capture()
    assert np.array_equal(a.data.qpos, b.data.qpos)
    assert np.array_equal(oa.depth, ob.depth) and np.array_equal(ga.segmentation, gb.segmentation)
    c = TabletopScene()
    c.reset(seed=8)
    assert not np.allclose(a.data.qpos, c.data.qpos)
    for s in (a, b, c):
        s.close()
