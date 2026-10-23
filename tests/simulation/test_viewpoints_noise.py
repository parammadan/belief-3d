import dataclasses

import numpy as np
import pytest

from perception.geometry import se3
from sim.cameras.noise import PRESETS, CalibrationError, NoiseConfig, corrupt
from sim.cameras.rgbd import RGBDSensor
from sim.cameras.viewpoints import ViewpointSet
from sim.environments.tabletop import PANDA_XML, TabletopScene


@pytest.fixture(scope="module")
def scene():
    s = TabletopScene(with_robot=PANDA_XML.exists())
    s.reset(seed=0)
    yield s
    s.close()


def test_viewpoints_are_valid_and_aim_at_workspace(scene):
    vs = ViewpointSet(scene)
    assert len(vs) >= 10
    for v in vs.views:
        assert se3.is_rotation(v.T_WC[:3, :3])
        assert v.T_WC[2, 2] < 0                      # optical axis points downward


def test_travel_time_properties(scene):
    vs = ViewpointSet(scene)
    n = len(vs)
    T = np.array([[vs.travel_time(i, j) for j in range(n)] for i in range(n)])
    assert np.all(np.diag(T) == 0)
    assert np.allclose(T, T.T)                       # symmetric motion model
    finite = T[np.isfinite(T) & (T > 0)]
    assert finite.min() >= vs.motion.t_settle


def test_move_camera_updates_reported_pose(scene):
    vs = ViewpointSet(scene)
    sensor = RGBDSensor(scene, "mobile")
    vs.move_camera(3)
    assert np.allclose(sensor.camera_pose(), vs.views[3].T_WC, atol=1e-6)


def test_clean_noise_is_identity(scene):
    obs, _ = RGBDSensor(scene, "mobile").capture()
    rng = np.random.default_rng(0)
    out = corrupt(obs, PRESETS["clean"], CalibrationError.sample(PRESETS["clean"], rng), rng)
    assert np.array_equal(out.rgb, obs.rgb) and np.array_equal(out.depth, obs.depth)
    assert np.allclose(out.T_WC, obs.T_WC) and out.intrinsics == obs.intrinsics


def test_noise_is_seeded_and_bounded(scene):
    obs, _ = RGBDSensor(scene, "mobile").capture()
    cfg = PRESETS["harsh"]

    def run(seed):
        rng = np.random.default_rng(seed)
        return corrupt(obs, cfg, CalibrationError.sample(cfg, rng), rng)

    a, b, c = run(1), run(1), run(2)
    assert np.array_equal(a.depth, b.depth) and np.array_equal(a.rgb, b.rgb) and np.allclose(a.T_WC, b.T_WC)
    assert not np.array_equal(a.depth, c.depth)
    both = (a.depth > 0) & (obs.depth > 0)
    err = a.depth[both] - obs.depth[both]
    sigma = cfg.depth_sigma0 + cfg.depth_sigma2 * obs.depth[both] ** 2
    assert abs(np.mean(err / sigma)) < 0.05 and abs(np.std(err / sigma) - 1) < 0.05
    assert np.isfinite(a.depth).all() and (a.depth >= 0).all()


def test_calibration_error_breaks_geometry_as_expected(scene):
    """A 1 deg extrinsic error must move back-projected points by ~range * 1 deg."""
    obs, gt = RGBDSensor(scene, "overhead").capture()
    cal = CalibrationError(se3.make_T(se3.so3_exp([np.radians(1.0), 0, 0]), np.zeros(3)), 1.0)
    cfg = dataclasses.replace(NoiseConfig())
    bad = corrupt(obs, cfg, cal, np.random.default_rng(0))
    m = gt.segmentation == 2
    P, _ = obs.intrinsics.depth_to_points(obs.depth, m)
    good_w = se3.transform_points(obs.T_WC, P)
    bad_w = se3.transform_points(bad.T_WC, P)
    shift = np.linalg.norm(bad_w - good_w, axis=1).mean()
    rng_m = np.linalg.norm(P, axis=1).mean()
    assert shift == pytest.approx(rng_m * np.radians(1.0), rel=0.05)
