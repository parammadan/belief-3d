"""Panda kinematics, IK and grasp physics.

The scripted grasp test reads object poses from the simulator. That is
allowed here only because this file *is* an evaluator-side physics check,
not the planning code (which must use perception estimates, M6).
"""
import mujoco
import numpy as np
import pytest

from perception.geometry import se3
from sim.environments.tabletop import PANDA_XML, SPAWN_BOUNDS, TABLE_TOP_Z, TabletopScene

pytestmark = pytest.mark.skipif(not PANDA_XML.exists(), reason="run scripts/fetch_assets.py")


@pytest.fixture()
def scene():
    s = TabletopScene(with_robot=True)
    s.reset(seed=0)
    yield s
    s.close()


def _step(s):
    return lambda: mujoco.mj_step(s.model, s.data)


def test_fk_matches_simulator_site(scene):
    from sim.robot.panda import OBSERVE_Q
    T = scene.robot.fk(OBSERVE_Q)
    scene.data.qpos[scene.robot.qpos_adr] = OBSERVE_Q
    mujoco.mj_forward(scene.model, scene.data)
    assert np.allclose(T, scene.robot.tcp_pose(), atol=1e-9)


def test_ik_does_not_touch_live_state(scene):
    q_before = scene.data.qpos.copy()
    scene.robot.ik(se3.make_T(np.diag([1.0, -1.0, -1.0]), [0.1, 0.0, TABLE_TOP_Z + 0.1]))
    assert np.array_equal(q_before, scene.data.qpos)


def test_ik_reaches_spawn_region_top_down(scene):
    from sim.robot.panda import R_TOP_DOWN
    lo, hi = SPAWN_BOUNDS
    results = []
    for x in np.linspace(lo[0], hi[0], 4):
        for y in np.linspace(lo[1], hi[1], 4):
            T = se3.make_T(R_TOP_DOWN, [x, y, TABLE_TOP_Z + 0.03])
            r = scene.robot.ik(T)
            results.append(r.success)
            if r.success:
                Tq = scene.robot.fk(r.q)
                assert np.linalg.norm(Tq[:3, 3] - T[:3, 3]) < 1e-3
                assert se3.geodesic_distance(Tq[:3, :3], T[:3, :3]) < 1e-2
                assert np.all(r.q >= scene.robot.q_lo - 1e-9) and np.all(r.q <= scene.robot.q_hi + 1e-9)
    assert np.mean(results) >= 0.9


def test_top_down_grasp_uses_yaw_symmetry(scene):
    """From the parked pose, a target yaw may need joint7 past its limit; yaw+pi must rescue it."""
    p = np.array([0.1, 0.05, TABLE_TOP_Z + 0.02])
    for yaw in np.linspace(-np.pi, np.pi, 9):
        pre, g, used = scene.robot.top_down_grasp(p, yaw)
        assert pre.success and g.success
        assert np.isclose(np.mod(used - yaw, np.pi), 0.0, atol=1e-9) or np.isclose(np.mod(used - yaw, np.pi), np.pi)


def test_unreachable_target_reports_failure(scene):
    r = scene.robot.ik(se3.make_T(np.eye(3), [2.0, 0.0, 2.0]), max_iters=50)
    assert not r.success and r.pos_err > 0.5


@pytest.mark.parametrize("idx", [0, 1, 3])   # boxes + cylinder; sphere grasp is M6
def test_scripted_top_down_grasp_lifts_object(scene, idx):
    from sim.robot.panda import GRIPPER_CLOSED, HOME_Q
    rb, bid = scene.robot, scene.body_ids[idx]
    p0 = scene.data.xpos[bid].copy()
    R = scene.data.xmat[bid].reshape(3, 3)
    r_pre, r_g, _ = rb.top_down_grasp(p0, np.arctan2(R[1, 0], R[0, 0]))
    assert r_pre.success and r_g.success
    rb.move_to(HOME_Q, 1.0, _step(scene))          # leave the parked pose over free space
    rb.move_to(r_pre.q, 1.0, _step(scene))
    rb.move_to(r_g.q, 1.0, _step(scene))
    rb.command(gripper=GRIPPER_CLOSED)
    for _ in range(400):
        mujoco.mj_step(scene.model, scene.data)
    rb.move_to(r_pre.q, 1.0, _step(scene))
    for _ in range(200):
        mujoco.mj_step(scene.model, scene.data)
    assert scene.data.xpos[bid][2] - p0[2] > 0.12
    assert rb.gripper_width() > 0.005      # fingers stopped on the object, not closed on air


def test_observe_pose_keeps_objects_visible(scene):
    from sim.cameras.rgbd import ROBOT_LABEL, RGBDSensor
    _, gt = RGBDSensor(scene, "overhead").capture()
    assert all(v > 0.9 for v in gt.visible_fraction.values())
    assert (gt.segmentation == ROBOT_LABEL).mean() < 0.1
