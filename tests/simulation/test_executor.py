"""Grasp executor physics check (evaluator-side test: uses GT poses to plan, which only tests
the executor and its proprioceptive verification, not perception)."""
import mujoco
import numpy as np
import pytest

from data.schemas.objects import CATALOG
from manipulation.grasping.executor import execute_grasp, release
from policies.viewpoint import plan_grasp
from sim.environments.tabletop import PANDA_XML, TabletopScene

pytestmark = pytest.mark.skipif(not PANDA_XML.exists(), reason="run scripts/fetch_assets.py")
SPEC = {o.name: o for o in CATALOG}


@pytest.fixture()
def scene():
    s = TabletopScene(with_robot=True)
    s.reset(seed=0)
    yield s
    s.close()


def _step(s):
    return lambda: mujoco.mj_step(s.model, s.data)


@pytest.mark.parametrize("idx", [0, 1, 3])
def test_planned_grasp_lifts_and_verifies(scene, idx):
    from sim.robot.panda import HOME_Q
    spec = scene.objects[idx]
    bid = scene.body_ids[idx]
    T = np.eye(4)
    T[:3, :3] = scene.data.xmat[bid].reshape(3, 3)
    T[:3, 3] = scene.data.xpos[bid]
    z0 = T[2, 3]
    g = plan_grasp(SPEC[spec.name], T)
    res = execute_grasp(scene.robot, _step(scene), g, home_q=HOME_Q)
    assert res.executed and res.verified, res
    assert scene.data.xpos[bid][2] - z0 > 0.10
    assert abs(res.finger_width - g["width"]) < 0.015
    release(scene.robot, _step(scene))


def test_grasp_on_empty_space_is_detected(scene):
    from sim.robot.panda import HOME_Q
    spec = scene.objects[0]
    bid = scene.body_ids[0]
    T = np.eye(4)
    T[:3, :3] = scene.data.xmat[bid].reshape(3, 3)
    T[:3, 3] = scene.data.xpos[bid] + np.array([0.0, 0.06, 0.0])     # 6 cm beside the object
    res = execute_grasp(scene.robot, _step(scene), plan_grasp(SPEC[spec.name], T), home_q=HOME_Q)
    assert res.executed and not res.verified and res.reason == "closed_on_nothing"
