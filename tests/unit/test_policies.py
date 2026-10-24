import numpy as np
import pytest

from data.schemas.objects import CATALOG
from perception.geometry import se3
from perception.geometry.camera import PinholeIntrinsics
from policies.predict import ModeSet, SignatureRenderer, mismatch, modes_of, predicted_posteriors, target_signature
from policies.viewpoint import (
    CovariancePolicy,
    EntropyPolicy,
    FixedPolicy,
    PolicyContext,
    RandomPolicy,
    TaskRiskPolicy,
    grasp_fails,
    plan_grasp,
)
from tracking.belief.particle import BeliefConfig, PoseBelief

SPEC = {o.name: o for o in CATALOG}
K = PinholeIntrinsics(216.5, 216.5, 159.5, 119.5, 320, 240)
PLANE = (np.array([0.0, 0.0, 1.0]), -0.40)
TOP = se3.look_at([0.0, 0.0, 1.25], [0.0, 0.0, 0.40], up=[1, 0, 0])
SIDE = se3.look_at([0.7, 0.0, 0.55], [0.0, 0.0, 0.42])


def test_grasp_failure_model():
    box = SPEC["box_red"]
    T = se3.make_T(se3.so3_exp([0, 0, 0.3]), [0.0, 0.0, 0.425])
    g = plan_grasp(box, T)
    assert not grasp_fails(box, g, T)
    assert grasp_fails(box, g, T @ se3.make_T(se3.so3_exp([0, 0, np.pi / 2]), np.zeros(3)))   # yaw off by 90
    assert not grasp_fails(box, g, T @ se3.make_T(se3.so3_exp([0, 0, np.pi]), np.zeros(3)))   # symmetric flip
    T2 = T.copy()
    T2[0, 3] += 0.02
    assert grasp_fails(box, g, T2)                                                             # 2 cm off
    cyl = SPEC["cyl_green"]
    Tc = se3.make_T(np.eye(3), [0.1, 0.0, 0.435])
    assert not grasp_fails(cyl, plan_grasp(cyl, Tc), Tc @ se3.make_T(se3.so3_exp([0, 0, 1.0]), np.zeros(3)))


def test_signature_mismatch_basics():
    r = SignatureRenderer(K)
    box = SPEC["box_red"]
    T = se3.make_T(np.eye(3), [0.0, 0.0, 0.425])
    s1 = target_signature(r, TOP, box, T, [], PLANE)
    assert len(s1) > 50
    assert mismatch(s1, s1, 0.002) == (0.0, len(s1))
    T_up = T.copy()
    T_up[2, 3] += 0.01
    m, _ = mismatch(s1, target_signature(r, TOP, box, T_up, [], PLANE), 0.002)
    assert m > 1.0


def test_occluder_hides_target():
    r = SignatureRenderer(K)
    box = SPEC["box_red"]
    T = se3.make_T(np.eye(3), [0.0, 0.0, 0.425])
    plate = (SPEC["box_yellow"], se3.make_T(np.eye(3), [0.0, 0.0, 0.60]))     # hovering estimate above it
    assert len(target_signature(r, TOP, box, T, [plate], PLANE)) < len(target_signature(r, TOP, box, T, [], PLANE))


def test_posteriors_identity_and_disambiguation():
    modes = ModeSet(np.array([0.5, 0.5]), np.stack([np.eye(4)] * 2), [np.arange(1), np.arange(1)])
    same = {(0, 0): 1.0, (0, 1): 1.0}
    P = predicted_posteriors(modes, [same, same], 0.002, 10.0)
    assert np.allclose(P, 0.5)
    other = {(5, 5): 1.0, (5, 6): 1.0}
    big = {k: 1.0 for k in [(i, j) for i in range(20) for j in range(20)]}
    P = predicted_posteriors(modes, [big, other], 0.002, 10.0)
    assert P[0, 0] > 0.99 and P[1, 1] > 0.99


def _ambiguous_belief():
    """Two equal modes of box_red: correct yaw vs yaw + 90 deg (a grasp-relevant ambiguity)."""
    box = SPEC["box_red"]
    T = se3.make_T(np.eye(3), [0.0, 0.0, 0.425])
    T90 = T @ se3.make_T(se3.so3_exp([0, 0, np.pi / 2]), np.zeros(3))
    b = PoseBelief(box, [T, T90], 0.0, BeliefConfig(n_particles=64, init_r_sigma=0.01, init_t_sigma=0.0005), seed=0)
    return box, b


def _ctx(views, current=0):
    box, b = _ambiguous_belief()
    return PolicyContext(box, b, [], PLANE, views, current, lambda i, j: 0.0 if i == j else 1.0,
                         SignatureRenderer(K))


def test_belief_has_two_modes():
    _, b = _ambiguous_belief()
    assert len(modes_of(b).weights) == 2


def test_info_policies_prefer_disambiguating_view_over_blind_view():
    # view 0 looks away from the object (sees nothing), view 1 looks at it from above
    away = se3.look_at([0.0, 0.0, 1.25], [0.6, 0.6, 0.40], up=[1, 0, 0])
    ctx = _ctx([away, TOP])
    rng = np.random.default_rng(0)
    for pol in (EntropyPolicy(), CovariancePolicy(), TaskRiskPolicy()):
        v, info = pol.choose(ctx, rng)
        assert v == 1, (pol.name, info)
    assert FixedPolicy().choose(ctx, rng)[0] == 0
    assert RandomPolicy().choose(ctx, rng)[0] in (0, 1)


def test_task_risk_observing_beats_grasping_blind():
    ctx = _ctx([TOP, SIDE])
    pol = TaskRiskPolicy(lam=0.0)
    v, info = pol.choose(ctx, np.random.default_rng(0))
    assert pol.grasp_now_loss(ctx) == pytest.approx(0.5, abs=0.15)      # coin flip between yaw modes
    assert info["scores"][v] < 0.1


def _cal_ctx(views, history, rot=np.radians(1.0), trans=0.005):
    box, b = _ambiguous_belief()
    one = PoseBelief(box, [se3.make_T(np.eye(3), [0.0, 0.0, 0.425])], 0.0,
                     BeliefConfig(n_particles=64, init_r_sigma=0.001, init_t_sigma=0.0002), seed=1)
    return PolicyContext(box, one, [], PLANE, views, history[-1], lambda i, j: 0.0 if i == j else 1.0,
                         SignatureRenderer(K), view_history=history, calib_rot_sigma=rot, calib_trans_sigma=trans)


def _ring(n=6, el=np.radians(30), r=0.75):
    c = np.array([0.0, 0.0, 0.40])
    return [se3.look_at(c + r * np.array([np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)]), c)
            for a in np.linspace(-np.pi, np.pi, n, endpoint=False)]


def test_calibration_bias_zero_without_prior_and_shrinks_with_diverse_views():
    from policies.predict import modes_of
    from policies.viewpoint import CalibratedTaskRiskPolicy
    views = _ring()
    pol = CalibratedTaskRiskPolicy(seed=0)
    ctx0 = _cal_ctx(views, [0], rot=0.0, trans=0.0)
    assert pol.calib_loss(ctx0, modes_of(ctx0.belief), [0, 0]) == 0.0
    ctx = _cal_ctx(views, [0])
    p = np.array([0.0, 0.0, 0.425])
    same = np.linalg.norm(pol._calib_bias(ctx, [0, 0, 0], p), axis=1).mean()
    diverse = np.linalg.norm(pol._calib_bias(ctx, [0, 2, 4], p), axis=1).mean()
    assert diverse < 0.8 * same                     # opposite-side views partially cancel the bias


def test_calibrated_policy_prefers_diversifying_view():
    from policies.viewpoint import CalibratedTaskRiskPolicy
    views = _ring()
    ctx = _cal_ctx(views, [0, 0], rot=np.radians(2.0), trans=0.01)
    pol = CalibratedTaskRiskPolicy(lam=0.0, seed=0)
    v, info = pol.choose(ctx, np.random.default_rng(0))
    assert v != 0 and info["scores"][v] < info["scores"][0]
