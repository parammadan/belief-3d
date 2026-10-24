import numpy as np

from benchmarks.metrics import rotation_error, sample_surface
from data.schemas.objects import CATALOG
from perception.geometry import se3
from perception.pose.completeness import surface_grid
from perception.pose.primitive_fit import fit_primitive, sdf
from policies.viewpoint import grasp_fails, plan_grasp
from sim.environments.scenario import PRESET_SCENARIOS, Scenario

MUG = next(o for o in CATALOG if o.shape == "mug")


def test_mug_sdf_and_samples():
    r, _, hl, _, _ = MUG.size
    assert sdf(MUG, np.zeros((1, 3)))[0] < 0                                   # inside body
    assert sdf(MUG, np.array([[r + hl / 2, 0, 0]]))[0] < 0                      # inside handle
    assert abs(sdf(MUG, np.array([[r + hl, 0, 0]]))[0]) < 1e-12                # handle tip on surface
    for P in (surface_grid(MUG, 0.004), sample_surface(MUG, 400)):
        assert len(P) > 100 and np.abs(sdf(MUG, P)).max() < 1e-7
        assert (P[:, 0] > r).any()                                              # handle is sampled


def test_mug_has_no_symmetry():
    R = se3.so3_exp([0, 0, np.pi])
    assert abs(rotation_error(MUG, R, np.eye(3)) - np.pi) < 1e-9


def test_mug_fit_recovers_handle_yaw_when_visible():
    T = se3.make_T(se3.so3_exp([0, 0, 1.1]), [0.05, 0.0, 0.40 + MUG.size[1]])
    P = se3.transform_points(T, sample_surface(MUG, 3000, seed=1))
    P = P[P[:, 2] > T[2, 3] - 0.3 * MUG.size[1]]
    est = fit_primitive(MUG, P, np.array([0.0, 0, 1]), -0.40)
    assert se3.geodesic_distance(est.T_WO[:3, :3], T[:3, :3]) < np.radians(3)


def test_mug_grasp_must_avoid_handle():
    T = se3.make_T(se3.so3_exp([0, 0, 0.4]), [0.0, 0.0, 0.44])
    g = plan_grasp(MUG, T)
    assert not grasp_fails(MUG, g, T)
    T_flip = T @ se3.make_T(se3.so3_exp([0, 0, np.pi]), np.zeros(3))
    assert not grasp_fails(MUG, g, T_flip)            # handle on the other side: still perpendicular
    T_90 = T @ se3.make_T(se3.so3_exp([0, 0, np.pi / 2]), np.zeros(3))
    assert grasp_fails(MUG, g, T_90)                  # handle now along the closing axis


def test_mugs_only_when_requested():
    for name in ("varied", "occlusion", "dynamic", "clutter"):
        for sd in range(30):
            sc = Scenario.sample(PRESET_SCENARIOS[name], np.random.default_rng(sd), 10)
            assert all(o.shape != "mug" for o in sc.objects)
    sc = Scenario.sample(PRESET_SCENARIOS["mugs"], np.random.default_rng(0), 10)
    assert sum(o.shape == "mug" for o in sc.objects) == 1
