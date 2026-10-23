import numpy as np
import pytest

from benchmarks.metrics import add_s, add_s_sampled, box_symmetry_group, pose_errors, rotation_error, sample_surface
from data.schemas.objects import CATALOG
from perception.geometry import se3
from perception.pose.primitive_fit import fit_primitive, half_extents, sdf

SPEC = {o.name: o for o in CATALOG}


@pytest.mark.parametrize("name", ["box_red", "cyl_green", "sphere_blue", "box_yellow"])
def test_sdf_zero_on_sampled_surface_and_signed(name):
    spec = SPEC[name]
    P = sample_surface(spec, 300)
    assert np.abs(sdf(spec, P)).max() < 1e-9
    assert sdf(spec, np.zeros((1, 3)))[0] < 0 and sdf(spec, np.array([[1.0, 0, 0]]))[0] > 0


def test_box_groups_are_groups():
    for ext, size in [((0.03, 0.02, 0.025), 4), ((0.04, 0.015, 0.015), 8), ((1, 1, 1), 24)]:
        G = box_symmetry_group(ext)
        assert len(G) == size
        for A in G:
            for B in G:
                assert any(np.allclose(A @ B, C) for C in G)          # closure


def test_rotation_error_is_symmetry_invariant():
    rng = np.random.default_rng(0)
    R = se3.so3_exp(rng.normal(size=3))
    box = SPEC["box_red"]
    for S in box_symmetry_group(box.size):
        assert rotation_error(box, R @ S, R) < 1e-7
    cyl = SPEC["cyl_green"]
    spin = se3.so3_exp([0, 0, 1.234])
    flip = se3.so3_exp([np.pi, 0, 0])
    assert rotation_error(cyl, R @ spin, R) < 1e-7 and rotation_error(cyl, R @ flip, R) < 1e-7
    tilt = se3.so3_exp([0.1, 0, 0])
    assert rotation_error(cyl, R @ tilt, R) == pytest.approx(0.1, abs=1e-9)
    assert np.isnan(rotation_error(SPEC["sphere_blue"], R, R))


def test_add_s_exact_vs_sampled_floor():
    box = SPEC["box_red"]
    T = se3.make_T(np.eye(3), [0.1, 0, 0.4])
    T_flip = T @ se3.make_T(se3.so3_exp([0, 0, np.pi]), np.zeros(3))
    assert add_s(box, T_flip, T) < 1e-9                     # exact: a symmetric flip is zero error
    assert add_s_sampled(box, T_flip, T) > 1e-4             # sampled: nonzero floor
    T_shift = T.copy()
    T_shift[0, 3] += 0.002
    assert 0.0 < add_s(box, T_shift, T) <= 0.002 + 1e-9


@pytest.mark.parametrize("name", ["box_red", "cyl_green", "sphere_blue", "box_yellow", "cyl_orange"])
def test_zero_noise_recovery_from_top_half_points(name):
    """Synthetic 'overhead' view: only upward-facing half of the surface is observed."""
    spec = SPEC[name]
    rng = np.random.default_rng(1)
    he = half_extents(spec)
    T_gt = se3.make_T(se3.so3_exp([0, 0, rng.uniform(-np.pi, np.pi)]), [0.05, -0.03, 0.40 + he[2]])
    P_O = sample_surface(spec, 2000, seed=2)
    P_W = se3.transform_points(T_gt, P_O)
    P_W = P_W[P_W[:, 2] > T_gt[2, 3] - 0.3 * he[2]]          # top and upper sides
    est = fit_primitive(spec, P_W, np.array([0.0, 0, 1]), -0.40)
    e = pose_errors(spec, est.T_WO, T_gt)
    assert e["t_err"] < 2e-4 and e["add_s"] < 2e-4 and est.rms < 1e-4
    if spec.shape != "sphere":
        assert e["r_err_deg"] < 0.1


def test_fit_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_primitive(SPEC["box_red"], np.zeros((5, 3)), np.array([0.0, 0, 1]), -0.4)
