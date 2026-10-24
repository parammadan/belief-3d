import numpy as np
import pytest

from data.schemas import calib_target as ct
from perception.calibration.fiducial import planar_pnp, project_W, triangulate
from perception.geometry import se3
from perception.geometry.camera import PinholeIntrinsics

K = PinholeIntrinsics(216.5, 216.5, 159.5, 119.5, 320, 240)


def _cams():
    T1 = se3.look_at([0.0, 0.0, 1.25], [0.08, 0.0, 0.4], up=[1, 0, 0])
    T2 = se3.look_at([0.6, -0.4, 0.9], [0.08, 0.0, 0.4])
    return T1, T2


def test_disc_grid_geometry():
    P = ct.disc_centres_W()
    assert P.shape == (12, 3) and np.ptp(P[:, 2]) == 0
    d = np.linalg.norm(P[1] - P[0])
    assert d == pytest.approx(ct.SPACING)


@pytest.mark.parametrize("cam", [0, 1])
def test_planar_pnp_exact_recovery(cam):
    T = _cams()[cam]
    P = ct.disc_centres_W()
    uv = project_W(K, T, P)
    est = planar_pnp(uv, P, K)                      # no initial guess: homography path
    assert se3.geodesic_distance(est.T_WC[:3, :3], T[:3, :3]) < 1e-6
    assert np.linalg.norm(est.T_WC[:3, 3] - T[:3, 3]) < 1e-6 and est.rms_px < 1e-6


def test_pnp_errors_are_consistent_with_its_covariance():
    """Monte Carlo: with 0.1 px noise, normalised pose errors should look chi2(6)."""
    T = _cams()[0]
    P = ct.disc_centres_W()
    rng = np.random.default_rng(0)
    d2 = []
    for _ in range(200):
        uv = project_W(K, T, P) + rng.normal(0, 0.1, (12, 2))
        est = planar_pnp(uv, P, K, T_init=T)
        dT = se3.inv_T(est.T_WC) @ T          # right perturbation from estimate to truth
        e = np.concatenate([se3.so3_log(dT[:3, :3]), dT[:3, 3]])
        d2.append(e @ np.linalg.solve(est.cov, e))
    assert np.mean(d2) == pytest.approx(6.0, rel=0.25)


def test_triangulation_exact():
    T1, T2 = _cams()
    P = ct.disc_centres_W()
    X = triangulate(project_W(K, T1, P), K, T1, project_W(K, T2, P), K, T2)
    assert np.abs(X - P).max() < 1e-9


def test_pnp_needs_four_points():
    with pytest.raises(ValueError):
        planar_pnp(np.zeros((3, 2)), ct.disc_centres_W()[:3], K)
