import numpy as np

from perception.calibration.self_calib import SelfCalibrator
from perception.geometry import se3

EL30 = float(np.radians(30.0))


def _views(n=6, el=EL30, r=0.75):
    c = np.array([0.05, 0.0, 0.42])
    return [se3.look_at(c + r * np.array([np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)]), c)
            for a in np.linspace(-np.pi, np.pi, n, endpoint=False)]


def _simulate(dT, views, P, noise=0.0, rng=None):
    """Return (key, reported T_WC, reported point) for every view and object: exact sim model."""
    out = []
    for T_true in views:
        T_rep = T_true @ dT
        for k, p in enumerate(P):
            p_C = se3.transform_points(se3.inv_T(T_true), p[None])[0]
            ph = se3.transform_points(T_rep, p_C[None])[0]
            if noise:
                ph = ph + rng.normal(0, noise, 3)
            out.append((k, T_rep, ph))
    return out


P = np.array([[0.0, 0.0, 0.43], [0.10, 0.05, 0.43], [-0.05, 0.10, 0.44], [0.12, -0.08, 0.42]])


def test_recovers_the_identifiable_view_varying_bias():
    """Self-consistency identifies only the part of the bias that differs between views. An error
    equivalent to one world-frame rigid motion common to all views (here, rotation about / shift
    along the ring's vertical axis) is a gauge freedom: indistinguishable from the objects being
    elsewhere. So compare per-view bias *differences* (relative to view 0), not xi itself."""
    xi_true = np.concatenate([np.radians([0.8, -0.6, 0.5]), [0.004, -0.003, 0.002]])
    dT = se3.make_T(se3.so3_exp(xi_true[:3]), xi_true[3:])
    views = _views() + _views(el=np.radians(55.0))
    data = _simulate(dT, views, P)
    sc = SelfCalibrator(sigma_m=0.0005)
    for k, T, ph in data:
        sc.add(k, T, ph)
    sc.solve()
    p = P[0]
    reps = [T @ dT for T in views]
    pred = np.array([sc.bias_at(p, T) for T in reps])
    true = np.array([ph for k, _, ph in data if k == 0]) - p
    rel_pred, rel_true = pred - pred[0], true - true[0]
    assert np.abs(rel_pred - rel_true).max() < 0.0015                   # view-varying part recovered
    assert np.abs(rel_true).max() > 0.005                               # ... and it is not trivially small


def test_correction_reduces_fused_position_error():
    rng = np.random.default_rng(0)
    errs_before, errs_after = [], []
    for trial in range(20):
        xi_true = np.concatenate([rng.normal(0, np.radians(1.0), 3), rng.normal(0, 0.005, 3)])
        dT = se3.make_T(se3.so3_exp(xi_true[:3]), xi_true[3:])
        views = _views(n=4)
        data = _simulate(dT, views, P, noise=0.002, rng=rng)
        sc = SelfCalibrator()
        for k, T, ph in data:
            sc.add(k, T, ph)
        sc.solve()
        reps = [T for k, T, _ in data if k == 0]
        fused = np.mean([ph for k, _, ph in data if k == 0], axis=0)
        errs_before.append(np.linalg.norm(fused - P[0]))
        errs_after.append(np.linalg.norm(sc.correct(fused, reps) - P[0]))
    assert np.mean(errs_after) < 0.6 * np.mean(errs_before)


def test_single_view_shrinks_to_prior_safely():
    xi_true = np.concatenate([np.radians([1.0, 0.0, 0.0]), [0.0, 0.0, 0.0]])
    dT = se3.make_T(se3.so3_exp(xi_true[:3]), xi_true[3:])
    sc = SelfCalibrator()
    for k, T, ph in _simulate(dT, _views(n=1), P):
        sc.add(k, T, ph)
    xi = sc.solve()
    assert np.all(np.isfinite(xi)) and np.linalg.norm(xi[:3]) < np.radians(3.0)
