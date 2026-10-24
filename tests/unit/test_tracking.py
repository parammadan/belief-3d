import numpy as np
import pytest

from tracking.association.hungarian import assign
from tracking.filters.kalman import CVKalman
from tracking.tracker import Detection, MultiObjectTracker, TrackerConfig


def det(p, cls="red", cam="overhead", t=0.0, rms=0.001):
    T = np.eye(4)
    T[:3, 3] = p
    return Detection(cam, t, cls, f"box_{cls}", T, rms, 100)


def test_kf_static_target_converges_and_is_consistent():
    rng = np.random.default_rng(0)
    p_true = np.array([0.1, -0.2, 0.43])
    R = np.eye(3) * 0.002**2
    kf = CVKalman.init(p_true + rng.normal(0, 0.002, 3), R, 0.0, q=1e-6)
    for k in range(1, 200):
        kf.predict(0.1 * k)
        kf.update(p_true + rng.normal(0, 0.002, 3), R)
    err = kf.x[:3] - p_true
    assert err @ np.linalg.solve(kf.P[:3, :3], err) < 14.16           # inside its own 99.7 % ellipsoid
    assert np.sqrt(np.trace(kf.P[:3, :3])) < 0.002                    # and tighter than one measurement
    assert np.mean(kf.nis_log) == pytest.approx(3.0, rel=0.25)      # chi2(3) mean
    assert np.allclose(kf.P, kf.P.T) and np.all(np.linalg.eigvalsh(kf.P) > 0)


def test_kf_tracks_constant_velocity():
    rng = np.random.default_rng(1)
    v = np.array([0.2, -0.1, 0.0])
    R = np.eye(3) * 0.001**2
    kf = CVKalman.init(np.zeros(3), R, 0.0, q=1e-3)
    for k in range(1, 60):
        t = 0.1 * k
        kf.predict(t)
        kf.update(v * t + rng.normal(0, 0.001, 3), R)
    err = kf.x[3:] - v
    assert err @ np.linalg.solve(kf.P[3:, 3:], err) < 14.16
    assert np.allclose(kf.x[3:], v, atol=0.02)


def test_kf_rejects_non_causal_predict():
    kf = CVKalman.init(np.zeros(3), np.eye(3) * 1e-6, 1.0)
    with pytest.raises(ValueError):
        kf.predict(0.5)


def test_assign_gates_and_handles_rectangular():
    C = np.array([[1.0, 50.0, 2.0], [0.5, 60.0, np.inf]])
    m, ur, uc = assign(C, gate=10.0)
    assert sorted(m) == [(0, 2), (1, 0)] and ur == [] and uc == [1]
    m, ur, uc = assign(np.zeros((0, 3)), 10.0)
    assert m == [] and uc == [0, 1, 2]


def _run(tracker, frames):
    out = []
    for t, dets in frames:
        out.append({tr.track_id: tr.position.copy() for tr in tracker.step(t, dets)})
    return out


def test_crossing_similar_objects_keep_ids():
    """Two same-class objects cross at constant velocity; CV prediction disambiguates."""
    rng = np.random.default_rng(2)
    frames = []
    for k in range(40):
        t = 0.1 * k
        a = np.array([-0.2 + 0.1 * t, 0.0, 0.43])
        b = np.array([0.2 - 0.1 * t, 0.003, 0.43])
        frames.append((t, [det(a + rng.normal(0, 1e-3, 3), t=t), det(b + rng.normal(0, 1e-3, 3), t=t)]))
    out = _run(MultiObjectTracker(TrackerConfig(q=1e-4)), frames)
    first, last = out[2], out[-1]
    ida = min(first, key=lambda i: first[i][0])           # starts on the left
    assert set(first) == set(last) and len(last) == 2
    assert last[ida][0] > 0.1                              # ended on the right: no swap


def test_track_survives_full_occlusion_and_reacquires_same_id():
    p = np.array([0.05, 0.05, 0.43])
    frames = [(0.1 * k, [] if 5 <= k < 12 else [det(p, t=0.1 * k)]) for k in range(20)]
    tr = MultiObjectTracker()
    out = _run(tr, frames)
    ids_before = set(out[4])
    assert ids_before and all(set(o) == ids_before for o in out[5:12])   # coasting, same ID
    assert set(out[-1]) == ids_before and tr._next_id == 1


def test_false_positive_is_pruned_and_never_confirmed():
    p = np.array([0.0, 0.0, 0.43])
    frames = [(0.1 * k, [det(p, t=0.1 * k)] + ([det([0.3, 0.3, 0.43], t=0.1 * k)] if k == 3 else []))
              for k in range(10)]
    tr = MultiObjectTracker()
    out = _run(tr, frames)
    assert all(len(o) <= 1 for o in out) and len(tr.tracks) == 1


def test_class_mismatch_never_associates_and_entry_spawns():
    tr = MultiObjectTracker()
    tr.step(0.0, [det([0, 0, 0.43], "red", t=0.0)])
    tr.step(0.1, [det([0, 0, 0.43], "red", t=0.1)])
    tr.step(0.2, [det([0, 0, 0.43], "red", t=0.2), det([0, 0, 0.43], "green", t=0.2)])
    tr.step(0.3, [det([0, 0, 0.43], "red", t=0.3), det([0, 0, 0.43], "green", t=0.3)])
    classes = sorted(t.color_class for t in tr.confirmed())
    assert classes == ["green", "red"]


def test_two_cameras_update_one_track():
    tr = MultiObjectTracker()
    for k in range(3):
        t = 0.1 * k
        tr.step(t, [det([0, 0, 0.43], cam="overhead", t=t), det([0.001, 0, 0.43], cam="mobile", t=t)])
    assert len(tr.tracks) == 1


def test_tracker_rejects_time_travel():
    tr = MultiObjectTracker()
    tr.step(1.0, [])
    with pytest.raises(ValueError):
        tr.step(0.5, [])


def test_calibration_term_prevents_duplicate_tracks_from_inconsistent_cameras():
    """Two cameras see one object 2.5 cm apart (independent extrinsic errors), ~0.8 m away."""
    def frames():
        for k in range(10):
            t = 0.1 * k
            a = det([0.0, 0.0, 0.43], cam="overhead", t=t)
            b = det([0.025, 0.0, 0.43], cam="mobile", t=t)
            yield t, [Detection(a.camera, t, "red", "box_red", a.T_WO, 0.001, 100, 1.0, 0.85),
                      Detection(b.camera, t, "red", "box_red", b.T_WO, 0.001, 100, 1.0, 0.80)]

    plain = MultiObjectTracker()
    for t, d in frames():
        plain.step(t, d)
    assert len(plain.confirmed()) == 2                         # the M2 failure mode, reproduced

    calib = MultiObjectTracker(TrackerConfig(calib_rot_sigma=np.radians(1.0), calib_trans_sigma=0.005))
    for t, d in frames():
        calib.step(t, d)
    assert len(calib.confirmed()) == 1
    x = calib.confirmed()[0].position[0]
    assert 0.0 < x < 0.025                                     # fused between the two reports


def test_meas_cov_adds_range_scaled_calibration_term():
    tr = MultiObjectTracker(TrackerConfig(calib_rot_sigma=0.01, calib_trans_sigma=0.002))
    T = np.eye(4)
    d = Detection("overhead", 0.0, "red", "box_red", T, 0.001, 100, 1.0, 2.0)
    R = tr.meas_cov(d)
    assert np.allclose(R, np.eye(3) * (0.002**2 + (0.01 * 2.0) ** 2 + 0.002**2))
