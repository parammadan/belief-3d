import numpy as np
import pytest

from perception.geometry import se3


def rand_rot(rng):
    return se3.so3_exp(rng.normal(size=3) * rng.uniform(0, np.pi) / np.sqrt(3))


def test_identity_transform():
    P = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]])
    assert np.allclose(se3.transform_points(np.eye(4), P), P)


def test_inverse_and_composition():
    rng = np.random.default_rng(0)
    for _ in range(100):
        T_AB = se3.make_T(rand_rot(rng), rng.normal(size=3))
        T_BC = se3.make_T(rand_rot(rng), rng.normal(size=3))
        assert np.allclose(se3.inv_T(T_AB) @ T_AB, np.eye(4), atol=1e-12)
        P_C = rng.normal(size=(5, 3))
        chained = se3.transform_points(T_AB, se3.transform_points(T_BC, P_C))
        assert np.allclose(se3.transform_points(T_AB @ T_BC, P_C), chained, atol=1e-12)


def test_analytic_frame_example():
    # B is A rotated +90 deg about z and shifted 1 m along A's x.
    T_AB = se3.make_T(se3.so3_exp([0, 0, np.pi / 2]), [1.0, 0.0, 0.0])
    # B's x-axis unit point lands on A's (1, 1, 0).
    assert np.allclose(se3.transform_points(T_AB, [[1.0, 0.0, 0.0]]), [[1.0, 1.0, 0.0]])


def test_quaternion_normalisation_and_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        R = rand_rot(rng)
        q = se3.rot_to_quat(R)
        assert abs(np.linalg.norm(q) - 1) < 1e-12 and q[0] >= 0
        assert np.allclose(se3.quat_to_rot(q), R, atol=1e-10)
    q = se3.quat_normalize([2.0, 0, 0, 0])
    assert np.allclose(q, [1, 0, 0, 0])
    assert np.allclose(se3.quat_normalize([-1.0, 0, 0, 0]), [1, 0, 0, 0])  # q ~ -q
    with pytest.raises(ValueError):
        se3.quat_normalize([0, 0, 0, 0])


def test_quaternion_matches_axis_angle():
    th = 0.7
    q = [np.cos(th / 2), 0, np.sin(th / 2), 0]
    assert np.allclose(se3.quat_to_rot(q), se3.so3_exp([0, th, 0]))


@pytest.mark.parametrize("theta", [0.0, 1e-10, 1e-6, 0.3, 2.0, np.pi - 1e-3, np.pi - 1e-6, np.pi])
def test_log_exp_roundtrip_including_degenerate(theta):
    rng = np.random.default_rng(2)
    for _ in range(20):
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        R = se3.so3_exp(theta * a)
        w = se3.so3_log(R)
        assert np.linalg.norm(w) <= np.pi + 1e-12
        assert np.allclose(se3.so3_exp(w), R, atol=1e-9)


def test_geodesic_distance():
    R = se3.so3_exp([0.1, -0.2, 0.3])
    for th in [0.0, 1e-9, 0.5, np.pi - 1e-7, np.pi]:
        R2 = R @ se3.so3_exp([0, 0, th])
        assert abs(se3.geodesic_distance(R, R2) - th) < 1e-7
    assert se3.geodesic_distance(R, R) == pytest.approx(0.0, abs=1e-12)


def test_project_to_so3():
    rng = np.random.default_rng(3)
    R = rand_rot(rng)
    noisy = R + 1e-3 * rng.normal(size=(3, 3))
    Rp = se3.project_to_so3(noisy)
    assert se3.is_rotation(Rp)
    assert se3.geodesic_distance(R, Rp) < 5e-3


def test_look_at_opencv_convention():
    T = se3.look_at([0, 0, 1], [0, 0, 0], up=[0, 1, 0])
    assert se3.is_rotation(T[:3, :3])
    assert np.allclose(T[:3, 2], [0, 0, -1])        # z forward toward target
    T = se3.look_at([1, 0, 0], [0, 0, 0])
    assert np.allclose(T[:3, 1], [0, 0, -1])        # y down in world for level view
