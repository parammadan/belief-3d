import numpy as np
import pytest

from perception.geometry import se3
from perception.geometry.camera import PinholeIntrinsics

K = PinholeIntrinsics(fx=400.0, fy=410.0, cx=159.5, cy=119.5, width=320, height=240)


def test_principal_axis_projects_to_principal_point():
    uv, front = K.project([[0.0, 0.0, 2.0]])
    assert front[0] and np.allclose(uv, [[159.5, 119.5]])


def test_known_point_projection():
    uv, _ = K.project([[0.1, -0.05, 1.0]])
    assert np.allclose(uv, [[159.5 + 40.0, 119.5 - 20.5]])


def test_behind_camera_flagged():
    uv, front = K.project([[0.0, 0.0, -1.0]])
    assert not front[0] and np.isnan(uv).all()


def test_roundtrip_projection_backprojection():
    rng = np.random.default_rng(0)
    P = np.column_stack([rng.uniform(-0.5, 0.5, 100), rng.uniform(-0.4, 0.4, 100), rng.uniform(0.2, 3.0, 100)])
    uv, _ = K.project(P)
    assert np.allclose(K.backproject(uv[:, 0], uv[:, 1], P[:, 2]), P, atol=1e-12)


def test_depth_image_reconstructs_plane_and_drops_invalid():
    depth = np.full((240, 320), 1.5, dtype=np.float32)
    depth[0, 0] = 0.0
    depth[1, 1] = np.nan
    P, px = K.depth_to_points(depth)
    assert len(P) == 240 * 320 - 2
    assert np.allclose(P[:, 2], 1.5)
    with pytest.raises(ValueError):
        K.depth_to_points(np.ones((10, 10)))


def test_world_to_pixel_through_extrinsics():
    T_WC = se3.look_at([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], up=[0, 1, 0])
    P_W = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    P_C = se3.transform_points(se3.inv_T(T_WC), P_W)
    assert np.allclose(P_C[0], [0, 0, 1])
    uv, _ = K.project(P_C)
    assert np.allclose(uv[0], [K.cx, K.cy])
    # 0.1 m lateral offset at 1 m range -> fx * 0.1 = 40 px from the principal point.
    assert np.isclose(abs(uv[1, 0] - K.cx), 40.0)


def test_extrinsic_perturbation_shifts_reprojection():
    T_WC = se3.look_at([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], up=[0, 1, 0])
    dT = se3.make_T(se3.so3_exp([0.0, 0.01, 0.0]), [0.0, 0.0, 0.0])   # 10 mrad yaw error
    P_W = np.array([[0.0, 0.0, 0.0]])
    uv0, _ = K.project(se3.transform_points(se3.inv_T(T_WC), P_W))
    uv1, _ = K.project(se3.transform_points(se3.inv_T(T_WC @ dT), P_W))
    assert np.isclose(np.linalg.norm(uv1 - uv0), 400.0 * np.tan(0.01), rtol=1e-6)
