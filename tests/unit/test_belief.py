import numpy as np
import pytest

from benchmarks.metrics import box_symmetry_group, rotation_error
from data.schemas.objects import CATALOG
from perception.geometry import se3
from perception.geometry.camera import PinholeIntrinsics
from perception.geometry.symmetry import align_to, sym_distance, symmetry_group
from tracking.belief.particle import BeliefConfig, DepthView, PoseBelief, _so3_exp_batch, systematic_resample

SPEC = {o.name: o for o in CATALOG}
PLANE = (np.array([0.0, 0.0, 1.0]), -0.40)


def top_face_points(spec, T, n=200, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    e = spec.size
    P = np.column_stack([rng.uniform(-e[0], e[0], n), rng.uniform(-e[1], e[1], n), np.full(n, e[2])])
    return se3.transform_points(T, P) + rng.normal(0, noise, (n, 3)) if noise else se3.transform_points(T, P)


def test_batch_exp_matches_scalar():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(50, 3)) * rng.uniform(0, 3, (50, 1))
    W[0] = 0
    W[1] = 1e-10
    B = _so3_exp_batch(W)
    for w, R in zip(W, B):
        assert np.allclose(R, se3.so3_exp(w), atol=1e-12)


def test_systematic_resampling_is_unbiased_and_low_variance():
    rng = np.random.default_rng(1)
    w = rng.random(100)
    w /= w.sum()
    counts = np.zeros(100)
    for _ in range(2000):
        counts += np.bincount(systematic_resample(w, rng), minlength=100)
    assert np.allclose(counts / 2000, 100 * w, atol=0.05)
    one = np.bincount(systematic_resample(w, rng), minlength=100)
    assert np.all(np.abs(one - 100 * w) < 1.0 + 1e-9)          # each count within 1 of N*w_i


@pytest.mark.parametrize("name", ["box_red", "box_yellow"])
def test_symmetry_contract_with_evaluator(name):
    spec = SPEC[name]
    G_inf, G_eval = symmetry_group(spec), box_symmetry_group(spec.size)
    assert len(G_inf) == len(G_eval)
    for A in G_inf:
        assert any(np.allclose(A, B) for B in G_eval)
    rng = np.random.default_rng(2)
    for _ in range(50):
        R1, R2 = se3.so3_exp(rng.normal(size=3)), se3.so3_exp(rng.normal(size=3))
        assert sym_distance(spec, R1, R2) == pytest.approx(rotation_error(spec, R1, R2), abs=1e-9)


def test_cylinder_contract_with_evaluator():
    spec = SPEC["cyl_green"]
    rng = np.random.default_rng(3)
    for _ in range(50):
        R1, R2 = se3.so3_exp(rng.normal(size=3)), se3.so3_exp(rng.normal(size=3))
        assert sym_distance(spec, R1, R2) == pytest.approx(rotation_error(spec, R1, R2), abs=1e-9)


def test_align_to_picks_closest_equivalent():
    spec = SPEC["box_red"]
    R = se3.so3_exp([0.2, -0.1, 0.4])
    Rs = np.array([R @ S for S in symmetry_group(spec)])
    A = align_to(spec, R, Rs)
    assert np.allclose(A, R[None], atol=1e-9)


def test_weights_normalised_and_ess_bounds():
    spec = SPEC["box_red"]
    T = se3.make_T(np.eye(3), [0.0, 0.0, 0.425])
    b = PoseBelief(spec, [T], 0.0, BeliefConfig(n_particles=128), seed=0)
    for k in range(5):
        b.predict(0.1 * (k + 1))
        ess = b.update(top_face_points(spec, T, seed=k), 0.001, PLANE)
        assert abs(b.w.sum() - 1) < 1e-12 and 1.0 <= ess <= 128 + 1e-9


def test_converges_from_wrong_and_right_hypotheses():
    """Hypotheses: truth and a 90-deg-wrong face. Side + top points disambiguate."""
    spec = SPEC["box_red"]
    T = se3.make_T(se3.so3_exp([0, 0, 0.5]), [0.05, -0.02, 0.425])
    wrong = T @ se3.make_T(se3.so3_exp([np.pi / 2, 0, 0]), [0, 0, -0.005])
    rng = np.random.default_rng(4)
    from benchmarks.metrics import sample_surface
    P_all = se3.transform_points(T, sample_surface(spec, 2000, seed=5))
    visible = P_all[P_all[:, 2] > T[2, 3] - 0.01]                # top + upper sides
    b = PoseBelief(spec, [T, wrong], 0.0, BeliefConfig(), seed=1)
    for k in range(8):
        b.predict(0.1 * (k + 1))
        b.update(visible + rng.normal(0, 0.001, visible.shape), 0.001, PLANE)
    E = b.estimate()
    assert np.linalg.norm(E[:3, 3] - T[:3, 3]) < 0.003
    assert rotation_error(spec, E[:3, :3], T[:3, :3]) < np.radians(5)
    assert b.n_modes() == 1


K = PinholeIntrinsics(216.5, 216.5, 159.5, 119.5, 320, 240)
T_WC_TOP = se3.look_at([0.0, 0.0, 1.25], [0.0, 0.0, 0.40], up=[1, 0, 0])


def zbuffer(points_W: np.ndarray) -> np.ndarray:
    """Tiny splat renderer for tests: nearest z per pixel."""
    P = se3.transform_points(se3.inv_T(T_WC_TOP), points_W)
    uv, front = K.project(P)
    ok = front & K.in_image(uv)
    ui, vi = np.round(uv[ok]).astype(int).T
    depth = np.full((K.height, K.width), np.inf)
    np.minimum.at(depth, (vi, ui), P[ok, 2])
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def grid(x0, x1, y0, y1, z, step):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])


def _two_face_hypotheses():
    spec = SPEC["box_red"]
    T = se3.make_T(np.eye(3), [0.0, 0.0, 0.425])                       # z-up, resting
    T_y = se3.make_T(se3.so3_exp([np.pi / 2, 0, 0]), [0.0, 0.0, 0.45 - spec.size[1]])  # y-up, same top height
    return spec, T, T_y


# Static-scene tests: no motion diffusion (a diffusive motion model legitimately erodes
# narrow modes between updates, which is not what these tests probe).
STATIC = {"sigma_r": 0.0, "sigma_t": 0.0005, "q_a": 1e-6, "jitter_v_floor": 0.0}
SEEDS = [0, 1, 2, 3, 4]


def _mass(b, up_axis):
    return float(b.w[np.abs(b.R[:, 2, up_axis]) > 0.9].sum())


@pytest.mark.parametrize("seed", SEEDS)
def test_free_space_rejects_wider_face_hypothesis(seed):
    """Full top view: y-up claims box where the camera sees table 50 mm lower."""
    spec, T, T_y = _two_face_hypotheses()
    table = grid(-0.15, 0.15, -0.15, 0.15, 0.40, 0.0008)
    top = grid(-0.03, 0.03, -0.02, 0.02, 0.45, 0.0005)
    depth = zbuffer(np.vstack([table, top]))
    pts = top[np.random.default_rng(0).choice(len(top), 200, replace=False)]
    view = DepthView(depth, K, T_WC_TOP)
    b = PoseBelief(spec, [T, T_y], 0.0, BeliefConfig(support_sigma=None, **STATIC), seed=seed)
    for k in range(4):
        b.predict(0.1 * (k + 1))
        b.update(pts, 0.001, PLANE, views=[view])
    assert _mass(b, 1) < 0.05 and _mass(b, 2) > 0.95


@pytest.mark.parametrize("seed", SEEDS)
def test_genuine_ambiguity_is_preserved(seed):
    """An occluder hides all but a 40 x 30 mm window of the top face. Both hypotheses fit
    every observation, so neither mode may go extinct (no support prior). The masses are not
    expected to be equal: y-up's wider face tolerates ~2x more positional slack, so it carries
    more posterior volume (Occam factor). The test checks survival, not balance."""
    spec, T, T_y = _two_face_hypotheses()
    table = grid(-0.15, 0.15, -0.15, 0.15, 0.40, 0.0008)
    top = grid(-0.03, 0.03, -0.02, 0.02, 0.45, 0.0005)                     # the true (z-up) top face
    window = top[(np.abs(top[:, 0]) < 0.02) & (np.abs(top[:, 1]) < 0.015)]
    occl = grid(-0.15, 0.15, -0.15, 0.15, 0.60, 0.0008)
    # Hole sized so that, through perspective, only the window of the top face is visible.
    occl = occl[~((np.abs(occl[:, 0]) < 0.0160) & (np.abs(occl[:, 1]) < 0.0120))]
    depth = zbuffer(np.vstack([table, top, occl]))
    pts = window[np.random.default_rng(1).choice(len(window), 200, replace=False)]
    view = DepthView(depth, K, T_WC_TOP)
    b = PoseBelief(spec, [T, T_y], 0.0, BeliefConfig(support_sigma=None, **STATIC), seed=seed)
    for k in range(6):
        b.predict(0.1 * (k + 1))
        b.update(pts, 0.001, PLANE, views=[view])
        assert _mass(b, 1) >= 0.03 and _mass(b, 2) >= 0.03, k     # survival, not balance
    assert b.n_modes(min_mass=0.03) >= 2


@pytest.mark.parametrize("seed", SEEDS)
def test_support_prior_resolves_the_window_ambiguity(seed):
    spec, T, T_y = _two_face_hypotheses()
    window = grid(-0.02, 0.02, -0.015, 0.015, 0.45, 0.0005)
    pts = window[np.random.default_rng(1).choice(len(window), 200, replace=False)]
    b = PoseBelief(spec, [T, T_y], 0.0, BeliefConfig(**STATIC), seed=seed)
    for k in range(4):
        b.predict(0.1 * (k + 1))
        b.update(pts, 0.001, PLANE)
    assert _mass(b, 2) > 0.95           # y-up would float 10 mm above the table


def test_prediction_diffuses_without_updates_and_is_causal():
    spec = SPEC["sphere_blue"]
    b = PoseBelief(spec, [se3.make_T(np.eye(3), [0, 0, 0.428])], 0.0, seed=3)
    s0 = np.trace(b.translation_cov())
    b.predict(1.0)
    assert np.trace(b.translation_cov()) > s0
    with pytest.raises(ValueError):
        b.predict(0.5)


@pytest.mark.parametrize("name", ["box_red", "cyl_green", "cyl_orange", "sphere_blue"])
def test_vectorised_lowest_point_matches_scalar(name):
    from perception.pose.primitive_fit import lowest_point_height
    from tracking.belief.particle import lowest_point_heights
    spec = SPEC[name]
    rng = np.random.default_rng(0)
    R = _so3_exp_batch(rng.normal(size=(30, 3)))
    x = rng.normal(0, 0.05, (30, 3)) + [0, 0, 0.45]
    v = lowest_point_heights(spec, R, x, *PLANE)
    s = [lowest_point_height(spec, Ri, xi, *PLANE) for Ri, xi in zip(R, x)]
    assert np.allclose(v, s, atol=1e-12)


def test_predict_keeps_particles_out_of_the_table():
    from tracking.belief.particle import lowest_point_heights
    spec = SPEC["cyl_orange"]
    T = se3.make_T(np.eye(3), [0.0, 0.0, 0.445])
    b = PoseBelief(spec, [T], 0.0, BeliefConfig(), seed=0)
    b.plane = PLANE
    b.v[:] = [0.0, 0.0, -0.3]                       # everything heading into the table
    for k in range(10):
        b.predict(0.1 * (k + 1))
    assert lowest_point_heights(spec, b.R, b.x, *PLANE).min() > -1e-9
    assert (b.v @ PLANE[0]).min() > -0.1            # into-plane velocity removed at contact


def test_fast_likelihood_matches_reference_einsum():
    """The matmul rewrite must equal the original einsum formulation exactly (numerically)."""
    from perception.geometry import se3 as _se3
    spec, T, T_y = _two_face_hypotheses()
    table = grid(-0.15, 0.15, -0.15, 0.15, 0.40, 0.0008)
    top = grid(-0.03, 0.03, -0.02, 0.02, 0.45, 0.0005)
    view = DepthView(zbuffer(np.vstack([table, top])), K, T_WC_TOP)
    b = PoseBelief(spec, [T, T_y], 0.0, BeliefConfig(), seed=5)
    T_CW = _se3.inv_T(view.T_WC)
    S_W = np.einsum("nij,mj->nmi", b.R, b._S_O) + b.x[:, None]
    N_W = np.einsum("nij,mj->nmi", b.R, b._N_O)
    S_C = S_W @ T_CW[:3, :3].T + T_CW[:3, 3]
    N_C = N_W @ T_CW[:3, :3].T
    facing = np.einsum("nmi,nmi->nm", N_C, S_C) < 0.0
    z = S_C[..., 2]
    front = z > 1e-6
    u = view.K.fx * S_C[..., 0] / np.where(front, z, 1.0) + view.K.cx
    v = view.K.fy * S_C[..., 1] / np.where(front, z, 1.0) + view.K.cy
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = front & facing & (ui >= 0) & (ui < view.K.width) & (vi >= 0) & (vi < view.K.height)
    z_obs = np.zeros_like(z)
    z_obs[inside] = view.depth[vi[inside], ui[inside]]
    viol = np.where(inside & (z_obs > 0), z_obs - z - b.cfg.fs_margin, 0.0)
    cos = np.abs(np.einsum("nmi,nmi->nm", N_C, S_C)) / np.maximum(np.linalg.norm(S_C, axis=2), 1e-9)
    pix = b._area_per_sample * cos * (view.K.fx / np.where(front, z, 1.0)) * (view.K.fy / np.where(front, z, 1.0))
    ref = -b.cfg.fs_weight * (pix * np.minimum(np.maximum(viol, 0.0) / 0.002, 3.0)).sum(1) / b.cfg.temper
    assert np.allclose(b.free_space_log_lik(view, 0.002), ref, rtol=1e-9, atol=1e-9)
