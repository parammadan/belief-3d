import numpy as np

from data.generators.episodes import EpisodeConfig, run_episode
from data.schemas.objects import CATALOG
from perception.geometry import se3
from perception.pipeline import mask_to_rle, perceive, rle_to_mask
from perception.pose.completeness import expected_pixels
from perception.pose.primitive_fit import fit_primitive, lowest_point_height

SPEC = {o.name: o for o in CATALOG}


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    for p in (0.0, 0.3, 1.0):
        m = rng.random((24, 32)) < p
        assert np.array_equal(rle_to_mask(mask_to_rle(m), m.shape), m)


def test_lowest_point_height_analytic():
    n, d = np.array([0.0, 0, 1]), -0.4
    box = SPEC["box_red"]
    assert lowest_point_height(box, np.eye(3), np.array([0, 0, 0.425]), n, d) == np.float64(0.0) or abs(
        lowest_point_height(box, np.eye(3), np.array([0, 0, 0.425]), n, d)) < 1e-12
    cyl = SPEC["cyl_green"]
    lying = se3.so3_exp([np.pi / 2, 0, 0])
    assert abs(lowest_point_height(cyl, lying, np.array([0, 0, 0.425]), n, d)) < 1e-12


def test_support_prior_resolves_box_face_ambiguity():
    """Only the top face is seen: z-up vs y-up differ by 5 mm in height; the prior picks the right one."""
    box = SPEC["box_red"]
    rng = np.random.default_rng(3)
    T_gt = se3.make_T(se3.so3_exp([0, 0, 0.4]), [0.0, 0.0, 0.425])
    P_O = np.column_stack([rng.uniform(-0.03, 0.03, 300), rng.uniform(-0.02, 0.02, 300), np.full(300, 0.025)])
    P_W = se3.transform_points(T_gt, P_O) + rng.normal(0, 0.002, (300, 3))
    est = fit_primitive(box, P_W, np.array([0.0, 0, 1]), -0.4, noise_sigma=0.002)
    assert abs(est.T_WO[2, 3] - 0.425) < 0.003


def test_expected_pixels_matches_rendered_object():
    frames = list(run_episode(1_500_000, EpisodeConfig(n_frames=1, scenario="varied", noise="clean")))
    obs, gt = frames[0].obs["overhead"], frames[0].gt["overhead"]
    for ob in gt.objects:
        if gt.visible_fraction[ob.instance_id] < 0.99:
            continue
        exp = expected_pixels(SPEC[ob.class_name], ob.T_WO, obs.T_WC, obs.intrinsics)
        seen = int((gt.segmentation == ob.instance_id).sum())
        assert abs(exp - seen) / seen < 0.15


def test_perceive_end_to_end_clean():
    frames = list(run_episode(1_500_001, EpisodeConfig(n_frames=1, scenario="varied", noise="clean")))
    obs, gt = frames[0].obs["overhead"], frames[0].gt["overhead"]
    res = perceive(obs)
    G = {ob.instance_id: ob for ob in gt.objects if (gt.segmentation == ob.instance_id).sum() > 50}
    assert len(res.detections) == len(G)
    for d in res.detections:
        dist = min(np.linalg.norm(d.T_WO[:3, 3] - ob.T_WO[:3, 3]) for ob in G.values())
        assert dist < 0.003 and d.completeness > 0.6


def test_batched_jacobian_gives_identical_fits():
    """PERF-M7-006: the batched finite-difference Jacobian must not change any fit."""
    from perception.pose.primitive_fit import fit_hypotheses
    rng = np.random.default_rng(4)
    n, d = np.array([0.0, 0, 1]), -0.4
    for name in ("box_red", "cyl_green", "sphere_blue", "mug_cyan"):
        spec = SPEC[name]
        P_O = rng.uniform(-0.03, 0.03, (300, 3))
        P_W = se3.transform_points(se3.make_T(se3.so3_exp([0, 0, 0.4]), [0.0, 0.0, 0.43]), P_O)
        for sup in (0.005, None):
            A = fit_hypotheses(spec, P_W, n, d, 0.002, support_sigma=sup, fast_jac=False)
            B = fit_hypotheses(spec, P_W, n, d, 0.002, support_sigma=sup, fast_jac=True)
            assert all(np.array_equal(a.T_WO, b.T_WO) and a.cost == b.cost for a, b in zip(A, B))
