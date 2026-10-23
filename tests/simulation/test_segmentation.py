"""Classical segmentation, scored against GT masks (GT used only for scoring)."""
import numpy as np
import pytest

from data.generators.episodes import EpisodeConfig, run_episode
from perception.segmentation.classical import (
    Plane,
    SegmenterConfig,
    estimate_depth_noise,
    fit_support_plane,
    rgb_to_hsv,
    segment,
)


def _score(noise, seeds=range(1_500_000, 1_500_003), scenario="varied"):
    tp = fn = fp = cls_err = 0
    ious = []
    for seed in seeds:
        for fr in run_episode(seed, EpisodeConfig(n_frames=1, scenario=scenario, noise=noise)):
            for c in ("overhead", "mobile"):
                obs, gt = fr.obs[c], fr.gt[c]
                segs, _ = segment(obs)
                used = set()
                for ob in gt.objects:
                    gm = gt.segmentation == ob.instance_id
                    if gm.sum() < 25:
                        continue
                    best = max(((np.logical_and(s.mask, gm).sum() / np.logical_or(s.mask, gm).sum(), i)
                                for i, s in enumerate(segs)), default=(0.0, -1))
                    if best[0] >= 0.5:
                        tp += 1
                        ious.append(best[0])
                        used.add(best[1])
                        spec = next(o for o in fr.scenario.objects if o.name == ob.class_name)
                        cls_err += segs[best[1]].color_class != spec.color_class
                    else:
                        fn += 1
                fp += len(segs) - len(used)
    return tp, fn, fp, cls_err, float(np.mean(ious))


def test_hsv_matches_colorsys():
    import colorsys
    rng = np.random.default_rng(0)
    px = rng.integers(0, 256, (200, 3)).astype(np.uint8)
    h, s, v = rgb_to_hsv(px[None])
    for i, p in enumerate(px):
        hh, ss, vv = colorsys.rgb_to_hsv(*(p / 255.0))
        if ss > 1e-6:
            assert abs(((h[0, i] - 360 * hh) + 180) % 360 - 180) < 1e-3
        assert abs(s[0, i] - ss) < 1e-5 and abs(v[0, i] - vv) < 1e-6


def test_plane_fit_and_noise_estimate_on_synthetic_points():
    rng = np.random.default_rng(1)
    P = np.column_stack([rng.uniform(-0.4, 0.4, 5000), rng.uniform(-0.3, 0.3, 5000), np.full(5000, 0.4)])
    P[:, 2] += rng.normal(0, 0.002, 5000)
    P[:500, 2] += rng.uniform(0.02, 0.1, 500)                          # 10 % clutter above the plane
    plane = fit_support_plane(P, SegmenterConfig(), rng)
    assert np.allclose(plane.n, [0, 0, 1], atol=0.01) and abs(-plane.d - 0.4) < 1e-3
    assert estimate_depth_noise(plane, P) == pytest.approx(0.002, rel=0.15)
    assert isinstance(plane, Plane)


def test_clean_segmentation_quality():
    _, fn, fp, cls_err, miou = _score("clean")
    assert fn == 0 and fp <= 2 and cls_err == 0 and miou > 0.93


def test_harsh_noise_degrades_gracefully():
    tp, fn, fp, _, miou = _score("harsh", scenario="dynamic")
    assert tp / (tp + fn) > 0.8 and fp <= 2 and miou > 0.65


def test_segmentation_ignores_robot_and_occluder():
    frames = list(run_episode(3, EpisodeConfig(n_frames=12, scenario="occlusion", noise="clean")))
    sc = frames[0].scenario
    for fr in frames[sc.occl_frames[0]:sc.occl_frames[1]]:
        obs, gt = fr.obs["overhead"], fr.gt["overhead"]
        segs, _ = segment(obs)
        robot_or_bg = gt.segmentation < 0
        for s in segs:
            assert (s.mask & robot_or_bg).sum() / s.n_pixels < 0.2
