"""M2 classical perception: Observation -> list[Detection].

segment (plane + seed-and-grow) -> per segment, fit every catalog primitive of
that hue class -> keep the best fit. The class decides *which shapes are
plausible*; the geometric fit decides the pose. Inputs are the Observation
and the catalog only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data.schemas.objects import CATALOG, ObjectSpec
from data.schemas.observation import Observation
from perception.geometry import se3
from perception.pose.completeness import completeness
from perception.pose.primitive_fit import fit_primitive
from perception.segmentation.classical import Segment, SegmenterConfig, estimate_depth_noise, segment
from tracking.tracker import Detection


@dataclass(frozen=True)
class PerceptionConfig:
    seg: SegmenterConfig = field(default_factory=SegmenterConfig)
    min_fit_points: int = 20
    max_rms: float = 0.01          # m; reject fits that explain the points poorly


@dataclass(frozen=True)
class FrameResult:
    detections: list[Detection]
    segments: list[Segment]
    depth_sigma: float


def perceive(obs: Observation, cfg: PerceptionConfig = PerceptionConfig(),  # noqa: B008
             catalog: tuple[ObjectSpec, ...] = CATALOG, seed: int = 0) -> FrameResult:
    segs, plane = segment(obs, cfg.seg, seed=seed, catalog=catalog)
    P_C, _ = obs.intrinsics.depth_to_points(obs.depth)
    sigma = estimate_depth_noise(plane, se3.transform_points(obs.T_WC, P_C))
    by_class: dict[str, list[ObjectSpec]] = {}
    for o in catalog:
        # Twins share shape and size; fitting both would be redundant.
        if all((o.shape, o.size) != (x.shape, x.size) for x in by_class.get(o.color_class, [])):
            by_class.setdefault(o.color_class, []).append(o)
    dets, kept = [], []
    for s in segs:
        specs = by_class.get(s.color_class)
        if not specs or len(s.points_W) < cfg.min_fit_points:
            continue
        fits = [fit_primitive(sp, s.points_W, plane.n, plane.d, sigma, seed=seed) for sp in specs]
        best = min(fits, key=lambda f: f.cost)
        if best.rms > cfg.max_rms:
            continue
        spec = next(sp for sp in specs if sp.name == best.spec_name)
        comp = completeness(s.n_pixels, spec, best.T_WO, obs.T_WC, obs.intrinsics)
        rng_m = float(np.linalg.norm(best.T_WO[:3, 3] - obs.T_WC[:3, 3]))
        dets.append(Detection(obs.camera_name, obs.timestamp, s.color_class, best.spec_name, best.T_WO,
                              best.rms, best.n_points, comp, rng_m))
        kept.append(s)
    return FrameResult(dets, kept, sigma)


def mask_to_rle(mask: np.ndarray) -> list[int]:
    """Row-major run lengths, alternating 0-runs and 1-runs, starting with a 0-run."""
    m = mask.ravel()
    change = np.flatnonzero(m[1:] != m[:-1]) + 1
    runs = np.diff(np.concatenate([[0], change, [m.size]])).tolist()
    return ([0] + runs) if m.size and m[0] else runs


def rle_to_mask(runs: list[int], shape: tuple[int, int]) -> np.ndarray:
    vals = np.zeros(shape[0] * shape[1], bool)
    pos = 0
    for k, r in enumerate(runs):
        if k % 2 == 1:
            vals[pos:pos + r] = True
        pos += r
    return vals.reshape(shape)
