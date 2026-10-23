"""Classical instance segmentation from one RGB-D Observation (M2 baseline).

1. Back-project valid depth to the world frame using the *reported* T_WC
   (so calibration error propagates exactly as it would on a real cell).
2. Fit the support plane with RANSAC, restricted to near-horizontal planes.
3. Seeds = saturated foreground pixels (between `min_height` and `max_height`
   above the plane). Table, robot, occluder and shadows are unsaturated.
4. Seeds grow at most `grow_iters` px into depth-continuous foreground. This
   recovers specular-washed object tops (saturation alone kept only 25-77 of
   ~200 px on the overhead view) without flooding into the arm. Touching
   fragments of the same hue class are merged.
5. Thresholds adapt to the depth noise, estimated from plane residuals (MAD),
   so nothing about the noise model is assumed.

Inputs: Observation + the object catalog. Nothing else.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from data.schemas.objects import CATALOG, ObjectSpec
from data.schemas.observation import Observation
from perception.geometry import se3


@dataclass(frozen=True)
class Plane:
    n: np.ndarray        # unit normal (world), oriented so n_z > 0
    d: float             # n . p + d = 0 on the plane

    def height(self, P: np.ndarray) -> np.ndarray:
        return P @ self.n + self.d


@dataclass(frozen=True)
class Segment:
    mask: np.ndarray             # (H, W) bool
    color_class: str             # catalog hue class or "unknown"
    hue_deg: float
    n_pixels: int
    points_W: np.ndarray         # (N, 3) world points of the mask with valid depth


@dataclass(frozen=True)
class SegmenterConfig:
    ransac_iters: int = 150
    ransac_inlier: float = 0.004       # m
    ransac_sample: int = 3000
    max_tilt_deg: float = 15.0         # plane normal vs world +z
    min_height: float = 0.004          # m above plane
    max_height: float = 0.30           # m; ignores the robot arm high above the table
    min_saturation: float = 0.40
    min_value: float = 0.12
    min_pixels: int = 25
    max_hue_dist_deg: float = 18.0
    depth_jump: float = 0.01           # m; floor for the neighbour depth step that stops growth
    min_seed: int = 5                  # saturated px in a seed component
    grow_iters: int = 8                # px of growth from saturated seeds


def catalog_hues(catalog: tuple[ObjectSpec, ...] = CATALOG) -> dict[str, float]:
    """Nominal hue (deg) per colour class, from catalog rgba."""
    out: dict[str, list[float]] = {}
    for o in catalog:
        out.setdefault(o.color_class, []).append(360.0 * colorsys.rgb_to_hsv(*o.rgba[:3])[0])
    return {k: float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(v)))))) % 360 for k, v in out.items()}


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised HSV; h in degrees [0, 360), s and v in [0, 1]."""
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    delta = mx - mn
    s = np.where(mx > 0, delta / np.maximum(mx, 1e-9), 0.0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    safe = np.maximum(delta, 1e-9)
    h = np.select(
        [mx == r, mx == g],
        [((g - b) / safe) % 6.0, (b - r) / safe + 2.0],
        (r - g) / safe + 4.0,
    ) * 60.0
    return np.where(delta > 0, h, 0.0), s, mx


def fit_support_plane(P: np.ndarray, cfg: SegmenterConfig, rng: np.random.Generator) -> Plane:
    if len(P) < 3:
        raise ValueError("not enough points for plane fit")
    S = P[rng.choice(len(P), size=min(cfg.ransac_sample, len(P)), replace=False)]
    cos_tilt = np.cos(np.radians(cfg.max_tilt_deg))
    best, best_n = None, -1
    for _ in range(cfg.ransac_iters):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n[2] < 0:
            n = -n
        if n[2] < cos_tilt:
            continue
        d = -n @ a
        inl = np.abs(S @ n + d) < cfg.ransac_inlier
        # Prefer the largest plane; among equals, the lowest-lying one cannot be
        # an object top, but the table dominates by area anyway.
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None:
        raise ValueError("no near-horizontal plane found")
    Q = S[best]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][-1]
    n = n if n[2] > 0 else -n
    return Plane(n, float(-n @ c))


def estimate_depth_noise(plane: Plane, P_W: np.ndarray, band: float = 0.03) -> float:
    """Robust sigma (m) of point-to-plane residuals near the support plane (MAD)."""
    r = plane.height(P_W)
    r = r[np.abs(r) < band]
    if len(r) < 50:
        return 0.0
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


_SHIFTS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _shift(a: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    out = np.full_like(a, fill)
    ys = slice(max(dy, 0), a.shape[0] + min(dy, 0))
    yd = slice(max(-dy, 0), a.shape[0] + min(-dy, 0))
    xs = slice(max(dx, 0), a.shape[1] + min(dx, 0))
    xd = slice(max(-dx, 0), a.shape[1] + min(-dx, 0))
    out[ys, xs] = a[yd, xd]
    return out


def segment(obs: Observation, cfg: SegmenterConfig = SegmenterConfig(), seed: int = 0,  # noqa: B008
            catalog: tuple[ObjectSpec, ...] = CATALOG) -> tuple[list[Segment], Plane]:
    K = obs.intrinsics
    valid = obs.depth > 0
    P_C, px = K.depth_to_points(obs.depth)
    P_W = se3.transform_points(obs.T_WC, P_C)
    plane = fit_support_plane(P_W, cfg, np.random.default_rng(seed))
    sigma = estimate_depth_noise(plane, P_W)
    min_h = max(cfg.min_height, 3.0 * sigma)
    jump = max(cfg.depth_jump, 4.0 * sigma)

    height = np.full(obs.depth.shape, -np.inf, np.float32)
    height[px[:, 1], px[:, 0]] = plane.height(P_W)
    pts_img = np.zeros((*obs.depth.shape, 3), np.float32)
    pts_img[px[:, 1], px[:, 0]] = P_W

    h, s, v = rgb_to_hsv(obs.rgb)
    fg = valid & (height > min_h) & (height < cfg.max_height)
    sat = fg & (s > cfg.min_saturation) & (v > cfg.min_value)
    labels, _ = ndimage.label(sat, structure=np.ones((3, 3), bool))
    labels[np.isin(labels, np.nonzero(np.bincount(labels.ravel()) < cfg.min_seed)[0])] = 0

    # Bounded, depth-continuous growth of the saturated seeds into the foreground.
    depth = obs.depth
    for _ in range(cfg.grow_iters):
        changed = False
        for dy, dx in _SHIFTS:
            cand = _shift(labels, dy, dx, 0)
            cd = _shift(depth, dy, dx, 0.0)
            take = (labels == 0) & fg & (cand > 0) & (np.abs(depth - cd) < jump)
            if take.any():
                labels[take] = cand[take]
                changed = True
        if not changed:
            break

    hues = catalog_hues(catalog)

    # Hue votes skip over-exposed pixels: a clipped channel shifts hue (orange
    # read as yellow under the key light, measured hue 54 deg vs nominal 28).
    unclipped = obs.rgb.max(-1) < 250

    def classify(mask: np.ndarray) -> tuple[str, float]:
        ms = mask & sat & unclipped
        if ms.sum() < cfg.min_seed:
            ms = mask & sat
        hue = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(h[ms])))))) % 360
        dist = {k: abs((hue - hv + 180) % 360 - 180) for k, hv in hues.items()}
        cls = min(dist, key=dist.get)
        return (cls if dist[cls] <= cfg.max_hue_dist_deg else "unknown"), hue

    # Merge touching fragments of the same hue class across depth-continuous borders.
    ids = [i for i in np.unique(labels) if i > 0]
    cls_of = {i: classify(labels == i)[0] for i in ids}
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for dy, dx in _SHIFTS[::2]:
        nb = _shift(labels, dy, dx, 0)
        nd = _shift(depth, dy, dx, 0.0)
        pairs = (labels > 0) & (nb > 0) & (labels != nb) & (np.abs(depth - nd) < jump)
        for a, b in set(zip(labels[pairs].tolist(), nb[pairs].tolist())):
            if cls_of[a] == cls_of[b]:
                parent[find(a)] = find(b)
    for i in ids:
        labels[labels == i] = find(i)

    segs = []
    for lab in np.unique(labels):
        if lab == 0:
            continue
        m = labels == lab
        cnt = int(m.sum())
        if cnt < cfg.min_pixels:
            continue
        cls, hue = classify(m)
        segs.append(Segment(m, cls, hue, cnt, pts_img[m]))
    return segs, plane


def _depth_edges(depth: np.ndarray, jump: float) -> np.ndarray:
    """Pixels whose 4-neighbour depth differs by more than `jump` (or borders no-return)."""
    e = np.zeros(depth.shape, bool)
    for ax in (0, 1):
        d = np.abs(np.diff(depth, axis=ax)) > jump
        if ax == 0:
            e[1:] |= d
            e[:-1] |= d
        else:
            e[:, 1:] |= d
            e[:, :-1] |= d
    return e
