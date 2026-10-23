"""Seeded sensor corruption applied to Observations (never to GroundTruth).

The coefficients are **parametric placeholders, not fitted to any real
sensor**. They exist to create controlled difficulty levels for experiments.
Two kinds of error:

* per-frame noise: RGB Gaussian noise, depth noise sigma(z) = s0 + s2 * z^2,
  random depth dropout, and dropout at depth discontinuities (where real
  RGB-D sensors lose returns);
* per-episode calibration error: the *reported* T_WC is the true pose composed
  with a fixed small SE(3) perturbation, and the reported fx/fy are scaled.
  The renderer still uses the true camera, so downstream geometry sees a
  genuine calibration mismatch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from data.schemas.observation import Observation
from perception.geometry import se3


@dataclass(frozen=True)
class NoiseConfig:
    rgb_sigma: float = 0.0             # intensity units (0-255)
    depth_sigma0: float = 0.0          # m
    depth_sigma2: float = 0.0          # m per m^2
    depth_dropout: float = 0.0         # probability a valid pixel loses its return
    edge_dropout_jump: float = np.inf  # m; drop pixels next to a depth jump larger than this
    extrinsic_rot_sigma: float = 0.0   # rad, per axis, per episode
    extrinsic_trans_sigma: float = 0.0 # m, per axis, per episode
    focal_scale_sigma: float = 0.0     # relative, per episode

    def to_dict(self) -> dict:
        return {k: (None if v == np.inf else v) for k, v in asdict(self).items()}


PRESETS: dict[str, NoiseConfig] = {
    "clean": NoiseConfig(),
    "nominal": NoiseConfig(rgb_sigma=3.0, depth_sigma0=0.001, depth_sigma2=0.0015, depth_dropout=0.01,
                           edge_dropout_jump=0.05),
    "harsh": NoiseConfig(rgb_sigma=8.0, depth_sigma0=0.002, depth_sigma2=0.004, depth_dropout=0.05,
                         edge_dropout_jump=0.03, extrinsic_rot_sigma=np.radians(1.0),
                         extrinsic_trans_sigma=0.005, focal_scale_sigma=0.01),
}


@dataclass(frozen=True)
class CalibrationError:
    """Per-episode, per-camera error between true and reported calibration."""
    dT: np.ndarray          # reported T_WC = true T_WC @ dT
    focal_scale: float

    @staticmethod
    def sample(cfg: NoiseConfig, rng: np.random.Generator) -> CalibrationError:
        w = rng.normal(0.0, cfg.extrinsic_rot_sigma, 3) if cfg.extrinsic_rot_sigma > 0 else np.zeros(3)
        t = rng.normal(0.0, cfg.extrinsic_trans_sigma, 3) if cfg.extrinsic_trans_sigma > 0 else np.zeros(3)
        s = 1.0 + (rng.normal(0.0, cfg.focal_scale_sigma) if cfg.focal_scale_sigma > 0 else 0.0)
        return CalibrationError(se3.make_T(se3.so3_exp(w), t), float(s))


def _edge_mask(depth: np.ndarray, jump: float) -> np.ndarray:
    valid = depth > 0
    m = np.zeros(depth.shape, bool)
    dx = np.abs(np.diff(depth, axis=1)) > jump
    dy = np.abs(np.diff(depth, axis=0)) > jump
    m[:, 1:] |= dx
    m[:, :-1] |= dx
    m[1:, :] |= dy
    m[:-1, :] |= dy
    return m & valid


def corrupt(obs: Observation, cfg: NoiseConfig, calib: CalibrationError, rng: np.random.Generator) -> Observation:
    rgb = obs.rgb
    if cfg.rgb_sigma > 0:
        rgb = np.clip(rgb.astype(np.float32) + rng.normal(0.0, cfg.rgb_sigma, rgb.shape), 0, 255).astype(np.uint8)
    depth = obs.depth.copy()
    valid = depth > 0
    if cfg.depth_sigma0 > 0 or cfg.depth_sigma2 > 0:
        sigma = cfg.depth_sigma0 + cfg.depth_sigma2 * depth**2
        depth = np.where(valid, depth + rng.normal(0.0, 1.0, depth.shape).astype(np.float32) * sigma, 0.0)
        depth = np.where(depth > 0, depth, 0.0).astype(np.float32)
    drop = np.zeros(depth.shape, bool)
    if np.isfinite(cfg.edge_dropout_jump):
        drop |= _edge_mask(obs.depth, cfg.edge_dropout_jump)
    if cfg.depth_dropout > 0:
        drop |= rng.random(depth.shape) < cfg.depth_dropout
    depth[drop] = 0.0
    K = obs.intrinsics
    K = replace(K, fx=K.fx * calib.focal_scale, fy=K.fy * calib.focal_scale)
    return Observation(obs.camera_name, obs.timestamp, rgb, depth.astype(np.float32), K, obs.T_WC @ calib.dT)
