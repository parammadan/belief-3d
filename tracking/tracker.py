"""Multi-object tracker (B1: unimodal CV-Kalman + Hungarian association).

Per frame: predict every track to the frame time, then for each camera
associate that camera's detections to tracks (one-to-one within a camera, so
a track can be updated once per camera), update, and spawn tentative tracks
from detections no track explains. Lifecycle per frame:

    tentative --(confirm_hits)--> confirmed --(miss)--> coasting --(max_misses)--> deleted

While a track coasts its covariance grows, so the gate widens and the same ID
can re-acquire the object after an occlusion. That is the recovery mechanism
measured in M2.

Causality: the tracker only sees Detections with timestamps. It raises on a
frame older than its current time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tracking.association.hungarian import assign
from tracking.filters.kalman import CVKalman

CHI2_3_997 = 14.16   # 3-dof chi-square, 99.7 %


@dataclass(frozen=True)
class Detection:
    camera: str
    timestamp: float
    color_class: str
    spec_name: str
    T_WO: np.ndarray
    rms: float
    n_points: int
    completeness: float = 1.0     # observed / model-expected pixels (perception.pose.completeness)

    def position_cov(self, floor: float = 0.002) -> np.ndarray:
        """Fit residuals understate error on partial views (a 34 %-visible box fit
        with rms 0.9 mm was 36 mm off), so std is inflated by 1/completeness."""
        s = max(floor, 2.0 * self.rms) / max(self.completeness, 0.1)
        return np.eye(3) * s * s


@dataclass
class Track:
    track_id: int
    color_class: str
    spec_name: str
    kf: CVKalman
    R_WO: np.ndarray
    hits: int = 1
    misses: int = 0
    age: int = 0
    confirmed: bool = False
    history: list[tuple[float, np.ndarray]] = field(default_factory=list)

    @property
    def position(self) -> np.ndarray:
        return self.kf.x[:3]

    @property
    def T_WO(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R_WO, self.position
        return T

    @property
    def status(self) -> str:
        if not self.confirmed:
            return "tentative"
        return "tracked" if self.misses == 0 else "coasting"


@dataclass(frozen=True)
class TrackerConfig:
    gate: float = CHI2_3_997
    confirm_hits: int = 2
    max_misses: int = 20          # frames a confirmed track may coast
    tentative_max_misses: int = 1
    min_spawn_completeness: float = 0.6   # partial views update tracks but never start them
    q: float = 0.05               # CV process noise (m^2/s^3)
    v_sigma0: float = 0.3         # initial velocity std (m/s)


class MultiObjectTracker:
    def __init__(self, cfg: TrackerConfig = TrackerConfig()):  # noqa: B008
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._next_id = 0
        self.t = -np.inf

    def step(self, t: float, detections: list[Detection]) -> list[Track]:
        if t < self.t - 1e-9:
            raise ValueError(f"non-causal step: t={t} < tracker time {self.t}")
        self.t = t
        for tr in self.tracks:
            tr.kf.predict(t)
            tr.age += 1
        updated: set[int] = set()
        for cam in sorted({d.camera for d in detections}):
            dets = [d for d in detections if d.camera == cam]
            cost = np.full((len(self.tracks), len(dets)), np.inf)
            for i, tr in enumerate(self.tracks):
                for j, d in enumerate(dets):
                    if d.color_class == tr.color_class:
                        cost[i, j] = tr.kf.mahalanobis2(d.T_WO[:3, 3], d.position_cov())
            matches, _, unmatched = assign(cost, self.cfg.gate)
            for i, j in matches:
                tr, d = self.tracks[i], dets[j]
                tr.kf.update(d.T_WO[:3, 3], d.position_cov())
                tr.R_WO = d.T_WO[:3, :3]
                tr.hits += 1          # each camera is an independent confirmation
                updated.add(tr.track_id)
            for j in unmatched:
                d = dets[j]
                if d.completeness < self.cfg.min_spawn_completeness:
                    continue
                tr = Track(self._next_id, d.color_class, d.spec_name,
                           CVKalman.init(d.T_WO[:3, 3], d.position_cov(), t, self.cfg.v_sigma0, self.cfg.q),
                           d.T_WO[:3, :3])
                self._next_id += 1
                self.tracks.append(tr)
                updated.add(tr.track_id)
        alive = []
        for tr in self.tracks:
            if tr.track_id in updated:
                tr.misses = 0
                if tr.hits >= self.cfg.confirm_hits:
                    tr.confirmed = True
            else:
                tr.misses += 1
            limit = self.cfg.max_misses if tr.confirmed else self.cfg.tentative_max_misses
            if tr.misses <= limit:
                tr.history.append((t, tr.position.copy()))
                alive.append(tr)
        self.tracks = alive
        return self.confirmed()

    def confirmed(self) -> list[Track]:
        return [tr for tr in self.tracks if tr.confirmed]
