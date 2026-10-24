"""B2: multi-object tracker whose tracks carry multimodal pose beliefs.

Association and lifecycle are the same as B1 (tracking/tracker.py), so the
comparison isolates the state representation: the gate uses the belief's
translation mean and covariance plus the detection's measurement covariance
(including the calibration term). A matched track's PoseBelief is updated
with the detection's raw segment points, plus free space against every
camera's depth image from that frame. New tracks start from all near-tie
pose hypotheses of the detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data.schemas.objects import CATALOG
from tracking.association.hungarian import assign
from tracking.belief.particle import BeliefConfig, DepthView, PoseBelief
from tracking.tracker import Detection, TrackerConfig

SPEC = {o.name: o for o in CATALOG}


@dataclass
class BeliefTrack:
    track_id: int
    color_class: str
    spec_name: str
    belief: PoseBelief
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    history: list = field(default_factory=list)

    @property
    def T_WO(self) -> np.ndarray:
        return self.belief.estimate()

    @property
    def position(self) -> np.ndarray:
        return self.belief.w @ self.belief.x

    @property
    def status(self) -> str:
        if not self.confirmed:
            return "tentative"
        return "tracked" if self.misses == 0 else "coasting"


class BeliefTracker:
    def __init__(self, cfg: TrackerConfig = TrackerConfig(), bcfg: BeliefConfig = BeliefConfig(),  # noqa: B008
                 seed: int = 0, use_free_space: bool = True):
        self.cfg, self.bcfg = cfg, bcfg
        self.tracks: list[BeliefTrack] = []
        self._next_id = 0
        self._seed = seed
        self.use_free_space = use_free_space
        self.t = -np.inf

    def meas_cov(self, d: Detection) -> np.ndarray:
        c = self.cfg
        return d.position_cov() + np.eye(3) * ((c.calib_rot_sigma * d.range_m) ** 2 + c.calib_trans_sigma**2)

    def _gate_cost(self, tr: BeliefTrack, d: Detection) -> float:
        y = d.T_WO[:3, 3] - tr.position
        S = tr.belief.translation_cov() + self.meas_cov(d)
        return float(y @ np.linalg.solve(S, y))

    def step(self, t: float, detections: list[Detection], views: dict[str, DepthView] | None = None) -> list[BeliefTrack]:
        if t < self.t - 1e-9:
            raise ValueError(f"non-causal step: t={t} < tracker time {self.t}")
        self.t = t
        for tr in self.tracks:
            tr.belief.predict(t)
        fs_views = list(views.values()) if (views and self.use_free_space) else None
        updated: set[int] = set()
        for cam in sorted({d.camera for d in detections}):
            dets = [d for d in detections if d.camera == cam]
            cost = np.full((len(self.tracks), len(dets)), np.inf)
            for i, tr in enumerate(self.tracks):
                for j, d in enumerate(dets):
                    if d.color_class == tr.color_class:
                        cost[i, j] = self._gate_cost(tr, d)
            matches, _, unmatched = assign(cost, self.cfg.gate)
            for i, j in matches:
                tr, d = self.tracks[i], dets[j]
                if d.points_W is not None and len(d.points_W) >= 10:
                    tr.belief.update(d.points_W, d.noise_sigma, d.plane, fs_views)
                tr.hits += 1
                updated.add(tr.track_id)
            for j in unmatched:
                d = dets[j]
                if d.completeness < self.cfg.min_spawn_completeness or d.points_W is None:
                    continue
                hyps = list(d.hypotheses) or [d.T_WO]
                b = PoseBelief(SPEC[d.spec_name], hyps, t, self.bcfg, seed=self._seed + self._next_id)
                b.update(d.points_W, d.noise_sigma, d.plane, fs_views)
                self.tracks.append(BeliefTrack(self._next_id, d.color_class, d.spec_name, b))
                self._next_id += 1
                updated.add(self.tracks[-1].track_id)
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
                alive.append(tr)
        self.tracks = alive
        return [tr for tr in self.tracks if tr.confirmed]
