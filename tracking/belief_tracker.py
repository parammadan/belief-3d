"""B2: multi-object tracker whose tracks carry multimodal pose beliefs.

Association and lifecycle are the same as B1 (tracking/tracker.py), so the
comparison isolates the state representation: the gate uses the belief's
translation mean and covariance plus the detection's measurement covariance
(including the calibration term). A matched track's PoseBelief is updated
with the detection's raw segment points, plus free space against every
camera's depth image from that frame. New tracks start from all near-tie
pose hypotheses of the detection.

Online relative extrinsic correction (on by default). Each camera's
calibration error shifts all of its world-frame measurements. B2's
high-resolution likelihood then fuses two inconsistent geometries: on val under
harsh calibration it reached a 9.7-14 deg median rotation error vs B1's 3.4-5
(measured). Per frame, every object seen confidently by both the reference
camera and another camera pulls both onto their consensus (see
_relative_offsets). The per-camera offset, shrunk towards 0 by the calibration
prior, is applied to that camera's points and to its depth view's pose. It is
re-estimated every frame because the mobile camera moves and its error rotates
with it. No camera is assumed better calibrated than another.

Reported covariance = particle covariance + calibration term / n_cameras. The
consensus averages n independent absolute errors, and the belief is
conditional on calibration (95 % coverage was 0.01 without the term on harsh
val).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from data.schemas.objects import CATALOG
from tracking.association.hungarian import assign
from tracking.belief.particle import BeliefConfig, DepthView, PoseBelief
from tracking.tracker import Detection, TrackerConfig

SPEC = {o.name: o for o in CATALOG}


def _shift(T: np.ndarray, b: np.ndarray) -> np.ndarray:
    T = np.array(T, dtype=float, copy=True)
    T[:3, 3] += b
    return T


@dataclass
class BeliefTrack:
    track_id: int
    color_class: str
    spec_name: str
    belief: PoseBelief
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    n_cams: int = 1                # cameras that updated this track in the latest frame
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
                 seed: int = 0, use_free_space: bool = True, relative_calibration: bool = True):
        self.cfg, self.bcfg = cfg, bcfg
        self.relative_calibration = relative_calibration
        self.reid_dist = 0.10          # m
        self.last_offsets: dict[str, np.ndarray] = {}
        self.tracks: list[BeliefTrack] = []
        self._next_id = 0
        self._seed = seed
        self.use_free_space = use_free_space
        self.t = -np.inf

    def meas_cov(self, d: Detection) -> np.ndarray:
        c = self.cfg
        return d.position_cov() + np.eye(3) * ((c.calib_rot_sigma * d.range_m) ** 2 + c.calib_trans_sigma**2)

    def calib_cov(self, range_m: float) -> np.ndarray:
        c = self.cfg
        return np.eye(3) * ((c.calib_rot_sigma * range_m) ** 2 + c.calib_trans_sigma**2)

    def reported_cov(self, tr: BeliefTrack, range_m: float = 0.85) -> np.ndarray:
        return tr.belief.translation_cov() + self.calib_cov(range_m) / max(1, tr.n_cams)

    def _gate_cost(self, tr: BeliefTrack, d: Detection) -> float:
        y = d.T_WO[:3, 3] - tr.position
        S = tr.belief.translation_cov() + self.meas_cov(d)
        return float(y @ np.linalg.solve(S, y))

    def _relative_offsets(self, detections: list[Detection]) -> dict[str, np.ndarray]:
        """Per camera: world offset that moves its measurements onto the cross-camera consensus.

        Every object confidently seen by >= 2 cameras gives a consensus position m (the
        completeness-weighted mean over cameras), and each camera c votes b_c = m - p_c.
        Aligning to the consensus instead of to one reference camera keeps the averaging
        of independent absolute errors. Aligning to the overhead camera alone made both
        cameras inherit its full bias: median translation error 10.2 -> 22.4 mm on harsh
        val (measured)."""
        good = [d for d in detections if d.completeness >= self.cfg.min_spawn_completeness]
        cams = sorted({d.camera for d in good})
        if len(cams) < 2:
            return {}
        groups = []                                    # clusters of same-class detections across cameras
        for d in good:
            for g in groups:
                if g[0].color_class == d.color_class and all(x.camera != d.camera for x in g) and \
                        np.linalg.norm(g[0].T_WO[:3, 3] - d.T_WO[:3, 3]) < 0.06:
                    g.append(d)
                    break
            else:
                groups.append([d])
        votes: dict[str, list[tuple[np.ndarray, float]]] = {c: [] for c in cams}
        for g in groups:
            if len(g) < 2:
                continue
            w = np.array([x.completeness for x in g])
            m = (w[:, None] * np.array([x.T_WO[:3, 3] for x in g])).sum(0) / w.sum()
            for x in g:
                votes[x.camera].append((m - x.T_WO[:3, 3], x.completeness))
        out = {}
        prior = self.cfg.calib_rot_sigma * 0.85 + self.cfg.calib_trans_sigma
        for c, vs in votes.items():
            if not vs or prior <= 0:
                continue
            V, w = np.array([v for v, _ in vs]), np.array([x for _, x in vs])
            b_hat = (w[:, None] * V).sum(0) / w.sum()
            noise = 0.004 / np.sqrt(len(V))            # ~2 mm position std per vote, both ends
            out[c] = prior**2 / (prior**2 + noise**2) * b_hat
        return out

    def step(self, t: float, detections: list[Detection], views: dict[str, DepthView] | None = None) -> list[BeliefTrack]:
        if t < self.t - 1e-9:
            raise ValueError(f"non-causal step: t={t} < tracker time {self.t}")
        self.t = t
        for tr in self.tracks:
            tr.belief.predict(t)
        if self.relative_calibration:
            offs = self._relative_offsets(detections)
            self.last_offsets = offs
            shifted = []
            for d in detections:
                b = offs.get(d.camera)
                if b is None:
                    shifted.append(d)
                    continue
                T = d.T_WO.copy()
                T[:3, 3] += b
                shifted.append(replace(d, T_WO=T, points_W=None if d.points_W is None else d.points_W + b,
                                       hypotheses=tuple(_shift(H, b) for H in d.hypotheses)))
            detections = shifted
            if views:
                views = {c: (DepthView(v.depth, v.K, _shift(v.T_WC, offs[c])) if c in offs else v)
                         for c, v in views.items()}
        fs_views = list(views.values()) if (views and self.use_free_space) else None
        updated: set[int] = set()
        cams_seen: dict[int, int] = {}
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
                cams_seen[tr.track_id] = cams_seen.get(tr.track_id, 0) + 1
            for j in unmatched:
                d = dets[j]
                if d.completeness < self.cfg.min_spawn_completeness or d.points_W is None:
                    continue
                hyps = list(d.hypotheses) or [d.T_WO]
                b = PoseBelief(SPEC[d.spec_name], hyps, t, self.bcfg, seed=self._seed + self._next_id)
                b.update(d.points_W, d.noise_sigma, d.plane, fs_views)
                # Re-identification: a confident unexplained detection near a same-class track that
                # was not updated this frame re-initialises that track (keeping its ID) instead of
                # spawning a duplicate (a coasting duplicate cost 10 FP + 1 ID switch on harsh val).
                stale = [tr for tr in self.tracks if tr.color_class == d.color_class and tr.confirmed
                         and tr.track_id not in updated
                         and np.linalg.norm(tr.position - d.T_WO[:3, 3]) < self.reid_dist]
                if stale:
                    old = min(stale, key=lambda tr: np.linalg.norm(tr.position - d.T_WO[:3, 3]))
                    old.belief, old.spec_name = b, d.spec_name
                    old.hits += 1
                    updated.add(old.track_id)
                    cams_seen[old.track_id] = cams_seen.get(old.track_id, 0) + 1
                    continue
                self.tracks.append(BeliefTrack(self._next_id, d.color_class, d.spec_name, b))
                self._next_id += 1
                updated.add(self.tracks[-1].track_id)
        alive = []
        for tr in self.tracks:
            tr.n_cams = cams_seen.get(tr.track_id, 1)
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
