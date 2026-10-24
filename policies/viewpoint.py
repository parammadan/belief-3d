"""Viewpoint policies over a finite candidate set (master prompt §14).

All policies see the same inputs: the target's PoseBelief, estimated poses of the other
tracked objects (as occluders), the support plane, the candidate camera poses, the
current view index and a travel-time function. None reads simulator state.

  FixedPolicy        stay at the current view                         (A)
  RandomPolicy       uniform over reachable views                     (B)
  EntropyPolicy      min expected entropy of mode weights             (C1, ActivePose-style)
  CovariancePolicy   min expected between-mode rotational variance    (C2, SE(3)-PoseFlow-style)
  TaskRiskPolicy     min expected grasp failure + lambda * travel     (D, proposed)

Grasp-failure model (shared, explicit, geometric): a top-down parallel-jaw grasp planned
on a pose estimate closes across the object's narrowest horizontal extent at the estimated
centre and height. Against a true pose it fails if
  * horizontal centre error > tol_xy, or height error > tol_z;
  * the closing axis is misaligned by more than tol_yaw from the true object's narrowest
    horizontal axis (yaw-free for upright cylinders and spheres);
  * the true width along the closing axis > the gripper's max opening.
Expected loss after a view: sum_j w_j L(g(MAP of p(. | j)), pose_j).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.schemas.objects import ObjectSpec
from perception.geometry import se3
from perception.geometry.symmetry import sym_distance
from policies.predict import ModeSet, SignatureRenderer, modes_of, predicted_posteriors, target_signature


@dataclass(frozen=True)
class GraspModel:
    tol_xy: float = 0.010
    tol_z: float = 0.020
    tol_yaw: float = np.radians(15.0)
    max_opening: float = 0.08


def horizontal_extents(spec: ObjectSpec, T_WO: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(unit narrowest horizontal axis in world, width along it, width along the other)."""
    R = T_WO[:3, :3]
    if spec.shape == "sphere":
        return np.array([1.0, 0.0, 0.0]), 2 * spec.size[0], 2 * spec.size[0]
    if spec.shape == "cylinder":
        r, hh = spec.size
        axis = R[:, 2]
        if abs(axis[2]) > 0.7:                                     # upright: round in plan
            return np.array([1.0, 0.0, 0.0]), 2 * r, 2 * r
        a = np.array([axis[0], axis[1], 0.0])
        a /= np.linalg.norm(a)
        return np.array([-a[1], a[0], 0.0]), 2 * r, 2 * hh        # close across the lying cylinder
    e = np.asarray(spec.size)
    cands = []
    for k in range(3):
        col = R[:, k]
        if abs(col[2]) < 0.7:                                       # a horizontal box axis
            h = np.array([col[0], col[1], 0.0])
            cands.append((2 * e[k], h / np.linalg.norm(h)))
    cands.sort(key=lambda c: c[0])
    if len(cands) < 2:
        return np.array([1.0, 0.0, 0.0]), 2 * e.min(), 2 * e.max()
    return cands[0][1], cands[0][0], cands[1][0]


def plan_grasp(spec: ObjectSpec, T_WO: np.ndarray) -> dict:
    axis, w, _ = horizontal_extents(spec, T_WO)
    return {"p": T_WO[:3, 3].copy(), "axis": axis, "width": w}


def grasp_fails(spec: ObjectSpec, g: dict, T_true: np.ndarray, m: GraspModel = GraspModel()) -> bool:  # noqa: B008
    d = g["p"] - T_true[:3, 3]
    if np.hypot(d[0], d[1]) > m.tol_xy or abs(d[2]) > m.tol_z:
        return True
    axis_t, w_t, _ = horizontal_extents(spec, T_true)
    upright_round = spec.shape == "sphere" or (spec.shape == "cylinder" and abs(T_true[2, 2]) > 0.7)
    if not upright_round:
        c = abs(float(g["axis"] @ axis_t))
        if np.arccos(np.clip(c, -1, 1)) > m.tol_yaw:
            # misaligned: the fingers meet the other extent, which may not fit or may slip
            return True
    return w_t > m.max_opening


@dataclass
class PolicyContext:
    spec: ObjectSpec
    belief: object                                   # PoseBelief of the target
    occluders: list[tuple[ObjectSpec, np.ndarray]]   # other tracked objects' estimates
    plane: tuple[np.ndarray, float] | None
    views: list[np.ndarray]                          # candidate T_WC (OpenCV convention)
    current: int
    travel_time: callable                            # (i, j) -> seconds (inf if blocked)
    renderer: SignatureRenderer
    noise_scale: float = 0.002                       # m, signature mismatch scale
    temper: float = 10.0


def _view_posteriors(ctx: PolicyContext, modes: ModeSet, v: int) -> np.ndarray:
    sigs = [target_signature(ctx.renderer, ctx.views[v], ctx.spec, T, ctx.occluders, ctx.plane)
            for T in modes.poses]
    return predicted_posteriors(modes, sigs, ctx.noise_scale, ctx.temper)


def _reachable(ctx: PolicyContext) -> list[int]:
    return [v for v in range(len(ctx.views)) if np.isfinite(ctx.travel_time(ctx.current, v))]


class Policy:
    name = "base"

    def score(self, ctx: PolicyContext, modes: ModeSet, v: int) -> float:
        raise NotImplementedError

    def choose(self, ctx: PolicyContext, rng: np.random.Generator) -> tuple[int, dict]:
        modes = modes_of(ctx.belief)
        cands = _reachable(ctx)
        scores = {v: self.score(ctx, modes, v) for v in cands}
        best = min(cands, key=lambda v: (scores[v], ctx.travel_time(ctx.current, v), v))
        return best, {"scores": scores, "n_modes": len(modes.weights)}


class FixedPolicy(Policy):
    name = "fixed"

    def choose(self, ctx, rng):
        return ctx.current, {}


class RandomPolicy(Policy):
    name = "random"

    def choose(self, ctx, rng):
        return int(rng.choice(_reachable(ctx))), {}


class EntropyPolicy(Policy):
    name = "entropy"

    def score(self, ctx, modes, v):
        if len(modes.weights) < 2:
            return 0.0
        P = _view_posteriors(ctx, modes, v)
        H = -(P * np.log(np.maximum(P, 1e-300))).sum(1)
        return float(modes.weights @ H)


class CovariancePolicy(Policy):
    name = "covariance"

    def score(self, ctx, modes, v):
        if len(modes.weights) < 2:
            return 0.0
        P = _view_posteriors(ctx, modes, v)
        R = modes.poses[:, :3, :3]
        D2 = np.array([[sym_distance(ctx.spec, R[a], R[b]) ** 2 for b in range(len(R))] for a in range(len(R))])
        # expected posterior rotational variance about the posterior MAP mode
        return float(sum(modes.weights[j] * (P[j] @ D2[int(np.argmax(P[j]))]) for j in range(len(R))))


class TaskRiskPolicy(Policy):
    """Expected grasp-failure probability after the view, plus lambda * travel time (s)."""
    name = "task_risk"

    def __init__(self, grasp: GraspModel = GraspModel(), lam: float = 0.02):  # noqa: B008
        self.grasp, self.lam = grasp, lam

    def expected_loss(self, ctx, modes, P: np.ndarray | None) -> float:
        loss = 0.0
        for j, wj in enumerate(modes.weights):
            i = int(np.argmax(P[j])) if P is not None else int(np.argmax(modes.weights))
            g = plan_grasp(ctx.spec, modes.poses[i])
            # within-mode risk: average failure over the true mode's own particles
            idx = modes.members[j]
            sub = idx[np.argsort(-ctx.belief.w[idx])[:16]]
            ww = ctx.belief.w[sub] / ctx.belief.w[sub].sum()
            fails = np.array([grasp_fails(ctx.spec, g, se3.make_T(ctx.belief.R[s], ctx.belief.x[s]), self.grasp)
                              for s in sub])
            loss += wj * float(ww @ fails)
        return loss

    def score(self, ctx, modes, v):
        P = _view_posteriors(ctx, modes, v) if len(modes.weights) > 1 else None
        return self.expected_loss(ctx, modes, P) + self.lam * ctx.travel_time(ctx.current, v)

    def grasp_now_loss(self, ctx) -> float:
        return self.expected_loss(ctx, modes_of(ctx.belief), None)


POLICIES = {p.name: p for p in (FixedPolicy, RandomPolicy, EntropyPolicy, CovariancePolicy, TaskRiskPolicy)}
