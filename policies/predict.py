"""Render-free prediction of what a candidate view would reveal about a belief.

For one target belief and candidate camera pose T_WC:
  1. Cluster the particles into modes (symmetry-aware). Each mode k has a weight w_k
     and a representative pose (its heaviest particle).
  2. Predict each mode's depth signature from that view: camera-facing points of the
     known model's surface grid, projected and z-buffered together with *estimated*
     occluders (other tracked objects' estimates and the support plane). No ground
     truth is used: occlusion comes from the belief, not the simulator.
  3. Mode mismatch m_ij = mean over the union of their pixels of min(|z_i - z_j|, cap) / s,
     where a pixel seen by only one mode costs the cap. Observing from the view while
     mode j is true gives the posterior over modes
         p(i | j) ∝ w_i exp(-n_ij * m_ij / T),
     with n_ij the union pixel count and T the same tempering as the belief likelihood.

Approximation (documented): mode-level reasoning only; within-mode spread is handled
by each policy's own term. Predictions use the camera's reported calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.schemas.objects import ObjectSpec
from perception.geometry import se3
from perception.geometry.camera import PinholeIntrinsics
from perception.pose.completeness import _normals, surface_grid


@dataclass(frozen=True)
class ModeSet:
    weights: np.ndarray          # (K,) normalised
    poses: np.ndarray            # (K, 4, 4) representative T_WO
    members: list[np.ndarray]    # particle indices per mode


def modes_of(belief, min_mass: float = 0.02) -> ModeSet:
    labels = belief.cluster_labels()
    mass = np.bincount(labels, weights=belief.w)
    keep = np.nonzero(mass >= min_mass)[0]
    poses, members = [], []
    for k in keep:
        idx = np.nonzero(labels == k)[0]
        j = idx[np.argmax(belief.w[idx])]
        poses.append(se3.make_T(belief.R[j], belief.x[j]))
        members.append(idx)
    w = mass[keep] / mass[keep].sum()
    return ModeSet(w, np.asarray(poses), members)


class SignatureRenderer:
    """Point-splat z-buffer of known models from a camera (inference-side)."""

    def __init__(self, K: PinholeIntrinsics, spacing: float = 0.003):
        self.K, self.spacing = K, spacing
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _grid(self, spec: ObjectSpec):
        if spec.name not in self._cache:
            S = surface_grid(spec, self.spacing)
            self._cache[spec.name] = (S, _normals(spec, S))
        return self._cache[spec.name]

    def depth(self, T_WC: np.ndarray, items: list[tuple[ObjectSpec, np.ndarray]],
              plane: tuple[np.ndarray, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(depth, label) images; label = index into items, -1 = plane/background."""
        K = self.K
        depth = np.full((K.height, K.width), np.inf)
        label = np.full((K.height, K.width), -1, dtype=np.int32)
        T_CW = se3.inv_T(T_WC)
        for idx, (spec, T_WO) in enumerate(items):
            S, N = self._grid(spec)
            T_CO = T_CW @ T_WO
            P = S @ T_CO[:3, :3].T + T_CO[:3, 3]
            Nc = N @ T_CO[:3, :3].T
            facing = np.einsum("ij,ij->i", Nc, P) < 0
            uv, front = K.project(P[facing])
            ok = front & K.in_image(uv)
            ui, vi = np.round(uv[ok]).astype(int).T
            z = P[facing][ok, 2]
            order = np.argsort(-z)                     # far first, near overwrites
            ui, vi, z = ui[order], vi[order], z[order]
            closer = z < depth[vi, ui]
            depth[vi[closer], ui[closer]] = z[closer]
            label[vi[closer], ui[closer]] = idx
        if plane is not None:
            # Pixels behind the support plane are hidden by it (table occludes the underside).
            n, d = plane
            vv, uu = np.mgrid[0:K.height, 0:K.width]
            rays_C = np.stack([(uu - K.cx) / K.fx, (vv - K.cy) / K.fy, np.ones_like(uu, float)], -1)
            rays_W = rays_C @ T_WC[:3, :3].T
            denom = rays_W @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                tp = -(T_WC[:3, 3] @ n + d) / denom
            z_plane = np.where((tp > 0) & np.isfinite(tp), tp, np.inf)    # z_C = t because ray z_C = 1
            hidden = z_plane < depth - 1e-4
            depth[hidden] = z_plane[hidden]
            label[hidden] = -1
        return depth, label


def target_signature(renderer: SignatureRenderer, T_WC: np.ndarray, spec: ObjectSpec, T_WO: np.ndarray,
                     occluders: list[tuple[ObjectSpec, np.ndarray]], plane) -> dict[tuple[int, int], float]:
    """Pixel -> depth for the *visible* part of the target (after estimated occlusion)."""
    depth, label = renderer.depth(T_WC, [(spec, T_WO)] + occluders, plane)
    v, u = np.nonzero(label == 0)
    return dict(zip(zip(v.tolist(), u.tolist()), depth[v, u].tolist()))


def mismatch(sig_a: dict, sig_b: dict, s: float, cap: float = 3.0) -> tuple[float, int]:
    keys = sig_a.keys() | sig_b.keys()
    if not keys:
        return 0.0, 0
    tot = 0.0
    for k in keys:
        if k in sig_a and k in sig_b:
            tot += min(abs(sig_a[k] - sig_b[k]) / s, cap)
        else:
            tot += cap
    return tot / len(keys), len(keys)


def predicted_posteriors(modes: ModeSet, sigs: list[dict], s: float, temper: float) -> np.ndarray:
    """P[j, i] = p(mode i | observation generated by mode j) for one view."""
    K = len(modes.weights)
    logL = np.zeros((K, K))
    for j in range(K):
        for i in range(K):
            if i != j:
                m, n = mismatch(sigs[i], sigs[j], s)
                logL[j, i] = -n * m / temper
    lp = np.log(np.maximum(modes.weights, 1e-300))[None] + logL
    lp -= lp.max(1, keepdims=True)
    P = np.exp(lp)
    return P / P.sum(1, keepdims=True)
