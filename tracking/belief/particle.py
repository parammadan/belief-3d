"""Multimodal 6D pose belief for one object: a particle filter on SE(3) x R^3 (B2 core).

State per particle:  R in SO(3), t in R^3 (world), v in R^3 (m/s).

Prediction (causal; damped velocity, i.e. Ornstein-Uhlenbeck with time constant tau):
    t <- t + v dt + N(0, (sigma_t^2 + q_a dt^3/3) I)
    v <- exp(-dt/tau) v + N(0, q_a dt I)
    R <- R exp(N(0, sigma_r^2 dt I))          right-perturbation on the manifold
Tabletop objects are mostly at rest and friction stops pushed ones. An
undamped random-walk velocity let the whole cloud share one drifting
velocity after resampling: the estimate climbed 30 mm in 8 updates (measured).

Observation likelihood, two-sided:

1. Surface agreement: the segment's world points P (from depth) lie on the
   particle's model,  sum_i -rho( sdf(R^T (p_i - t)) / s ) / T_lik.
2. Free space (optional, needs the depth image): camera-facing model surface
   samples are projected into each view. Where the observed depth is *behind*
   the model surface by more than a margin, the camera saw past where the
   particle claims the object is:  -lambda_fs * sum_j a_j min(viol_j / s, 3) / T_lik.
   Here a_j is the number of image pixels sample j stands for (surface area per
   sample * (f/z)^2 * |cos|), so both terms count *pixels* of evidence. The
   surface term is scaled by n_observed / n_used for the same reason. In
   per-sample units a 44-pixel contradiction scored only -1.2 nats (measured).
   Observed depth *in front* means something else occludes that pixel, so there
   is no penalty. Without this term the belief preferred a wider-faced wrong
   pose: it explained the same points with more slack, so it had more posterior
   volume (measured: 99.6 % mass on the wrong mode).

with a soft-L1 rho and scale s from the estimated depth noise. The points are
evaluated against each particle's model directly, so symmetric poses score
identically by construction and partial views leave a set of plausible poses
instead of one fit. T_lik >= 1 tempers the over-confidence that comes from
correlated pixel errors (points are not independent).

Update by progressive correction (Oudjane & Musso) with resample-move MCMC steps. The likelihood is
applied as L^beta in stages. Each beta is the largest (found by bisection)
that keeps ESS >= ess_frac * N. Between stages the particles are
systematically resampled and jittered with a kernel scaled to their current
spread. A one-shot update with this likelihood collapsed ESS to ~2 of 256 and
killed genuine alternative modes (measured); progressive correction keeps
diversity and moves particles into the modes.

Resampling preserves modes (mixture tracking, Vermaak et al. 2003): particles
are clustered symmetry-aware, each cluster is resampled internally with its
mass kept, and each is jittered with its *own* spread. A global jitter kernel
scales with the inter-mode distance (a 90 deg ambiguity gave ~13 deg jitter)
and wiped out the tighter mode (measured).

Rotation never averages matrices naively: the estimate aligns particles to
the best one through the symmetry group, takes the chordal mean and projects
it back onto SO(3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.schemas.objects import ObjectSpec
from perception.geometry import se3
from perception.geometry.camera import PinholeIntrinsics
from perception.geometry.symmetry import align_to
from perception.pose.completeness import _normals, surface_grid
from perception.pose.primitive_fit import lowest_point_height, sdf


@dataclass(frozen=True)
class DepthView:
    """What the free-space term needs from one Observation (no labels)."""
    depth: np.ndarray            # (H, W) z_C meters, 0 = no return
    K: PinholeIntrinsics
    T_WC: np.ndarray


@dataclass(frozen=True)
class BeliefConfig:
    n_particles: int = 256
    sigma_t: float = 0.002          # m, per-step translation jitter
    sigma_r: float = 0.05           # rad / sqrt(s), rotation diffusion
    q_a: float = 0.01               # m^2/s^3, acceleration noise
    tau_v: float = 0.5              # s, velocity damping time constant
    jitter_v_floor: float = 0.005   # m/s
    lik_scale_floor: float = 0.002  # m
    temper: float = 10.0            # T_lik: effective points = N / temper
    max_points: int = 200
    ess_frac: float = 0.5
    max_stages: int = 10            # progressive-correction stages per update
    jitter_frac: float = 0.3        # jitter std as a fraction of the current particle spread
    jitter_t_floor: float = 0.0003  # m
    jitter_r_floor: float = 0.005   # rad
    support_sigma: float | None = 0.005   # same resting prior as the pose fit; None disables
    init_t_sigma: float = 0.005
    init_r_sigma: float = 0.15            # rad around each initial hypothesis
    fs_weight: float = 1.0                # lambda_fs; 0 disables the free-space term
    fs_spacing: float = 0.004             # m between model surface samples (~1 px at 0.8 m)
    fs_margin: float = 0.005              # m behind the model surface before a violation counts


def surface_area(spec: ObjectSpec) -> float:
    if spec.shape == "sphere":
        return 4 * np.pi * spec.size[0] ** 2
    if spec.shape == "box":
        a, b, c = (2 * x for x in spec.size)
        return 2 * (a * b + b * c + a * c)
    r, hh = spec.size
    return 2 * np.pi * r * (2 * hh) + 2 * np.pi * r * r


def _so3_exp_batch(W: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(W, axis=1)
    K = np.zeros((len(W), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -W[:, 2], W[:, 1], -W[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = W[:, 2], -W[:, 1], W[:, 0]
    small = th < 1e-8
    a = np.where(small, 1.0, np.sin(th) / np.where(small, 1.0, th))
    b = np.where(small, 0.5, (1 - np.cos(th)) / np.where(small, 1.0, th**2))
    return np.eye(3)[None] + a[:, None, None] * K + b[:, None, None] * (K @ K)


def systematic_resample(w: np.ndarray, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    """n (default len(w)) systematic draws of indices with probabilities w."""
    n = len(w) if n is None else n
    c = np.cumsum(w)
    c[-1] = 1.0
    return np.searchsorted(c, (rng.random() + np.arange(n)) / n)


class PoseBelief:
    def __init__(self, spec: ObjectSpec, hypotheses: list[np.ndarray], t0: float,
                 cfg: BeliefConfig = BeliefConfig(), seed: int = 0):  # noqa: B008
        """Initialise around one or more pose hypotheses T_WO (equal mass each)."""
        if not hypotheses:
            raise ValueError("need at least one hypothesis")
        self.spec, self.cfg, self.t = spec, cfg, t0
        self.rng = np.random.default_rng(seed)
        n = cfg.n_particles
        k = self.rng.integers(len(hypotheses), size=n)
        H = np.asarray(hypotheses)
        self.R = H[k, :3, :3] @ _so3_exp_batch(self.rng.normal(0, cfg.init_r_sigma, (n, 3)))
        self.x = H[k, :3, 3] + self.rng.normal(0, cfg.init_t_sigma, (n, 3))
        self.v = np.zeros((n, 3))
        self.w = np.full(n, 1.0 / n)
        self._S_O = surface_grid(spec, cfg.fs_spacing)
        self._N_O = _normals(spec, self._S_O)
        self._area_per_sample = surface_area(spec) / len(self._S_O)
        self.log_lik_last = np.zeros(n)

    # ------------------------------------------------------------------ predict
    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt < -1e-9:
            raise ValueError(f"non-causal predict: {t} < {self.t}")
        if dt <= 0:
            return
        c, n = self.cfg, len(self.w)
        self.x = self.x + self.v * dt + self.rng.normal(0, np.sqrt(c.sigma_t**2 + c.q_a * dt**3 / 3), (n, 3))
        self.v = np.exp(-dt / c.tau_v) * self.v + self.rng.normal(0, np.sqrt(c.q_a * dt), (n, 3))
        self.R = self.R @ _so3_exp_batch(self.rng.normal(0, c.sigma_r * np.sqrt(dt), (n, 3)))
        self.t = t

    # ------------------------------------------------------------------ update
    def free_space_log_lik(self, view: DepthView, s: float) -> np.ndarray:
        c = self.cfg
        T_CW = se3.inv_T(view.T_WC)
        S_W = np.einsum("nij,mj->nmi", self.R, self._S_O) + self.x[:, None]      # (N, M, 3)
        N_W = np.einsum("nij,mj->nmi", self.R, self._N_O)
        S_C = S_W @ T_CW[:3, :3].T + T_CW[:3, 3]
        N_C = N_W @ T_CW[:3, :3].T
        facing = np.einsum("nmi,nmi->nm", N_C, S_C) < 0.0
        z = S_C[..., 2]
        front = z > 1e-6
        u = view.K.fx * S_C[..., 0] / np.where(front, z, 1.0) + view.K.cx
        v = view.K.fy * S_C[..., 1] / np.where(front, z, 1.0) + view.K.cy
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = front & facing & (ui >= 0) & (ui < view.K.width) & (vi >= 0) & (vi < view.K.height)
        z_obs = np.zeros_like(z)
        z_obs[inside] = view.depth[vi[inside], ui[inside]]
        viol = np.where(inside & (z_obs > 0), z_obs - z - c.fs_margin, 0.0)
        cos = np.abs(np.einsum("nmi,nmi->nm", N_C, S_C)) / np.maximum(np.linalg.norm(S_C, axis=2), 1e-9)
        pix = self._area_per_sample * cos * (view.K.fx / np.where(front, z, 1.0)) * (view.K.fy / np.where(front, z, 1.0))
        return -c.fs_weight * (pix * np.minimum(np.maximum(viol, 0.0) / s, 3.0)).sum(1) / c.temper

    def log_likelihood(self, P_W: np.ndarray, noise_sigma: float, plane: tuple[np.ndarray, float] | None,
                       views: list[DepthView] | None = None) -> np.ndarray:
        c = self.cfg
        n_obs = len(P_W)
        if n_obs > c.max_points:
            P_W = P_W[self.rng.choice(n_obs, c.max_points, replace=False)]
        scale = n_obs / len(P_W)                                      # back to pixel units
        s = max(c.lik_scale_floor, 2.0 * noise_sigma)
        Q = P_W[None] - self.x[:, None]                              # (N, M, 3)
        P_O = np.einsum("nmi,nij->nmj", Q, self.R)                    # R^T (p - t)
        d = sdf(self.spec, P_O.reshape(-1, 3)).reshape(len(self.w), -1) / s
        rho = 2.0 * (np.sqrt(1.0 + d * d) - 1.0)                      # soft-L1
        ll = -scale * rho.sum(1) / c.temper
        if views and c.fs_weight:
            for view in views:
                ll = ll + self.free_space_log_lik(view, s)
        if plane is not None and c.support_sigma:
            n_, d_ = plane
            gap = np.array([lowest_point_height(self.spec, R, x, n_, d_) for R, x in zip(self.R, self.x)])
            ll += -0.5 * (gap / c.support_sigma) ** 2
        return ll

    @staticmethod
    def _ess_of(lw: np.ndarray) -> float:
        w = np.exp(lw - lw.max())
        w /= w.sum()
        return float(1.0 / np.sum(w**2))

    def _sym_dist_to(self, R_ref: np.ndarray, Rs: np.ndarray) -> np.ndarray:
        """Vectorised symmetry-aware angle from R_ref to each of Rs."""
        if self.spec.shape == "sphere":
            return np.zeros(len(Rs))
        if self.spec.shape == "cylinder":
            return np.arccos(np.clip(np.abs(Rs[:, :, 2] @ R_ref[:, 2]), -1.0, 1.0))
        A = align_to(self.spec, R_ref, Rs)
        return np.arccos(np.clip((np.einsum("ij,nij->n", R_ref, A) - 1.0) / 2.0, -1.0, 1.0))

    def cluster_labels(self, rot_thresh: float = np.radians(20), t_thresh: float = 0.01) -> np.ndarray:
        """Greedy symmetry-aware clustering: repeatedly seed at the heaviest unassigned particle."""
        labels = np.full(len(self.w), -1)
        k = 0
        while (labels < 0).any():
            free = np.nonzero(labels < 0)[0]
            i = free[np.argmax(self.w[free])]
            near = (np.linalg.norm(self.x[free] - self.x[i], axis=1) < t_thresh) & \
                   (self._sym_dist_to(self.R[i], self.R[free]) < rot_thresh)
            labels[free[near]] = k
            k += 1
        return labels

    def _resample(self) -> tuple[np.ndarray, np.ndarray]:
        """Mode-preserving resampling. Returns (parent indices, cluster label of each new particle)."""
        n = len(self.w)
        labels = self.cluster_labels()
        mass = np.bincount(labels, weights=self.w)
        counts = np.bincount(systematic_resample(mass / mass.sum(), self.rng, n), minlength=len(mass))
        parents, plab = [], []
        for k, m_k in enumerate(counts):
            if m_k == 0:
                continue
            members = np.nonzero(labels == k)[0]
            wk = self.w[members] / self.w[members].sum()
            parents.append(members[systematic_resample(wk, self.rng, int(m_k))])
            plab.append(np.full(m_k, k))
        return np.concatenate(parents), np.concatenate(plab)

    def _kernel_stds(self, members: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
        c = self.cfg
        t_std, r_std, v_std = c.jitter_t_floor, c.jitter_r_floor, c.jitter_v_floor
        if len(members) > 1:
            wk = w / w.sum()
            t_std = max(t_std, c.jitter_frac * float(np.sqrt(np.trace(np.atleast_2d(np.cov(self.x[members].T))) / 3)))
            v_std = max(v_std, c.jitter_frac * float(np.sqrt(np.trace(np.atleast_2d(np.cov(self.v[members].T))) / 3)))
            if self.spec.shape != "sphere":
                d = self._sym_dist_to(self.R[members[np.argmax(wk)]], self.R[members])
                r_std = max(r_std, c.jitter_frac * float(wk @ d))
        return t_std, r_std, v_std

    def _resample_move(self, beta_cum: float, ll: np.ndarray, lik_args: tuple) -> np.ndarray:
        """Resample, then one Metropolis-Hastings jitter move per particle that targets
        prior x L^beta_cum (resample-move, Gilks & Berzuini 2001). Returns the log-lik of
        the new particle set.

        Approximation: the dynamics-prior density ratio of the move is ignored. This is
        valid while kernel stds << the predictive spread. Without the MH step the jitter
        blurs each mode and favours wide plateaus (measured: a mode 2 nats/update worse
        kept 29 % mass instead of < 0.1 %)."""
        n = len(self.w)
        stds = {}
        labels = self.cluster_labels()
        for k in np.unique(labels):
            members = np.nonzero(labels == k)[0]
            stds[k] = self._kernel_stds(members, self.w[members])
        parents, _ = self._resample()
        lab_of_parent = labels[parents]
        R, x, v, ll_cur = self.R[parents], self.x[parents], self.v[parents], ll[parents]
        t_s = np.array([stds[k][0] for k in lab_of_parent])[:, None]
        r_s = np.array([stds[k][1] for k in lab_of_parent])[:, None]
        v_s = np.array([stds[k][2] for k in lab_of_parent])[:, None]
        self.R = R @ _so3_exp_batch(self.rng.normal(0, 1, (n, 3)) * r_s)
        self.x = x + self.rng.normal(0, 1, (n, 3)) * t_s
        self.v = v + self.rng.normal(0, 1, (n, 3)) * v_s
        ll_prop = self.log_likelihood(*lik_args)
        accept = np.log(self.rng.random(n)) < beta_cum * (ll_prop - ll_cur)
        self.R = np.where(accept[:, None, None], self.R, R)
        self.x = np.where(accept[:, None], self.x, x)
        self.v = np.where(accept[:, None], self.v, v)
        self.w = np.full(n, 1.0 / n)
        self.last_accept_rate = float(accept.mean())
        return np.where(accept, ll_prop, ll_cur)

    def update(self, P_W: np.ndarray, noise_sigma: float = 0.0, plane: tuple[np.ndarray, float] | None = None,
               views: list[DepthView] | None = None) -> float:
        """Bayes update with one segment's points (progressive correction + resample-move).

        Returns the ESS of the final stage's reweighting."""
        c, n = self.cfg, len(self.w)
        lik_args = (P_W, noise_sigma, plane, views)
        remaining, cum, ess = 1.0, 0.0, float(n)
        ll = self.log_likelihood(*lik_args)
        for stage in range(c.max_stages):
            lw0 = np.log(np.maximum(self.w, 1e-300))
            target = c.ess_frac * n
            if stage == c.max_stages - 1 or self._ess_of(lw0 + remaining * ll) >= target:
                beta = remaining
            else:
                lo, hi = 0.0, remaining
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if self._ess_of(lw0 + mid * ll) >= target else (lo, mid)
                beta = max(lo, 1e-6)
            lw = lw0 + beta * ll
            w = np.exp(lw - lw.max())
            self.w = w / w.sum()
            ess = self.ess()
            remaining -= beta
            cum += beta
            self.log_lik_last = ll
            if remaining <= 1e-9:
                break
            ll = self._resample_move(cum, ll, lik_args)
        if ess < c.ess_frac * n:
            self._resample_move(1.0, ll, lik_args)
        return ess

    # ------------------------------------------------------------------ outputs
    def ess(self) -> float:
        return float(1.0 / np.sum(self.w**2))

    def estimate(self) -> np.ndarray:
        """Point estimate T_WO: weighted mean translation; symmetry-aligned chordal mean rotation."""
        t = self.w @ self.x
        ref = self.R[int(np.argmax(self.w))]
        A = align_to(self.spec, ref, self.R)
        R = se3.project_to_so3(np.einsum("n,nij->ij", self.w, A))
        return se3.make_T(R, t)

    def translation_cov(self) -> np.ndarray:
        d = self.x - self.w @ self.x
        return (self.w[:, None] * d).T @ d

    def rotation_spread(self) -> float:
        """Weighted mean symmetry-aware angle (rad) of particles from the estimate."""
        if self.spec.shape == "sphere":
            return 0.0
        R0 = self.estimate()[:3, :3]
        A = align_to(self.spec, R0, self.R)
        cos = np.clip((np.einsum("ij,nij->n", R0, A) - 1.0) / 2.0, -1.0, 1.0)
        return float(self.w @ np.arccos(cos))

    def n_modes(self, min_mass: float = 0.05) -> int:
        """Number of symmetry-aware clusters holding >= min_mass of the weight."""
        mass = np.bincount(self.cluster_labels(), weights=self.w)
        return int((mass >= min_mass).sum())
