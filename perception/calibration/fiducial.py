"""Calibration checks against the fiducial board (master prompt §9).

* detect_discs: magenta components -> sub-pixel centroids.
* match_to_model: correspondences via the *reported* calibration (projection
  + Hungarian). Valid while drift is below about half the disc spacing in pixels.
* planar_pnp: homography (DLT, normalised) -> [r1 r2 t] -> nearest rotation,
  then Levenberg-Marquardt on reprojection error. The target is planar, so a
  general 3D DLT would be degenerate.
* triangulate: linear two-view triangulation with the reported calibrations.

Measured conditioning (M2, 320x240, board spanning ~60 px overhead): with a
0.1 px centroid error, single-image PnP leaves ~1 deg of tilt coupled with
~1 cm of translation essentially unobservable. That matches its covariance.
The *reprojection error under the reported calibration* is the sensitive
drift signal (0.1 px clean -> 3.4 px at 0.9 deg drift).

Inputs: an Observation and the known board geometry only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares, linear_sum_assignment

from data.schemas import calib_target as ct
from data.schemas.observation import Observation
from perception.geometry import se3
from perception.geometry.camera import PinholeIntrinsics
from perception.segmentation.classical import rgb_to_hsv


def detect_discs(obs: Observation, hue_deg: float = 300.0, tol: float = 25.0, min_px: int = 4) -> np.ndarray:
    h, s, v = rgb_to_hsv(obs.rgb)
    m = (np.abs((h - hue_deg + 180) % 360 - 180) < tol) & (s > 0.3) & (v > 0.2)
    lab, n = ndimage.label(m)
    if n == 0:
        return np.zeros((0, 2))
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum(m, lab, idx)
    cents = np.array(ndimage.center_of_mass(m, lab, idx))[:, ::-1]      # (row, col) -> (u, v)
    return cents[sizes >= min_px]


def project_W(K: PinholeIntrinsics, T_WC: np.ndarray, P_W: np.ndarray) -> np.ndarray:
    return K.project(se3.transform_points(se3.inv_T(T_WC), P_W))[0]


def match_to_model(uv: np.ndarray, K: PinholeIntrinsics, T_WC: np.ndarray, P_W: np.ndarray,
                   max_px: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (model indices, detection indices)."""
    pred = project_W(K, T_WC, P_W)
    D = np.linalg.norm(pred[:, None] - uv[None], axis=2)
    r, c = linear_sum_assignment(np.where(D < max_px, D, 1e6))
    ok = D[r, c] < max_px
    return r[ok], c[ok]


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    def norm(p):
        m, s = p.mean(0), np.sqrt(2) / np.mean(np.linalg.norm(p - p.mean(0), axis=1))
        T = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
        return (np.column_stack([p, np.ones(len(p))]) @ T.T), T
    a, Ta = norm(src)
    b, Tb = norm(dst)
    A = []
    for (x, y, w), (u, v, z) in zip(a, b):
        A.append([0, 0, 0, -z * x, -z * y, -z * w, v * x, v * y, v * w])
        A.append([z * x, z * y, z * w, 0, 0, 0, -u * x, -u * y, -u * w])
    H = np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 3)
    return np.linalg.inv(Tb) @ H @ Ta


@dataclass(frozen=True)
class PnPResult:
    T_WC: np.ndarray
    rms_px: float
    n: int
    cov: np.ndarray          # 6x6 covariance of the right-perturbation [w, t] (rad, m)

    @property
    def rot_std_deg(self) -> float:
        return float(np.degrees(np.sqrt(np.trace(self.cov[:3, :3]))))

    @property
    def trans_std_mm(self) -> float:
        return float(1e3 * np.sqrt(np.trace(self.cov[3:, 3:])))


def planar_pnp(uv: np.ndarray, P_W: np.ndarray, K: PinholeIntrinsics, T_init: np.ndarray | None = None) -> PnPResult:
    """Camera pose from >= 4 coplanar world points (plane z = const)."""
    if len(uv) < 4:
        raise ValueError("need >= 4 correspondences")
    z0 = float(P_W[:, 2].mean())
    if np.ptp(P_W[:, 2]) > 1e-6:
        raise ValueError("planar_pnp expects points on a z = const plane")
    xy = P_W[:, :2]
    norm_img = (np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(K.K).T)[:, :2]
    H = _homography(xy, norm_img)
    lam = 1.0 / np.linalg.norm(H[:, 0])
    r1, r2, t = lam * H[:, 0], lam * H[:, 1], lam * H[:, 2]
    if t[2] < 0:                              # points must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    R_CP = se3.project_to_so3(np.column_stack([r1, r2, np.cross(r1, r2)]))
    # plane frame P: world translated by z0, so T_CW = T_CP @ T_PW
    T_CP = se3.make_T(R_CP, t)
    T_PW = se3.make_T(np.eye(3), [0, 0, -z0])
    T0 = se3.inv_T(T_CP @ T_PW) if T_init is None else T_init

    def resid(x):
        T = T0 @ se3.make_T(se3.so3_exp(x[:3]), x[3:])
        return (project_W(K, T, P_W) - uv).ravel()

    sol = least_squares(resid, np.zeros(6), method="lm")
    T = T0 @ se3.make_T(se3.so3_exp(sol.x[:3]), sol.x[3:])
    r = resid(sol.x)
    dof = max(1, len(r) - 6)
    sigma2 = float(r @ r) / dof
    cov = sigma2 * np.linalg.pinv(sol.jac.T @ sol.jac)
    return PnPResult(T, float(np.sqrt((r.reshape(-1, 2) ** 2).sum(1).mean())), len(uv), cov)


def triangulate(uv1: np.ndarray, K1: PinholeIntrinsics, T1: np.ndarray,
                uv2: np.ndarray, K2: PinholeIntrinsics, T2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation; uv1[i] and uv2[i] are the same point."""
    P1 = K1.K @ se3.inv_T(T1)[:3]
    P2 = K2.K @ se3.inv_T(T2)[:3]
    out = []
    for (u1, v1), (u2, v2) in zip(uv1, uv2):
        A = np.array([u1 * P1[2] - P1[0], v1 * P1[2] - P1[1], u2 * P2[2] - P2[0], v2 * P2[2] - P2[1]])
        X = np.linalg.svd(A)[2][-1]
        out.append(X[:3] / X[3])
    return np.asarray(out)


@dataclass(frozen=True)
class CalibCheck:
    n_matched: int
    reproj_rms_reported_px: float     # model points projected with the reported calibration
    reproj_rms_estimated_px: float    # after re-estimating T_WC by PnP
    drift_rot_deg: float              # reported vs re-estimated extrinsic
    drift_trans_mm: float
    T_WC_est: np.ndarray
    est_rot_std_deg: float            # PnP 1-sigma (from the Gauss-Newton covariance)
    est_trans_std_mm: float


def check_camera(obs: Observation) -> CalibCheck:
    P = ct.disc_centres_W()
    uv = detect_discs(obs)
    mi, di = match_to_model(uv, obs.intrinsics, obs.T_WC, P)
    if len(mi) < 4:
        raise ValueError(f"only {len(mi)} discs matched")
    Pm, uvm = P[mi], uv[di]
    rep = project_W(obs.intrinsics, obs.T_WC, Pm) - uvm
    est = planar_pnp(uvm, Pm, obs.intrinsics, T_init=obs.T_WC)
    dT = se3.inv_T(obs.T_WC) @ est.T_WC
    return CalibCheck(len(mi), float(np.sqrt((rep**2).sum(1).mean())), est.rms_px,
                      float(np.degrees(np.linalg.norm(se3.so3_log(dT[:3, :3])))),
                      float(1e3 * np.linalg.norm(est.T_WC[:3, 3] - obs.T_WC[:3, 3])), est.T_WC,
                      est.rot_std_deg, est.trans_std_mm)
