"""EXP-M2-002: effect of extrinsic calibration drift (seeded sweep).

For each drift level s, per seed: draw an SE(3) error with rotation std s/sqrt(3) per axis
(so the expected magnitude is about s) and translation std 5 mm * s/1deg per axis. Apply it to
the *reported* extrinsics of both cameras (the renderer keeps the truth). Then measure:
  * reprojection RMS of the fiducial discs under the reported calibration (px)
  * PnP re-estimation error vs the true pose (deg, mm)
  * two-view triangulation error of the discs with reported calibrations (mm)
  * M2 object localisation error, overhead camera, on a test-split object scene (mm)

Usage:  MUJOCO_GL=cgl PYTHONPATH=. python -m benchmarks.calibration.calib_sweep --seeds 10 --out reports/m2/calib_sweep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.schemas import calib_target as ct
from data.schemas.splits import seeds_for
from perception.calibration.fiducial import check_camera, detect_discs, match_to_model, triangulate
from perception.geometry import se3
from perception.pipeline import perceive
from sim.cameras.noise import CalibrationError, NoiseConfig, corrupt
from sim.cameras.rgbd import RGBDSensor
from sim.cameras.viewpoints import ViewpointSet
from sim.environments.scenario import PRESET_SCENARIOS, Scenario
from sim.environments.tabletop import TabletopScene

LEVELS_DEG = (0.0, 0.25, 0.5, 1.0, 2.0)


def _err(level: float, rng: np.random.Generator) -> CalibrationError:
    w = rng.normal(0, np.radians(level) / np.sqrt(3), 3)
    t = rng.normal(0, 0.005 * level / np.sqrt(3), 3)
    return CalibrationError(se3.make_T(se3.so3_exp(w), t), 1.0)


def run(n_seeds: int) -> dict:
    board = TabletopScene(objects=(), with_robot=True, calib_board=True)
    board.reset(0)
    views = ViewpointSet(board)
    clean = NoiseConfig()
    out = {}
    for level in LEVELS_DEG:
        rows = []
        for seed in seeds_for("test", n_seeds, offset=500):
            rng = np.random.default_rng(seed)
            views.move_camera(int(rng.integers(len(views))))
            obs = {c: RGBDSensor(board, c).capture()[0] for c in ("overhead", "mobile")}
            errs = {c: _err(level, rng) for c in obs}
            rep = {c: corrupt(o, clean, errs[c], rng) for c, o in obs.items()}
            row = {}
            for c in obs:
                chk = check_camera(rep[c])
                row[f"{c}_reproj_px"] = chk.reproj_rms_reported_px
                row[f"{c}_pnp_rot_err_deg"] = float(np.degrees(se3.geodesic_distance(chk.T_WC_est[:3, :3], obs[c].T_WC[:3, :3])))
                row[f"{c}_pnp_trans_err_mm"] = float(1e3 * np.linalg.norm(chk.T_WC_est[:3, 3] - obs[c].T_WC[:3, 3]))
                row[f"{c}_true_drift_deg"] = float(np.degrees(np.linalg.norm(se3.so3_log(errs[c].dT[:3, :3]))))
            P = ct.disc_centres_W()
            uv = {c: detect_discs(rep[c]) for c in rep}
            mi1, di1 = match_to_model(uv["overhead"], rep["overhead"].intrinsics, rep["overhead"].T_WC, P, 30)
            mi2, di2 = match_to_model(uv["mobile"], rep["mobile"].intrinsics, rep["mobile"].T_WC, P, 30)
            common = sorted(set(mi1) & set(mi2))
            if common:
                a = uv["overhead"][[di1[list(mi1).index(i)] for i in common]]
                b = uv["mobile"][[di2[list(mi2).index(i)] for i in common]]
                X = triangulate(a, rep["overhead"].intrinsics, rep["overhead"].T_WC, b, rep["mobile"].intrinsics,
                                rep["mobile"].T_WC)
                row["triangulation_err_mm"] = float(1e3 * np.linalg.norm(X - P[common], axis=1).mean())
            # object localisation under the same overhead drift
            sc = Scenario.sample(PRESET_SCENARIOS["varied"], np.random.default_rng(seed), 1)
            sc_scene = TabletopScene(sc.objects, with_robot=True)
            sc.apply_appearance(sc_scene)
            sc_scene.reset(seed)
            o, g = RGBDSensor(sc_scene, "overhead").capture()
            res = perceive(corrupt(o, clean, errs["overhead"], rng))
            G = np.array([ob.T_WO[:3, 3] for ob in g.objects])
            loc = [float(np.linalg.norm(G - d.T_WO[:3, 3], axis=1).min()) for d in res.detections]
            row["object_loc_err_mm"] = 1e3 * float(np.mean(loc)) if loc else float("nan")
            sc_scene.close()
            rows.append(row)
        keys = rows[0].keys()
        out[str(level)] = {k: {"mean": float(np.nanmean([r.get(k, np.nan) for r in rows])),
                               "std": float(np.nanstd([r.get(k, np.nan) for r in rows]))} for k in keys}
        out[str(level)]["n"] = len(rows)
    board.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="reports/m2/calib_sweep.json")
    a = ap.parse_args()
    res = run(a.seeds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for lvl, r in res.items():
        print(lvl, {k: round(v["mean"], 3) for k, v in r.items() if isinstance(v, dict)})
