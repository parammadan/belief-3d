"""M4 inference: one perception pass per frame feeds BOTH trackers.

B1 = MultiObjectTracker (unimodal CV-Kalman), B2 = BeliefTracker (multimodal particles).
Both get the same detections and the same calibration prior, so differences come
only from the state representation. Reads ONLY obs_* files.
Usage:  PYTHONPATH=. python scripts/run_m4_inference.py <dataset> <out_dir> [--cameras overhead,mobile]
            [--calib-rot-deg 1 --calib-trans-mm 5] [--particles 256] [--no-free-space]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data.loaders import load_observation
from perception.pipeline import mask_to_rle, perceive
from tracking.belief.particle import BeliefConfig, DepthView
from tracking.belief_tracker import BeliefTracker
from tracking.tracker import MultiObjectTracker, TrackerConfig


def _track_rows(tracks, belief: bool, tracker=None):
    rows = []
    for tr in tracks:
        T = tr.T_WO
        if belief:
            cov, extra = tracker.reported_cov(tr), {"rot_spread_deg": float(np.degrees(tr.belief.rotation_spread())),
                                                       "n_modes": tr.belief.n_modes()}
        else:
            cov, extra = tr.kf.P[:3, :3], {}
        rows.append({"track_id": tr.track_id, "color_class": tr.color_class, "spec_name": tr.spec_name,
                     "T_WO": T.tolist(), "status": tr.status, "pos_cov": cov.tolist()} | extra)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--cameras", default=None)
    ap.add_argument("--calib-rot-deg", type=float, default=1.0)
    ap.add_argument("--calib-trans-mm", type=float, default=5.0)
    ap.add_argument("--particles", type=int, default=256)
    ap.add_argument("--no-free-space", action="store_true")
    ap.add_argument("--no-relative-calib", action="store_true")
    a = ap.parse_args()
    ds, out = Path(a.dataset), Path(a.out)
    m = json.loads((ds / "manifest.json").read_text())
    cams = a.cameras.split(",") if a.cameras else m["config"]["cameras"]
    tcfg = TrackerConfig(calib_rot_sigma=np.radians(a.calib_rot_deg), calib_trans_sigma=a.calib_trans_mm / 1e3)
    bcfg = BeliefConfig(n_particles=a.particles)
    meta = {"cameras": cams, "dataset": m["name"], "dataset_config_hash": m["config_hash"],
            "tracker": {"calib_rot_deg": a.calib_rot_deg, "calib_trans_mm": a.calib_trans_mm},
            "belief": {"n_particles": a.particles, "free_space": not a.no_free_space,
                       "relative_calibration": not a.no_relative_calib}}
    for tag in ("b1", "b2"):
        (out / tag).mkdir(parents=True, exist_ok=True)
        (out / tag / "meta.json").write_text(json.dumps(meta | {"method": tag}))
    timing = {"perceive_s": [], "b1_s": [], "b2_s": []}
    for ep in m["episodes"]:
        b1, b2 = MultiObjectTracker(tcfg), BeliefTracker(tcfg, bcfg, seed=ep["seed"], use_free_space=not a.no_free_space,
                                                      relative_calibration=not a.no_relative_calib)
        fr1, fr2 = [], []
        for k in range(ep["n_frames"]):
            dets, seg_rows, views, t_frame = [], [], {}, None
            for c in cams:
                obs = load_observation(ds / ep["episode_id"], k, c)
                t_frame = obs.timestamp
                t0 = time.perf_counter()
                res = perceive(obs, seed=k)
                timing["perceive_s"].append(time.perf_counter() - t0)
                views[c] = DepthView(obs.depth, obs.intrinsics, obs.T_WC)
                for d, s in zip(res.detections, res.segments):
                    seg_rows.append({"camera": c, "color_class": d.color_class, "spec_name": d.spec_name,
                                     "T_WO": d.T_WO.tolist(), "rms": d.rms, "n_points": d.n_points,
                                     "completeness": d.completeness, "range_m": d.range_m,
                                     "n_hypotheses": len(d.hypotheses), "mask_rle": mask_to_rle(s.mask)})
                dets += res.detections
            t0 = time.perf_counter()
            tr1 = b1.step(t_frame, dets)
            timing["b1_s"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            tr2 = b2.step(t_frame, dets, views)
            timing["b2_s"].append(time.perf_counter() - t0)
            fr1.append({"frame": k, "timestamp": t_frame, "detections": seg_rows, "tracks": _track_rows(tr1, False)})
            fr2.append({"frame": k, "timestamp": t_frame, "detections": seg_rows, "tracks": _track_rows(tr2, True, b2),
                        "calib_offsets": {c: v.tolist() for c, v in b2.last_offsets.items()}})
        for tag, frs in (("b1", fr1), ("b2", fr2)):
            (out / tag / f"{ep['episode_id']}.json").write_text(json.dumps({"episode_id": ep["episode_id"], "frames": frs}))
    summary = {k: {"median_ms": 1e3 * float(np.median(v)), "p95_ms": 1e3 * float(np.percentile(v, 95)), "n": len(v)}
               for k, v in timing.items()}
    for tag in ("b1", "b2"):
        (out / tag / "timing.json").write_text(json.dumps(summary))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
