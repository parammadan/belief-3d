"""M2 inference: read ONLY obs_* files, run perception + tracking, write predictions.

Usage:  PYTHONPATH=. python scripts/run_m2_inference.py <dataset_root> <pred_dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from data.loaders import load_observation
from perception.pipeline import mask_to_rle, perceive
from tracking.tracker import MultiObjectTracker


def run(dataset: Path, out: Path) -> dict:
    manifest = json.loads((dataset / "manifest.json").read_text())
    cams = manifest["config"]["cameras"]
    out.mkdir(parents=True, exist_ok=True)
    timing = {"perceive_s": [], "track_s": []}
    for ep in manifest["episodes"]:
        tracker = MultiObjectTracker()
        frames = []
        for k in range(ep["n_frames"]):
            dets, seg_out = [], []
            t_frame = None
            for c in cams:
                obs = load_observation(dataset / ep["episode_id"], k, c)
                t_frame = obs.timestamp
                t0 = time.perf_counter()
                res = perceive(obs, seed=k)
                timing["perceive_s"].append(time.perf_counter() - t0)
                for d, s in zip(res.detections, res.segments):
                    seg_out.append({"camera": c, "color_class": d.color_class, "spec_name": d.spec_name,
                                    "T_WO": d.T_WO.tolist(), "rms": d.rms, "n_points": d.n_points, "completeness": d.completeness,
                                    "mask_rle": mask_to_rle(s.mask)})
                dets += res.detections
            t0 = time.perf_counter()
            tracks = tracker.step(t_frame, dets)
            timing["track_s"].append(time.perf_counter() - t0)
            frames.append({"frame": k, "timestamp": t_frame, "detections": seg_out,
                           "tracks": [{"track_id": tr.track_id, "color_class": tr.color_class,
                                       "spec_name": tr.spec_name, "T_WO": tr.T_WO.tolist(), "status": tr.status,
                                       "pos_cov": tr.kf.P[:3, :3].tolist()} for tr in tracks]})
        (out / f"{ep['episode_id']}.json").write_text(json.dumps({"episode_id": ep["episode_id"], "frames": frames}))
    return timing


if __name__ == "__main__":
    tm = run(Path(sys.argv[1]), Path(sys.argv[2]))
    import numpy as np
    print({k: f"median {1e3 * np.median(v):.1f} ms, p95 {1e3 * np.percentile(v, 95):.1f} ms, n={len(v)}"
           for k, v in tm.items()})
