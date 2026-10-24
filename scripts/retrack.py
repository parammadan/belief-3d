"""Re-run ONLY the tracker on stored detections (perception is not repeated).

Lets tracker settings be studied cheaply and exactly: same detections, different
tracker config. Reads a pred dir written by run_m2_inference.py; writes a new one.
Usage:  PYTHONPATH=. python scripts/retrack.py <pred_in> <pred_out> --calib-rot-deg 1.0 --calib-trans-mm 5
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from tracking.tracker import Detection, MultiObjectTracker, TrackerConfig


def retrack(src: Path, dst: Path, cfg: TrackerConfig, cameras: list[str] | None = None) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    if cameras:
        if not set(cameras) <= set(meta["cameras"]):
            raise ValueError(f"{cameras} not all in source cameras {meta['cameras']}")
        meta["cameras"] = cameras
    meta["tracker_config"] = cfg.__dict__ | {}
    (dst / "meta.json").write_text(json.dumps(meta))
    for f in sorted(src.glob("ep_*.json")):
        P = json.loads(f.read_text())
        tr = MultiObjectTracker(cfg)
        for fr in P["frames"]:
            dets = [Detection(d["camera"], fr["timestamp"], d["color_class"], d["spec_name"], np.array(d["T_WO"]),
                              d["rms"], d["n_points"], d.get("completeness", 1.0), d.get("range_m", 1.0))
                    for d in fr["detections"] if d["camera"] in meta["cameras"]]
            tracks = tr.step(fr["timestamp"], dets)
            fr["tracks"] = [{"track_id": t.track_id, "color_class": t.color_class, "spec_name": t.spec_name,
                             "T_WO": t.T_WO.tolist(), "status": t.status, "pos_cov": t.kf.P[:3, :3].tolist()}
                            for t in tracks]
        (dst / f.name).write_text(json.dumps(P))
    if (src / "timing.json").exists():
        shutil.copy(src / "timing.json", dst / "timing.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--calib-rot-deg", type=float, default=0.0)
    ap.add_argument("--calib-trans-mm", type=float, default=0.0)
    ap.add_argument("--cameras", default=None)
    a = ap.parse_args()
    retrack(Path(a.src), Path(a.dst), TrackerConfig(calib_rot_sigma=np.radians(a.calib_rot_deg),
                                                    calib_trans_sigma=a.calib_trans_mm / 1e3),
            a.cameras.split(",") if a.cameras else None)
