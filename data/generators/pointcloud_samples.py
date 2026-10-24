"""Training samples for the neural pose proposer: per-object point clouds + labels.

Each visible object instance in each camera frame becomes one sample:
  points  (P, 3) float16  segment points, world frame via the *reported* extrinsics
                          (exactly what inference sees), centred on their centroid
  centroid (3,)  float32  the centre that was subtracted
  cls      int            index into CATALOG
  T_WO     (4, 4) float32 ground-truth pose (label)
  vis      float          visible fraction (metadata only)
GT masks pick the points at *generation* time (labels are legal for training);
inference uses the classical segmentation masks instead. Storing ~3 KB per
sample instead of ~270 KB per image frame is what makes large training sets
feasible.

Usage (one shard; launch many in parallel as a CPU array job):
  MUJOCO_GL=egl PYTHONPATH=. python -m data.generators.pointcloud_samples --split train --offset 0 \\
      --episodes 50 --frames 10 --out shards/train_0000.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from data.generators.episodes import EpisodeConfig, run_episode
from data.schemas.objects import CATALOG
from data.schemas.splits import seeds_for
from perception.geometry import se3

CLS = {o.name: i for i, o in enumerate(CATALOG)}
N_POINTS = 512


def sample_points(P: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return P[rng.choice(len(P), n, replace=len(P) < n)]


def generate(split: str, offset: int, episodes: int, frames: int, noise: str, scenario: str,
             min_vis: float = 0.1, min_px: int = 30) -> dict[str, np.ndarray]:
    rows = {k: [] for k in ("points", "centroid", "cls", "T_WO", "vis", "seed", "frame", "cam")}
    cams = ("overhead", "mobile")
    for seed in seeds_for(split, episodes, offset):
        rng = np.random.default_rng(seed + 7_000_000)
        for fr in run_episode(seed, EpisodeConfig(n_frames=frames, noise=noise, scenario=scenario)):
            for ci, c in enumerate(cams):
                obs, gt = fr.obs[c], fr.gt[c]
                for ob in gt.objects:
                    m = gt.segmentation == ob.instance_id
                    if m.sum() < min_px or gt.visible_fraction.get(ob.instance_id, 0) < min_vis:
                        continue
                    P_C, _ = obs.intrinsics.depth_to_points(obs.depth, m)
                    if len(P_C) < min_px:
                        continue
                    P_W = se3.transform_points(obs.T_WC, P_C)
                    cen = P_W.mean(0)
                    rows["points"].append(sample_points(P_W - cen, N_POINTS, rng).astype(np.float16))
                    rows["centroid"].append(cen.astype(np.float32))
                    rows["cls"].append(CLS[ob.class_name])
                    rows["T_WO"].append(ob.T_WO.astype(np.float32))
                    rows["vis"].append(gt.visible_fraction[ob.instance_id])
                    rows["seed"].append(seed)
                    rows["frame"].append(fr.index)
                    rows["cam"].append(ci)
    return {k: np.asarray(v) for k, v in rows.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--noise", default="nominal")
    ap.add_argument("--scenario", default="dynamic")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()
    d = generate(a.split, a.offset, a.episodes, a.frames, a.noise, a.scenario)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    meta = {"args": vars(a), "n_samples": len(d["cls"]), "seconds": round(time.perf_counter() - t0, 1),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "catalog": [o.name for o in CATALOG]}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
