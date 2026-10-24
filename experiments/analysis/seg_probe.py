"""Val-only segmentation scoring vs GT masks (TP/FN/FP at IoU >= 0.5, mean IoU), per scenario.

Usage: MUJOCO_GL=cgl PYTHONPATH=. python experiments/analysis/seg_probe.py --scenario clutter --episodes 8
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from data.generators.episodes import EpisodeConfig, run_episode
from data.schemas.splits import seeds_for
from perception.segmentation.classical import SegmenterConfig, segment


def score(scenario, noise, episodes, cfg):
    tp = fn = fp = 0
    ious = []
    for seed in seeds_for("val", episodes, 900):
        for fr in run_episode(seed, EpisodeConfig(n_frames=1, scenario=scenario, noise=noise)):
            for c in ("overhead", "mobile"):
                obs, gt = fr.obs[c], fr.gt[c]
                segs, _ = segment(obs, cfg)
                used = set()
                for ob in gt.objects:
                    g = gt.segmentation == ob.instance_id
                    if g.sum() < 25:
                        continue
                    best = max(((np.logical_and(s.mask, g).sum() / np.logical_or(s.mask, g).sum(), i)
                                for i, s in enumerate(segs)), default=(0.0, -1))
                    if best[0] >= 0.5:
                        tp += 1
                        ious.append(best[0])
                        used.add(best[1])
                    else:
                        fn += 1
                fp += len(segs) - len(used)
    return {"scenario": scenario, "noise": noise, "tp": tp, "fn": fn, "fp": fp, "miou": round(float(np.mean(ious)), 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="clutter")
    ap.add_argument("--noise", default="nominal")
    ap.add_argument("--episodes", type=int, default=8)
    a = ap.parse_args()
    print(json.dumps(score(a.scenario, a.noise, a.episodes, SegmenterConfig()) | {"hue_gate": True}))
    print(json.dumps(score(a.scenario, a.noise, a.episodes, SegmenterConfig(grow_hue_min_sat=2.0)) | {"hue_gate": False}))
