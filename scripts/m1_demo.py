"""M1 demo: an occlusion episode seen from both cameras, as one PNG strip.

Top row overhead, bottom row mobile; columns are frames before, during and
after the scheduled occlusion of one object. Also writes per-frame visibility.
Usage:  MUJOCO_GL=cgl PYTHONPATH=. python scripts/m1_demo.py --seed 3
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from data.generators.episodes import EpisodeConfig, run_episode
from scripts.m0_demo import write_png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="reports/m1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    frames = list(run_episode(args.seed, EpisodeConfig(n_frames=12, scenario="occlusion", noise="nominal")))
    sc = frames[0].scenario
    a, b = sc.occl_frames
    pick = [a - 1, a, b - 1, b]
    rows = [np.concatenate([frames[k].obs[c].rgb for k in pick], axis=1) for c in ("overhead", "mobile")]
    strip = np.concatenate(rows, axis=0)[::2, ::2]           # half-res keeps the committed PNG small
    write_png(f"{args.out}/occlusion_strip.png", np.ascontiguousarray(strip))
    log = {
        "seed": args.seed, "objects": [o.name for o in sc.objects], "occluded": sc.objects[sc.occl_target].name,
        "occl_frames": [a, b], "columns": pick,
        "visible_fraction": {c: [round(fr.gt[c].visible_fraction[sc.occl_target], 3) for fr in frames]
                             for c in ("overhead", "mobile")},
    }
    with open(f"{args.out}/occlusion_log.json", "w") as f:
        json.dump(log, f, indent=1)
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
