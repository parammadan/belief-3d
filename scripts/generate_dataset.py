"""Generate a versioned synthetic dataset and report throughput + storage.

Usage:  MUJOCO_GL=cgl PYTHONPATH=. python scripts/generate_dataset.py \\
            --out datasets/smoke --split train --episodes 3 --frames 10 --noise nominal
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data.generators.episodes import EpisodeConfig, write_dataset
from data.schemas.splits import seeds_for
from data.validators.validate import validate_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--noise", default="nominal")
    ap.add_argument("--scenario", default="static")
    ap.add_argument("--no-robot", action="store_true")
    args = ap.parse_args()
    cfg = EpisodeConfig(n_frames=args.frames, noise=args.noise, scenario=args.scenario, with_robot=not args.no_robot)
    seeds = seeds_for(args.split, args.episodes, args.offset)
    t0 = time.perf_counter()
    m = write_dataset(Path(args.out), args.name or Path(args.out).name, seeds, cfg)
    gen_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    errs = validate_dataset(Path(args.out))
    val_s = time.perf_counter() - t0
    n_img = args.episodes * args.frames * len(cfg.cameras)
    size = sum(f.stat().st_size for f in Path(args.out).rglob("*") if f.is_file())
    stats = {
        "seeds": [seeds[0], seeds[-1]], "camera_frames": n_img, "generate_s": round(gen_s, 2),
        "camera_frames_per_s": round(n_img / gen_s, 2), "bytes_total": size,
        "kib_per_camera_frame": round(size / n_img / 1024, 1), "validate_s": round(val_s, 2),
        "validation_errors": len(errs), "config_hash": m["config_hash"], "git_commit": m["git_commit"],
    }
    (Path(args.out) / "generation_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    for e in errs[:20]:
        print("  ", e)


if __name__ == "__main__":
    main()
