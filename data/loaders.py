"""Readers for datasets written by data.generators.episodes.

`load_observation` touches only obs_* files and returns an Observation, so
inference code can be fed from disk without any path to labels.
`load_ground_truth` is for evaluators.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.generators.episodes import dequantize_depth
from data.schemas.observation import GroundTruth, ObjectState, Observation
from perception.geometry.camera import PinholeIntrinsics


def load_observation(episode_dir: Path, frame: int, camera: str) -> Observation:
    stem = Path(episode_dir) / f"obs_{frame:04d}_{camera}"
    arr = np.load(f"{stem}.npz")
    meta = json.loads(Path(f"{stem}.json").read_text())
    return Observation(camera, meta["timestamp"], arr["rgb"], dequantize_depth(arr["depth_u16"]),
                       PinholeIntrinsics(**meta["intrinsics"]), np.array(meta["T_WC"]))


def load_ground_truth(episode_dir: Path, frame: int, camera: str) -> GroundTruth:
    stem = Path(episode_dir) / f"gt_{frame:04d}_{camera}"
    seg = np.load(f"{stem}.npz")["segmentation"].astype(np.int32)
    meta = json.loads(Path(f"{stem}.json").read_text())
    objs = tuple(ObjectState(o["instance_id"], o["class_name"], np.array(o["T_WO"]), o["symmetry"])
                 for o in meta["objects"])
    vis = {o["instance_id"]: o["visible_fraction"] for o in meta["objects"]}
    return GroundTruth(camera, meta["timestamp"], seg, objs, vis)
