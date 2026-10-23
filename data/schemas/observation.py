"""Data schemas that enforce the inference / evaluation split.

`Observation` is the ONLY thing the inference stack may consume: sensor data
plus the camera state the robot itself would know (calibrated intrinsics and
the commanded/kinematic camera extrinsics). `GroundTruth` carries simulator
labels and is consumed by dataset writers and evaluators only. The two types
share no fields that carry labels, and `Observation` rejects any attempt to
attach extra attributes (frozen + slots).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from perception.geometry.camera import PinholeIntrinsics

SCHEMA_VERSION = "0.1.0"

# Field names an Observation must never carry. Checked by tests/unit/test_leakage.py.
FORBIDDEN_OBSERVATION_FIELDS = frozenset(
    {"segmentation", "seg", "object_poses", "poses", "visibility", "occlusion", "instance_ids", "gt"}
)


@dataclass(frozen=True, slots=True)
class Observation:
    camera_name: str
    timestamp: float                 # simulation time, seconds
    rgb: np.ndarray                  # (H, W, 3) uint8
    depth: np.ndarray                # (H, W) float32, z_C in meters; 0 = no return
    intrinsics: PinholeIntrinsics
    T_WC: np.ndarray                 # (4, 4) camera pose in world, OpenCV camera frame

    def __post_init__(self) -> None:
        h, w = self.intrinsics.height, self.intrinsics.width
        if self.rgb.shape != (h, w, 3) or self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be ({h},{w},3) uint8, got {self.rgb.shape} {self.rgb.dtype}")
        if self.depth.shape != (h, w):
            raise ValueError(f"depth must be ({h},{w}), got {self.depth.shape}")
        if self.T_WC.shape != (4, 4):
            raise ValueError("T_WC must be 4x4")


@dataclass(frozen=True)
class ObjectState:
    instance_id: int
    class_name: str
    T_WO: np.ndarray                 # (4, 4) object pose in world
    symmetry: str                    # "none" | "cylindrical_z" | "spherical" | "box"


@dataclass(frozen=True)
class GroundTruth:
    camera_name: str
    timestamp: float
    segmentation: np.ndarray         # (H, W) int32 instance id; -1 = background/static
    objects: tuple[ObjectState, ...]
    visible_fraction: dict[int, float] = field(default_factory=dict)
