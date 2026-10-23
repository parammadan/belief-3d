"""The inference API must only ever see an Observation, never GroundTruth."""
import dataclasses

import numpy as np
import pytest

from data.schemas.observation import FORBIDDEN_OBSERVATION_FIELDS, GroundTruth, Observation
from perception.geometry.camera import PinholeIntrinsics

K = PinholeIntrinsics(100.0, 100.0, 1.5, 1.5, 4, 4)


def make_obs():
    return Observation("c", 0.0, np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.float32), K, np.eye(4))


def test_observation_has_no_label_fields():
    names = {f.name for f in dataclasses.fields(Observation)}
    assert not names & FORBIDDEN_OBSERVATION_FIELDS
    assert not names & {f.name for f in dataclasses.fields(GroundTruth)} - {"camera_name", "timestamp"}


def test_observation_rejects_attached_labels():
    obs = make_obs()
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        obs.segmentation = np.zeros((4, 4))  # type: ignore[attr-defined]


def test_observation_validates_shapes():
    with pytest.raises(ValueError):
        Observation("c", 0.0, np.zeros((3, 4, 3), np.uint8), np.zeros((4, 4), np.float32), K, np.eye(4))
