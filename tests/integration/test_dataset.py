import itertools
import json
import shutil

import numpy as np
import pytest

from data.generators.episodes import DEPTH_SCALE, EpisodeConfig, dequantize_depth, quantize_depth, write_dataset
from data.loaders import load_ground_truth, load_observation
from data.schemas.splits import SPLITS, seeds_for, split_of
from data.validators.validate import validate_dataset

CFG = EpisodeConfig(n_frames=3, noise="harsh")


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds") / "a"
    write_dataset(root, "a", [0, 1], CFG)
    return root


def test_splits_are_disjoint_and_cover_seeds():
    ranges = sorted(SPLITS.values())
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(ranges))
    assert split_of(seeds_for("test", 3)[0]) == "test"
    with pytest.raises(ValueError):
        split_of(1_200_000)


def test_depth_quantisation_bound():
    d = np.random.default_rng(0).uniform(0.0, 5.0, (50, 50)).astype(np.float32)
    d[0, 0] = 0.0
    r = dequantize_depth(quantize_depth(d))
    assert np.abs(r - d).max() <= DEPTH_SCALE / 2 + 1e-6 and r[0, 0] == 0.0
    with pytest.raises(ValueError):
        quantize_depth(np.full((2, 2), 7.0, np.float32))


def test_generated_dataset_validates(ds):
    assert validate_dataset(ds) == []


def test_replay_is_bit_identical(ds, tmp_path):
    write_dataset(tmp_path / "b", "b", [0, 1], CFG)
    ma = json.loads((ds / "manifest.json").read_text())
    mb = json.loads((tmp_path / "b" / "manifest.json").read_text())
    for ea, eb in zip(ma["episodes"], mb["episodes"]):
        assert ea["files"] == eb["files"]          # every file's sha256 matches
    assert ma["episodes"][0]["files"] != ma["episodes"][1]["files"]


def test_loaders_roundtrip(ds):
    ep = ds / "ep_0000000"
    obs = load_observation(ep, 2, "mobile")
    gt = load_ground_truth(ep, 2, "mobile")
    assert obs.rgb.shape == (240, 320, 3) and gt.segmentation.shape == (240, 320)
    assert obs.timestamp == gt.timestamp and len(gt.objects) == 4


@pytest.mark.parametrize("corruption, expect", [
    ("nan_depth", "non-finite"),
    ("tamper_hash", "sha256 mismatch"),
    ("rename_object", "identities changed"),
    ("move_object", "projects outside its mask"),
    ("bad_rotation", "invalid T_WC"),
    ("time_reversal", "timestamp"),
])
def test_validator_catches_corruption(ds, tmp_path, corruption, expect):
    root = tmp_path / "c"
    shutil.copytree(ds, root)
    ep = root / "ep_0000000"
    check_hashes = corruption == "tamper_hash"
    if corruption == "nan_depth":
        a = dict(np.load(ep / "obs_0001_overhead.npz"))
        # uint16 cannot hold NaN: emulate a writer bug by storing float depth
        np.savez_compressed(ep / "obs_0001_overhead.npz", rgb=a["rgb"],
                            depth_u16=np.full(a["depth_u16"].shape, np.nan, np.float32))
    elif corruption == "tamper_hash":
        p = ep / "obs_0001_overhead.json"
        p.write_text(p.read_text() + " ")
    else:
        p = ep / ("obs_0001_overhead.json" if corruption in ("bad_rotation", "time_reversal") else "gt_0001_overhead.json")
        j = json.loads(p.read_text())
        if corruption == "rename_object":
            j["objects"][0]["class_name"] = "impostor"
        elif corruption == "move_object":
            j["objects"][1]["T_WO"][0][3] += 0.2
        elif corruption == "bad_rotation":
            j["T_WC"][0][0] *= 2.0
        elif corruption == "time_reversal":
            j["timestamp"] = -1.0
        p.write_text(json.dumps(j))
    errs = validate_dataset(root, check_hashes=check_hashes)
    assert any(expect in e for e in errs), errs
