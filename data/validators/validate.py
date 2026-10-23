"""Dataset validator (master prompt §8). Returns a list of human-readable errors.

Usage:  python -m data.validators.validate <dataset_root>
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from data.generators.episodes import dequantize_depth
from data.schemas.splits import split_of
from perception.geometry import se3
from perception.geometry.camera import PinholeIntrinsics


def _K(d: dict) -> PinholeIntrinsics:
    return PinholeIntrinsics(**d)


def validate_dataset(root: Path, check_hashes: bool = True) -> list[str]:
    root = Path(root)
    errs: list[str] = []
    m = json.loads((root / "manifest.json").read_text())
    cams = m["config"]["cameras"]
    for ep in m["episodes"]:
        eid, d = ep["episode_id"], root / ep["episode_id"]
        if split_of(ep["seed"]) != ep["split"]:
            errs.append(f"{eid}: split {ep['split']} != split_of(seed)")
        if check_hashes:
            for f, h in ep["files"].items():
                if not (d / f).exists():
                    errs.append(f"{eid}/{f}: missing")
                elif hashlib.sha256((d / f).read_bytes()).hexdigest() != h:
                    errs.append(f"{eid}/{f}: sha256 mismatch")
        identities: dict[int, str] | None = None
        for c in cams:
            last_t = -math.inf
            for k in range(ep["n_frames"]):
                stem = f"{k:04d}_{c}"
                where = f"{eid}/{stem}"
                o = np.load(d / f"obs_{stem}.npz")
                oj = json.loads((d / f"obs_{stem}.json").read_text())
                g = np.load(d / f"gt_{stem}.npz")
                gj = json.loads((d / f"gt_{stem}.json").read_text())
                K, Kt = _K(oj["intrinsics"]), _K(gj["true_intrinsics"])
                rgb, depth, seg = o["rgb"], dequantize_depth(o["depth_u16"]), g["segmentation"]
                # finite / valid depth
                if not np.isfinite(depth).all() or (depth < 0).any():
                    errs.append(f"{where}: non-finite or negative depth")
                # dimensions
                hw = (K.height, K.width)
                if rgb.shape != (*hw, 3) or depth.shape != hw or seg.shape != hw:
                    errs.append(f"{where}: shape mismatch rgb{rgb.shape} depth{depth.shape} seg{seg.shape} K{hw}")
                    continue
                # transforms
                for name, T in (("T_WC", oj["T_WC"]), ("true_T_WC", gj["true_T_WC"])):
                    T = np.array(T)
                    if T.shape != (4, 4) or not se3.is_rotation(T[:3, :3]) or not np.allclose(T[3], [0, 0, 0, 1]):
                        errs.append(f"{where}: invalid {name}")
                # calibration consistency: reported vs true within the declared noise envelope
                Tr, Tt = np.array(oj["T_WC"]), np.array(gj["true_T_WC"])
                rot_sig = m["noise"]["extrinsic_rot_sigma"] or 0.0
                tr_sig = m["noise"]["extrinsic_trans_sigma"] or 0.0
                if se3.geodesic_distance(Tr[:3, :3], Tt[:3, :3]) > 6 * rot_sig * math.sqrt(3) + 1e-9:
                    errs.append(f"{where}: reported extrinsic rotation outside noise envelope")
                if np.linalg.norm(Tr[:3, 3] - Tt[:3, 3]) > 6 * tr_sig * math.sqrt(3) + 1e-9:
                    errs.append(f"{where}: reported extrinsic translation outside noise envelope")
                if (Kt.cx, Kt.cy, Kt.width, Kt.height) != (K.cx, K.cy, K.width, K.height):
                    errs.append(f"{where}: principal point / size differ between reported and true K")
                # timestamps strictly increasing
                if not oj["timestamp"] > last_t or oj["timestamp"] != gj["timestamp"]:
                    errs.append(f"{where}: timestamp not increasing or obs/gt mismatch")
                last_t = oj["timestamp"]
                # identity consistency
                ids = {ob["instance_id"]: ob["class_name"] for ob in gj["objects"]}
                if identities is None:
                    identities = ids
                elif ids != identities:
                    errs.append(f"{where}: object identities changed")
                # label alignment: well-visible object centres project into their own mask (true calib)
                for ob in gj["objects"]:
                    T_WO = np.array(ob["T_WO"])
                    if not se3.is_rotation(T_WO[:3, :3]):
                        errs.append(f"{where}: invalid T_WO for {ob['class_name']}")
                    mask = seg == ob["instance_id"]
                    if ob["visible_fraction"] < 0.9 or mask.sum() < 30:
                        continue
                    uv, front = Kt.project(se3.transform_points(se3.inv_T(Tt), T_WO[:3, 3][None]))
                    u, v = np.round(uv[0]).astype(int) if front[0] else (-1, -1)
                    if not (0 <= u < K.width and 0 <= v < K.height) or not mask[v, u]:
                        errs.append(f"{where}: {ob['class_name']} centre projects outside its mask")
                    bb = ob["bbox_visible_xyxy"]
                    ys, xs = np.nonzero(mask)
                    if bb != [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]:
                        errs.append(f"{where}: bbox does not match mask for {ob['class_name']}")
    return errs


if __name__ == "__main__":
    problems = validate_dataset(Path(sys.argv[1]))
    print("\n".join(problems) if problems else "OK: no validation errors")
    sys.exit(1 if problems else 0)
