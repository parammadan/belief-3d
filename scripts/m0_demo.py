"""M0 demo: render one RGB-D frame per camera and validate its geometry.

Writes reports/m0/<cam>_rgb.png, <cam>_depth.png and reports/m0/geometry_report.json.
Usage:  MUJOCO_GL=cgl python scripts/m0_demo.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import zlib

import numpy as np

from perception.geometry import se3
from sim.cameras.rgbd import RGBDSensor
from sim.environments.tabletop import TABLE_TOP_Z, TabletopScene


def write_png(path: str, img: np.ndarray) -> None:
    """Minimal PNG writer (uint8 gray or RGB) so M0 needs no imaging dependency."""
    h, w = img.shape[:2]
    ctype = 2 if img.ndim == 3 else 0
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def validate(obs, gt) -> dict:
    """Geometry checks using GT — this is evaluator code, not inference."""
    out = {"objects": {}}
    for o in gt.objects:
        P, _ = obs.intrinsics.depth_to_points(obs.depth, gt.segmentation == o.instance_id)
        Pw = se3.transform_points(obs.T_WC, P)
        entry = {"pixels": len(P)}
        if o.symmetry == "spherical":
            r = np.abs(np.linalg.norm(Pw - o.T_WO[:3, 3], axis=1) - 0.028)
            entry |= {"sphere_surface_err_mean_mm": float(r.mean() * 1e3), "sphere_surface_err_max_mm": float(r.max() * 1e3)}
        out["objects"][o.class_name] = entry
    P, _ = obs.intrinsics.depth_to_points(obs.depth, gt.segmentation == -1)
    z = se3.transform_points(obs.T_WC, P)[:, 2]
    on = np.abs(z - TABLE_TOP_Z) < 0.02
    out["table_plane_err_mean_mm"] = float(np.abs(z[on] - TABLE_TOP_Z).mean() * 1e3)
    out["table_pixels"] = int(on.sum())
    out["depth_no_return_fraction"] = float((obs.depth == 0).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/m0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scene = TabletopScene()
    scene.reset(seed=args.seed)
    report = {"seed": args.seed, "mujoco_gl": os.environ.get("MUJOCO_GL"), "cameras": {}}
    for cam in ("overhead", "mobile"):
        obs, gt = RGBDSensor(scene, cam).capture()
        write_png(f"{args.out}/{cam}_rgb.png", obs.rgb)
        d = obs.depth.copy()
        valid = d > 0
        lo, hi = d[valid].min(), d[valid].max()
        vis = np.zeros(d.shape, np.uint8)
        vis[valid] = (255 * (1 - (d[valid] - lo) / (hi - lo))).astype(np.uint8)
        write_png(f"{args.out}/{cam}_depth.png", vis)
        report["cameras"][cam] = {
            "intrinsics": {k: getattr(obs.intrinsics, k) for k in ("fx", "fy", "cx", "cy", "width", "height")},
            "T_WC": np.round(obs.T_WC, 6).tolist(),
            "geometry": validate(obs, gt),
        }
    scene.close()
    with open(f"{args.out}/geometry_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["cameras"], indent=1))


if __name__ == "__main__":
    main()
