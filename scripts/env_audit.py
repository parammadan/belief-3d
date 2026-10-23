"""Record the actual development environment (M0). Facts only — nothing assumed.

Usage:  MUJOCO_GL=cgl python scripts/env_audit.py [--out reports/env_audit.json]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys

PACKAGES = ["mujoco", "numpy", "scipy", "cv2", "open3d", "torch", "onnxruntime", "tensorrt", "pybind11", "pytest"]


def _cmd(*args: str) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    try:
        return subprocess.run(args, check=False, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception as e:  # noqa: BLE001 — audit must never crash
        return f"error: {e}"


def _render_backends() -> dict[str, str]:
    out = {}
    for gl in ("cgl", "glfw", "egl", "osmesa"):
        code = (
            "import mujoco;m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\".1\"/>"
            "<camera name=\"c\" pos=\"0 0 1\"/></worldbody></mujoco>');d=mujoco.MjData(m);mujoco.mj_forward(m,d);"
            "r=mujoco.Renderer(m,32,32);r.enable_depth_rendering();r.update_scene(d,'c');print(float(r.render()[16,16]))"
        )
        p = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": gl})
        out[gl] = f"ok depth={float(p.stdout.strip()):.4f}" if p.returncode == 0 else p.stderr.strip().splitlines()[-1][:120]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/env_audit.json")
    args = ap.parse_args()
    pkgs = {}
    for name in PACKAGES:
        try:
            pkgs[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except Exception:  # noqa: BLE001
            pkgs[name] = None
    du = shutil.disk_usage(os.path.expanduser("~"))
    report = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cmd("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "cxx": _cmd("c++", "--version"),
        "cmake": _cmd("cmake", "--version"),
        "nvidia_smi": _cmd("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
        "disk_free_gb": round(du.free / 1e9, 2),
        "packages": pkgs,
        "mujoco_render_backends": _render_backends(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
