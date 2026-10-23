# PROJECT_STATE

_Last updated: 2026-10-23_

## Current milestone

**M0: Environment and research audit. Exit criteria met.**
**Next: M1, simulation and sensors** (robot arm, viewpoint set, dataset writer, noise).

## M0 checklist

- [x] Environment inspected → `reports/env_audit.json` (`scripts/env_audit.py`)
- [x] GPU access located: NU Explorer `gpu:a100` partition (query only, **no job run**)
- [x] MuJoCo 3.17.0 installed; `cgl` offscreen backend verified headless
- [x] Initial literature scan + novelty assessment → `docs/literature_review.md`
- [x] `docs/research_question.md`, `docs/architecture.md`, `docs/coordinate_frames.md`, `docs/limitations.md`
- [x] Requirements traceability → `docs/traceability.md`
- [x] Repo skeleton (master prompt §21)
- [x] SE(3)/camera CPU reference + unit tests
- [x] Minimal MuJoCo scene, RGB + depth + segmentation from 2 cameras
- [x] Geometry validated against simulator state (sub-mm)
- [x] Reproducible demo → `reports/m0/` (`scripts/m0_demo.py`, EXP-M0-001)
- [x] CI workflow (GitHub Actions, Linux + OSMesa)

## Implemented modules

| Module | Purpose |
|---|---|
| `perception/geometry/se3.py` | hat/vee, exp/log (degenerate-safe), quaternions, SE(3), geodesic, look_at |
| `perception/geometry/camera.py` | `PinholeIntrinsics`: project, back-project, depth → points |
| `data/schemas/observation.py` | `Observation` (inference) / `GroundTruth` (eval) split |
| `sim/environments/tabletop.py` | Table, 4 primitives, fixed overhead cam, mocap mobile cam, seeded reset |
| `sim/cameras/rgbd.py` | RGB-D-seg capture, intrinsics from fovy, OpenCV extrinsics, max-range dropout |

## Verified tests (actually run)

`MUJOCO_GL=cgl .venv/bin/python -m pytest -q` → **36 passed** (macOS, 2026-10-23).
CI (ubuntu-latest, OSMesa): **36 passed**, ruff clean — https://github.com/parammadan/belief-3d/actions/runs/40039989867

## Environment

Apple M1, 8 cores, 8 GB RAM, macOS 26.4, **~2.4 GB free disk**. Python 3.12.13 venv. Apple clang 21,
**no cmake**. No CUDA locally. Not installed: torch, OpenCV, Open3D, onnxruntime, pybind11.
Cluster: `ssh explorer` works non-interactively; A100/H200/V100/T4 GPU partitions; CUDA 12.3–13.2 modules.

## Experimental results

| ID | Result |
|---|---|
| EXP-M0-001 | Sphere back-projection error mean 0.146 mm / max 0.291 mm (overhead), 0.161 / 0.289 mm (mobile). Table-plane error 0.001 mm (overhead), 0.067 mm (oblique mobile). |

These are geometry sanity checks, not research results.

## Open defects / risks

1. Disk: 2.4 GB free. Datasets must be generated on the cluster, or kept to smoke-test size here.
2. CGL depth precision is limited (`ARB_clip_control` warning). Returns >5 m are clamped to no-return.
3. Two `mujoco.Renderer`s in one process corrupt segmentation on CGL. Use `scene.renderer` only.
4. Leakage protection is schema-level only. The import-graph test is due in M2.
5. The literature check is abstract-level. SE(3)-PoseFlow is a serious overlap risk (see review).

## Remaining work (next steps, M1)

1. Add an arm + parallel gripper (candidate: MuJoCo Menagerie Franka Panda or UR5e + Robotiq;
   check licenses and disk size first).
2. A finite set of reachable, collision-free viewpoints for camera B.
3. Depth noise / dropout / RGB noise / extrinsic-perturbation models (seeded).
4. Dataset writer + versioned manifest + validators (§8). Measure throughput and MB/frame first.
5. Replay test: the same seed and actions give bit-identical observations across runs.

## Exact commands to continue

```bash
cd ~/belief-3d
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # if .venv is missing
export MUJOCO_GL=cgl PYTHONPATH=.
.venv/bin/python -m pytest -q
.venv/bin/python scripts/env_audit.py
.venv/bin/python scripts/m0_demo.py --seed 0
```
