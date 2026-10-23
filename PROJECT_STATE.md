# PROJECT_STATE

_Last updated: 2026-10-23_

## Current milestone

**M1: Simulation and sensors. Exit criteria met.** Scene replay is bit-identical (same seed →
identical sha256 for every dataset file), and projected ground truth aligns with rendered
observations (validator label-alignment check, 0 errors).

**Next: M2, classical perception baseline** (colour/geometry segmentation, 3D centroid and
pose from depth, symmetry-aware error metrics, EKF + Hungarian tracking, an independent
evaluation script on held-out seeds).

## Milestones

- [x] **M0**: environment audit, geometry core, minimal scene, validated camera geometry
- [x] **M1**: robot, viewpoints, noise, scenarios, dataset writer/validator
- [ ] M2 classical perception · [ ] M3 neural pose · [ ] M4 belief tracking · [ ] M5 active perception
- [ ] M6 closed-loop manipulation · [ ] M7 systems optimisation · [ ] M8 final study

## M1 checklist

- [x] Franka Panda (Menagerie @ `f3b6013`, Apache-2.0) fetched by `scripts/fetch_assets.py`, sha256-locked
- [x] Panda attached via `MjSpec`; `panda/tcp` site; parked observe pose (arm covers 4 % of overhead view vs 47 % at home)
- [x] DLS IK on a scratch `MjData` (never perturbs the sim), joint limits, yaw-symmetric top-down grasp IK
- [x] Physics check: scripted top-down grasps lift both boxes and the cylinder > 0.12 m
- [x] Finite viewpoint set for camera B: 14/16 valid (2 blocked by the robot), straight-path clearance via `mj_ray`, travel-time model
- [x] Seeded noise: RGB, range-dependent depth noise, random + edge dropout, per-episode extrinsic/focal errors (`clean`/`nominal`/`harsh`)
- [x] Per-object visible fraction (render-in-isolation), robot segmentation label −2
- [x] Scenarios: 3–6 objects from a 7-object catalog incl. look-alike twins; light/colour jitter; scheduled full occlusion; object kicks
- [x] Dataset writer: obs_*/gt_* file split, uint16 depth (0.1 mm), versioned manifest with git commit, config hash, per-file sha256
- [x] Validator: finite depth, shapes, transforms, calibration envelope, timestamps, identities, label alignment, bbox, split, hashes. Negative-tested on 6 corruptions
- [x] Import-graph leakage test: inference packages cannot import `mujoco`, `sim`, or GT types
- [x] Throughput and storage measured (EXP-M1-001); demo (EXP-M1-002)

## Implemented modules

| Module | Purpose |
|---|---|
| `perception/geometry/{se3,camera}.py` | SE(3)/SO(3) and pinhole CPU reference |
| `data/schemas/observation.py`, `splits.py` | Observation/GroundTruth split; disjoint seed-range splits |
| `data/generators/episodes.py`, `data/loaders.py` | Deterministic episodes → on-disk dataset; obs-only / GT-only loaders |
| `data/validators/validate.py` | §8 dataset validation (`python -m data.validators.validate <root>`) |
| `sim/environments/{tabletop,scenario}.py` | Scene (+ optional Panda, occluder, kicks); seeded scenario sampling |
| `sim/robot/panda.py` | Proprioception, FK, IK, grasp IK, joint-space moves, gripper |
| `sim/cameras/{rgbd,viewpoints,noise}.py` | RGB-D-seg + visibility; viewpoint set + motion model; noise |

## Verified tests (actually run)

`MUJOCO_GL=cgl .venv/bin/python -m pytest -q` → **92 passed** (macOS, 2026-10-23); `ruff check .` clean.
CI (ubuntu-latest, OSMesa, Panda fetched): **92 passed**, ruff clean — https://github.com/parammadan/belief-3d/actions/runs/40041526869

## Environment

Apple M1, 8 cores, 8 GB RAM, macOS 26.4. **Disk: ~1.8 GB free** (falling). Python 3.12.13 venv. Apple clang 21,
no cmake. No CUDA locally; torch/OpenCV/Open3D not installed. Cluster: `ssh explorer`, `gpu:a100` partition
(no job run yet; GPU use requires Param's explicit OK).

## Experimental results (sanity/infra measurements, not research results)

| ID | Result |
|---|---|
| EXP-M0-001 | Back-projection onto true sphere surface: mean 0.15 mm, max 0.29 mm. Table plane 0.001–0.067 mm. |
| EXP-M1-001 | 15.6–15.7 camera-frames/s (2 cams, 320×240, 4–6 objects + isolation renders, M1 Mac). 268–270 KiB per camera frame. 0 validation errors. |
| EXP-M1-002 | Occlusion demo seed 3: sphere overhead visibility 1.0 → 0.0 for frames 4–7 → 1.0; mobile still sees 11–100 %. |

## Open defects / risks

1. **Storage:** ~270 KiB per camera frame, because sensor noise is incompressible. 100k frames ≈ 26 GB, so large sets must go on the cluster.
2. CGL depth precision is limited; returns >5 m are clamped to no-return.
3. One `mujoco.Renderer` per process on CGL (a second one corrupts segmentation).
4. The camera B motion model is abstract (gantry assumption). Path clearance is a straight-line ray test, not a swept volume.
5. The occluder is contact-free ("hovering"). It is a visibility device, not a physical obstacle.
6. The Panda's joint-space `move_to` has no collision checking yet (M6). The physics test routes via HOME_Q.
7. Primitive objects only; held-out *meshes/textures* per split are not implemented yet (only held-out seeds).
8. Literature review is still abstract-level (SE(3)-PoseFlow is the main overlap risk).
9. Spawn region changed in M1 (robot reach). M0 scenes rendered at the current commit differ from `reports/m0/` (EXP-M0-001 is tied to its commit).

## Remaining work: M2

1. Classical detector: colour + depth segmentation → instance masks (no GT masks at inference).
2. 3D centroid / plane removal / outlier rejection; known-geometry pose fit (primitive-specific, then ICP).
3. Symmetry-aware pose error (box 8-fold proper group, cylinder continuous about z + flip, sphere translation-only).
4. EKF tracker (constant velocity) + Hungarian association; ID-switch / fragmentation metrics.
5. `benchmarks/perception/eval_m2.py`: an independent evaluator reading GT files only, on held-out `val` seeds.

## Exact commands to continue

```bash
cd ~/belief-3d
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # if .venv is missing
.venv/bin/python scripts/fetch_assets.py                         # Panda assets (35 MB, git-ignored)
export MUJOCO_GL=cgl PYTHONPATH=.
.venv/bin/python -m pytest -q
.venv/bin/python scripts/generate_dataset.py --out datasets/smoke --episodes 3 --frames 10
.venv/bin/python scripts/m1_demo.py --seed 3
```
