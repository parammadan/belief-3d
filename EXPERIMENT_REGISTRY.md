# EXPERIMENT_REGISTRY

Each experiment ID is immutable. A rerun with changed code or config gets a new ID.
Record the git commit, command, seed(s), environment and outputs. Generated outputs are
git-ignored unless they are small reports.

| ID | Date | Commit | Command | Seeds | Env | Output | Status |
|---|---|---|---|---|---|---|---|
| EXP-M0-001 | 2026-10-23 | see `git log -- scripts/m0_demo.py` | `MUJOCO_GL=cgl PYTHONPATH=. python scripts/m0_demo.py --seed 0` | 0 | M1 Mac, mujoco 3.17.0, cgl | `reports/m0/geometry_report.json`, 4 PNGs | done |

## EXP-M0-001: render-geometry consistency

- **Question:** do rendered depth and segmentation, back-projected with our intrinsics and
  extrinsics, land on the simulator's true surfaces?
- **Measured:** sphere-surface |‖p−c‖−r| mean 0.146 mm / max 0.291 mm (overhead, 170 px);
  0.161 / 0.289 mm (mobile, 164 px). Table-plane |z−0.40| mean 0.001 mm (overhead, 40,137 px);
  0.067 mm (mobile, 32,087 px). Mobile no-return fraction 5.7 % (sky beyond 5 m).
- **Principal-point ablation:** shifting (cx, cy) by ±0.5 px raises the overhead sphere error from
  0.146 mm to about 1.1 mm. We therefore adopt (W−1)/2, (H−1)/2.
- **Caveat:** one seed and one scene. This is a sanity check, not a statistical result.

| ID | Date | Commit | Command | Seeds | Env | Output | Status |
|---|---|---|---|---|---|---|---|
| EXP-M1-001 | 2026-10-23 | `e29df26` | `write_dataset` with `EpisodeConfig(n_frames=10, scenario=S, noise='nominal')`, S ∈ {static, dynamic} (inline script, equivalent to `scripts/generate_dataset.py`) | train 0–4 | M1 Mac, mujoco 3.17.0, cgl, Panda on | stats below; datasets deleted (regenerable bit-exactly) | done |
| EXP-M1-002 | 2026-10-23 | see `git log -- scripts/m1_demo.py` | `MUJOCO_GL=cgl PYTHONPATH=. python scripts/m1_demo.py --seed 3` | 3 | same | `reports/m1/occlusion_strip.png`, `occlusion_log.json` | done |

## EXP-M1-001: generation throughput and storage

| Scenario | Camera frames | Wall time | Frames/s | KiB / camera frame | Validation errors | Config hash |
|---|---|---|---|---|---|---|
| static | 100 | 6.36 s | 15.72 | 267.7 | 0 | 7b80109a6de2827d |
| dynamic | 100 | 6.41 s | 15.61 | 270.3 | 0 | 7bd39b537a8451fc |

- Throughput includes physics (0.1 s per frame), 3 render passes per camera plus one isolation
  render per object, noise and compressed writes. It was not profiled per stage (that is M7).
- Storage: about 162 KiB noisy RGB plus about 99 KiB noisy uint16 depth per camera frame
  (zlib). Noise is close to incompressible.
- Projection: 100k camera frames ≈ 26 GB and ≈ 1.8 h on this Mac. The Mac's 1.8 GB free disk
  rules out more than ~6k frames locally.

## EXP-M1-002: controlled occlusion demo

Seed 3, `occlusion` preset, 6 objects. `sphere_blue` is occluded for frames [4, 8). Overhead
visible fraction: 1, 1, 1, 1, **0, 0, 0, 0**, 1, 1, 1, 1. The mobile camera keeps partial or full
sight during the interval, because its viewpoint changes randomly per frame. This is the
situation active viewpoint selection (M5) must exploit.
