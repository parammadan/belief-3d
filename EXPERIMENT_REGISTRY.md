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
