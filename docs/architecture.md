# Architecture

Status key: **[done]** implemented and tested · **[M#]** planned for that milestone.

```
sim.environments.TabletopScene + Scenario  [done: table, Panda, 3–6 objects, 2 cameras, occluder, kicks, seeded]
      │  (MuJoCo state — only sim/ and evaluators may read it)
      ▼
sim.cameras.RGBDSensor.capture() ──► (Observation, GroundTruth)   [done]
      │ Observation only                       │ GroundTruth only
      ▼                                        ▼
perception.geometry (SE3, pinhole)  [done]    data writer + validator [done] / evaluators [M2]
perception.calibration              [M2]
perception.detection / segmentation [M2 classical, M3 neural]
perception.pose (ICP/PnP → neural)  [M2 / M3]
tracking.association (Hungarian)    [M2]
tracking.filters (EKF), tracking.belief (PF / mixture)  [M2 / M4]
policies.{fixed,random,information_gain,risk_aware}     [M5]
sim.robot.Panda (FK/IK/gripper)  [done]  →  manipulation.{grasping,motion,recovery}  [M6]
      │ action (camera pose / grasp)
      └──────────────► TabletopScene.set_mobile_camera / robot  ───► feedback loop
```

## The leakage boundary

- `data.schemas.Observation` is frozen and slotted. It holds RGB, depth, intrinsics, `T_WC` and a
  timestamp, and nothing else. Every inference-side module takes `Observation` (plus robot
  proprioception from M1).
- `GroundTruth` (segmentation, object poses, visibility) is produced in the same call. Only
  dataset writers and evaluators may consume it. `tests/unit/test_leakage.py` checks the schema;
  `tests/unit/test_import_leakage.py` statically checks that nothing under `perception/`,
  `tracking/`, `policies/`, `manipulation/` or `models/` imports `mujoco`, `sim`, `GroundTruth`
  or `load_ground_truth`.
- On disk the same split holds: `obs_*` files (RGB, depth, *reported* K and T_WC) versus `gt_*`
  files (segmentation, poses, visibility, *true* K and T_WC). `data.loaders.load_observation`
  never opens a `gt_*` file.
- The robot interface `sim.robot.Panda` exposes only proprioception and kinematics. Planning code
  (M6) will receive it as an injected object, so the robot arm can be used without importing
  `sim`.
- Camera B's pose in `Observation` is the *commanded* pose. Calibration perturbations (M2) are
  applied to what the inference stack is told, not to the renderer.

## Verified environment choices (2026-10-23)

| Choice | Evidence |
|---|---|
| Dev machine: Apple M1, 8 CPU, 8 GB RAM, macOS 26.4 | `reports/env_audit.json` |
| Python 3.12 venv (system default is 3.14) | MuJoCo wheels installed and ran on 3.12 |
| Renderer: MuJoCo `cgl` (headless, macOS) | Analytic depth test 0.9016 m returned; `egl`/`osmesa` are not valid on macOS |
| One `mujoco.Renderer` per scene | Two renderers in one process returned corrupt segmentation ids on CGL |
| CPU reference geometry in NumPy | 36/36 tests; C++ deferred, `cmake` not installed |
| GPU: NU Explorer cluster has `gpu:a100` nodes (plus V100/H200/T4), CUDA modules up to 13.2 | `sinfo` query only, no job run; GPU use needs Param's explicit OK |
| No torch / OpenCV / Open3D yet | Only **~2.4 GB free disk** on the Mac. Neural work goes to the cluster |
