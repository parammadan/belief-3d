# Requirements → module → test traceability

| Req (master prompt §) | Module | Test / evidence | Status |
|---|---|---|---|
| §5.1 rotation/quaternion conversions | `perception/geometry/se3.py` | `test_se3.py::test_quaternion_*` | done |
| §5.1 SE(3) compose/inverse/points | `se3.py` | `test_inverse_and_composition`, `test_analytic_frame_example` | done |
| §5.1 projection / back-projection | `perception/geometry/camera.py` | `test_camera.py` | done |
| §5.1 extrinsic transforms + perturbation | `camera.py`, `se3.py` | `test_world_to_pixel_through_extrinsics`, `test_extrinsic_perturbation_shifts_reprojection` | done |
| §5.1 geodesic distance, near-degenerate rotations | `se3.py` | `test_log_exp_roundtrip_including_degenerate`, `test_geodesic_distance` | done |
| §7 RGB, depth, segmentation, geometric consistency | `sim/cameras/rgbd.py` | `tests/simulation/test_render_geometry.py` | done |
| §7 deterministic reset by seed | `sim/environments/tabletop.py` | `test_reset_is_deterministic` | done |
| §7 movable camera B | `TabletopScene.set_mobile_camera` | `test_commanded_mobile_pose_is_reported_back` | done (free pose; finite reachable viewpoint set in M1) |
| §23 GT leakage | `data/schemas/observation.py`, `data/loaders.py` | `test_leakage.py`, `test_import_leakage.py` | done (schema + import graph + obs/gt file split) |
| §7 robot arm + gripper | `sim/robot/panda.py` | `tests/simulation/test_robot.py` (FK, IK, grasp-lift physics) | done |
| §7 finite reachable viewpoints, motion model | `sim/cameras/viewpoints.py` | `test_viewpoints_noise.py` | done |
| §7 noise, dropout, extrinsic perturbation | `sim/cameras/noise.py` | `test_viewpoints_noise.py` | done |
| §7 variable counts, lighting/material, occlusion intervals, object motion, similar instances | `sim/environments/scenario.py` | `test_scenario.py` | done (static obstacles: occluder only; stacking: not yet) |
| §7 visibility/occlusion metadata | `sim/cameras/rgbd.py` | `test_robot.py::test_observe_pose_keeps_objects_visible`, `test_scenario.py` | done |
| §8 dataset + manifests + validators | `data/generators`, `data/validators` | `tests/integration/test_dataset.py` (replay, 6 negative cases) | done (held-out meshes/textures: not yet) |
| §8 splits by seed | `data/schemas/splits.py` | `test_splits_are_disjoint_and_cover_seeds` | done |
| §5.1 symmetry-aware errors, §10 pose | `perception/pose` | — | M2 |
| §11 tracking | `tracking/` | — | M2 |
| §12 belief tracking | `tracking/belief` | — | M4 |
| §14 policies | `policies/` | — | M5 |
| §19 C++ components | `cpp/` | — | M7 (cmake not installed) |
