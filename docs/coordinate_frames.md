# Coordinate frames and conventions

All code in `perception/geometry/` implements these; `tests/unit/test_se3.py`,
`tests/unit/test_camera.py` and `tests/simulation/test_render_geometry.py` pin them.

## Notation

`T_AB ∈ SE(3)` maps coordinates expressed in frame **B** into frame **A**:

    p_A = T_AB · p_B,   T_AB = [R_AB t_AB; 0 1],   T_AC = T_AB · T_BC,   T_BA = T_AB⁻¹

`t_AB` is the origin of B expressed in A. Code names follow the symbol:
`T_WC`, `T_WO`, `R_W_gl`.

## Frames

| Frame | Symbol | Definition |
|---|---|---|
| World | W | MuJoCo world. +z up (gravity −z). Origin at floor centre under the table. Table top at z = 0.40 m. |
| Camera | C | **OpenCV**: +x right, +y down, +z forward (optical axis). |
| Camera (MuJoCo) | C_gl | MuJoCo/OpenGL camera: looks along −z, +y up. `R_W_C = R_W_Cgl · diag(1,−1,−1)`. Never used outside `sim/`. |
| Robot base | B | (M1) base link of the arm. |
| End-effector | E | (M1) gripper TCP, z along approach direction. |
| Object | O | Body frame of the object's MuJoCo body = geometric centre of the primitive/mesh. |

## Units and representations

- Translations: **meters**. Angles: **radians**. Time: simulation seconds.
- Quaternions: **scalar-first `[w, x, y, z]`** (MuJoCo's order), unit norm, returned with `w ≥ 0`.
  `q` and `−q` are the same rotation; never compare quaternions element-wise.
- Rotation error: geodesic angle `θ = atan2(‖vee(R₁ᵀR₂)‖, (tr(R₁ᵀR₂)−1)/2)` (stable at 0 and π).
- Never average rotation matrices or quaternions naively. Use tangent-space (`so3_log`) means
  about a reference or chordal mean + `project_to_so3`, and say which one.

## Images

- Pixel `(u, v)`: u = column, v = row, **centre of the top-left pixel is (0, 0)**.
- Intrinsics `K = [fx 0 cx; 0 fy cy; 0 0 1]`. For MuJoCo cameras: `fx = fy = (H/2)/tan(fovy/2)`,
  `cx = (W−1)/2`, `cy = (H−1)/2`. The ±0.5 px alternatives were measured to give ~7× larger
  back-projection error (`test_principal_point_convention_is_the_measured_best`).
- Depth images store **z_C (meters)**, not ray length. `0` means no return (beyond `max_range`,
  default 5 m, or dropped). Measured: table-plane back-projection error 0.001 mm (overhead).
- RGB is `(H, W, 3) uint8`, row 0 = top of image.

## Symmetries (for evaluation)

`ObjectState.symmetry ∈ {none, box, cylindrical_z, spherical}`. Pose errors for symmetric objects
must be taken as the minimum over the proper symmetry group:
- box with three distinct extents: D2, **4** rotations. (An earlier version of this doc said 8,
  which wrongly counted reflections.)
- box with two equal extents: 8 rotations; a cube: 24.
- cylinder: continuous rotation about local z plus a π flip.
- sphere: rotation unobservable, so only translation is evaluated.

Implemented in `benchmarks/metrics.py` (M2) and tested in `tests/unit/test_pose_metrics.py`.
