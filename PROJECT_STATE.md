# PROJECT_STATE

_Last updated: 2026-10-24 (late)_

## Current milestone

**M4: Probabilistic tracking. Exit criteria met.** The belief tracker is mathematically tested
(24 seed-parametrised property tests) and uncertainty metrics are reported. The pre-registered
EXP-M4-001 confirmed both hypotheses:
- **H1:** rotation success on partially visible objects 64.8 % → 87.0 %; paired +0.266,
  95 % CI [0.129, 0.410], p = 0.0074.
- **H2:** no MOTA loss in any condition.

B2 is not uniformly better: about 0.4 mm worse translation under nominal noise, and about
5,000× slower.

**M3: Neural pose estimation. Exit criteria met.** Reproducible outputs, checkpoint (on the
cluster, sha256-recorded) and baseline comparison tables exist.
- **EXP-M3-001:** neural proposals + SDF refinement score 0.985 [0.956, 1.0] success on
  < 50 %-visible test samples, vs 0.853 for the classical fit, at half its time. The network
  alone is worse (0.735).
- **Not run:** FoundationPose (level 3, optional).
- **Checkpoint:** not yet published publicly (needs Param's OK for a GitHub release).

**M5: Active perception. Exit criteria met** (evidence for and against the hypothesis, no
leakage; two registered studies).
- **EXP-M5-001: H3 NOT supported.** Task-risk without a calibration model never moved
  (0.487 = fixed); random was best (0.627).
- **EXP-M5-002: confirmed (Holm p = 0.0078 on all three).** Calibration-aware task-risk (D2,
  frozen on val) reached 0.667, vs 0.520 for D and 0.533 for entropy / covariance; vs random
  +0.033 (n.s.).
- **Mechanism:** D2 picks views whose calibration biases cancel, giving the lowest centre
  error (median 12.7 mm).

**M6: Closed-loop manipulation. Exit criteria met.** Randomised pick-and-place from
perception-derived targets: physics-executed grasp, proprioceptive verification, recovery,
tray placement.
- **EXP-M6-002 (100 paired test episodes):** success 0.50 (fixed) / 0.64 (D2) with recovery.
- **H6 (recovery helps) not supported:** +0.03 / +0.05, Holm p = 0.45; retry success ~14 %.
- **EXP-M6-001** was compromised by two harness bugs (both fixed and recorded).
- **Calibration bias attacked directly:** online self-calibration
  (`perception/calibration/self_calib.py`). EXP-M5-003 (pre-registered, 150 test episodes):
  D2 + self-cal 0.767 vs D2 0.640 (+0.127, p = 0.0034); control (fixed camera) exactly 0 effect.
  Full system vs fixed: 0.513 → 0.767.
Literature check done: SE(3)-PoseFlow, ActivePose and POTAMP were read in full and the narrow
claim stands (see literature_review).

## Milestones

- [x] **M0**: environment audit, geometry core, minimal scene, validated camera geometry
- [x] **M1**: robot, viewpoints, noise, scenarios, dataset writer/validator
- [x] **M2**: classical segmentation, SDF pose fit, completeness, CV-KF + Hungarian tracking, calibration checks, independent evaluator
- [x] **M3** neural pose · [x] **M4** belief tracking · [x] **M5** active perception (EXP-M5-001 null; EXP-M5-002 confirmed)
- [x] **M6** closed-loop manipulation · [ ] M7 systems optimisation · [ ] M8 final study

## M8 status

- EXP-M8-001 pre-registered (8 arms incl. new B0/B1 grasp-source arms; 300 test seeds, offset 4000–4299).
- Running on Explorer: arrays 11354631 (sc0, 6 arms) + 11354632 (sc1, selfcal, 2 arms) → /scratch/madan.pa/belief3d/reports/m8/.
- When done: copy with `ssh explorer 'cd /scratch/madan.pa/belief3d/reports/m8 && tar cf - exp_m8_001_sc0 exp_m8_001_sc1' | tar xf - -C reports/m8`,
  then run the analysis command in the registry. Next: results entry, ablation write-up, paper.

## M7 checklist

- [x] PERF-M7-001: belief update 4.3× faster (matmul, cached symmetry groups)
- [x] PERF-M7-002/003: proposer inference on V100 (torch fp32/fp16, ORT CUDA 0.48 ms at batch 1)
- [x] PERF-M7-004: TensorRT fp16 on A100, 0.19 ms at b1 (4.9× torch fp32), 161k samples/s at b256, accuracy held
- [x] PERF-M7-005: C++ free-space kernel (pybind11 + CMake, CI-built), 4.8–6.9× kernel and 3.4× per update, identical particles
- [x] PERF-M7-006: pose-fit Jacobian batched, 306 → 237 ms per frame (1.29×), bit-identical. Still not real-time

## M2 checklist

- [x] Catalog moved to `data/schemas/objects.py` (allowed CAD prior; inference may not import `sim`)
- [x] Segmentation: RANSAC table plane, MAD depth-noise estimate, saturated seeds + bounded depth-continuous growth, same-class fragment merge, over-exposure-safe hue vote
- [x] Pose: analytic primitive SDFs, robust SE(3) least squares, symmetric initialisations, soft support prior (ablatable)
- [x] Completeness = observed / model-rendered pixels (corr. 0.885 with true visibility on val)
- [x] Tracking: CV Kalman (Joseph form), Hungarian with χ² gate + class compatibility, tentative/confirmed/coasting lifecycle, completeness-aware noise + spawn gating
- [x] Metrics: symmetry groups (D2 / D4 / cube), cylinder axis error, exact SDF-based ADD-S, CLEAR-MOT, occlusion recovery, bootstrap CIs
- [x] Calibration: fiducial board, disc detection, planar PnP + LM + covariance, triangulation, drift sweep
- [x] Obs-only inference runner and independent evaluator; registered test-seed run

## M1 checklist (done)

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
| `perception/segmentation/classical.py` | Plane + seed-and-grow instance segmentation |
| `perception/pose/{primitive_fit,completeness}.py` | SDF pose fit with support prior; model-based completeness |
| `perception/calibration/fiducial.py` | Disc detection, planar PnP (+cov), triangulation, drift check |
| `perception/pipeline.py` | Observation → Detections |
| `tracking/{filters/kalman,association/hungarian,tracker}.py` | B1 unimodal multi-object tracker |
| `benchmarks/metrics.py`, `benchmarks/perception/eval_m2.py`, `benchmarks/calibration/calib_sweep.py` | Evaluator side only |
| `data/schemas/{observation,objects,splits,calib_target}.py` | Schemas and shared priors |
| `data/generators/episodes.py`, `data/loaders.py`, `data/validators/validate.py` | Dataset I/O |
| `sim/environments/{tabletop,scenario}.py`, `sim/robot/panda.py`, `sim/cameras/{rgbd,viewpoints,noise}.py` | Simulation |

## Verified tests (actually run)

`MUJOCO_GL=cgl .venv/bin/python -m pytest -q` → **139 passed** (macOS, 2026-10-24); `ruff check .` clean.
CI (ubuntu-latest, OSMesa, Panda fetched): **139 passed**, ruff clean — https://github.com/parammadan/belief-3d/actions/runs/40102852938

## Environment

Apple M1, 8 cores, 8 GB RAM, macOS 26.4. **Disk: ~2.9 GB free**. Python 3.12.13 venv. Apple clang 21,
no cmake. No CUDA locally; torch/OpenCV/Open3D not installed. Cluster: `ssh explorer`, `gpu:a100` partition
(no job run yet; GPU use requires Param's explicit OK).

## Experimental results (simulation only; see EXPERIMENT_REGISTRY.md for full details)

| ID | Result |
|---|---|
| EXP-M0-001 | Back-projection onto true sphere surface: mean 0.15 mm, max 0.29 mm. |
| EXP-M1-001 | 15.6–15.7 camera-frames/s; 268–270 KiB per camera frame; 0 validation errors. |
| EXP-M1-002 | Scheduled occlusion: overhead visibility 1 → 0 → 1; mobile keeps 11–100 %. |
| EXP-M2-001 | Nominal noise: pose t-err median 0.25–0.60 mm, rot 0.19–0.83°, success 0.96–0.99; MOTA 0.85–0.88 overhead-only vs 0.98–1.00 two-camera; 13/13 hidden objects recovered same-ID. Harsh: t-err 16–22 mm (calibration-dominated); **two-camera MOTA drops (0.767 → 0.736, 26 ID sw.)** because calibration uncertainty is not modelled. |
| EXP-M2-002 | Object loc. error ≈ 11.6 mm/deg of drift; reprojection detects ≥ 0.25°; single-image PnP floor 0.85° (overhead) / 0.17° (oblique). |

## Open defects / risks

1. **Calibration-inconsistent multi-camera fusion** spawns duplicate tracks under drift (EXP-M2-001 harsh). Fix: add a calibration term to R (val-developed, M4).
2. Perception runs at ~260–350 ms per camera frame on CPU (pose fit is most of it). Profiling and optimisation belong to M7. Not real-time.
3. The support prior assumes resting objects; it is wrong for stacked or lifted ones (stacking not in scenarios yet).
4. Same-class twins that touch merge into one segment, so one detection is produced.
5. Partial views (< 50 % visible) can give 1–4 cm, ~90° pose errors in single-view fits. Completeness down-weights them, but they are not corrected. This is M4's job.
6. Expected-pixel model over-counts by ~8 % (rasterisation approximation), so a fully visible object scores completeness ≈ 0.8.
7. Storage ~270 KiB per camera frame; Mac disk ~2.9 GB free. Large datasets go on the cluster.
8. CGL depth precision limited; one Renderer per process; camera-B motion abstract; occluder contact-free.
9. Literature review is still abstract-level (SE(3)-PoseFlow is the main overlap risk).

## Remaining work: M3 (+ M4 prerequisite)

1. Ask Param before any GPU use. Set up torch on Explorer (A100), off-GPU first, with pre-cached weights.
2. Choose a level-2 neural pose proposal model (license + compatibility check). Consider FoundationPose as level 3 (verify the NVlabs license first).
3. Generate a train split on the cluster (~26 GB per 100k frames) with a provenance manifest.
4. CPU side: calibration-aware measurement covariance, then rerun EXP-M2-001 harsh as a *new* experiment ID.

## Exact commands to continue

```bash
cd ~/belief-3d
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # if .venv is missing
.venv/bin/python scripts/fetch_assets.py                         # Panda assets (35 MB, git-ignored)
export MUJOCO_GL=cgl PYTHONPATH=.
.venv/bin/python -m pytest -q
.venv/bin/python scripts/generate_dataset.py --out datasets/smoke --episodes 3 --frames 10
.venv/bin/python scripts/m1_demo.py --seed 3
bash experiments/scripts/exp_m2_001.sh && .venv/bin/python experiments/analysis/m2_tables.py   # ~10 min
.venv/bin/python -m benchmarks.calibration.calib_sweep --seeds 10
```
