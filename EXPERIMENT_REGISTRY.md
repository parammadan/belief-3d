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

| ID | Date | Commit | Command | Seeds | Env | Output | Status |
|---|---|---|---|---|---|---|---|
| EXP-M2-001 | 2026-10-24 | `777bce5` | `bash experiments/scripts/exp_m2_001.sh` then `python experiments/analysis/m2_tables.py` | test 2,000,000–2,000,009 (10 eps × 15 frames) | M1 Mac, cgl, Panda on | `reports/m2/m2_test_*.json`, `reports/tables/m2_results.md` | done |
| EXP-M2-002 | 2026-10-24 | `777bce5` | `python -m benchmarks.calibration.calib_sweep --seeds 10` | test 2,000,500–2,000,509 | same | `reports/m2/calib_sweep.json` | done |

## EXP-M2-001: classical perception + B1 tracking on held-out test seeds

- **Protocol.** All segmentation, pose and tracker thresholds were tuned only on **val** seeds
  (1,500,000+). The test seeds were run once, after the code was frozen at the commit above.
- **Methods.** (a) overhead camera only (fixed view, B1); (b) overhead plus mobile camera
  following the random viewpoint schedule.
- **Conditions.** static / occlusion / dynamic with nominal noise, plus dynamic with harsh noise.
  Full tables are in `reports/tables/m2_results.md`.
- **Nominal-noise findings.**
  - Detection precision ≥ 0.993; recall 0.943–0.973 (most misses are heavily occluded objects).
  - Median translation error 0.25–0.60 mm; median symmetry-aware rotation error 0.19–0.83°.
  - Pose success (t < 1 cm and r < 5°) 0.958–0.990.
  - Tracking MOTA 0.847–0.882 with overhead only versus 0.977–0.997 with two cameras.
  - With overhead only, all 13 fully hidden objects were recovered under the same track ID,
    with zero frames of latency (coasting).
- **Harsh-noise findings.**
  - Median translation error rises to 16–22 mm and pose success collapses to 0.08–0.13. That
    is consistent with EXP-M2-002: the harsh preset's ~1° extrinsic error predicts ~12 mm,
    plus ~8 mm from its translation offset.
- **Negative result.** Under harsh calibration error, adding the mobile camera *lowers* MOTA
  (0.767 → 0.736, 95 % CI [0.363, 0.991]) with 26 ID switches and 133 FP. Each camera's
  calibration error is independent, so the same object is reported 2–3 cm apart. That exceeds
  the gate, because measurement covariance currently models fit residual and completeness but
  **not calibration uncertainty**, so duplicate tracks spawn. Not fixed here, to avoid tuning
  on test; it is carried to M4 as a val-developed fix.

## EXP-M2-002: extrinsic calibration drift sweep

- **Object localisation degrades linearly:** about 11.6 mm per degree of drift (overhead),
  and two-view triangulation about 18 mm per degree.
- **Reprojection RMS under the reported calibration:** 0.11 px clean, 0.72 px at 0.25°,
  2.8 px at 1°. It is a sensitive drift detector.
- **Single-image planar PnP re-estimation** has a constant error floor of 0.85° (overhead) /
  0.17° (oblique) independent of drift. The estimator is verified exact on synthetic data and
  covariance-consistent (mean NIS ≈ 6 over 200 trials), so the floor is ~0.1 px systematic
  centroid error amplified by poor conditioning at 320×240. Re-calibrating from one image
  therefore only helps once drift exceeds ≈ 1° (overhead) or ≈ 0.2° (oblique).

## EXP-M4P-002 — PRE-REGISTERED (written before running)

- **Hypothesis:** with σ_rot = 1°, σ_trans = 5 mm (frozen; chosen on val by VAL-M4P-001),
  two-camera MOTA on `dynamic`/`harsh` is higher than with σ = 0.
- **Data:** 40 new test seeds, offsets 10–49 (2,000,010–2,000,049), never evaluated before;
  15 frames per episode.
- **Analysis:** per-episode paired difference (σ = 1 minus σ = 0). Report the mean, a paired
  bootstrap 95 % CI and an exact two-sided sign test. The overhead-only arm is reported as a
  control (expected ≈ 0 effect).
- **Commitment:** the result is reported as-is, whatever its direction.

| ID | Date | Commit | Command | Seeds | Env | Output | Status |
|---|---|---|---|---|---|---|---|
| VAL-M4P-001 | 2026-10-24 | `b294403` | `bash experiments/scripts/val_calib_sweep.sh` | val 1,500,200–1,500,207 | M1 Mac | `reports/m4p/val_dynamic_*.json` | done (selection) |
| EXP-M4P-001 | 2026-10-24 | `b294403` | `bash experiments/scripts/exp_m4p_001.sh` + `experiments/analysis/paired.py` | test 2,000,000–2,000,009 | M1 Mac | `reports/m4p/test_*.json`, `paired_test.txt` | done |
| EXP-M4P-002 | 2026-10-24 | `b294403` | `bash experiments/scripts/exp_m4p_002.sh` (pre-registered above) | test 2,000,010–2,000,049 | M1 Mac | `reports/m4p/m4p2_*`, `exp_m4p_002_paired.txt` | done |

## VAL-M4P-001: choosing the tracker's calibration-uncertainty prior (val only)

Swept σ_rot ∈ {0, 0.5, 1, 1.5, 2, 3}° with σ_trans = 5 mm per degree, for both camera modes,
on `dynamic`/`nominal` and `dynamic`/`harsh`.
- **Harsh, two cameras:** MOTA 0.937 → 0.972 for any σ ≥ 0.5°.
- **Nominal, overhead only:** unchanged up to 1°, but drops to ~0.84 at ≥ 1.5°. Inflating R for
  what is really a *constant per-camera bias* slows the filter's response to kicked objects.
- **Selection rule:** the largest σ that leaves every nominal condition unchanged, among those
  achieving the best harsh two-camera MOTA. That gives **σ_rot = 1°, σ_trans = 5 mm**.

## EXP-M4P-001: same 10 test episodes as EXP-M2-001

- **Reproducibility check:** σ = 0 reproduces EXP-M2-001 exactly in 8/8 conditions, from
  bit-identically regenerated data.
- **σ = 1°, harsh, two cameras:** MOTA 0.736 → 0.984, ID switches 26 → 0, FP 133 → 2. No
  condition got worse.
- **Paired test:** the gain comes from 3 of 10 episodes (the rest are tied). Mean +0.254,
  bootstrap 95 % CI [0.000, 0.619], sign test p = 0.25. **Not significant at n = 10**, which
  motivated the pre-registered EXP-M4P-002.

## EXP-M4P-002 result (pre-registered hypothesis: CONFIRMED)

| Arm | MOTA σ = 0 → σ = 1° | Mean paired Δ | 95 % CI | Better / worse / tied (of 40) | Sign test p |
|---|---|---|---|---|---|
| overhead + mobile | 0.806 → 0.972 | **+0.156** | [0.079, 0.242] | 14 / 0 / 26 | **0.00012** |
| overhead only (control) | 0.717 → 0.715 | −0.002 | [−0.006, 0.000] | 0 / 1 / 39 | 1.0 |

- **Two-camera ID switches:** 61 → 7; FP 437 → 37.
- **Control arm:** no effect, as predicted. Its one worse episode is consistent with the
  responsiveness cost seen on val.
- **Limitation:** R-inflation approximates a correlated, constant per-camera bias as white
  noise. The principled model estimates per-camera extrinsic bias as a state; this is left as
  future work.

| ID | Date | Commit | Command | Seeds | Env | Output | Status |
|---|---|---|---|---|---|---|---|
| VAL-M4-001 | 2026-10-24 | `19bb083` | `bash experiments/scripts/val_m4_001.sh` | val 1,500,400–1,500,405 (6 eps × 15 frames per condition) | M1 Mac | `reports/m4/val_m4_*_{b1,b2}.json` | done (dev) |
| VAL-M4-003 | 2026-10-24 | `e1f5589` | `bash experiments/scripts/val_m4_003.sh` | same datasets as VAL-M4-001 | M1 Mac | `reports/m4/val_m4_*_v3_*.json` | running |

## VAL-M4-001: B1 (CV-Kalman) vs B2 (particle belief), same detections, val only

- **Nominal noise.** Both trackers reach MOTA 0.995–0.997 with 0 ID switches.
  - **Partially visible objects** (< 50 % visible, n = 11, occlusion scenario): B1 median
    rotation error **83.4°** (single-view face flips) versus B2 **1.47°**; success 0.455 vs
    1.000. This is the hypothesised benefit, but n = 11 is tiny.
  - **Overall:** B2's median translation error is worse (0.64–0.75 mm vs 0.24–0.26 mm). This
    is Monte Carlo approximation error from 256 particles.
  - **Uncertainty:** B2's 95 % coverage is 0.97 (≈ calibrated). B1's is 1.00 at every level,
    i.e. over-conservative, because its R carries the 1° calibration prior even when
    calibration is perfect.
- **Harsh noise: B2 v1 FAILED.**
  - MOTA 0.933 vs 0.979, 6 vs 1 ID switches; median rotation 9.7–14.0° vs 3.4–5.0°.
  - **95 % coverage 0.01–0.02:** the particle belief ignores calibration bias, so it is badly
    overconfident.
  - **Cause:** B2's high-resolution likelihood fuses two cameras whose calibration errors
    disagree by 2–3 cm; free space from one camera contradicts the other camera's points.
- **Cost:** B2 takes 2.3–2.8 s per frame versus 0.5 ms for B1 (CPU, 256 particles, free space
  on). That is M7 work.
- **Fix under development (val only).** Per-frame cross-camera consensus alignment from
  co-observed objects, plus reported covariance that includes calibration uncertainty /
  n_cameras. A first variant aligned to the overhead camera; it restored MOTA (0.979), rotation
  (2.09°) and coverage (0.92) but doubled translation error (22.4 mm), because both cameras
  inherited one camera's bias. Hence the consensus version, VAL-M4-003.

## VAL-M4-004 (val; final B2 development state, commit `67bf0bf`)

B2 = particle belief + consensus relative calibration + honest covariance + support constraint in
predict + re-identification. On all four val conditions, B2 equals B1 in MOTA, ID switches and FP.
- **Rotation (median, all tracked frames):** nominal 0.57° / 0.74° (B1 0.43° / 0.62°);
  harsh 2.37° / 2.24° (B1 3.44° / 5.03°).
- **Partially visible objects:** occlusion nominal B1 83.4° vs B2 1.03°; occlusion harsh 39.2° vs
  2.18° (small n).
- **Translation:** B2 is worse by 0.4–0.7 mm (Monte Carlo approximation noise).
- **Coverage:** 95 % coverage is 1.00 for B2 everywhere (over-conservative under nominal noise,
  where calibration is actually perfect).
- **Cost:** 2.1–2.7 s per frame.

## EXP-M4-001 — PRE-REGISTERED (written before running; B2 frozen at commit `67bf0bf`)

- **H1 (primary).** On tracked object-frames with visible fraction < 0.5 (max over cameras), B2's
  rate of symmetry-aware rotation error < 5° is higher than B1's. Pooled over `occlusion`/
  `nominal` and `occlusion`/`harsh`.
  - *Unit:* episodes that contain ≥ 1 partial-visibility frame.
  - *Statistic:* per-episode mean of (B2 − B1); paired bootstrap 95 % CI; exact two-sided sign
    test.
- **H2 (non-inferiority).** In every condition, MOTA(B2) ≥ MOTA(B1) − 0.01.
- **Secondary (descriptive, no claims):** median translation and rotation error over all tracked
  frames; 50/80/95 % coverage; runtime.
- **Data:** never-evaluated test seeds, offset 100 onward (2,000,100+). occlusion_nominal and
  occlusion_harsh: 30 episodes each; dynamic_nominal and dynamic_harsh: 10 episodes each;
  15 frames per episode; both cameras; identical detections fed to B1 and B2.
- **Commitment:** report as-is, whatever the direction.

| ID | Date | Commit | Command | Data | Env | Output | Status |
|---|---|---|---|---|---|---|---|
| DATA-M3-001 | 2026-10-24 | `012d6cc` | `sbatch experiments/cluster/gen_shards.sbatch` (+ resubmit of 10 timed-out tasks with 1.5 h limit, same seeds) | train 0–1,999 (20 × 100 eps), val 1,001,000–1,001,199, test 2,003,000–2,003,199; 10 frames, both cameras | Explorer CPU nodes, OSMesa (conda-forge mesalib < 25) | `/scratch/madan.pa/belief3d/shards/*.npz` (+ sha256 json) | done |
| RUN-M3-smoke | 2026-10-24 | `012d6cc` | `train_pose.sbatch` MAXSTEPS=60 | 13 train / 3 val shards | A100-80GB | `runs/smoke` | done (pipeline OK) |
| RUN-M3-v100fail | 2026-10-24 | `e1f5589`+ | `train_pose.sbatch` EPOCHS=60 on V100 | full | V100-32GB | cancelled at step ~300 | **failed: bf16 on V100** |
| RUN-M3-diag1 | 2026-10-24 | see git | `train_pose.sbatch` MAXSTEPS=1500 | full | A100-80GB | `runs/diag1` | done |
| RUN-M3-pose_v1 | 2026-10-24 | see git | `train_pose.sbatch` EPOCHS=60 → `eval_m3.sbatch` | full | A100 (any GPU allowed) | `runs/pose_v1`, `reports/m3/eval_m3_pose_v1.json` | running |

## DATA-M3-001: neural-pose training data

- **Samples:** 175,599 train / 17,599 val / 17,789 test. Each sample is one visible object
  instance in one camera frame: 512 segment points (float16) plus the label. Total 218 MB.
- **Throughput:** about 23 min per 100-episode shard on one CPU core under OSMesa. 10 of 28
  tasks exceeded the 45 min cap on slower nodes and were rerun with the same seeds, so the
  data is identical.
- **Rendering:** EGL is unusable on Explorer CPU nodes (no EGL device). Current conda-forge
  Mesa (26.x) dropped OSMesa; `mesalib<25` restores it. Its depth test matches the Mac
  exactly (0.9016 m).

## RUN-M3-v100fail / diag1: training diagnostics

- **V100 failure.** 1,304 samples/s at ~0 % utilisation vs 7,618 on an A100 smoke run,
  because V100s have no bf16 units. Cancelled. Fix: bf16 where supported, otherwise fp16
  with a GradScaler.
- **Idle GPU at startup.** Loading compressed shards took ~5 min of GPU-allocated idle time.
  Fix: a packed uncompressed cache built once.
- **Learning.** Inputs were rescaled ×20 (centred coordinates are ~±3 cm). diag1 (1,500
  steps, 1 min 23 s wall, 22.4k samples/s, A100) reached val best-of-K mean 3.25 mm
  (median 2.17) and top-1 mean 5.59 mm (median 3.04), with 47 % of top-1 predictions under
  2 mm. The top-1 vs best-of-K gap shows a weak confidence head.
