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
| VAL-M4-003 | 2026-10-24 | `e1f5589` | `bash experiments/scripts/val_m4_003.sh` | same datasets as VAL-M4-001 | M1 Mac | superseded by v4 (interrupted by session close) | superseded |

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

## EXP-M4-001 RESULT (pre-registered above; B2 frozen at `c45e7e0`; analysed with the pre-committed `experiments/analysis/m4_h1.py`)

| Hypothesis | Result |
|---|---|
| **H1 (primary): CONFIRMED.** Rotation-success rate (< 5°) on partially visible tracked frames, pooled over occlusion nominal + harsh | B1 64.8 % → **B2 87.0 %** (108 frames, 42 episodes). Per-episode paired mean **+0.266**, bootstrap 95 % CI **[0.129, 0.410]**; 13 episodes better / 2 worse / 27 tied; exact sign test **p = 0.0074** |
| **H2 (non-inferiority): CONFIRMED.** MOTA(B2) ≥ MOTA(B1) − 0.01 in every condition | occlusion/nominal 0.983 = 0.983; occlusion/harsh 0.977 → 0.979; dynamic/nominal 0.995 = 0.995; dynamic/harsh 0.969 = 0.969 |

**Secondary (descriptive; full table in `reports/m4/exp_m4_001_secondary.txt`).**
- **Nominal noise favours B1 overall.** Median translation B1 0.24–0.25 mm vs B2 0.62–0.64 mm
  (Monte Carlo noise); median rotation 0.52–0.57° vs 0.67–0.81°; overall success
  (1 cm & 5°) B1 0.985 / 0.995 vs B2 0.972 / 0.966.
- **Harsh noise:** B2 has the better rotation (2.52 / 3.03° vs 3.85 / 4.38°) and overall success
  (0.269 / 0.188 vs 0.206 / 0.177); translation is equal (~13 mm, calibration-limited).
- **Coverage:** both are over-conservative under nominal (1.00 at every level). Under harsh, B2
  is 0.80/0.97/1.00 vs B1 0.73/0.92/0.97 at 50/80/95 %.
- **Cost:** B2 takes 2.4–3.0 s per frame vs 0.5–0.6 ms for B1 (CPU, 256 particles).

**Interpretation.** The multimodal belief delivers its targeted benefit — correct orientation when
single views are ambiguous — with statistically significant effect and no tracking penalty. It does
not dominate: under nominal conditions its Monte Carlo approximation costs ~0.4 mm and a small
amount of overall success, and it is ~5,000× slower. Both are M7 targets (refinement of the
reported estimate; vectorised/compiled likelihood).

## RUN-M3-pose_v1 + EXP-M3-001 (done)

**Training.** 60 epochs, 20,520 steps, batch 512, A100-80GB, 6 min 41 s wall at 32k samples/s,
bf16. Checkpoint `/scratch/madan.pa/belief3d/runs/pose_v1/best.pt` (sha256 prefix and size:
00f937ddfc470885 2.3M ). Run metadata (data hashes, git commit) is in `reports/m3/pose_v1_meta.json`.
Final val: best-of-K mean 2.64 mm (median 1.56); top-1 mean 4.78 mm (median 2.41); 49.2 % of
top-1 predictions under 2 mm (surface distance).

**EXP-M3-001: pose comparison on 4,000 held-out TEST samples** (`eval_m3.sbatch`; same GT-mask
points for every method, which isolates pose estimation from segmentation). Not pre-registered;
it is the M3 baseline comparison, run once. Report: `reports/m3/eval_m3_pose_v1.json`.

| Method | Success all (95 % CI) | Success, vis < 0.5 (n = 68) | ADD-S p90, vis < 0.5 | Median time |
|---|---|---|---|---|
| classical SDF fit (level 1) | 0.997 [0.995, 0.998] | 0.853 [0.765, 0.926] | 4.67 mm | 189 ms |
| neural top-1 (level 2, no refinement) | 0.987 [0.983, 0.990] | 0.735 [0.632, 0.838] | 7.16 mm | 4.5 ms |
| **neural proposals + SDF refinement** | **0.999 [0.998, 1.000]** | **0.985 [0.956, 1.000]** | **1.28 mm** | 98 ms (+4.5) |
| neural best-of-K (ORACLE: selects with GT) | 0.991 [0.987, 0.994] | 0.838 [0.750, 0.912] | 3.34 mm | — |

- **The network alone is worse than geometry.** Top-1 translation is 0.52 vs 0.31 mm median,
  and on partial views 0.735 vs 0.853.
- **As initialisations for the same geometric refinement, it is the best level.** On partial
  views 0.985 vs 0.853, with non-overlapping CIs, at half the classical fit's time (4 good
  initialisations instead of ~6 generic ones).
- **Refinement beats the oracle.** Refined-and-cost-selected beats the network's own best
  hypothesis, so the geometry corrects the network's residual error.
- **Caveats.** Few partial-visibility test samples (102 below 0.8, 68 below 0.5); GT masks (no
  segmentation error); one training seed; simulation only.
- **Level 3 (FoundationPose) not run.** Its NVIDIA licence is research-only and allowed for this
  project, but its GPU stack is heavy; deferred as an optional comparison.

## PERF-M7-001: belief-update profiling (early M7; benchmark = 5 updates, 256 particles, 2 views)

- **Before:** 726 ms per update. `free_space_log_lik` took 76 %, almost all in `np.einsum`
  (no BLAS path for batched 3×3 × point-set rotations), and symmetry groups were rebuilt on
  every call (0.57 s per 5 updates).
- **Change:** per-particle camera←object rotations composed first, then batched `np.matmul`;
  symmetry groups `lru_cache`d per model (read-only arrays).
- **After:** 170 ms per update (**4.3×**). Equivalence is tested:
  `test_fast_likelihood_matches_reference_einsum` (rtol 1e-9). The remaining cost is
  element-wise free-space arithmetic on N × M = 256 × 902 arrays.
- **Measured on the M1 Mac while a concurrent M5 val job was running**; treat the absolute
  numbers as indicative.

## VAL-M5-001 / -002 / probe (val; M5 design, no claims)

- **VAL-M5-001** (occlusion / nominal, overhead + mobile, 8 eps × 5 policies, budget 2):
  every policy succeeds 8/8. 1/8 episodes was ambiguous at t₀, and its modes were
  grasp-irrelevant (risk 0). There, task-risk stayed put (0 s travel) while entropy and
  covariance moved (0.21 s) and random travelled 4.45 s.
- **VAL-M5-002** (occlusion / nominal, mobile camera only): every policy 10/10. The first
  attempt skipped every episode because one frame cannot confirm a track (2 hits needed);
  fixed with a warm-up frame shared by all policies.
- **Ambiguity probe** (`experiments/analysis/m5_ambiguity_probe.py`, 12 val episodes per regime;
  target = highest grasp-now risk):

  | Regime | Multi-mode at t₀ | Mean grasp-now risk | Risk > 0.3 |
  |---|---|---|---|
  | occlusion / nominal / mobile | 0/12 | 0.000 | 0 |
  | occlusion / harsh / mobile | 0/12 | 0.000 | 0 |
  | clutter / nominal / mobile | 3/12 | 0.000 | 0 |
  | clutter / harsh / mobile | 2/12 | 0.000 | 0 |
  | clutter / nominal / overhead + mobile | 1/12 | 0.051 | 1 |

- **Finding (negative for the current setup).** With box / cylinder / sphere primitives and
  good depth, grasp-relevant ambiguity essentially never survives B2 tracking. No viewpoint
  policy can change grasp outcomes here (ceiling); the only differences are travel cost.
- **Defect found.** Task-risk evaluates loss over belief particles, which exclude calibration
  uncertainty. It predicts ~0 risk under harsh calibration, where real errors (~13 mm) exceed
  the 1 cm grasp tolerance.
- **Design change (val-motivated; hypothesis unchanged; recorded before any M5 test):**
  (1) add a mug primitive (cylinder + handle) whose handle can self-occlude, creating
  grasp-relevant yaw ambiguity — SE(3)-PoseFlow's example case; (2) make task-risk
  calibration-aware.

## VAL-M5-003: mug regime + a rejected perception change (val)

- **Mug probe** (12 val episodes): ambiguity at t₀ is 0/12 with the mobile camera only and
  1/12 with overhead + mobile.
- **Diagnosis** (8 episodes): in 7/7 detected cases the mug's handle was visible and yaw
  was correct, with a single mode and 1.1–2.3° error. From the 35–60° elevated candidate
  views, a horizontally protruding handle stays visible even when it points away from the
  camera.
- **Finding:** with elevated cameras, known CAD and good depth, grasp-relevant visual
  ambiguity is rare. Active viewpoint selection has little to act on there.
- **Detection miss** (val seed 1,500,800): a touching neighbour merged into the mug's
  segment; the merged blob's fit RMS was 10.5 mm, over the 10 mm cutoff, so it was dropped.
- **Hypothesised fix, rejected on evidence:** hue-gated seed growth, scored with
  `experiments/analysis/seg_probe.py` on 6 val episodes per scenario.

  | Scenario | TP / FN / FP, with gate | without gate |
  |---|---|---|
  | clutter | 70 / 4 / 3 | 71 / 3 / 2 |
  | mugs | 58 / 4 / 1 | 59 / 3 / 0 |
  | varied | 53 / 1 / 0 | 53 / 1 / 0 |

  No improvement, so the change was **reverted**; merges in tight clutter stay a known
  limitation.

## EXP-M5-001 — PRE-REGISTERED (written before running; code frozen at `6326137`)

- **Question (H3, docs/research_question.md).** At equal observation budgets, does choosing
  views by expected grasp failure (task_risk, D) beat choosing them by information
  objectives — mode-weight entropy (C1, ActivePose-style) and expected rotational variance
  (C2, SE(3)-PoseFlow-style) — on grasp success?
- **Regime** (selected on val by VAL-M5-001..003 and the ambiguity probes, before any test):
  - `mugs` scenario (4–6 objects, one mug, 9 cm spacing), `harsh` noise;
  - mobile camera only, candidate views at 12° / 25° / 45° elevation (24 views);
  - budget 2 camera moves after one shared warm-up frame;
  - B2 belief tracker with the calibration prior; target = highest grasp-now risk.
  - Val probe: 9/24 episodes ambiguous at t₀, mostly with low grasp risk.
- **Primary outcome.** *Physical* grasp success: the final grasp, planned from the target belief's
  MAP mode, is executed once on the simulated Panda; success = the target-class object is lifted
  > 8 cm (evaluator reads simulator state). No recovery attempts in this experiment.
- **Data.** 150 never-evaluated test episodes (test offset 1000–1149, seeds 2,001,000–2,001,149),
  all 5 policies per episode (paired).
- **Primary analysis.** `experiments/analysis/m5_analysis.py`: for D vs C1 and D vs C2, the
  paired per-episode success difference, bootstrap 95 % CI, exact two-sided McNemar test;
  Holm correction across the two.
- **Secondary (descriptive):**
  - D vs fixed and D vs random;
  - success on the ambiguous-at-t₀ subset;
  - mean camera travel time and decision time;
  - agreement between the geometric proxy and physical outcomes.
- **Stated expectation.** Val ambiguity was mostly grasp-irrelevant, so success differences
  may be small or null. A null H3 will be reported as such, with an analysis of when D and
  C1/C2 choose different views.

## EXP-M5-001 RESULT (pre-registered above; 150 paired test episodes, physical grasp scoring)

| Policy | Physical success | Ambiguous-at-t₀ subset (n = 28) | Mean travel | Mean decide time |
|---|---|---|---|---|
| fixed | 0.487 | 0.571 | 0 s | 5 ms |
| **task_risk (D)** | **0.487** | 0.571 | **0 s** | 56 ms |
| entropy (C1) | 0.507 | 0.679 | 0.54 s | 33 ms |
| covariance (C2) | 0.507 | 0.679 | 0.54 s | 34 ms |
| random | **0.627** | 0.643 | 3.51 s | 6 ms |

- **H3 (primary): NOT SUPPORTED.** D vs C1 and D vs C2 are both −0.020, bootstrap 95 % CI
  [−0.060, 0.020]; 3 D-only vs 6 baseline-only successes; McNemar p = 0.51, Holm p = 1.0.
- **Task-risk never moved.** It predicted ~zero grasp risk under the belief in every episode,
  so it behaved exactly like fixed.
- **Secondary, unexpected and significant: random beats D** by +0.140 (95 % CI of D − random
  [−0.22, −0.06]; 11 vs 32 discordant; McNemar p = 0.0019).
- **Mechanism analysis** (`reports/m5/exp_m5_001/mechanism.txt`).
  - **Centre error:** random's grasp-centre error is lower than D's (paired mean −3.84 mm,
    median −2.38 mm), and lower in 73 % of episodes.
  - **Moving helps:** when the information policies moved (29 episodes) they succeeded 0.66,
    vs 0.47 when they stayed.
  - **View diversity:** random visits 1.93 distinct views on average, vs ~1.0 for the others.
  - **Interpretation:** under `harsh` noise the dominant error is the per-episode extrinsic
    calibration error. It is fixed in the *camera* frame, so moving the camera re-samples its
    world-frame effect, and diverse views average it out. None of the objectives model this:
    the belief is conditional on calibration. The earlier claim (in the M5 design notes) that
    calibration blindness is "benign for view choice" was **wrong**. It assumed a
    view-independent offset.
- **Proxy vs physics.** The 1 cm geometric proxy agrees with physical outcomes in only
  64–73 % of cases (185 proxy-fail / physics-success, 35 proxy-success / physics-fail over all
  750 rows), which confirms that switching to physical scoring was necessary.
- **Conclusion.** In this regime grasp success is limited by systematic calibration error,
  not by pose ambiguity. A task-aware objective that ignores the dominant error source does no
  better than staying put. The principled next step (a new method variant, to be developed on
  val and re-registered) is to model view-dependent calibration bias in the predicted task
  loss, so the objective values view diversity.

## VAL-M5-004: D2 (calibration-aware task-risk) on val (40 paired episodes, EXP-M5-001 regime)

| Policy | Success | Median centre error | Travel | Distinct views |
|---|---|---|---|---|
| fixed / task_risk (D) | 0.450 | 19.8 mm | 0.00 s | 1.00 |
| entropy | 0.450 | 19.5 mm | 0.32 s | 1.00 |
| **task_risk_cal (D2, λ = 0.02)** | **0.550** | **15.3 mm** | 3.61 s | 1.60 |
| random | 0.600 | 15.3 mm | 3.97 s | 1.98 |

D2 matches random's centre-error reduction by deliberate choice but visits fewer distinct
views, apparently because the travel penalty outweighs the calibration benefit of a third view.

**VAL-M5-005 selection rule (fixed BEFORE the λ-sweep results were seen):** choose
λ ∈ {0, 0.005, 0.02} with the highest val physical success. Break ties by lower mean travel,
then by larger λ. The chosen λ is frozen for EXP-M5-002.

## VAL-M5-005 result (λ sweep) → D2 frozen

| D2 variant (40 paired val episodes) | Success | Travel | Distinct views |
|---|---|---|---|
| λ = 0.02 | 0.550 | 3.61 s | 1.60 |
| λ = 0.005 | 0.575 | 4.15 s | 1.82 |
| λ = 0 | 0.575 | 4.15 s | 1.82 |

By the pre-committed rule (highest success; tie → lower travel → larger λ), **D2 = λ = 0.005**
(`task_risk_cal_l005`). Random scored 0.600 on the same episodes.

## EXP-M5-002 — PRE-REGISTERED (written before running; code frozen at `0436e79`)

- **Question.** Does a task-risk objective that models the dominant, view-dependent error source
  (calibration bias) improve physical grasp success over the original task-risk (D) and over
  the information objectives (C1 entropy, C2 covariance)?
- **Regime.** Identical to EXP-M5-001: mugs / harsh / mobile only / 12°, 25°, 45° views /
  budget 2 / warm-up 1 / physical single-attempt scoring.
- **Data.** 150 never-evaluated test episodes, test offset 1200–1349
  (seeds 2,001,200–2,001,349). Policies: fixed, random, entropy, covariance, task_risk,
  task_risk_cal_l005, all paired.
- **Primary.**
  - D2 vs D, D2 vs C1, D2 vs C2.
  - Paired success difference with bootstrap 95 % CI, exact McNemar, Holm across the three.
  - `python experiments/analysis/m5_analysis.py reports/m5/exp_m5_002 --treat task_risk_cal_l005
    --primary task_risk,entropy,covariance --secondary random,fixed`
- **Secondary (descriptive, two-sided).** D2 vs random (on val 0.575 vs 0.600, so no
  superiority is expected); D2 vs fixed; travel; distinct views; centre error.
- **Stated expectation.** D2 > D and D2 > C1/C2, driven by view diversity, and D2 ≈ random.
  Any outcome is reported as-is.
