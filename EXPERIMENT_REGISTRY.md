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

## EXP-M5-002 RESULT (pre-registered above; 150 paired test episodes; physical scoring)

| Primary (Holm across 3) | Δ success | Bootstrap 95 % CI | Discordant (D2 / base) | McNemar p | Holm p |
|---|---|---|---|---|---|
| **D2 vs D (task_risk)** | **+0.147** | [0.060, 0.240] | 36 / 14 | 0.0026 | **0.0078** |
| **D2 vs C1 (entropy)** | **+0.133** | [0.053, 0.220] | 32 / 12 | 0.0037 | **0.0078** |
| **D2 vs C2 (covariance)** | **+0.133** | [0.053, 0.220] | 32 / 12 | 0.0037 | **0.0078** |

**All three primary hypotheses CONFIRMED.**

| Policy | Success | Median centre error | Distinct views | Travel | Decide time |
|---|---|---|---|---|---|
| fixed | 0.520 | 18.29 mm | 1.00 | 0.00 s | 5 ms |
| task_risk (D) | 0.520 | 18.29 mm | 1.00 | 0.00 s | 56 ms |
| entropy (C1) | 0.533 | 17.12 mm | 1.06 | 0.57 s | 37 ms |
| covariance (C2) | 0.533 | 17.12 mm | 1.06 | 0.57 s | 38 ms |
| random | 0.633 | 14.38 mm | 1.95 | 3.51 s | 7 ms |
| **task_risk_cal (D2)** | **0.667** | **12.66 mm** | 1.62 | 3.90 s | 185 ms |

- **Secondary.** D2 vs random +0.033, 95 % CI [−0.053, 0.120], p = 0.54 (not significant,
  as stated in advance). D2 vs fixed +0.147, p = 0.0026.
- **Mechanism.** D2 has the lowest grasp-centre error of all policies (paired vs random:
  −2.13 mm mean). It achieves this with *fewer* distinct views than random (1.62 vs 1.95):
  it picks views whose calibration biases cancel rather than simply moving more.
- **Reading of the two M5 studies.** A task-aware objective beats information objectives
  *only when its loss model includes the error source that actually limits the task*. Here
  that is view-dependent extrinsic calibration bias, not pose ambiguity. The same objective
  without it (D, EXP-M5-001) never moved and tied with fixed. Simulation only; one regime
  (mugs / harsh / mobile camera / low views); a single calibration-error model.

## EXP-M6-001 — PRE-REGISTERED (written before running; code frozen at `f587448`)

- **Task.** Closed-loop pick-and-place (`scripts/run_m6.py`). The target is chosen from the
  belief; the policy picks views (budget 2); the grasp comes from the target belief's MAP mode
  and is executed on the simulated Panda. A proprioceptive check (finger width) decides
  held / not held. On failure: park the arm, re-observe with the policy, re-plan, retry. If held,
  the object is placed in a walled tray, then the arm retreats vertically and parks.
- **Success (evaluator, simulator state).** A target-class object ends inside the tray.
- **Regime.** mugs / harsh / mobile camera / default candidate views (35°, 60°) / budget 2.
- **Arms.** max_attempts = 1 vs max_attempts = 2, on the *same* 100 never-evaluated test seeds
  (offset 1400–1499). Policies: fixed and task_risk_cal_l005 (D2).
- **Primary (H6).** Per policy, success with 2 attempts > with 1. Exact McNemar on paired
  outcomes; Holm across the two policies. `experiments/analysis/m6_analysis.py`.
- **Descriptive.** Success rates, mean attempts, retry success rate, verification errors (held
  but not placed; not held but placed).
- **Commitment.** Report as-is.

## EXP-M6-001 RESULT: COMPROMISED by two harness defects (reported, not used for claims)

1. **D2 arm invalid.** `run_m6.py` built the PolicyContext without the view history and
   calibration prior, so D2's calibration term was 0 and it degenerated to plain task-risk.
   Its results were identical to `fixed` in every field.
2. **Crashes biased the sample.** `modes_of` could return an empty mode set for a spread-out
   belief (all clusters < 2 % mass), which crashed 12 of 20 retry-arm tasks. Only 87/100
   episodes paired, plausibly the easier ones.
- **What the valid fixed arm shows (descriptive only):** success 0.529 with 1 attempt and
  0.529 with 2 (1 / 1 discordant); retry success 3/40 (7.5 %). Retrying from the same
  belief without new information repeats the same calibration-biased grasp.
- **Fixes** (commit `256fa35`): context passes view history and calibration prior; `modes_of`
  always keeps the heaviest cluster (regression test added). Re-run as EXP-M6-002.

## EXP-M6-002 — PRE-REGISTERED (written before running; code frozen at `256fa35`)

Same task, regime, arms, policies, analysis and commitment as EXP-M6-001. New,
never-evaluated test seeds, offset 1500–1599. H6: per policy, success with 2 attempts > with 1
(exact McNemar, Holm across fixed and D2). Descriptive: D2 vs fixed at each attempt budget
(no correction claimed).

## EXP-M6-002 RESULT (pre-registered above; 100 paired test episodes per arm, all complete)

| Policy | Success, 1 attempt | Success, 2 attempts | Gain (recovered / lost) | McNemar p | Holm p | Retry success | Mean attempts |
|---|---|---|---|---|---|---|---|
| fixed | 0.47 | 0.50 | +0.03 (5 / 2) | 0.453 | 0.453 | 14.0 % (7/50) | 1.50 |
| D2 (task_risk_cal_l005) | 0.59 | 0.64 | +0.05 (8 / 3) | 0.227 | 0.453 | 13.9 % (5/36) | 1.36 |

- **H6 (recovery raises success): NOT SUPPORTED** for either policy.
- **Why retries rarely help.** Retrying re-plans from essentially the same belief, and under
  `harsh` noise failures are dominated by calibration bias, which a re-plan does not remove.
- **Descriptive, uncorrected.** D2 beats fixed by +0.12 at 1 attempt and +0.14 at 2,
  consistent with EXP-M5-002, now on the full pick-and-place task.
- **Proprioceptive verification errors.** "Held" but not placed in 5–7 % of episodes;
  "not held" but actually placed in 3–4 %.
- **M6 exit criterion met.** The robot completes randomised pick-and-place episodes from
  perception-derived targets, with success 0.50 (fixed) / 0.64 (D2) under `harsh` calibration
  error. The limiting error is calibration, which is the motivation for online self-calibration.

## VAL-M5-006: online extrinsic self-calibration (val, 40 paired episodes, EXP-M5 regime)

`perception/calibration/self_calib.py` solves a linear Gaussian MAP for the camera's
extrinsic error from repeated observations of static tracked objects across views; the grasp
centre is then corrected by the estimated mean bias over the target's views.

| Policy | Without self-cal | With self-cal | Median centre error | Median correction |
|---|---|---|---|---|
| fixed | 0.525 | 0.550 | 17.9 → 17.9 mm | 0.0 mm |
| random | 0.700 | 0.775 | 13.4 → 9.4 mm | 10.8 mm |
| D2 (task_risk_cal_l005) | 0.675 | 0.775 | 13.0 → 9.6 mm | 8.5 mm |

- **Fixed:** one view gives nothing to estimate from, so the prior holds the correction at 0
  (fails safe).
- **Diverse views:** about 30 % lower centre error.
- **Identifiability (unit-tested):** only the *view-varying* part of the bias is identifiable;
  the part equivalent to a common world-frame motion is a gauge freedom and needs an external
  reference.

## EXP-M5-003 — PRE-REGISTERED (written before running; code frozen at `2036068`)

- **Regime.** Identical to EXP-M5-001/002.
- **Data.** 150 never-evaluated test episodes, offset 1600–1749. Each run twice (without and
  with `--selfcal`) for policies fixed, random and D2.
- **Primary (H7).** D2+selfcal > D2 on physical grasp success. Exact McNemar on paired
  outcomes; single primary test, no correction.
- **Secondary (descriptive).** random+selfcal vs random; fixed+selfcal vs fixed (control,
  expected ≈ 0); D2+selfcal vs random+selfcal; centre-error change; median correction size.
- **Analysis.**
  `python experiments/analysis/m5_analysis.py "reports/m5/exp_m5_003_sc0+reports/m5/exp_m5_003_sc1"
  --treat task_risk_cal_l005+selfcal --primary task_risk_cal_l005
  --secondary random+selfcal,random,fixed+selfcal,fixed`
- **Commitment.** Report as-is.

## EXP-M5-003 RESULT (pre-registered above; 150 paired test episodes; physical scoring)

**Data-integrity note.** The first automatic copy from the cluster failed midway (`scp:
Connection closed`) and the analysis ran on 645 of 900 rows (65 episodes). That partial output
(Δ = +0.169) was **discarded unreported**. All 60 tasks were confirmed COMPLETED, all 900 rows
were re-copied (tar over ssh), and the analysis below uses the full sample.

| Comparison | Success | Δ (bootstrap 95 % CI) | Discordant | McNemar p |
|---|---|---|---|---|
| **H7 primary: D2+selfcal vs D2** | **0.767 vs 0.640** | **+0.127 [0.047, 0.207]** | 29 / 10 | **0.0034** |
| control: fixed+selfcal vs fixed | 0.513 vs 0.513 | 0.000 | — | — |
| random+selfcal vs random | 0.747 vs 0.627 | +0.120 | — | — |
| D2+selfcal vs random+selfcal | 0.767 vs 0.747 | +0.020 [−0.060, 0.100] | 21 / 18 | 0.749 |
| D2+selfcal vs random | 0.767 vs 0.627 | +0.140 [0.047, 0.233] | 37 / 16 | 0.0055 |
| D2+selfcal vs fixed | 0.767 vs 0.513 | +0.253 [0.160, 0.347] | 48 / 10 | < 0.001 |

- **H7 CONFIRMED.**
- **Control behaved as predicted:** with a single view there is nothing to estimate, so the
  correction is 0 mm and the effect is exactly 0.
- **Centre error** (`reports/m5/exp_m5_003_sc1/centre_error.txt`): D2 12.79 → 9.13 mm median
  (median correction 6.7 mm); random 14.73 → 9.33 mm (11.6 mm); fixed 17.54 → 17.65 mm.
- **Reading.** The improvements stack. Calibration-aware view selection (D2) beats a fixed camera
  (0.513 → 0.640); self-calibration exploits the view diversity D2 provides (0.640 → 0.767).
  Together that is +0.253 absolute success under ~1° calibration error. Once self-calibrated,
  D2 ≈ random (n.s.): most of the benefit comes from view *diversity* plus estimating the bias it
  reveals, not from fine view choice.
- Simulation only; one regime; one calibration-error model.

## PERF-M7-002: pose-proposer inference benchmark (model-only; V100-SXM2-32GB; 200 iters, 20 warm-up, CUDA-synchronised)

| Backend | Batch | Median | p95 | p99 | Throughput | Peak memory | Top-1 surface error (2,048 val) |
|---|---|---|---|---|---|---|---|
| PyTorch fp32 | 1 | 1.16 ms | 1.17 ms | 1.18 ms | 861 /s | 29 MB | 4.920 mm (reference) |
| PyTorch fp32 | 32 | 1.22 ms | 1.24 ms | 1.25 ms | 26.1k /s | 78 MB | |
| PyTorch fp32 | 256 | 6.76 ms | 6.79 ms | 6.80 ms | 37.8k /s | 432 MB | |
| PyTorch fp16 autocast | 1 | 1.20 ms | 1.33 ms | 1.35 ms | 834 /s | 30 MB | 4.915 mm |
| PyTorch fp16 autocast | 32 | 1.20 ms | 1.21 ms | 1.21 ms | 26.7k /s | 71 MB | |
| PyTorch fp16 autocast | 256 | 4.94 ms | 4.95 ms | 5.03 ms | 51.8k /s | 366 MB | |

- **fp16:** 1.37× faster at batch 256, no gain (slightly slower) at batch 1, where launch
  overhead dominates. Accuracy is preserved (4.915 vs 4.920 mm). argmax agreement 0.9995:
  0.05 % of samples flip the top-ranked hypothesis (max |ΔR| 0.31 comes from those flips).
- **ONNX export:** ok (opset 17; `torch.cross` replaced by an explicit, identical cross product).
- **ONNX Runtime GPU: not yet measured.** Run 11351691 *silently fell back to the CPU provider*.
  Its numbers are CPU timings under GPU labels and are **discarded**; the benchmark now refuses
  to report a fallback. Root cause (run 11352235): onnxruntime-gpu 1.31 targets CUDA 13
  (`libcublas.so.13`), while torch cu124 bundles CUDA 12. Being re-done with a version-consistent
  stack (onnxruntime-gpu 1.22, CUDA 12 build; TensorRT 10.x).

## PERF-M7-003: fair ONNX Runtime comparison (V100; onnxruntime-gpu 1.22 / CUDA 12; GPU-resident I/O via IOBinding)

| Backend (device-resident I/O) | b1 median / p99 | b32 median | b256 median | Accuracy vs fp32 |
|---|---|---|---|---|
| PyTorch fp32 | 1.049 / 1.064 ms | 1.333 ms | 6.770 ms | reference (4.920 mm) |
| PyTorch fp16 autocast | 1.200 / 1.216 ms | 1.201 ms | 4.935 ms | 4.915 mm; argmax agree 0.9995 |
| **ONNX Runtime CUDA** | **0.482 / 0.493 ms** | 1.886 ms | 13.722 ms | 4.920 mm; max \|ΔR\| 2.3e-4; argmax agree 1.000 |

- **ORT CUDA is 2.2× faster at batch 1** (lower per-call dispatch overhead) but slower at batch
  32 (1.4×) and batch 256 (2.0×), where its kernels lose to PyTorch's.
- **Host copies cost little** (b1 0.551 vs 0.482 ms; b256 14.1 vs 13.7 ms).
- **Deployment reading.** The perception pipeline calls the proposer per segment (batch ≈ 1–6),
  so ORT CUDA is the better backend for this workload, and for large offline batches PyTorch
  fp16 is best (51.9k samples/s).
- **TensorRT on V100: not supported.** TensorRT 10.16 rejects SM 70 ("Target GPU SM 70 is not
  supported by this TensorRT release"). The earlier 200 ms "TensorRT" timings were ORT partially
  falling back and are **discarded**. Being re-run on an A100 (SM 80).

## PERF-M7-005: C++ free-space likelihood (pybind11; M1 Mac, single thread, Apple clang -O3, no fast-math)

`cpp/src/free_space.cpp` is an exact port of `PoseBelief.free_space_log_lik`, enabled with
`BeliefConfig(fs_backend="cpp")`. The default stays `"numpy"`, so earlier runs replay unchanged.
The kernel fuses the per-particle × per-sample loop, which removes the N × M temporaries.
Rounding uses `nearbyint` (round-half-to-even, like `np.round`), so pixel lookups match.

| Object (M samples) | N=256 NumPy → C++ | N=1024 NumPy → C++ |
|---|---|---|
| box_red (902) | 7.22 → 1.46 ms (4.9×) | 35.2 → 5.92 ms (5.9×) |
| cyl_green (1007) | 7.88 → 1.64 ms (4.8×) | 36.3 → 6.95 ms (5.2×) |
| sphere_blue (615) | 5.25 → 1.05 ms (5.0×) | 23.7 → 4.51 ms (5.2×) |
| mug_cyan (1633) | 13.6 → 2.86 ms (4.8×) | 94.4 → 11.3 ms (8.3×)* |

(All 8 catalogue objects are in `reports/m7/free_space_bench_m1.json`; the range is 4.8–6.9×,
excluding the starred row.) *The mug N=1024 NumPy time was 51.6 ms in an earlier run of the
same script (4.7×), so treat 8.3× as timing noise.

- **Equivalence:** max |Δ| ≤ 2.7e-13 nats over every row.
  `tests/unit/test_cpp_kernels.py` checks 4 shapes × 3 seeds with wide clouds (rtol 1e-9). CI
  now builds the extension, so the test runs there instead of skipping.
- **End to end (5 updates, 256 particles, 2 views):** 163.8 → 48.0 ms per update (**3.4×**).
  With the same seed, the particle sets after 5 updates are identical between backends (atol
  1e-9). The surface SDF term, resampling and MCMC are still NumPy.
- **Not done:** multithreading. Apple clang ships without OpenMP, and the loop over particles is
  embarrassingly parallel, so this is the next step if it is needed.

## PERF-M7-006: pose-fit Jacobian (M1 Mac; 6 harsh frames, 5 reps, median)

- **Profile before:** `perceive()` took ~278–306 ms per frame. Pose fitting was 87 % of that, and
  scipy's 2-point finite-difference Jacobian was ~60 % (6 separate residual calls per Jacobian,
  each with Python overhead).
- **Change:** `fit_hypotheses(..., fast_jac=True)` (the default). It applies the same 2-point
  rule with the same steps, but the base point and all 6 perturbations go through one batched
  `sdf` call. The three translation columns reuse the base rotation.
- **Result:** 306 → 237 ms per frame (**1.29×**). Every pose and cost is **bit-identical** to
  scipy's Jacobian (`test_batched_jacobian_gives_identical_fits`; benchmark check over all
  detections), so earlier experiments replay unchanged.
- **What remains:** scipy's trust-region internals and small-array NumPy calls, about 24
  iterations × 4.5 initialisations per segment. Further gains need an algorithmic change, such as
  fewer initialisations (pruning by early cost) or an analytic Jacobian. Either would change
  results, so it needs its own accuracy check. Perception is still not real-time on CPU.

## EXP-M8-001 — PRE-REGISTERED final study (written before running; code frozen at `c276e52`)

- **Question (RQ, docs/research_question.md).** At equal camera, compute and grasp-attempt budgets,
  does the full system beat the baselines on simulated grasp reliability, and which components
  carry the gain? The full system is the calibrated task-risk view policy D2 (λ=0.005), the B2
  belief, and online self-calibration.
- **Regime.** Identical to EXP-M5-001/002/003, so results are comparable:
  - `mugs` scenario, `harsh` noise, mobile camera only;
  - candidate views at 12° / 25° / 45°; budget 2 moves after one shared warm-up frame;
  - physical grasp scoring, one attempt.
- **Arms (8, all paired on the same seeds).**

  | Arm | Views | Grasp pose from | Self-cal |
  |---|---|---|---|
  | `fixed@b0` (B0) | fixed | latest single-frame fit | no |
  | `fixed@b1` (B1) | fixed | Kalman tracker fed the same detections | no |
  | `fixed` (B2) | fixed | belief MAP mode | no |
  | `random` | random | belief | no |
  | `entropy` (B3, info gain) | mode-weight entropy | belief | no |
  | `task_risk_cal_l005` (D2) | calibrated task risk | belief | no |
  | `fixed+selfcal` | fixed | belief | yes |
  | `task_risk_cal_l005+selfcal` (**full**) | calibrated task risk | belief | yes |

  B0 and B1 share B2's target choice, which picks the highest-risk target from the belief. The
  fixed policy's views don't depend on the belief, so the only difference between B0, B1 and B2 is
  the pose the grasp is planned from (same observations). That isolates the estimator; it is not a
  fully independent B1 pipeline.
- **Data.** 300 never-evaluated test episodes, offset 4000–4299 (seeds 2,004,000–2,004,299).
- **Primary family (Holm-corrected, exact two-sided McNemar on paired outcomes).**
  - **P1** full system > B1. This is the headline claim against the standard tracker baseline.
  - **P2** D2 > B3 entropy. Replicates EXP-M5-002 on new seeds.
  - **P3** full system > D2. Self-calibration's contribution; replicates H7 (EXP-M5-003).
  - **P4** B2 > B1 with fixed views. H1 (belief vs Kalman) in the closed loop.
- **Secondary (descriptive, uncorrected).**
  - B1 vs B0; D2 vs random. EXP-M5-002 found them ≈ equal, so no claim is made either way.
  - fixed+selfcal vs fixed: a control, expected ≈ 0.
  - full vs B2 fixed; full vs B0.
  - Every arm on the ambiguous-at-t₀ subset.
- **Analysis.** `python experiments/analysis/m8_analysis.py
  "reports/m8/exp_m8_001_sc0+reports/m8/exp_m8_001_sc1" --out reports/m8/exp_m8_001_analysis.json`
- **Compute.** Explorer CPU array of 60 tasks × 5 episodes each. sc0 runs 6 arms; sc1 runs 2 arms
  with `SELFCAL=1`. No GPU.
- **Commitment.** Report every outcome as-is, including nulls and reversals. Any episode lost to
  infrastructure failure is reported, and its seed is re-run unchanged (no seed replacement).

## EXP-M8-002 — PRE-REGISTERED coverage calibration (H2 of docs/research_question.md; code frozen at `4c5c4d5`)

The "H2" in EXP-M4-001 was a different hypothesis (MOTA non-inferiority). This entry tests the
research question's H2: after held-out calibration, B2's credible regions reach nominal coverage
within ±5 pp at 50/80/95 %.
- **Measure.** NEES of the reported 3×3 position covariance per matched track-frame. Scaling the
  covariance by s divides NEES by s.
- **Calibration data.** The calibration split (never used before), offset 0, 20 episodes × 15
  frames, both cameras, for each of occlusion/dynamic × nominal/harsh.
- **Test data.** 20 never-evaluated test episodes per condition, offset 4400–4419 (after
  EXP-M8-001's 4000–4299).
- **Primary.** One scalar per noise preset, s_p = q₀.₈(NEES_calib) / χ²₃(0.8). H2 holds if test
  coverage is within ±5 pp of nominal at all three levels in both presets (6 checks).
  Episode-cluster bootstrap CIs are reported.
- **Secondary (descriptive).**
  - raw (uncalibrated) coverage;
  - one global scale, pooling both presets;
  - per-level split-conformal scales (calibrates each level separately);
  - the same numbers for B1 (Kalman).
- **Stated expectation, from val (VAL data only, scratch probe).**
  - Raw nominal coverage is 1.00 at every level: the covariance carries the calibration
    allowance (1°, 5 mm), and nominal noise has no calibration error.
  - The per-preset scale fixes 80 % by construction, but on val it left harsh at 50 % = 0.58 and
    nominal at 95 % = 0.91. The NEES distribution's shape is not χ²₃, so **H2 is expected to fail**
    as written.
  - A global scale fails badly (harsh 0.30/0.60/0.73).
  - Per-level conformal should hold by construction, up to calibration-to-test shift.
- **Analysis.** `python experiments/analysis/m8_h2.py reports/m8/h2 --out reports/m8/exp_m8_002_analysis.json`
- **Compute.** `sbatch --array=0-7 --export=ALL,N=20,CALIB_OFFSET=0,TEST_OFFSET=4400
  experiments/cluster/h2.sbatch`. CPU only.
- **Commitment.** Report as-is.

## PERF-M7-004: TensorRT on A100 (A100-SXM4-80GB; ORT 1.22 TensorRT EP, TRT 10.16, fp16; fresh engine cache; device-resident I/O)

| Backend | b1 median | b32 median | b256 median (throughput) | Accuracy (top-1 mm; argmax agree) |
|---|---|---|---|---|
| PyTorch fp32 | 0.943 ms | 0.950 ms | 2.664 ms (96k/s) | 4.917 (reference) |
| PyTorch fp16 autocast | 1.085 ms | 1.105 ms | 2.894 ms (88k/s) | 4.915; 0.9985 |
| ONNX Runtime CUDA | 0.422 ms | 0.808 ms | 4.369 ms (59k/s) | 4.921; 0.9995 |
| **TensorRT fp16** | **0.192 ms** | **0.310 ms** | **1.586 ms (161k/s)** | 4.921; 0.9990 |

- **TensorRT is fastest at every batch size.** Against PyTorch fp32 it is 4.9× at b1, 3.1× at b32
  and 1.7× at b256; against ORT CUDA it is 2.2× at b1. Host copies add 0.06 ms at b1.
- **Accuracy holds.** Top-1 error is 4.921 mm vs 4.917 mm, and argmax agreement is 0.9990. The
  largest elementwise rotation difference over all hypotheses is 0.67, which means a few
  low-weight fp16 hypotheses flip. The selected pose is unaffected at this rate.
- **Integrity notes.**
  - The first A100 attempt failed on a harness bug: the accuracy pass fed a 2048 batch beyond the
    engine profile (max 256). It is fixed by chunking.
  - The second attempt reused an engine built on an A100-40GB, which TensorRT does not support,
    and was **discarded**. The benchmark now clears its engine cache on every run.
  - The run above built its engines fresh.
- `reports/m7/infer_bench_a100.json`. Model-only latency; the perception pipeline still costs
  ~237 ms per frame on CPU (PERF-M7-006).
