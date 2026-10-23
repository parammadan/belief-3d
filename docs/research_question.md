# Research question

**RQ.** Under visual ambiguity (occlusion, symmetry, depth corruption, calibration drift), does
choosing camera actions by *expected task loss under a calibrated multimodal pose belief* improve
simulated grasp reliability over (B0) fixed-view single-frame pose, (B1) fixed-view temporal
tracking, (B2) multimodal belief without active vision, and (B3) information-gain active vision,
**at equal camera, compute and grasp-attempt budgets**?

This is a hypothesis, not a claim. See `literature_review.md` for the prior work that bounds any
novelty claim.

## Hypotheses (registered before any policy evaluation)

- **H1 (belief).** A multimodal belief (B2) has lower symmetry-aware pose error and better coverage
  than a unimodal EKF (B1) under severe and complete occlusion. Equal performance under low
  occlusion is expected.
- **H2 (calibration).** After held-out calibration, the belief's credible regions reach nominal
  coverage within ±5 pp at 50/80/95 %.
- **H3 (policy).** At equal budgets, B4 (task-risk) beats B3 (information gain) on grasp success.
  If H3 fails, we report it and analyse when the two objectives choose different views.
- **H4 (mechanism).** Any B4 > B3 gap shrinks when calibration is removed (A2). That would mean
  calibration, not the objective form, carries the benefit.

## Decision objective (starting point, to be finalised in M5)

    a* = argmin_{a ∈ A}  E_{x ~ b_t, z ~ p(z|x,a)} [ L_task(π_grasp(b_{t+1}(z)), x) ] + λ C(a)

Here `L_task` is the predicted grasp failure (0/1) of the grasp the system *would* execute from
the updated belief `b_{t+1}`, and `C(a)` is camera travel time plus render/inference time.
The separate `β R_uncertainty` term from the master prompt is **dropped by default**. The expected
task loss already integrates over belief uncertainty, so adding a variance term double-counts
it (§5.4). It returns only as a labelled variant: CVaR_α of `L_task` replacing the expectation.
Every expectation is estimated from belief particles with the observation model the tracker
itself uses. **No simulator state appears in the objective.**

## Data protocol

- Seeds are split by range: train `[0, 10⁶)`, calibration `[10⁶, 1.1·10⁶)`, test `[2·10⁶, …)`.
  Object meshes, textures and clutter configurations are held out per split (from M1).
- All methods are compared with paired seeds; we report means with 95 % bootstrap CIs and
  McNemar or paired-bootstrap tests for success rates.
