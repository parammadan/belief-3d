# Literature review (M0, initial)

Status: **initial scan, 2026-10-23.** Items marked *abstract-level* were checked only on the
arXiv abstract page. Their method details are not yet verified and must not be cited for
specifics until the full paper has been read. Items marked *from memory* need a DOI check
before they go into the paper.

## Closest prior work: possible novelty conflicts

| Work | What it does | Overlap with RCABT |
|---|---|---|
| **SE(3)-PoseFlow**: Jin, Funk, Prasad, Li, Franzius, Peters, Chalvatzaki. arXiv:2511.01501 (Nov 2025). *Full HTML read 2026-10-24.* | Flow matching on SE(3) outputs pose *samples*. NBV (their Eq. 7): v* = argmin_v E[tr(Σ_R \| v)], i.e. **minimise predicted rotational covariance** over viewpoints sampled on a sphere. Grasps marginalise EquiGraspFlow velocity fields over hypotheses. Real Franka: mug grasping 19/20 multi-hypothesis vs 15/20 single. | **Closest work, but its view objective is uncertainty-only and "not linked" to grasp outcomes.** It reports no quantitative active-perception numbers and no Bayesian filtering over time; it lists "how to incorporate Bayesian inference for principled utilization of these samples" as an open question. Its objective becomes our B3 variant ("minimise expected rotational covariance"). |
| **ActivePose**: Liu et al. arXiv:2509.11364 (Sep 2025, rev. Mar 2026). *Full HTML read 2026-10-24.* | Score S_j = 0.6 · normalised entropy of FoundationPose's top-N hypotheses + 0.4 · VLM ambiguity probability, over 12 rendered candidate views (argmin). Single-view: **no temporal Bayesian belief**. Reported success at 5 mm / 5°: 92.5–97.5 % vs Entropy-NBV 42–64 % (sim + real Franka). | Entropy/ambiguity NBV for 6D pose. Neither term is task- or grasp-based. It is our "entropy" comparator (B3b). Its gains come partly from a VLM prior that we do not use. |
| **TVVE**: Bai et al. "Learning to See and Act: Task-Aware Virtual View Exploration". arXiv:2508.05186 (CVPR 2026 per arXiv page). *abstract-level* | Learned, task-aware virtual viewpoint selection for VLA manipulation (RLBench). | Medium. It is task-aware but learned end-to-end, with no explicit pose belief stated in the abstract. |
| **PoseRBPF**: Deng, Mousavian, Xiang, Xia, Bretl, Fox. RSS 2019, arXiv:1905.09304. | Rao-Blackwellized particle filter: rotation as a discretised distribution, translation per particle; handles symmetry and occlusion. | It is the canonical multimodal 6D tracker. Our belief tracker is **not** a contribution over it; it is a baseline to reimplement or compare against. |
| **FoundationPose**: Wen, Yang, Kautz, Birchfield. CVPR 2024, arXiv:2312.08344. | Unified model-based/model-free 6D pose estimation and tracking. | Level-3 published baseline (§10). The license is **unverified**: a third-party weights mirror says CC-BY-NC-4.0. Check the official NVlabs LICENSE before use. |
| **POTAMP**: Curtis, Matheos, Gothoskar, Mansinghka, Tenenbaum, Lozano-Pérez, Kaelbling, "Partially Observable Task and Motion Planning with Uncertainty and Risk Awareness", arXiv:2403.10454 (2024). *Full HTML read (first 100k chars) 2026-10-24.* | Belief-space stochastic shortest path solved with LAO\*. **Information gathering emerges from goal-directed planning** (Look/Inspect controllers are chosen because they improve goal achievement). Risk = irreversible probabilistic outcomes, no CVaR or threshold. Pose beliefs come from Bayes3D / SMC. Sim cube-stacking with Gaussian pose noise: return 0.66–0.70. | **Conceptual precedent for task-driven observation: we must not claim the principle as new.** It operates on abstract task-level beliefs; it does not select camera viewpoints by expected grasp loss over a continuous multimodal 6D belief, and reports no viewpoint-level comparison against entropy or covariance objectives. |
| Multi-view 6D pose estimation and camera motion planning (ICCV 2017 workshop, labicvl). *abstract-level* | Hypothesis accumulation plus entropy-based NBV with movement cost. | It is an early precedent for "information gain + action cost". |
| Active Perception and Representation for Robotic Manipulation. arXiv:2003.06734. *abstract-level* | Learned active camera for grasping; reports gains over a fixed camera. | It is a precedent showing active > fixed for grasping. |

Evaluation references (*from memory*): BOP benchmark (Hodaň et al., ECCV 2018) for VSD/MSSD/MSPD
symmetry-aware pose metrics; HOTA (Luiten et al., IJCV 2021) for tracking.

## Novelty assessment (honest, preliminary)

- These are **not novel** on their own: multimodal 6D pose beliefs (PoseRBPF, SE(3)-PoseFlow),
  entropy/information-gain NBV (ActivePose, the ICCV-W 2017 work), uncertainty-aware grasping from
  pose samples (SE(3)-PoseFlow), and risk-aware observe-before-grasp planning (arXiv:2403.10454).
- **Updated after full reads (2026-10-24).** The decision-theoretic principle (observe because it
  lowers expected task loss) is established: POTAMP, and classical belief-space planning. The
  candidate contribution is therefore the *viewpoint-level instantiation and its controlled test*:
  expected-grasp-loss view selection over a calibrated, temporal, multimodal SE(3) belief,
  compared at equal budgets against the two objectives the closest 6D works use (rotational
  covariance trace, SE(3)-PoseFlow; hypothesis entropy, ActivePose), plus fixed and random.
- **Narrowest candidate that may be defensible (original wording):** a controlled study of whether choosing views by
  *expected task (grasp) failure under a calibrated multimodal belief* beats *information gain*
  when both run under **equal camera, compute and grasp budgets** and paired seeds, together with
  an explicit grasp / observe / abstain rule. It would also test whether **calibration** of the
  belief is what drives any difference (ablation A2). This is closer to an empirical contribution
  than an algorithmic one.
- **Risk resolved (2026-10-24):** SE(3)-PoseFlow's view objective is rotational-covariance
  reduction, not grasp quality, and it has no temporal Bayesian belief. The narrow claim stands,
  provided the remaining checks (ActivePose, arXiv:2403.10454, a value-of-information search)
  don't overturn it.
- A negative result ("task-aware ≈ information gain once beliefs are calibrated") is acceptable
  and must be reported as such.

## To do before M5

1. ~~SE(3)-PoseFlow~~ (done). Read ActivePose and arXiv:2403.10454 in full; record each one's objective.
2. Search for task-driven NBV / value-of-information in grasping (e.g. "value of information
   grasp", "decision-theoretic active perception manipulation").
3. Verify the FoundationPose code and weights licenses from the official repository.
