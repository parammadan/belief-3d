# Literature review (M0, initial)

Status: **initial scan, 2026-10-23.** Items marked *abstract-level* were checked only on the
arXiv abstract page. Their method details are not yet verified and must not be cited for
specifics until the full paper has been read. Items marked *from memory* need a DOI check
before they go into the paper.

## Closest prior work: possible novelty conflicts

| Work | What it does | Overlap with RCABT |
|---|---|---|
| **SE(3)-PoseFlow**: Jin, Funk, Prasad, Li, Franzius, Peters, Chalvatzaki. arXiv:2511.01501 (Nov 2025). *abstract-level* | Flow matching on SE(3) to output pose *distributions*; active perception to disambiguate uncertain views; grasp synthesis from pose samples. | **High.** It already combines a multimodal pose distribution, active viewing and uncertainty-aware grasping. We must read the full paper to find out whether its view objective is uncertainty- or task-based. |
| **ActivePose**: Liu et al. arXiv:2509.11364 (Sep 2025, rev. Mar 2026). *abstract-level* | FoundationPose hypothesis entropy + VLM ambiguity score; NBV over rendered candidate views; diffusion-policy camera tracking. | High for "entropy/ambiguity-driven NBV for 6D pose". This is a natural information-gain-style comparator. |
| **TVVE**: Bai et al. "Learning to See and Act: Task-Aware Virtual View Exploration". arXiv:2508.05186 (CVPR 2026 per arXiv page). *abstract-level* | Learned, task-aware virtual viewpoint selection for VLA manipulation (RLBench). | Medium. It is task-aware but learned end-to-end, with no explicit pose belief stated in the abstract. |
| **PoseRBPF**: Deng, Mousavian, Xiang, Xia, Bretl, Fox. RSS 2019, arXiv:1905.09304. | Rao-Blackwellized particle filter: rotation as a discretised distribution, translation per particle; handles symmetry and occlusion. | It is the canonical multimodal 6D tracker. Our belief tracker is **not** a contribution over it; it is a baseline to reimplement or compare against. |
| **FoundationPose**: Wen, Yang, Kautz, Birchfield. CVPR 2024, arXiv:2312.08344. | Unified model-based/model-free 6D pose estimation and tracking. | Level-3 published baseline (§10). The license is **unverified**: a third-party weights mirror says CC-BY-NC-4.0. Check the official NVlabs LICENSE before use. |
| POTAMP with uncertainty and risk awareness. arXiv:2403.10454 (2024). *abstract-level, authors not yet recorded* | Risk-aware partially observable task and motion planning; includes reducing pose uncertainty before grasping. | Medium-high for the "risk-aware observe-vs-act" framing. |
| Multi-view 6D pose estimation and camera motion planning (ICCV 2017 workshop, labicvl). *abstract-level* | Hypothesis accumulation plus entropy-based NBV with movement cost. | It is an early precedent for "information gain + action cost". |
| Active Perception and Representation for Robotic Manipulation. arXiv:2003.06734. *abstract-level* | Learned active camera for grasping; reports gains over a fixed camera. | It is a precedent showing active > fixed for grasping. |

Evaluation references (*from memory*): BOP benchmark (Hodaň et al., ECCV 2018) for VSD/MSSD/MSPD
symmetry-aware pose metrics; HOTA (Luiten et al., IJCV 2021) for tracking.

## Novelty assessment (honest, preliminary)

- These are **not novel** on their own: multimodal 6D pose beliefs (PoseRBPF, SE(3)-PoseFlow),
  entropy/information-gain NBV (ActivePose, the ICCV-W 2017 work), uncertainty-aware grasping from
  pose samples (SE(3)-PoseFlow), and risk-aware observe-before-grasp planning (arXiv:2403.10454).
- **Narrowest candidate that may be defensible:** a controlled study of whether choosing views by
  *expected task (grasp) failure under a calibrated multimodal belief* beats *information gain*
  when both run under **equal camera, compute and grasp budgets** and paired seeds, together with
  an explicit grasp / observe / abstain rule. It would also test whether **calibration** of the
  belief is what drives any difference (ablation A2). This is closer to an empirical contribution
  than an algorithmic one.
- **Main risk:** if SE(3)-PoseFlow's view objective already scores views by downstream grasp
  quality, even this narrow claim weakens. Reading that paper in full is the first M5-blocking task.
- A negative result ("task-aware ≈ information gain once beliefs are calibrated") is acceptable
  and must be reported as such.

## To do before M5

1. Read SE(3)-PoseFlow, ActivePose and arXiv:2403.10454 in full; record each one's objective.
2. Search for task-driven NBV / value-of-information in grasping (e.g. "value of information
   grasp", "decision-theoretic active perception manipulation").
3. Verify the FoundationPose code and weights licenses from the official repository.
