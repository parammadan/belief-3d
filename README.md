# BELIEF-3D

**Belief-space 3D perception and risk-aware active vision for robust robotic manipulation**,
built on MuJoCo. Simulation only.

The research question: when object pose is ambiguous (occlusion, symmetry, bad depth,
calibration drift), does picking camera views by *expected grasp failure under a calibrated
multimodal pose belief* beat information-gain view selection at equal budgets? This is a
hypothesis. Prior work that overlaps with it is listed in
[`docs/literature_review.md`](docs/literature_review.md).

## Status

**M0–M2 complete** (simulation only):
- **M0:** environment audit, geometry core, validated camera geometry.
- **M1:** Franka Panda with IK and grasp physics, movable-camera viewpoints, seeded noise,
  occlusion/motion scenarios, versioned datasets with a validator.
- **M2:** classical perception baseline (plane + seed-and-grow segmentation, SDF-based 6D
  fitting of known primitives), unimodal Kalman tracking, calibration checks, and an
  independent evaluator.

Neural pose (M3), belief tracking (M4) and active policies (M5) are not implemented yet. See
[`PROJECT_STATE.md`](PROJECT_STATE.md).

**M2 headline (held-out test seeds, nominal noise; [full tables](reports/tables/m2_results.md)):**
- Median pose error is **0.25–0.60 mm** in translation and **0.19–0.83°** in rotation
  (symmetry-aware).
- Tracking MOTA is **0.85–0.88** with the overhead camera only, and **0.98–1.00** when the
  mobile camera is added.

Two caveats:
- With ~1° calibration drift, error grows to **16–22 mm**, and adding the second camera then
  *hurts* tracking. The cause is that calibration uncertainty isn't modelled yet. This is a
  recorded negative result ([EXP-M2-001](EXPERIMENT_REGISTRY.md)).
- Localisation degrades by about **11.6 mm per degree** of extrinsic drift
  ([EXP-M2-002](EXPERIMENT_REGISTRY.md)).

Controlled occlusion (EXP-M1-002). The top row is overhead, where the plate fully hides the
blue sphere. The bottom row is the mobile camera, which can still see it:

![](reports/m1/occlusion_strip.png)

| Overhead camera | Mobile camera |
|---|---|
| ![](reports/m0/overhead_rgb.png) | ![](reports/m0/mobile_rgb.png) |

Depth back-projected through the camera model lands within 0.3 mm of the true sphere surface
([EXP-M0-001](EXPERIMENT_REGISTRY.md)).

## Quickstart

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python scripts/fetch_assets.py      # Franka Panda from MuJoCo Menagerie (pinned)
export MUJOCO_GL=cgl        # macOS; use egl or osmesa on Linux
export PYTHONPATH=.
.venv/bin/python -m pytest -q
.venv/bin/python scripts/m0_demo.py --seed 0
```

## Docs

- [Coordinate frames](docs/coordinate_frames.md): `T_AB` notation, OpenCV camera frame, quaternion order
- [Architecture](docs/architecture.md): modules and the inference/ground-truth leakage boundary
- [Research question](docs/research_question.md): hypotheses H1–H4 and the data protocol
- [Limitations](docs/limitations.md)

## License

MIT
