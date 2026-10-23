# BELIEF-3D

**Belief-space 3D perception and risk-aware active vision for robust robotic manipulation**,
built on MuJoCo. Simulation only.

The research question: when object pose is ambiguous (occlusion, symmetry, bad depth,
calibration drift), does picking camera views by *expected grasp failure under a calibrated
multimodal pose belief* beat information-gain view selection at equal budgets? This is a
hypothesis. Prior work that overlaps with it is listed in
[`docs/literature_review.md`](docs/literature_review.md).

## Status

**M0 complete** (environment audit, geometry core, minimal MuJoCo RGB-D scene, validated camera
geometry). See [`PROJECT_STATE.md`](PROJECT_STATE.md) for exactly what is and isn't built.
Nothing beyond M0 is implemented yet.

| Overhead camera | Mobile camera |
|---|---|
| ![](reports/m0/overhead_rgb.png) | ![](reports/m0/mobile_rgb.png) |

Depth back-projected through the camera model lands within 0.3 mm of the true sphere surface
([EXP-M0-001](EXPERIMENT_REGISTRY.md)).

## Quickstart

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
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
