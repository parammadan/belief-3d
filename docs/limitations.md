# Limitations (living document)

- **Simulation only.** No result here is hardware-validated. MuJoCo's rasterised RGB has no
  rolling shutter, motion blur, multipath, flying pixels or IR interference. Depth is currently
  perfect up to GL precision. The noise models (M1/M2) will be parametric guesses, not fitted to
  a real sensor.
- **Depth precision on macOS CGL.** MuJoCo warns `ARB_clip_control unavailable`, so the depth
  buffer has reduced precision. Measured error at about 1 m is under 0.3 mm (max), but far-range
  values are unreliable: one mobile-camera frame showed 120 m with zfar = 50 m. Returns beyond
  5 m are therefore marked as no return.
- **Primitive objects only (M0).** Box, cylinder and sphere. Meshes arrive later; spheres and
  cylinders make rotation partly unobservable by construction.
- **Mac resources.** 8 GB RAM and ~2.4 GB free disk bound dataset size on the Mac. Large
  generation and training must run on the cluster.
- **Camera B motion is abstract.** It is assumed to ride a gantry or second arm with bounded
  speeds. Clearance is checked with straight-line rays, not swept volumes.
- **The occluder hovers and has no contact.** It produces exact, controllable visibility
  intervals, but no physical interaction.
- **Grasp physics is MuJoCo soft contact.** Lift success in simulation is not evidence of
  real-world grasp robustness.
- **Noise presets are parametric guesses.** `nominal`/`harsh` are not fitted to any sensor.
- **Classical perception relies on colour.** Detection assumes saturated, catalog-hued objects
  on an unsaturated support plane. It would not transfer to grey metal parts.
- **The support prior assumes resting objects.** The pose fit uses a soft "touching the
  table" prior. Disable it (`support_sigma=None`) for stacking or in-hand scenes.
- **Not real-time.** Classical perception takes ~260–350 ms per camera frame on the M1 CPU.
- **Calibration limits accuracy.** At about 11.6 mm per degree of extrinsic drift, calibration
  dominates pose error as soon as drift exceeds ~0.05°. Single-image re-calibration at 320×240
  cannot recover drift below ~1°.
- **All M2 numbers are simulation-only, over 10 test episodes per condition.** The CIs are
  bootstrap over episodes and are wide where only a few episodes drive the errors.
