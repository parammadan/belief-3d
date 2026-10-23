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
