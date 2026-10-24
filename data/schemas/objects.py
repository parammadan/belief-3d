"""Known object models: the CAD-level prior both the simulator and the
inference stack may use (master prompt: "known CAD meshes").

Knowing the *catalog* is allowed at inference. Knowing *which* catalog
objects are in a scene, or where they are, is not.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    shape: str               # "box" | "cylinder" | "sphere" | "mug"
    # MuJoCo geom size (half-extents / radius, half-height). Mug: (r, hh, handle_len, handle_hw, handle_hh):
    # cylinder body (r, hh) plus a handle box along local +x spanning [r, r + handle_len].
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float]
    symmetry: str
    color_class: str        # nominal hue class used by the classical detector


DEFAULT_OBJECTS: tuple[ObjectSpec, ...] = (
    ObjectSpec("box_red", "box", (0.030, 0.020, 0.025), (0.85, 0.15, 0.15, 1), "box", "red"),
    ObjectSpec("cyl_green", "cylinder", (0.025, 0.035), (0.15, 0.75, 0.25, 1), "cylindrical_z", "green"),
    ObjectSpec("sphere_blue", "sphere", (0.028,), (0.15, 0.30, 0.85, 1), "spherical", "blue"),
    ObjectSpec("box_yellow", "box", (0.040, 0.015, 0.015), (0.90, 0.80, 0.10, 1), "box", "yellow"),
)


# Catalog for scenario sampling. "_b" entries look identical to their twin
# (same shape, size, colour) to create similar-instance ambiguity.
CATALOG: tuple[ObjectSpec, ...] = DEFAULT_OBJECTS + (
    ObjectSpec("box_red_b", "box", (0.030, 0.020, 0.025), (0.85, 0.15, 0.15, 1), "box", "red"),
    ObjectSpec("cyl_green_b", "cylinder", (0.025, 0.035), (0.15, 0.75, 0.25, 1), "cylindrical_z", "green"),
    ObjectSpec("cyl_orange", "cylinder", (0.020, 0.045), (0.95, 0.50, 0.10, 1), "cylindrical_z", "orange"),
    # Mug: the handle makes yaw fully observable *only when the handle is visible*, and it decides
    # the safe grasp (fingers must close perpendicular to it): grasp-relevant ambiguity (M5).
    ObjectSpec("mug_cyan", "mug", (0.030, 0.040, 0.030, 0.006, 0.022), (0.10, 0.75, 0.80, 1), "none", "cyan"),
)
