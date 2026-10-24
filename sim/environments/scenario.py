"""Seeded scenario sampling: which objects, appearance, and scheduled events.

A Scenario is fully determined by (ScenarioConfig, seed). Events are keyed by
frame index so episodes replay exactly:

* occlusion: from frame `occl_start` for `occl_len` frames the hovering plate
  sits on the ray from the overhead camera to one target object, `occl_height`
  above it. Overhead this is a complete occlusion; the mobile camera may still
  see the object from the side.
* kick: at frame `kick_frame` one object gets a planar velocity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from data.schemas.objects import CATALOG, DEFAULT_OBJECTS, ObjectSpec
from sim.environments.tabletop import TabletopScene


@dataclass(frozen=True)
class ScenarioConfig:
    n_objects: tuple[int, int] = (4, 4)        # inclusive range, sampled per episode
    use_catalog: bool = False                  # False: DEFAULT_OBJECTS in order (M0-compatible)
    force_similar_pair: bool = False           # always include box_red + box_red_b
    light_jitter: float = 0.0                  # relative diffuse/ambient jitter
    color_jitter: float = 0.0                  # relative brightness jitter on object/table rgba
    occlusion_prob: float = 0.0
    occl_len: tuple[int, int] = (3, 6)         # frames
    occl_height: float = 0.12                  # m above the target object's centre
    kick_prob: float = 0.0
    kick_speed: tuple[float, float] = (0.2, 0.5)  # m/s
    min_sep: float = 0.11                      # m between spawned centres; < ~0.09 creates contact clutter
    # Mugs were added to the catalog after EXP-M2/M3/M4. They are excluded from the sampling
    # pool unless asked for, so every earlier experiment's scenes replay bit-identically.
    n_mugs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


PRESET_SCENARIOS: dict[str, ScenarioConfig] = {
    "static": ScenarioConfig(),
    "varied": ScenarioConfig(n_objects=(3, 6), use_catalog=True, light_jitter=0.25, color_jitter=0.12),
    "occlusion": ScenarioConfig(n_objects=(3, 6), use_catalog=True, light_jitter=0.25, color_jitter=0.12,
                                occlusion_prob=1.0),
    "mugs": ScenarioConfig(n_objects=(4, 6), use_catalog=True, light_jitter=0.25, color_jitter=0.12,
                           min_sep=0.09, n_mugs=1),
    "clutter": ScenarioConfig(n_objects=(5, 7), use_catalog=True, light_jitter=0.25, color_jitter=0.12,
                              min_sep=0.075, force_similar_pair=True),
    "dynamic": ScenarioConfig(n_objects=(3, 6), use_catalog=True, light_jitter=0.25, color_jitter=0.12,
                              occlusion_prob=0.5, kick_prob=1.0, force_similar_pair=True),
}


@dataclass(frozen=True)
class Scenario:
    objects: tuple[ObjectSpec, ...]
    light_scale: float
    color_scale: tuple[float, ...]             # per object, then table last
    occl_target: int | None
    occl_frames: tuple[int, int] | None        # [start, end)
    kick_target: int | None
    kick_frame: int | None
    kick_v: tuple[float, float] | None
    occl_height: float = 0.12
    min_sep: float = 0.11

    @staticmethod
    def sample(cfg: ScenarioConfig, rng: np.random.Generator, n_frames: int) -> Scenario:
        if cfg.use_catalog:
            n = int(rng.integers(cfg.n_objects[0], cfg.n_objects[1] + 1))
            names = [o.name for o in CATALOG]
            pool = [i for i, o in enumerate(CATALOG) if o.shape != "mug"]
            chosen: list[int] = [i for i, o in enumerate(CATALOG) if o.shape == "mug"][: cfg.n_mugs]
            if cfg.force_similar_pair:
                chosen += [names.index("box_red"), names.index("box_red_b")]
                pool = [i for i in pool if i not in chosen]
            chosen += list(rng.choice(pool, size=max(0, n - len(chosen)), replace=False))
            objects = tuple(CATALOG[i] for i in sorted(chosen))
        else:
            objects = DEFAULT_OBJECTS
        n = len(objects)
        light = 1.0 + (rng.uniform(-cfg.light_jitter, cfg.light_jitter) if cfg.light_jitter else 0.0)
        colors = tuple(1.0 + (rng.uniform(-cfg.color_jitter, cfg.color_jitter) if cfg.color_jitter else 0.0)
                       for _ in range(n + 1))
        occl_t = occl = None
        if n_frames > 2 and rng.random() < cfg.occlusion_prob:
            occl_t = int(rng.integers(n))
            length = int(rng.integers(cfg.occl_len[0], cfg.occl_len[1] + 1))
            start = int(rng.integers(1, max(2, n_frames - length)))
            occl = (start, min(n_frames, start + length))
        kick_t = kick_f = kick_v = None
        if n_frames > 1 and rng.random() < cfg.kick_prob:
            kick_t = int(rng.integers(n))
            kick_f = int(rng.integers(1, n_frames))
            ang, spd = rng.uniform(-np.pi, np.pi), rng.uniform(*cfg.kick_speed)
            kick_v = (float(spd * np.cos(ang)), float(spd * np.sin(ang)))
        return Scenario(objects, float(light), colors, occl_t, occl, kick_t, kick_f, kick_v, cfg.occl_height, cfg.min_sep)

    def apply_appearance(self, scene: TabletopScene) -> None:
        scene.min_sep = self.min_sep
        m = scene.model
        m.light_diffuse[:] = np.clip(m.light_diffuse * self.light_scale, 0, 1)
        m.vis.headlight.ambient[:] = np.clip(np.asarray(m.vis.headlight.ambient) * self.light_scale, 0, 1)
        for inst, bid in enumerate(scene.body_ids):
            g = m.geom_bodyid == bid
            m.geom_rgba[g, :3] = np.clip(m.geom_rgba[g, :3] * self.color_scale[inst], 0, 1)
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        m.geom_rgba[tid, :3] = np.clip(m.geom_rgba[tid, :3] * self.color_scale[-1], 0, 1)

    def before_frame(self, scene: TabletopScene, k: int) -> None:
        """Apply events scheduled for frame k (call before stepping into frame k)."""
        if self.kick_frame == k and self.kick_target is not None:
            scene.kick(self.kick_target, np.array(self.kick_v))
        if self.occl_frames is not None and self.occl_target is not None:
            if self.occl_frames[0] <= k < self.occl_frames[1]:
                c = scene.data.xpos[scene.body_ids[self.occl_target]].copy()
                cam = scene.data.cam_xpos[mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")]
                alpha = self.occl_height / (cam[2] - c[2])
                scene.set_occluder(c + alpha * (cam - c))
            else:
                scene.set_occluder(None)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["objects"] = [o.name for o in self.objects]
        return d
