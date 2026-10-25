"""M5 closed-loop active-viewpoint episodes (simulation in the loop).

Per (seed, policy) the episode is replayed from the same seed, so the initial
observations are bit-identical across policies (paired design):

  t0   capture both cameras -> perceive -> B2 tracker step
  tgt  target = confirmed track with the highest grasp-now risk (belief-only rule, same for all policies)
  k=1..B  policy picks a viewpoint for camera B -> sim moves it -> step physics -> capture -> perceive -> update
  end  grasp planned from the target belief's MAP mode. The EVALUATOR scores it (a) with the
       shared geometric failure model ("proxy_success") and, with --physical, (b) by executing it on
       the simulated Panda ("success" = target-class object lifted > 8 cm). The proxy's 1 cm centre
       tolerance floored success under harsh calibration on val, so physics is the primary score.

The policy only receives the PolicyContext (belief, estimated occluders, plane, candidate
views, travel times). Ground truth is read only in `evaluate_grasp`.

Usage:  MUJOCO_GL=cgl PYTHONPATH=. python scripts/run_m5.py --split val --offset 600 --episodes 4 \\
            --policies fixed,random,entropy,covariance,task_risk --budget 2 --out reports/m5/dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from data.schemas.objects import CATALOG
from data.schemas.splits import seeds_for
from perception.pipeline import perceive
from policies.predict import SignatureRenderer, modes_of
from policies.viewpoint import POLICIES, PolicyContext, TaskRiskPolicy, grasp_fails, plan_grasp
from sim.cameras.noise import PRESETS, CalibrationError, corrupt
from sim.cameras.rgbd import RGBDSensor
from sim.cameras.viewpoints import ViewpointSet
from sim.environments.scenario import PRESET_SCENARIOS, Scenario
from sim.environments.tabletop import PANDA_XML, TabletopScene
from tracking.belief.particle import BeliefConfig, DepthView
from tracking.belief_tracker import BeliefTracker
from tracking.tracker import MultiObjectTracker, TrackerConfig

SPEC = {o.name: o for o in CATALOG}


class Episode:
    """Simulator side of one episode (owns ground truth; never handed to the policy)."""

    def __init__(self, seed: int, scenario: str, noise: str, cameras: tuple[str, ...] = ("overhead", "mobile"),
                 elevations: tuple[float, ...] = (35.0, 60.0), tray: bool = False):
        ss = np.random.SeedSequence(seed).spawn(5)
        self.layout_seed = int(ss[0].generate_state(1)[0])
        self.view_rng, self.calib_rng, self.noise_rng, scen_rng = (np.random.default_rng(s) for s in ss[1:])
        self.scenario = Scenario.sample(PRESET_SCENARIOS[scenario], scen_rng, n_frames=6)
        self.scene = TabletopScene(self.scenario.objects, with_robot=PANDA_XML.exists(), tray=tray)
        self.scenario.apply_appearance(self.scene)
        self.scene.reset(seed=self.layout_seed)
        self.views = ViewpointSet(self.scene, elevations_deg=elevations)
        self.noise = PRESETS[noise]
        self.sensors = {c: RGBDSensor(self.scene, c, compute_visibility=False) for c in cameras}
        self.calib = {c: CalibrationError.sample(self.noise, self.calib_rng) for c in self.sensors}
        self.view = int(self.view_rng.integers(len(self.views)))
        self.views.move_camera(self.view)
        self.k = 0

    def observe(self):
        self.scenario.before_frame(self.scene, self.k)
        if self.k > 0:
            for _ in range(round(0.1 / self.scene.model.opt.timestep)):
                mujoco.mj_step(self.scene.model, self.scene.data)
        out = {}
        for c, s in self.sensors.items():
            clean, gt = s.capture()
            out[c] = (corrupt(clean, self.noise, self.calib[c], self.noise_rng), gt)
        self.k += 1
        return out

    def move(self, v: int) -> float:
        t = self.views.travel_time(self.view, v)
        self.views.move_camera(v)
        self.view = v
        return t

    def close(self):
        self.scene.close()


def step_tracker(tracker, frames, k, shadow=None, last=None):
    """B2 step. `shadow` (B1 Kalman) gets the same detections; `last` collects this frame's detections (B0)."""
    dets, views = [], {}
    t = None
    for c, (obs, _gt) in frames.items():
        res = perceive(obs, seed=k)
        dets += res.detections
        views[c] = DepthView(obs.depth, obs.intrinsics, obs.T_WC)
        t = obs.timestamp
    if shadow is not None:
        shadow.step(t, dets)
    if last is not None:
        last[:] = dets
    return tracker.step(t, dets, views), next(iter(frames.values()))[0].intrinsics


def evaluate_grasp(ep: Episode, spec, g, gt_frame) -> dict:
    """EVALUATOR: true pose of the object nearest the grasp target (same class)."""
    best = None
    for ob in gt_frame.objects:
        if SPEC[ob.class_name].color_class != spec.color_class:
            continue
        d = np.linalg.norm(ob.T_WO[:3, 3] - g["p"])
        if best is None or d < best[0]:
            best = (d, ob)
    if best is None:
        return {"success": False, "reason": "no such object"}
    return {"success": not grasp_fails(SPEC[best[1].class_name], g, best[1].T_WO), "centre_err_m": float(best[0]),
            "true_class": best[1].class_name}


def physical_grasp(ep: Episode, g: dict, target_color: str) -> dict:
    """Execute the planned grasp on the simulated Panda; the EVALUATOR checks which object rose."""
    from manipulation.grasping.executor import execute_grasp
    from sim.robot.panda import HOME_Q
    d = ep.scene.data
    z0 = [float(d.xpos[b][2]) for b in ep.scene.body_ids]
    res = execute_grasp(ep.scene.robot, lambda: mujoco.mj_step(ep.scene.model, d), g, home_q=HOME_Q)
    rose = [ep.scene.objects[i].name for i, b in enumerate(ep.scene.body_ids) if d.xpos[b][2] - z0[i] > 0.08]
    ok = any(SPEC[n].color_class == target_color for n in rose)
    return {"success": bool(ok), "lifted": rose, "exec_reason": res.reason, "verified": res.verified,
            "finger_width": res.finger_width}


def feed_selfcal(calibs: dict, tracker, frames) -> None:
    """Add this frame's matched track centres, with each camera's reported pose, to its calibrator."""
    for tid, cam, p in tracker.last_matches:
        calibs[cam].add(tid, frames[cam][0].T_WC, p)


def run_episode(seed: int, policy_name: str, budget: int, scenario: str, noise: str, rng_seed: int,
                cameras: tuple[str, ...] = ("overhead", "mobile"), warmup: int = 1,
                elevations: tuple[float, ...] = (35.0, 60.0), physical: bool = False, selfcal: bool = False,
                grasp_source: str = "belief") -> dict:
    """grasp_source: "belief" (B2 MAP mode), "kf" (B1: Kalman track fed the same detections) or "frame"
    (B0: the latest frame's best fit). Views and target are chosen from the B2 belief in every case, so
    only the pose the grasp is planned from differs (paired, same observations)."""
    ep = Episode(seed, scenario, noise, cameras, elevations)
    tcfg = TrackerConfig(calib_rot_sigma=np.radians(1.0), calib_trans_sigma=0.005)
    tracker = BeliefTracker(tcfg, BeliefConfig(), seed=seed)
    shadow, last = MultiObjectTracker(tcfg), []
    t0 = time.perf_counter()
    history: list[int] = []              # views observed this episode (calibration-aware policy input)
    from perception.calibration.self_calib import SelfCalibrator
    calibs = {c: SelfCalibrator() for c in cameras}
    frames = ep.observe()
    tracks, K = step_tracker(tracker, frames, 0, shadow, last)
    feed_selfcal(calibs, tracker, frames)
    history.append(ep.view)
    # Warm-up: one more frame from the SAME starting view for every policy, so single-camera
    # runs can confirm tracks (confirmation needs 2 hits). Identical across policies; not part
    # of the movement budget. (Without it, all mobile-only episodes were skipped on val.)
    for w in range(warmup):
        frames = ep.observe()
        tracks, _ = step_tracker(tracker, frames, 100 + w, shadow, last)
        feed_selfcal(calibs, tracker, frames)
    renderer = SignatureRenderer(K)
    risk = TaskRiskPolicy()
    plane = next((tr.belief.plane for tr in tracks if tr.belief.plane is not None), None)

    def ctx_for(target):
        occl = [(SPEC[o.spec_name], o.T_WO) for o in tracker.tracks if o is not target]
        return PolicyContext(SPEC[target.spec_name], target.belief, occl, plane,
                             [v.T_WC for v in ep.views.views], ep.view, ep.views.travel_time, renderer,
                             view_history=list(history), calib_rot_sigma=np.radians(1.0), calib_trans_sigma=0.005)

    if not tracks:
        ep.close()
        return {"seed": seed, "policy": policy_name, "skipped": "no confirmed tracks"}
    # Target: highest grasp-now risk (belief-only; identical across policies by construction).
    target = max(tracks, key=lambda tr: (risk.grasp_now_loss(ctx_for(tr)), -tr.track_id))
    tid = target.track_id
    init_modes = len(modes_of(target.belief).weights)
    init_risk = risk.grasp_now_loss(ctx_for(target))
    policy = POLICIES[policy_name]()
    prng = np.random.default_rng(rng_seed)
    travel, chosen, decide_s = 0.0, [], []
    for k in range(1, budget + 1):
        tgt = next((tr for tr in tracker.tracks if tr.track_id == tid), None)
        if tgt is None:
            break
        td = time.perf_counter()
        v, _ = policy.choose(ctx_for(tgt), prng)
        decide_s.append(time.perf_counter() - td)
        travel += ep.move(v)
        chosen.append(v)
        history.append(v)
        frames = ep.observe()
        step_tracker(tracker, frames, k, shadow, last)
        feed_selfcal(calibs, tracker, frames)
    tgt = next((tr for tr in tracker.tracks if tr.track_id == tid), None)
    gt_frame = next(iter(frames.values()))[1]
    if tgt is None:
        result = {"success": False, "reason": "target track lost"}
        final_modes, final_risk = 0, None
    else:
        modes = modes_of(tgt.belief)
        i = int(np.argmax(modes.weights))
        T_grasp, spec_g = modes.poses[i], SPEC[tgt.spec_name]
        if grasp_source != "belief":
            # Nearest same-colour estimate to the B2 target's MAP position (estimates only, no ground truth).
            pool = ([(o.T_WO, o.spec_name) for o in shadow.confirmed() if o.color_class == tgt.color_class]
                    if grasp_source == "kf" else
                    [(d.T_WO, d.spec_name) for d in last if d.color_class == tgt.color_class])
            if pool:
                T_grasp, name = min(pool, key=lambda e: np.linalg.norm(e[0][:3, 3] - T_grasp[:3, 3]))
                spec_g = SPEC[name]
            else:
                T_grasp = None
        if T_grasp is None:
            g = None
        else:
            g = plan_grasp(spec_g, T_grasp)
        selfcal_shift = None
        if selfcal and g is not None:
            # Correct the grasp centre by the estimated mean calibration bias over the views in which
            # the target was observed (per camera; summed over cameras weighted by observation count).
            p0 = g["p"].copy()
            shifts, counts = [], []
            for sc in calibs.values():
                views_t = [T for key, T, _ in sc.obs if key == tid]
                if views_t:
                    sc.solve()
                    shifts.append(p0 - sc.correct(p0, views_t))
                    counts.append(len(views_t))
            if shifts:
                bias = np.average(np.array(shifts), axis=0, weights=counts)
                g["p"] = p0 - bias
                selfcal_shift = bias.tolist()
        if g is None:
            result = {"success": False, "proxy_success": False, "reason": f"no {grasp_source} estimate"}
        else:
            result = evaluate_grasp(ep, SPEC[tgt.spec_name], g, gt_frame)
            result["proxy_success"] = result.pop("success")
            if physical:
                result |= physical_grasp(ep, g, tgt.color_class)
            else:
                result["success"] = result["proxy_success"]
        final_modes, final_risk = len(modes.weights), risk.grasp_now_loss(ctx_for(tgt))
    ep.close()
    return {"seed": seed, "policy": policy_name, "budget": budget, "scenario": scenario, "noise": noise,
            "cameras": list(cameras), "warmup": warmup, "elevations": list(elevations), "physical": physical,
            "selfcal": selfcal, "grasp_source": grasp_source, "selfcal_shift": selfcal_shift if tgt is not None else None,
            "target_class": target.spec_name, "init_modes": init_modes, "init_risk": init_risk,
            "ambiguous_t0": bool(init_modes >= 2 or init_risk > 0.1), "views": chosen, "travel_s": travel,
            "final_modes": final_modes, "final_risk": final_risk, "decide_ms_mean": 1e3 * float(np.mean(decide_s)) if decide_s else None,
            "wall_s": time.perf_counter() - t0} | result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--offset", type=int, default=600)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--policies", default="fixed,random,entropy,covariance,task_risk")
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--scenario", default="occlusion")
    ap.add_argument("--noise", default="nominal")
    ap.add_argument("--cameras", default="overhead,mobile")
    ap.add_argument("--elevations", default="35,60", help="candidate-view elevations in degrees")
    ap.add_argument("--physical", action="store_true", help="score the final grasp by executing it in MuJoCo")
    ap.add_argument("--selfcal", action="store_true", help="correct the grasp centre with online self-calibration")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cams = tuple(a.cameras.split(","))
    elev = tuple(float(x) for x in a.elevations.split(","))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():                                   # resumable
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["seed"], r["policy"]))
    with out.open("a") as f:
        for seed in seeds_for(a.split, a.episodes, a.offset):
            for p in a.policies.split(","):
                if (seed, p) in done:
                    continue
                pol, _, src = p.partition("@")            # "fixed@b1" = fixed views, B1 grasp pose
                r = run_episode(seed, pol, a.budget, a.scenario, a.noise, rng_seed=seed + 17, cameras=cams,
                                elevations=elev, physical=a.physical, selfcal=a.selfcal,
                                grasp_source={"": "belief", "b1": "kf", "b0": "frame"}[src])
                r["policy"] = p
                if a.selfcal:
                    r["policy"] = p + "+selfcal"
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(json.dumps({k: r.get(k) for k in ("seed", "policy", "ambiguous_t0", "views", "success", "wall_s")}))


if __name__ == "__main__":
    main()
