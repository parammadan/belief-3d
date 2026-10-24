"""M6 closed-loop pick-and-place with perception-derived targets and recovery.

Per (seed, policy), replayed from the same seed (paired):
  observe (warm-up) -> policy views (budget B) -> plan grasp from the target belief's MAP mode
  -> execute on the simulated Panda -> verify from proprioception (finger width)
  -> on failure: park the arm (out of the cameras' way), re-observe with the policy, re-plan, retry
     (max_attempts) -> on success: carry to the place pose and release.
Robot proprioception and kinematics are used for control; object state is read only by `judge`.

Usage:  MUJOCO_GL=cgl PYTHONPATH=. python scripts/run_m6.py --split val --offset 700 --episodes 3 \\
            --policies fixed,task_risk --cameras mobile --out reports/m6/dev.jsonl
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
from manipulation.grasping.executor import execute_grasp, release
from perception.geometry import se3
from policies.predict import SignatureRenderer, modes_of
from policies.viewpoint import POLICIES, PolicyContext, TaskRiskPolicy, plan_grasp
from scripts.run_m5 import Episode, step_tracker
from sim.environments.tabletop import TRAY_HALF, TRAY_XY
from sim.robot.panda import HOME_Q, OBSERVE_Q
from tracking.belief.particle import BeliefConfig
from tracking.belief_tracker import BeliefTracker
from tracking.tracker import TrackerConfig

SPEC = {o.name: o for o in CATALOG}
PLACE_XY = np.array(TRAY_XY)                # tray centre (world)
PLACE_RADIUS = TRAY_HALF + 0.01


def judge(ep: Episode, target_color: str, z0: dict[int, float]) -> dict:
    """EVALUATOR (simulator state): which object, if any, ended up placed."""
    d = ep.scene.data
    placed = []
    for i, bid in enumerate(ep.scene.body_ids):
        p = d.xpos[bid]
        lifted_once = z0.get(i, p[2])
        if np.linalg.norm(p[:2] - PLACE_XY) < PLACE_RADIUS:
            placed.append((i, ep.scene.objects[i].name, float(p[2] - lifted_once)))
    ok = [x for x in placed if SPEC[x[1]].color_class == target_color]
    return {"placed": [x[1] for x in placed], "success": bool(ok)}


def run(seed: int, policy_name: str, budget: int, scenario: str, noise: str, cameras: tuple[str, ...],
        max_attempts: int) -> dict:
    ep = Episode(seed, scenario, noise, cameras, tray=True)
    robot = ep.scene.robot

    def step():
        mujoco.mj_step(ep.scene.model, ep.scene.data)

    tracker = BeliefTracker(TrackerConfig(calib_rot_sigma=np.radians(1.0), calib_trans_sigma=0.005),
                            BeliefConfig(), seed=seed)
    t0 = time.perf_counter()
    frames = ep.observe()
    tracks, K = step_tracker(tracker, frames, 0)
    frames = ep.observe()
    tracks, _ = step_tracker(tracker, frames, 100)
    renderer = SignatureRenderer(K)
    plane = next((tr.belief.plane for tr in tracks if tr.belief.plane is not None), None)
    risk = TaskRiskPolicy()

    def ctx_for(target):
        occl = [(SPEC[o.spec_name], o.T_WO) for o in tracker.tracks if o is not target]
        return PolicyContext(SPEC[target.spec_name], target.belief, occl, plane,
                             [v.T_WC for v in ep.views.views], ep.view, ep.views.travel_time, renderer)

    if not tracks:
        ep.close()
        return {"seed": seed, "policy": policy_name, "skipped": "no confirmed tracks"}
    target = max(tracks, key=lambda tr: (risk.grasp_now_loss(ctx_for(tr)), -tr.track_id))
    tid, tcolor = target.track_id, target.color_class
    policy = POLICIES[policy_name]()
    prng = np.random.default_rng(seed + 17)
    z0 = {i: float(ep.scene.data.xpos[b][2]) for i, b in enumerate(ep.scene.body_ids)}
    travel, attempts, log = 0.0, 0, []
    k = 1
    held = False
    while attempts < max_attempts and not held:
        for _ in range(budget):
            tgt = next((tr for tr in tracker.tracks if tr.track_id == tid), None)
            if tgt is None:
                break
            v, _ = policy.choose(ctx_for(tgt), prng)
            travel += ep.move(v)
            frames = ep.observe()
            step_tracker(tracker, frames, k)
            k += 1
        tgt = next((tr for tr in tracker.tracks if tr.track_id == tid), None)
        if tgt is None:
            log.append("target lost")
            break
        modes = modes_of(tgt.belief)
        g = plan_grasp(SPEC[tgt.spec_name], modes.poses[int(np.argmax(modes.weights))])
        attempts += 1
        res = execute_grasp(robot, step, g, home_q=HOME_Q)
        log.append(res.reason)
        held = res.verified
        if not held:
            release(robot, step)
            robot.move_to(OBSERVE_Q, 1.5, step)            # park out of view, then re-observe
            for _ in range(200):
                step()
    if held:
        T_place = se3.make_T(np.diag([1.0, -1.0, -1.0]), [PLACE_XY[0], PLACE_XY[1], 0.40 + 0.07])
        r = robot.ik_any([T_place])[1]
        if r.success:
            robot.move_to(r.q, 1.5, step)
        release(robot, step)
        # Retreat straight up before parking: sweeping from the place pose to OBSERVE_Q knocked the
        # just-placed object out of the region (measured on val seed 1,500,700).
        T_up = T_place.copy()
        T_up[2, 3] += 0.15
        up = robot.ik(T_up, q0=r.q)
        if up.success:
            robot.move_to(up.q, 1.0, step)
        robot.move_to(OBSERVE_Q, 1.5, step)
        for _ in range(300):
            step()
    verdict = judge(ep, tcolor, z0)
    ep.close()
    return {"seed": seed, "policy": policy_name, "cameras": list(cameras), "budget": budget, "target": target.spec_name,
            "attempts": attempts, "log": log, "travel_s": travel, "held": held, "wall_s": time.perf_counter() - t0} | verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--offset", type=int, default=700)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--policies", default="fixed,task_risk")
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--scenario", default="occlusion")
    ap.add_argument("--noise", default="nominal")
    ap.add_argument("--cameras", default="mobile")
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(x)["seed"], json.loads(x)["policy"]) for x in out.read_text().splitlines()} if out.exists() else set()
    with out.open("a") as f:
        for seed in seeds_for(a.split, a.episodes, a.offset):
            for p in a.policies.split(","):
                if (seed, p) in done:
                    continue
                r = run(seed, p, a.budget, a.scenario, a.noise, tuple(a.cameras.split(",")), a.max_attempts)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(json.dumps({x: r.get(x) for x in ("seed", "policy", "attempts", "log", "success", "placed", "wall_s")}))


if __name__ == "__main__":
    main()
