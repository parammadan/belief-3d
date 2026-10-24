"""Execute a planned top-down grasp on an injected robot and verify it from proprioception.

The robot is passed in (duck-typed: the Panda interface from sim.robot.panda), so this
module never imports the simulator. Inputs are the grasp from perception (policies.viewpoint
plan_grasp: centre p, closing axis, planned width) and the robot's own kinematics.

Sequence: HOME -> pre-grasp (+approach) -> descend -> close -> lift. Verification uses only
the finger width the robot reports: a held object stops the fingers near the planned width,
and a miss closes them (width ~0). This is the signal recovery acts on. The evaluator
separately checks lift with simulator state.

Convention (measured in M1, tests/simulation/test_robot.py): fingers close perpendicular to
the TCP yaw passed to Panda.top_down_grasp, so closing axis a gives yaw = atan2(a_y, a_x) - pi/2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecConfig:
    approach: float = 0.15          # m above the grasp point
    lift: float = 0.15              # m
    move_s: float = 1.0             # per joint-space segment
    close_steps: int = 400
    settle_steps: int = 200
    min_hold_width: float = 0.004   # m; below this the fingers closed on nothing
    width_tol: float = 0.015        # m; |measured - planned| above this => wrong object / bad grasp


@dataclass(frozen=True)
class GraspResult:
    executed: bool                  # IK succeeded and the motion ran
    verified: bool                  # proprioceptive success signal
    finger_width: float
    reason: str


def yaw_for_axis(axis: np.ndarray) -> float:
    return float(np.arctan2(axis[1], axis[0]) - np.pi / 2)


def execute_grasp(robot, step_fn, grasp: dict, cfg: ExecConfig = ExecConfig(), home_q=None,  # noqa: B008
                  open_ctrl: float = 255.0, closed_ctrl: float = 0.0) -> GraspResult:
    """robot: Panda-like (top_down_grasp, move_to, command, gripper_width, q). step_fn(): advance physics once."""
    pre, g, _ = robot.top_down_grasp(np.asarray(grasp["p"], float), yaw_for_axis(np.asarray(grasp["axis"])),
                                     approach=cfg.approach)
    if not (pre.success and g.success):
        return GraspResult(False, False, robot.gripper_width(), "ik_failed")
    lift_target = np.asarray(grasp["p"], float).copy()
    lift_target[2] += cfg.lift
    robot.command(gripper=open_ctrl)
    if home_q is not None:
        robot.move_to(np.asarray(home_q), cfg.move_s, step_fn)
    robot.move_to(pre.q, cfg.move_s, step_fn)
    robot.move_to(g.q, cfg.move_s, step_fn)
    robot.command(gripper=closed_ctrl)
    for _ in range(cfg.close_steps):
        step_fn()
    robot.move_to(pre.q, cfg.move_s, step_fn)
    for _ in range(cfg.settle_steps):
        step_fn()
    w = float(robot.gripper_width())
    if w < cfg.min_hold_width:
        return GraspResult(True, False, w, "closed_on_nothing")
    if abs(w - float(grasp["width"])) > cfg.width_tol:
        return GraspResult(True, False, w, "unexpected_width")
    return GraspResult(True, True, w, "held")


def release(robot, step_fn, open_ctrl: float = 255.0, steps: int = 300) -> None:
    robot.command(gripper=open_ctrl)
    for _ in range(steps):
        step_fn()
