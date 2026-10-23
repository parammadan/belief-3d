"""Franka Panda interface: proprioception, kinematics, IK and actuation.

Everything here uses the robot's own model and joint state, which the real
robot would also have. Nothing reads object state. IK runs on a scratch
MjData so solving never perturbs the simulation.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from perception.geometry import se3

ARM_JOINTS = tuple(f"panda/joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("panda/finger_joint1", "panda/finger_joint2")
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
# Arm parked to the -y side so it barely enters the overhead view (measured:
# 3.3k of 76.8k overhead pixels vs 36k at HOME_Q, which hides objects).
OBSERVE_Q = np.array([-1.57, -0.5, 0.0, -2.6, 0.0, 2.1, 0.785])
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0   # actuator8 ctrl range; maps to 0.04 m .. 0 m per finger
# TCP z-axis points out of the fingers. A top-down grasp therefore has
# R_WE[:, 2] = -z_W.
R_TOP_DOWN = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    success: bool
    pos_err: float      # meters
    rot_err: float      # radians
    iters: int


class Panda:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in jid])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in jid])
        self.finger_adr = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in FINGER_JOINTS]
        )
        self.q_lo, self.q_hi = model.jnt_range[jid, 0], model.jnt_range[jid, 1]
        self.act_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"panda/actuator{i}") for i in range(1, 9)]
        )
        self.tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "panda/tcp")
        self._scratch = mujoco.MjData(model)
        self._robot_bodies = {
            b for b in range(model.nbody) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b).startswith("panda/")
        }

    # ---------- proprioception ----------
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def gripper_width(self) -> float:
        return float(self.data.qpos[self.finger_adr].sum())

    def tcp_pose(self, data: mujoco.MjData | None = None) -> np.ndarray:
        d = data or self.data
        return se3.make_T(d.site_xmat[self.tcp_site].reshape(3, 3), d.site_xpos[self.tcp_site])

    def is_robot_geom(self, geom_ids: np.ndarray) -> np.ndarray:
        bodies = self.model.geom_bodyid[geom_ids]
        return np.isin(bodies, list(self._robot_bodies))

    # ---------- state setting (reset only) ----------
    def set_joint_state(self, q: np.ndarray) -> None:
        """Teleport to q with matching targets and an open gripper. Reset only."""
        q = np.clip(q, self.q_lo, self.q_hi)
        self.data.qpos[self.qpos_adr] = q
        self.data.qpos[self.finger_adr] = 0.04
        self.data.ctrl[self.act_ids[:7]] = q
        self.data.ctrl[self.act_ids[7]] = GRIPPER_OPEN

    # ---------- kinematics ----------
    def fk(self, q: np.ndarray) -> np.ndarray:
        d = self._scratch
        d.qpos[:] = self.data.qpos
        d.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return self.tcp_pose(d)

    def ik(
        self,
        T_target: np.ndarray,
        q0: np.ndarray | None = None,
        max_iters: int = 200,
        damping: float = 1e-2,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
    ) -> IKResult:
        """Damped least-squares IK on the TCP site, respecting joint limits."""
        q = (self.q() if q0 is None else np.asarray(q0, dtype=float)).copy()
        d = self._scratch
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        pe = re = np.inf
        for it in range(1, max_iters + 1):
            T = self.fk(q)
            e_p = T_target[:3, 3] - T[:3, 3]
            e_r = T[:3, :3] @ se3.so3_log(T[:3, :3].T @ T_target[:3, :3])   # world-frame rotation error
            pe, re = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
            if pe < pos_tol and re < rot_tol:
                return IKResult(q, True, pe, re, it)
            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.tcp_site)
            J = np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])
            e = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.q_lo, self.q_hi)
        return IKResult(q, False, pe, re, max_iters)

    def ik_any(self, targets: list[np.ndarray], seeds: list[np.ndarray] | None = None) -> tuple[int, IKResult]:
        """Solve for each (target, seed) pair; return the successful solution
        closest to HOME_Q in joint space (or the lowest-error failure)."""
        seeds = seeds or [HOME_Q, self.q()]
        best: tuple[float, int, IKResult] | None = None
        for k, T in enumerate(targets):
            for q0 in seeds:
                r = self.ik(T, q0=q0)
                score = float(np.linalg.norm(r.q - HOME_Q)) if r.success else 1e6 + r.pos_err + r.rot_err
                if best is None or score < best[0]:
                    best = (score, k, r)
        assert best is not None
        return best[1], best[2]

    def top_down_grasp(self, p_W: np.ndarray, yaw: float, approach: float = 0.15) -> tuple[IKResult, IKResult, float]:
        """IK for a top-down grasp at p_W with finger-closing yaw.

        A parallel-jaw grasp is invariant to yaw + pi, so both are tried.
        Returns (pre-grasp, grasp, chosen yaw). Callers check `.success`.
        """
        yaws = [yaw, yaw + np.pi]
        Ts = [se3.make_T(R_TOP_DOWN @ se3.so3_exp([0.0, 0.0, -y]), p_W) for y in yaws]
        pres = [T.copy() for T in Ts]
        for T in pres:
            T[2, 3] += approach
        k, pre = self.ik_any(pres)
        grasp = self.ik(Ts[k], q0=pre.q)
        return pre, grasp, yaws[k]

    # ---------- actuation ----------
    def command(self, q: np.ndarray | None = None, gripper: float | None = None) -> None:
        if q is not None:
            self.data.ctrl[self.act_ids[:7]] = np.clip(q, self.q_lo, self.q_hi)
        if gripper is not None:
            self.data.ctrl[self.act_ids[7]] = float(np.clip(gripper, GRIPPER_CLOSED, GRIPPER_OPEN))

    def move_to(self, q_goal: np.ndarray, duration: float, step_fn) -> None:
        """Linear joint-space interpolation of position targets over `duration` s."""
        q_start = self.data.ctrl[self.act_ids[:7]].copy()
        n = max(1, round(duration / self.model.opt.timestep))
        for k in range(1, n + 1):
            self.command(q=q_start + (q_goal - q_start) * k / n)
            step_fn()
