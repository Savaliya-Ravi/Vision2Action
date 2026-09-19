"""Convert a seven-value end-effector action to bounded G1 joint targets.

This module contains robot kinematics and actuator limits. It does not select a
target, prescribe a trajectory, or control the fridge joint. The VLA supplies
every end-effector and hand command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np


RIGHT_ARM = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

WAIST = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

RIGHT_HAND_CLOSED = {
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": -0.8,
    "right_hand_thumb_2_joint": -1.4,
    "right_hand_index_0_joint": 1.2,
    "right_hand_index_1_joint": 1.4,
    "right_hand_middle_0_joint": 1.2,
    "right_hand_middle_1_joint": 1.4,
}


@dataclass(frozen=True)
class AdapterLimits:
    max_translation_m: float = 0.025
    max_rotation_rad: float = 0.10
    max_joint_step_rad: float = 0.04
    damping: float = 0.03
    physics_steps: int = 5


def _bounded_vector(values: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values * min(1.0, limit / norm) if norm > 0 else values


class G1ActionAdapter:
    """Interpret Octo's 7D output as a local SE(3) delta and gripper scalar."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        limits: AdapterLimits = AdapterLimits(),
        *,
        control_waist: bool = False,
    ):
        self.model = model
        self.data = data
        self.limits = limits
        self.site_id = model.site("right_grasp_site").id
        self.body_id = model.body("robot").id
        self.free_joint = model.joint("robot_free").id
        self.control_joint_names = (WAIST + RIGHT_ARM) if control_waist else RIGHT_ARM
        self.arm_joint_ids = [model.joint(name).id for name in self.control_joint_names]
        self.arm_actuator_ids = [model.actuator(name).id for name in self.control_joint_names]
        self.arm_dof_ids = [int(model.jnt_dofadr[j]) for j in self.arm_joint_ids]
        self.hand_ids = [(model.joint(name).id, model.actuator(name).id, closed)
                         for name, closed in RIGHT_HAND_CLOSED.items()]
        self.base_pose = data.qpos[model.jnt_qposadr[self.free_joint]:model.jnt_qposadr[self.free_joint] + 7].copy()

    def pin_base(self) -> None:
        q = int(self.model.jnt_qposadr[self.free_joint])
        v = int(self.model.jnt_dofadr[self.free_joint])
        self.data.qpos[q:q + 7] = self.base_pose
        self.data.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def apply(self, action: np.ndarray, on_step: Callable[[], None] | None = None) -> None:
        """Run one receding-horizon action; reject malformed model output."""
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("VLA action must be seven finite values")

        translation = _bounded_vector(action[:3], self.limits.max_translation_m)
        rotation = _bounded_vector(action[3:6], self.limits.max_rotation_rad)
        # Spatial channels are robot-local deltas. V4 demonstrations and the
        # learned head are trained with this G1 embodiment mapping.
        base_rotation = self.data.xmat[self.body_id].reshape(3, 3)
        desired = np.concatenate((base_rotation @ translation, base_rotation @ rotation))
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.site_id)
        jacobian = np.vstack((jac_pos[:, self.arm_dof_ids], jac_rot[:, self.arm_dof_ids]))
        damping = self.limits.damping ** 2 * np.eye(6)
        joint_delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping, desired)
        joint_delta = np.clip(joint_delta, -self.limits.max_joint_step_rad, self.limits.max_joint_step_rad)

        for joint_id, actuator_id, delta in zip(self.arm_joint_ids, self.arm_actuator_ids, joint_delta):
            q = int(self.model.jnt_qposadr[joint_id])
            lo, hi = self.model.jnt_range[joint_id]
            self.data.ctrl[actuator_id] = np.clip(self.data.qpos[q] + delta, lo, hi)

        # Octo's gripper channel is 0=open, 1=closed; the G1 has seven finger actuators.
        close = bool(action[6] >= 0.5)
        for _, actuator_id, closed_value in self.hand_ids:
            self.data.ctrl[actuator_id] = closed_value if close else 0.0

        for _ in range(self.limits.physics_steps):
            mujoco.mj_step(self.model, self.data)
            self.pin_base()
            if on_step is not None:
                on_step()


def hold_current_joint_positions(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Initialize position actuators to the scene's initial posture."""
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]
