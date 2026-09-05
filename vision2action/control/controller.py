from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class RobotState:
    x: float
    y: float
    yaw: float


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Quaternion [w, x, y, z] for yaw rotation about Z-axis."""
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def initialize_robot_state(model: mujoco.MjModel, data: mujoco.MjData) -> RobotState:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    pos = data.xpos[body_id]
    x = float(pos[0])
    y = float(pos[1])
    yaw = 0.0
    return RobotState(x=x, y=y, yaw=yaw)


def apply_robot_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: RobotState,
    robot_height: float = 0.793,
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]

    data.qpos[qpos_adr : qpos_adr + 3] = np.array([state.x, state.y, robot_height], dtype=float)
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = yaw_to_quat(state.yaw)
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    mujoco.mj_forward(model, data)


def rotate_in_place(state: RobotState, yaw_step: float = 0.06) -> None:
    state.yaw += yaw_step


def move_forward(state: RobotState, forward_step: float = 0.005) -> None:
    """Move robot forward along current yaw and clamp inside room bounds."""
    state.x += forward_step * math.cos(state.yaw)
    state.y += forward_step * math.sin(state.yaw)
    state.x = float(np.clip(state.x, -3.7, 3.7))
    state.y = float(np.clip(state.y, -3.7, 3.7))


def distance_to_world_target(state: RobotState, target_xy: np.ndarray) -> float:
    """Return planar distance to a target position estimated from perception."""
    target_xy = np.asarray(target_xy, dtype=float)
    if target_xy.shape != (2,) or not np.all(np.isfinite(target_xy)):
        return float("nan")
    robot_xy = np.array([state.x, state.y], dtype=float)
    return float(np.linalg.norm(target_xy - robot_xy))


def robot_to_target_distance(model: mujoco.MjModel, data: mujoco.MjData, target_body_name: str) -> float:
    """Planar distance (meters) between robot body center and target body center."""
    robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    robot_xy = data.xpos[robot_id][:2]
    target_xy = data.xpos[target_id][:2]
    return float(np.linalg.norm(target_xy - robot_xy))


def camera_forward_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: RobotState,
    target_body_name: str,
    cam_local_xy: tuple[float, float] = (0.18, 0.0),
) -> float:
    """Distance from camera along robot forward (+x local) to target body in meters."""
    rx, ry = state.x, state.y
    yaw = state.yaw

    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=float)

    cam_local = np.array([cam_local_xy[0], cam_local_xy[1]], dtype=float)
    cam_world = np.array([rx, ry], dtype=float) + rot.dot(cam_local)

    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    target_xy = data.xpos[target_id][:2]

    vec = target_xy - cam_world
    forward = np.array([c, s], dtype=float)
    return float(np.dot(vec, forward))
