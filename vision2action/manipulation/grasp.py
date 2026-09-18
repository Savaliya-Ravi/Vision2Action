from __future__ import annotations

import math

import mujoco
import numpy as np

from vision2action.control.controller import apply_robot_state


RIGHT_ARM_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

ARM_HOME = {
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_elbow_joint": 1.28,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": 1.28,
}

RIGHT_HAND_JOINTS = {
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": -0.8,
    "right_hand_thumb_2_joint": -1.4,
    "right_hand_index_0_joint": 1.2,
    "right_hand_index_1_joint": 1.4,
    "right_hand_middle_0_joint": 1.2,
    "right_hand_middle_1_joint": 1.4,
}


def _id(model, kind, name):
    return mujoco.mj_name2id(model, kind, name)


def _set_joint(model, data, name, value):
    joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    actuator = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    data.qpos[model.jnt_qposadr[joint]] = value
    if actuator >= 0:
        data.ctrl[actuator] = value


def initialize_gripper(model, data):
    for name, value in ARM_HOME.items():
        _set_joint(model, data, name, value)
    for name in RIGHT_HAND_JOINTS:
        _set_joint(model, data, name, 0.0)
    mujoco.mj_forward(model, data)


def _move_hand(model, data, viewer, target, steps=600, held_can=False):
    site = _id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site")
    joints = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in RIGHT_ARM_JOINTS]
    dofs = [model.jnt_dofadr[joint] for joint in joints]

    for _ in range(steps):
        error = np.asarray(target) - data.site_xpos[site]
        if np.linalg.norm(error) < 0.015:
            return True

        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian, None, site)
        arm_jacobian = jacobian[:, dofs]
        damping = 0.01 * np.eye(3)
        change = arm_jacobian.T @ np.linalg.solve(
            arm_jacobian @ arm_jacobian.T + damping,
            np.clip(error, -0.01, 0.01),
        )
        change = np.clip(change, -0.008, 0.008)

        for name, joint, amount in zip(RIGHT_ARM_JOINTS, joints, change):
            qpos = model.jnt_qposadr[joint]
            value = np.clip(
                data.qpos[qpos] + amount,
                model.jnt_range[joint, 0],
                model.jnt_range[joint, 1],
            )
            _set_joint(model, data, name, value)

        mujoco.mj_forward(model, data)
        if held_can:
            _attach_can(model, data)
        viewer.sync()
    return False


def _attach_can(model, data):
    site = _id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site")
    joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_can_red_free")
    qpos = model.jnt_qposadr[joint]
    qvel = model.jnt_dofadr[joint]
    data.qpos[qpos:qpos + 3] = data.site_xpos[site] - np.array([0.0, 0.0, 0.0575])
    data.qpos[qpos + 3:qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[qvel:qvel + 6] = 0.0
    mujoco.mj_forward(model, data)


def sync_held_red_can(model, data):
    """Keep the V2 assisted grasp aligned after the floating base moves."""
    _attach_can(model, data)


def _move_base_closer(model, data, viewer, robot_state, perceived_xy, held_can=False):
    target = np.asarray(perceived_xy, dtype=float)
    forward = np.array([math.cos(robot_state.yaw), math.sin(robot_state.yaw)])
    right = np.array([math.sin(robot_state.yaw), -math.cos(robot_state.yaw)])
    goal = target - 0.35 * forward - 0.18 * right
    for _ in range(180):
        current = np.array([robot_state.x, robot_state.y])
        difference = goal - current
        distance = np.linalg.norm(difference)
        if distance <= 0.01:
            return True
        current += difference / distance * min(0.005, distance)
        robot_state.x, robot_state.y = current
        apply_robot_state(model, data, robot_state)
        if held_can:
            _attach_can(model, data)
        viewer.sync()
    return False


def _close_right_hand(model, data):
    for name, value in RIGHT_HAND_JOINTS.items():
        _set_joint(model, data, name, value)
    mujoco.mj_forward(model, data)


def _open_right_hand(model, data):
    for name in RIGHT_HAND_JOINTS:
        _set_joint(model, data, name, 0.0)
    mujoco.mj_forward(model, data)


def stow_right_arm(model, data, viewer, *, held_can=False, open_hand=False):
    """Bring the V2 arm to its neutral pose, preserving an assisted grasp."""
    targets = {name: ARM_HOME.get(name, 0.0) for name in RIGHT_ARM_JOINTS}
    for _ in range(400):
        remaining = 0.0
        for name, target in targets.items():
            joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            current = float(data.qpos[model.jnt_qposadr[joint]])
            delta = float(np.clip(target - current, -0.015, 0.015))
            remaining = max(remaining, abs(target - current))
            _set_joint(model, data, name, current + delta)
        mujoco.mj_forward(model, data)
        if held_can:
            _attach_can(model, data)
        viewer.sync()
        if remaining <= 0.015:
            break
    if open_hand:
        _open_right_hand(model, data)
        viewer.sync()


def _placement_target(model, data, robot_state, home_pose, put_back):
    home_pose = np.asarray(home_pose, dtype=float)
    if home_pose.shape != (7,) or not np.all(np.isfinite(home_pose)):
        raise ValueError("Expected the can's original seven-value free-joint pose")
    table_xy = home_pose[:2]
    table_z = float(home_pose[2])
    counter_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "counter_top")
    counter_center = data.geom_xpos[counter_geom][:2]
    counter_z = float(data.geom_xpos[counter_geom][2] + model.geom_size[counter_geom][2])
    choices = [("table", table_xy, table_z)]
    if not put_back:
        # These two clear areas flank the sink and appliances on the counter.
        for offset_x in (-1.35, 0.95):
            counter_xy = counter_center + np.array([offset_x, -0.27])
            choices.append(("counter", counter_xy, counter_z))
    robot_xy = np.array([robot_state.x, robot_state.y], dtype=float)
    surface, xy, z = min(choices, key=lambda item: np.linalg.norm(item[1] - robot_xy))
    if np.linalg.norm(xy - robot_xy) > 1.2:
        return None
    return surface, np.array([xy[0], xy[1], z + 0.0575]), z


def put_down_red_can(model, data, viewer, robot_state, home_pose, *, put_back=False):
    """Set the carried can on a nearby surface and retract the V2 arm.

    Returns ``(surface_name, can_base_z)`` or ``None`` if no surface is in reach.
    The assisted grasp remains active in the caller on failure.
    """
    placement = _placement_target(model, data, robot_state, home_pose, put_back)
    if placement is None:
        return None
    surface, target, can_base_z = placement
    if not _move_base_closer(model, data, viewer, robot_state, target[:2], held_can=True):
        stow_right_arm(model, data, viewer, held_can=True)
        return None
    above = target + np.array([0.0, 0.0, 0.15])
    if not _move_hand(model, data, viewer, above, held_can=True):
        stow_right_arm(model, data, viewer, held_can=True)
        return None
    if not _move_hand(model, data, viewer, target, held_can=True):
        stow_right_arm(model, data, viewer, held_can=True)
        return None
    _open_right_hand(model, data)
    can_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_can_red_free")
    can_qpos = int(model.jnt_qposadr[can_joint])
    can_qvel = int(model.jnt_dofadr[can_joint])
    data.qpos[can_qpos:can_qpos + 3] = [target[0], target[1], can_base_z]
    data.qvel[can_qvel:can_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)
    stow_right_arm(model, data, viewer, held_can=False)
    return surface, can_base_z


def pick_up_red_can(model, data, viewer, perceived_xy, robot_state, can_base_z=0.795):
    if not _move_base_closer(model, data, viewer, robot_state, perceived_xy):
        return False

    center = np.array([perceived_xy[0], perceived_xy[1], can_base_z + 0.0575])
    above = center + np.array([0.0, 0.0, 0.18])
    if not _move_hand(model, data, viewer, above):
        return False
    if not _move_hand(model, data, viewer, center):
        return False

    _close_right_hand(model, data)
    _attach_can(model, data)
    lift = center + np.array([0.0, 0.0, 0.25])
    if not _move_hand(model, data, viewer, lift, steps=800, held_can=True):
        return False
    stow_right_arm(model, data, viewer, held_can=True)
    return True
