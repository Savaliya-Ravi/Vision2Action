import argparse
import logging
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message="TensorFlow and JAX classes are deprecated.*")

import jax
import mujoco
import numpy as np
from absl import logging as absl_logging
from octo.model.octo_model import OctoModel
from transformers.utils import logging as transformers_logging

from test_octo_mujoco import camera_image, initialize_panda, object_id


logging.getLogger().setLevel(logging.ERROR)
absl_logging.set_verbosity(absl_logging.FATAL)
transformers_logging.set_verbosity_error()

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCENE = PROJECT_DIR / "vision2action" / "env" / "scene.xml"
CHECKPOINT = PROJECT_DIR / "checkpoints" / "octo-small-1.5"
HOME_JOINTS = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
POST_GRASP_SPEED = 0.25


def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def panda_ids(model):
    joint_ids = [
        object_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{number}")
        for number in range(1, 8)
    ]
    actuator_ids = [
        object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{number}")
        for number in range(1, 8)
    ]
    return joint_ids, actuator_ids


def keep_can_between_jaws(model, data, can_id):
    site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
    can_joint_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "target_can_red_free",
    )
    qpos_id = model.jnt_qposadr[can_joint_id]
    qvel_id = model.jnt_dofadr[can_joint_id]
    data.qpos[qpos_id : qpos_id + 3] = data.site_xpos[site_id] - np.array(
        [0.0, 0.0, 0.0575]
    )
    data.qpos[qpos_id + 3 : qpos_id + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[qvel_id : qvel_id + 6] = 0.0
    mujoco.mj_forward(model, data)


def move_hand(
    model,
    data,
    target,
    viewer,
    base_pose,
    steps=300,
    physics=False,
    held_can_id=None,
    speed=1.0,
):
    site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
    joint_ids, actuator_ids = panda_ids(model)
    dof_ids = [model.jnt_dofadr[joint_id] for joint_id in joint_ids]

    for _ in range(steps):
        current = data.site_xpos[site_id].copy()
        error = target - current
        if np.linalg.norm(error) < 0.008:
            return True

        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian, None, site_id)
        arm_jacobian = jacobian[:, dof_ids]
        damping = 0.01 * np.eye(3)
        joint_change = arm_jacobian.T @ np.linalg.solve(
            arm_jacobian @ arm_jacobian.T + damping,
            np.clip(error, -0.02, 0.02),
        )
        joint_change = np.clip(joint_change, -0.025, 0.025) * speed

        for joint_id, actuator_id, change in zip(
            joint_ids, actuator_ids, joint_change
        ):
            qpos_id = model.jnt_qposadr[joint_id]
            joint_target = data.qpos[qpos_id] + change
            joint_target = np.clip(
                joint_target,
                model.jnt_range[joint_id, 0],
                model.jnt_range[joint_id, 1],
            )
            data.qpos[qpos_id] = joint_target
            data.ctrl[actuator_id] = joint_target

        data.qpos[:7] = base_pose
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)
        if physics:
            mujoco.mj_step(model, data)
            data.qpos[:7] = base_pose
            data.qvel[:6] = 0.0
            mujoco.mj_forward(model, data)
        if held_can_id is not None:
            keep_can_between_jaws(model, data, held_can_id)

        if viewer is not None:
            viewer.sync()

    return False


def hold(model, data, viewer, base_pose, steps, held_can_id=None):
    for _ in range(steps):
        mujoco.mj_step(model, data)
        data.qpos[:7] = base_pose
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)
        if held_can_id is not None:
            keep_can_between_jaws(model, data, held_can_id)
        if viewer is not None:
            viewer.sync()


def can_center_position(model, data, can_id):
    rotation = data.xmat[can_id].reshape(3, 3)
    return data.xpos[can_id] + rotation @ np.array([0.0, 0.0, 0.0575])


def open_fridge(model, data, viewer, can_id):
    door_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "fridge_door_hinge",
    )
    qpos_id = model.jnt_qposadr[door_id]
    start = data.qpos[qpos_id]

    total_steps = int(120 / POST_GRASP_SPEED)
    for step in range(total_steps):
        amount = (step + 1) / total_steps
        data.qpos[qpos_id] = start + amount * (1.83 - start)
        data.qvel[model.jnt_dofadr[door_id]] = 0.0
        mujoco.mj_forward(model, data)
        keep_can_between_jaws(model, data, can_id)
        if viewer is not None:
            viewer.sync()


def drive_to_fridge(model, data, viewer, base_pose, can_id):
    start = base_pose.copy()
    target = np.array(
        [-3.25, 2.08, 0.12, np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    )

    total_steps = int(300 / POST_GRASP_SPEED)
    for step in range(total_steps):
        amount = (step + 1) / total_steps
        data.qpos[:7] = start + amount * (target - start)
        data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)
        keep_can_between_jaws(model, data, can_id)
        if viewer is not None:
            viewer.sync()

    return target


def can_is_inside_fridge(model, data, can_id):
    fridge_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "fridge_carcass")
    can = data.xpos[can_id]
    fridge = data.xpos[fridge_id]
    inside_x = abs(can[0] - fridge[0]) < 0.36
    inside_y = abs(can[1] - fridge[1]) < 0.34
    inside_z = 0.04 < can[2] < 1.75
    return bool(inside_x and inside_y and inside_z)


def main():
    args = read_arguments()
    print("Loading Octo on:", jax.devices())
    octo = OctoModel.load_pretrained(str(CHECKPOINT))
    task = octo.create_tasks(texts=["pick up the red can"])

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    initialize_panda(model, data)
    base_pose = data.qpos[:7].copy()

    renderer = mujoco.Renderer(model, height=256, width=256)
    viewer = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)

    can_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "target_can_red")
    can_start = data.xpos[can_id].copy()
    can_center = can_start + np.array([0.0, 0.0, 0.058])
    above_can = can_center + np.array([0.0, 0.0, 0.30])
    grasp_can = can_center.copy()

    print("Can coordinates:", can_center.round(3).tolist())
    print("Moving above the can")
    reached_above = move_hand(model, data, above_can, viewer, base_pose)

    print("Lowering to the can")
    reached_grasp = move_hand(model, data, grasp_can, viewer, base_pose)

    image = camera_image(renderer, data)
    observation = {
        "image_primary": image[None, None],
        "timestep_pad_mask": np.ones((1, 1), dtype=bool),
    }
    actions = octo.sample_actions(
        observation,
        task,
        rng=jax.random.PRNGKey(0),
    )
    octo_action = np.asarray(jax.device_get(actions))[0, 0]
    octo_says_close = bool(octo_action[6] > 0)
    print("Octo gripper suggestion:", "close" if octo_says_close else "open")

    finger_id = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    data.ctrl[finger_id] = 0.0
    hold(model, data, viewer, base_pose, 60)
    move_hand(model, data, grasp_can, viewer, base_pose, steps=150)
    site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
    center_at_grasp = can_center_position(model, data, can_id)
    print("Jaw-to-can-center error:", round(float(np.linalg.norm(
        data.site_xpos[site_id] - center_at_grasp
    )), 4), "m")
    keep_can_between_jaws(model, data, can_id)

    print("Lifting the can")
    lift_target = can_center + np.array([0.0, 0.0, 0.45])
    reached_lift = move_hand(
        model,
        data,
        lift_target,
        viewer,
        base_pose,
        steps=int(400 / POST_GRASP_SPEED),
        physics=True,
        held_can_id=can_id,
        speed=POST_GRASP_SPEED,
    )
    hold(model, data, viewer, base_pose, 100, held_can_id=can_id)

    can_end = data.xpos[can_id].copy()
    hand_end = data.site_xpos[site_id].copy()
    can_center_end = can_center_position(model, data, can_id)
    lifted = can_end[2] > can_start[2] + 0.12
    print("Reached waypoints:", reached_above, reached_grasp, reached_lift)
    print("Final can position:", can_end.round(3).tolist())
    print("Final hand position:", hand_end.round(3).tolist())
    print("Final can center:", can_center_end.round(3).tolist())
    print("Can lifted:", lifted)

    print("Opening the fridge door")
    open_fridge(model, data, viewer, can_id)

    print("Driving the mobile base to the fridge")
    base_pose = drive_to_fridge(model, data, viewer, base_pose, can_id)

    print("Moving the can into the fridge")
    fridge_entry = np.array([-3.25, 2.62, 0.65])
    fridge_drop = np.array([-3.25, 2.88, 0.28])
    reached_entry = move_hand(
        model,
        data,
        fridge_entry,
        viewer,
        base_pose,
        steps=int(400 / POST_GRASP_SPEED),
        held_can_id=can_id,
        speed=POST_GRASP_SPEED,
    )
    reached_drop = move_hand(
        model,
        data,
        fridge_drop,
        viewer,
        base_pose,
        steps=int(400 / POST_GRASP_SPEED),
        held_can_id=can_id,
        speed=POST_GRASP_SPEED,
    )

    print("Opening the fingers and releasing the can")
    data.ctrl[finger_id] = 255.0
    hold(model, data, viewer, base_pose, 250)

    can_final = data.xpos[can_id].copy()
    inside = can_is_inside_fridge(model, data, can_id)
    print("Reached fridge waypoints:", reached_entry, reached_drop)
    print("Released can position:", can_final.round(3).tolist())
    print("Can inside fridge:", inside)

    renderer.close()
    if viewer is not None:
        print("Close the viewer window to finish")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        viewer.close()


if __name__ == "__main__":
    main()
