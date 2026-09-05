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


logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
absl_logging.set_verbosity(absl_logging.FATAL)
transformers_logging.set_verbosity_error()


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCENE = PROJECT_DIR / "vision2action" / "env" / "scene.xml"
CHECKPOINT = PROJECT_DIR / "checkpoints" / "octo-small-1.5"
IMAGE_SIZE = 256
MOVE_SCALE = 0.01
ROTATION_SCALE = 0.03
HOME_JOINTS = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]


def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", default="pick up the red can")
    parser.add_argument(
        "--decisions",
        type=int,
        default=0,
        help="Number of Octo decisions. Use 0 to run until the viewer closes.",
    )
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def camera_image(renderer, data):
    renderer.update_scene(data, camera="wrist_cam")
    return renderer.render().copy()


def object_id(model, object_type, name):
    return mujoco.mj_name2id(model, object_type, name)


def initialize_panda(model, data):
    for number, value in enumerate(HOME_JOINTS, start=1):
        joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{number}")
        actuator_id = object_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{number}"
        )
        data.qpos[model.jnt_qposadr[joint_id]] = value
        data.ctrl[actuator_id] = value

    for name in ("finger_joint1", "finger_joint2"):
        joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = 0.04

    finger_id = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    data.ctrl[finger_id] = 255.0
    mujoco.mj_forward(model, data)


def apply_action(model, data, action):
    movement = np.clip(action[:3], -1.0, 1.0) * MOVE_SCALE
    rotation = np.clip(action[3:6], -1.0, 1.0) * ROTATION_SCALE

    site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))
    mujoco.mj_jacSite(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        site_id,
    )

    joint_ids = [
        object_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{number}")
        for number in range(1, 8)
    ]
    dof_ids = [model.jnt_dofadr[joint_id] for joint_id in joint_ids]
    jacobian = np.vstack(
        [jacobian_position[:, dof_ids], jacobian_rotation[:, dof_ids]]
    )
    desired_change = np.concatenate([movement, rotation])
    damping = 0.01 * np.eye(6)
    joint_change = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping,
        desired_change,
    )
    joint_change = np.clip(joint_change, -0.04, 0.04)

    for number, (joint_id, change) in enumerate(
        zip(joint_ids, joint_change), start=1
    ):
        qpos_id = model.jnt_qposadr[joint_id]
        actuator_id = object_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{number}"
        )
        data.ctrl[actuator_id] = data.qpos[qpos_id] + change

    finger_value = 0.0 if action[6] > 0 else 255.0
    finger_id = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    data.ctrl[finger_id] = finger_value

    for _ in range(20):
        mujoco.mj_step(model, data)

    return movement, finger_value


def main():
    args = read_arguments()
    if args.headless and args.decisions == 0:
        raise ValueError("Headless mode needs --decisions greater than 0")

    print("Loading Octo on:", jax.devices())
    octo = OctoModel.load_pretrained(str(CHECKPOINT))
    task = octo.create_tasks(texts=[args.instruction])

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_SIZE, width=IMAGE_SIZE)
    initialize_panda(model, data)

    viewer = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)

    print("Instruction:", args.instruction)
    print("Octo is controlling the Panda hand and fingers")

    try:
        decision = 0
        while viewer is None or viewer.is_running():
            if args.decisions > 0 and decision >= args.decisions:
                break

            image = camera_image(renderer, data)
            observation = {
                "image_primary": image[None, None],
                "timestep_pad_mask": np.ones((1, 1), dtype=bool),
            }
            actions = octo.sample_actions(
                observation,
                task,
                rng=jax.random.PRNGKey(decision),
            )
            action = np.asarray(jax.device_get(actions))[0, 0]
            movement, finger_value = apply_action(model, data, action)

            print(
                f"Step {decision + 1}: move {movement.round(4).tolist()}, "
                f"finger target {finger_value:.2f}"
            )

            decision += 1

            if viewer is not None:
                viewer.sync()
                time.sleep(0.15)
    finally:
        renderer.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
