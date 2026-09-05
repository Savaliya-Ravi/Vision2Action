from pathlib import Path

import mujoco
import numpy as np


SCENE = Path(__file__).resolve().parents[1] / "vision2action" / "env" / "scene.xml"


def body_position(model, data, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body_id].copy()


def move_mocap(model, data, target, steps=200):
    start = data.mocap_pos[0].copy()

    for step in range(steps):
        amount = (step + 1) / steps
        data.mocap_pos[0] = start + amount * (target - start)
        mujoco.mj_step(model, data)


def hold(model, data, steps=200):
    for _ in range(steps):
        mujoco.mj_step(model, data)


def can_is_inside_fridge(model, data):
    can = body_position(model, data, "target_can_red")
    fridge = body_position(model, data, "fridge_carcass")

    inside_x = abs(can[0] - fridge[0]) < 0.38
    inside_y = abs(can[1] - fridge[1]) < 0.36
    inside_z = fridge[2] + 0.04 < can[2] < fridge[2] + 1.80
    return inside_x and inside_y and inside_z


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    door_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fridge_door_hinge")
    door_qpos = model.jnt_qposadr[door_id]
    data.qpos[door_qpos] = np.deg2rad(105)
    mujoco.mj_forward(model, data)

    can = body_position(model, data, "target_can_red")
    grasp_point = can + np.array([0.0, 0.0, 0.06])
    gripper_offset = np.array([0.38, 0.0, -0.45])

    above_can = grasp_point - gripper_offset + np.array([0.0, 0.0, 0.45])
    grasp_can = grasp_point - gripper_offset

    data.ctrl[:] = 0.0
    move_mocap(model, data, above_can)
    move_mocap(model, data, grasp_can)

    data.ctrl[:] = 0.52
    hold(model, data, 300)

    grasp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "can_grasp_assist")
    data.eq_active[grasp_id] = 1
    hold(model, data, 50)

    lift = grasp_can + np.array([0.0, 0.0, 0.8])
    move_mocap(model, data, lift)

    fridge_drop = np.array([-3.25, 2.9, 1.25])
    above_fridge = fridge_drop - gripper_offset
    move_mocap(model, data, above_fridge, steps=500)

    data.eq_active[grasp_id] = 0
    data.ctrl[:] = 0.0
    hold(model, data, 500)

    can = body_position(model, data, "target_can_red")
    inside = can_is_inside_fridge(model, data)
    print(f"Final can position: {can.round(3).tolist()}")
    print(f"Can inside fridge: {inside}")


if __name__ == "__main__":
    main()
