from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from vision2action.control.controller import RobotState, apply_robot_state
from vision2action.env.randomization import randomize_scene_objects
from vision2action.manipulation import grasp
from vision2action.manipulation.grasp import (
    ARM_HOME,
    initialize_gripper,
    pick_up_red_can,
    put_down_red_can,
    sync_held_red_can,
)
from vision2action.main import _handle_command
from vision2action.world.world_model import WorldModel


SCENE = Path(__file__).resolve().parents[1] / "vision2action" / "env" / "scene.xml"


class HeadlessViewer:
    def sync(self):
        pass


def setup_can_trial():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    placements = randomize_scene_objects(model, data, seed=1)
    initialize_gripper(model, data)
    joint = model.joint("target_can_red_free")
    home_pose = data.qpos[joint.qposadr[0]:joint.qposadr[0] + 7].copy()
    state = RobotState(x=3.0, y=-2.5, yaw=-0.65)
    apply_robot_state(model, data, state)
    return model, data, placements["can_red"][:2], home_pose, state


def test_carried_can_follows_navigation_and_can_be_placed_on_counter():
    model, data, can_xy, home_pose, state = setup_can_trial()
    viewer = HeadlessViewer()
    can_body = model.body("target_can_red").id
    hand_site = model.site("right_grasp_site").id

    assert pick_up_red_can(model, data, viewer, can_xy, state)
    for name, target in ARM_HOME.items():
        if name.startswith("right_"):
            assert np.isclose(data.qpos[model.joint(name).qposadr[0]], target)
    np.testing.assert_allclose(data.site_xpos[hand_site] - data.xpos[can_body], [0, 0, 0.0575], atol=1e-6)

    state.x, state.y, state.yaw = 0.75, 2.4, np.pi / 2
    apply_robot_state(model, data, state)
    sync_held_red_can(model, data)
    np.testing.assert_allclose(data.site_xpos[hand_site] - data.xpos[can_body], [0, 0, 0.0575], atol=1e-6)
    assert data.xpos[can_body][1] > 2.0

    assert put_down_red_can(model, data, viewer, state, home_pose) == ("counter", 0.935)
    placed = data.xpos[can_body].copy()
    np.testing.assert_allclose(placed[:2], [1.55, 3.08], atol=1e-6)
    assert np.isclose(placed[2], 0.935)
    for name, target in ARM_HOME.items():
        if name.startswith("right_"):
            assert np.isclose(data.qpos[model.joint(name).qposadr[0]], target)

    state.x -= 0.5
    apply_robot_state(model, data, state)
    np.testing.assert_allclose(data.xpos[can_body], placed, atol=1e-6)


def test_put_back_requires_returning_near_original_table():
    model, data, can_xy, home_pose, state = setup_can_trial()
    viewer = HeadlessViewer()
    assert pick_up_red_can(model, data, viewer, can_xy, state)
    state.x, state.y, state.yaw = 0.75, 2.4, np.pi / 2
    apply_robot_state(model, data, state)
    sync_held_red_can(model, data)
    before = data.qpos.copy()

    assert put_down_red_can(model, data, viewer, state, home_pose, put_back=True) is None
    np.testing.assert_array_equal(data.qpos, before)

    state.x, state.y, state.yaw = 3.0, -2.5, -0.65
    apply_robot_state(model, data, state)
    sync_held_red_can(model, data)
    assert put_down_red_can(model, data, viewer, state, home_pose, put_back=True) == ("table", home_pose[2])
    can_qpos = model.joint("target_can_red_free").qposadr[0]
    np.testing.assert_allclose(data.qpos[can_qpos:can_qpos + 3], home_pose[:3], atol=1e-6)


def test_failed_placement_keeps_can_in_hand_and_retracts_arm(monkeypatch):
    model, data, can_xy, home_pose, state = setup_can_trial()
    viewer = HeadlessViewer()
    assert pick_up_red_can(model, data, viewer, can_xy, state)
    state.x, state.y, state.yaw = 0.75, 2.4, np.pi / 2
    apply_robot_state(model, data, state)
    sync_held_red_can(model, data)
    monkeypatch.setattr(grasp, "_move_hand", lambda *args, **kwargs: False)

    assert put_down_red_can(model, data, viewer, state, home_pose) is None
    hand = data.site_xpos[model.site("right_grasp_site").id]
    can = data.xpos[model.body("target_can_red").id]
    np.testing.assert_allclose(hand - can, [0, 0, 0.0575], atol=1e-6)
    for name, target in ARM_HOME.items():
        if name.startswith("right_"):
            assert np.isclose(data.qpos[model.joint(name).qposadr[0]], target)


def test_commands_preserve_carry_until_explicit_put_down():
    class MissionStub:
        config = SimpleNamespace(nav=SimpleNamespace(stop_distance_m=0.85))
        detector = SimpleNamespace(name="OracleDetector(omniscient)")
        query = None

        def reset(self):
            self.query = None

        def start(self, query):
            self.query = query

    mission = MissionStub()
    world = WorldModel()
    state = {
        "pick_up": False,
        "put_down": None,
        "held_can": True,
        "target_xy": None,
        "normal_stop": 0.85,
    }

    assert _handle_command("go to the sink", mission, world, state)
    assert mission.query.category == "sink"
    assert state["held_can"] and state["put_down"] is None

    assert _handle_command("put it down", mission, world, state)
    assert mission.query is None
    assert state["held_can"] and state["put_down"] == "down"

    assert _handle_command("stop", mission, world, state)
    assert state["held_can"] and state["put_down"] is None

    assert _handle_command("put the can back", mission, world, state)
    assert state["put_down"] == "back"
