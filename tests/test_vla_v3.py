from pathlib import Path
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
import pytest

from vision2action.vla import __main__ as vla_main
from vision2action.vla.__main__ import initialize_fridge_trial, run_interactive
from vision2action.vla.g1_adapter import G1ActionAdapter, RIGHT_ARM
from vision2action.vla.octo_policy import OctoPolicy


SCENE = Path(__file__).resolve().parents[1] / "vision2action" / "env" / "scene.xml"


def make_trial():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    initialize_fridge_trial(model, data)
    return model, data


def test_v3_starts_with_closed_door_and_reachable_handle():
    model, data = make_trial()
    door_q = model.joint("fridge_door_hinge").qposadr[0]
    hand = data.site_xpos[model.site("right_grasp_site").id]
    handle = data.geom_xpos[model.geom("fridge_handle_up").id]

    assert data.qpos[door_q] == 0.0
    assert np.linalg.norm(hand - handle) < 0.04
    assert model.camera("vla_fridge_cam").id >= 0
    assert model.camera("vla_wrist_cam").id >= 0


def test_v3_adapter_rejects_invalid_model_action_without_motion():
    model, data = make_trial()
    adapter = G1ActionAdapter(model, data)
    qpos_before = data.qpos.copy()
    ctrl_before = data.ctrl.copy()

    for invalid in (np.zeros(6), np.array([0, 0, 0, 0, 0, 0, np.nan])):
        with pytest.raises(ValueError):
            adapter.apply(invalid)

    np.testing.assert_array_equal(data.qpos, qpos_before)
    np.testing.assert_array_equal(data.ctrl, ctrl_before)


def test_v3_adapter_binds_model_output_to_arm_and_hand_only():
    model, data = make_trial()
    adapter = G1ActionAdapter(model, data)
    ctrl_before = data.ctrl.copy()
    base_before = data.qpos[:7].copy()
    adapter.apply(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    arm_ids = [model.actuator(name).id for name in RIGHT_ARM]
    hand_ids = [model.actuator(name).id for name in (
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    )]
    assert any(abs(data.ctrl[i] - ctrl_before[i]) > 1e-5 for i in arm_ids)
    assert any(abs(data.ctrl[i]) > 0.1 for i in hand_ids)
    untouched = [i for i in range(model.nu) if i not in arm_ids + hand_ids]
    np.testing.assert_array_equal(data.ctrl[untouched], ctrl_before[untouched])
    np.testing.assert_allclose(data.qpos[:7], base_before)
    assert np.all(np.isfinite(data.qpos))


def test_interactive_instruction_goes_to_model_and_resets_history():
    class FakeModel:
        def create_tasks(self, texts):
            return {"texts": texts}

    policy = OctoPolicy.__new__(OctoPolicy)
    policy.model = FakeModel()
    policy.primary_history = [np.zeros((256, 256, 3), dtype=np.uint8)]
    policy.wrist_history = [np.zeros((128, 128, 3), dtype=np.uint8)]
    policy.decision = 7
    policy.set_instruction("  open the fridge door  ")

    assert policy.task == {"texts": ["open the fridge door"]}
    assert policy.primary_history == []
    assert policy.wrist_history == []
    assert policy.decision == 0


def test_interactive_viewer_rejects_headless_egl(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    with pytest.raises(RuntimeError, match="without MUJOCO_GL=egl"):
        run_interactive(1)


def test_interactive_viewer_rejects_unreachable_display(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":99999")
    with pytest.raises(RuntimeError, match="Cannot connect to the desktop display"):
        run_interactive(1)


def test_interactive_command_drives_policy_and_simulation(monkeypatch):
    received = []

    class FakeRenderer:
        def __init__(self, model, height, width):
            self.shape = (height, width, 3)

        def update_scene(self, data, camera):
            pass

        def render(self):
            return np.zeros(self.shape, dtype=np.uint8)

        def close(self):
            pass

    class FakeViewer:
        cam = SimpleNamespace(type=None, fixedcamid=None)
        closed = False
        syncs = 0

        def is_running(self):
            return self.syncs < 4

        def sync(self):
            self.syncs += 1

        def close(self):
            self.closed = True

    class FakePolicy:
        def __init__(self, checkpoint, instruction):
            received.append(instruction)
            self.devices = ["fake"]

        def predict(self, primary, wrist):
            assert primary.shape == (256, 256, 3)
            assert wrist.shape == (128, 128, 3)
            return np.array([0.01, 0, 0, 0, 0, 0, 1.0])

    class ImmediateThread:
        def __init__(self, target, args, daemon):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    viewer = FakeViewer()
    monkeypatch.setattr(vla_main, "_require_desktop_viewer", lambda: None)
    monkeypatch.setattr(vla_main, "_read_terminal_commands", lambda commands: commands.put("open the fridge door"))
    monkeypatch.setattr(vla_main, "OctoPolicy", FakePolicy)
    monkeypatch.setattr(vla_main.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda model, data: viewer)

    sessions = run_interactive(1)

    assert received == ["open the fridge door"]
    assert sessions[0]["instruction"] == "open the fridge door"
    assert len(sessions[0]["actions"]) == 1
    assert sessions[0]["outcome"] == "limit_reached"
    assert viewer.closed
