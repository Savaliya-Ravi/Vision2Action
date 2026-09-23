from pathlib import Path

import mujoco
import numpy as np

from vision2action.vla.g1_adapter import G1ActionAdapter, WAIST
from vision2action.vla.v4 import (
    EXPERT_ACTION,
    SCENE,
    _expert_action,
    _load_dataset,
    expert_physics_rollout,
    initialize_v4_trial,
    run_v4_trial,
)
from vision2action.vla.v4_policy import (
    ACTION_DIM,
    FEATURE_DIM,
    LearnedActionHead,
    LearnedCompletionHead,
    LearnedIntentHead,
)


def make_trial():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    initialize_v4_trial(model, data)
    return model, data


def test_v4_start_is_reachable_and_clear_of_the_door_swing():
    model, data = make_trial()
    hand = data.site_xpos[model.site("right_grasp_site").id]
    handle = data.geom_xpos[model.geom("fridge_handle_up").id]
    robot = data.xpos[model.body("robot").id]

    assert np.linalg.norm(hand - handle) < 0.05
    assert robot[1] < 1.9
    assert data.qpos[model.joint("fridge_door_hinge").qposadr[0]] == 0.0


def test_v4_action_adapter_adds_waist_without_moving_the_base():
    model, data = make_trial()
    adapter = G1ActionAdapter(model, data, control_waist=True)
    base_q = model.joint("robot_free").qposadr[0]
    base_before = data.qpos[base_q:base_q + 7].copy()
    ctrl_before = data.ctrl.copy()

    adapter.apply(EXPERT_ACTION)

    waist_actuators = [model.actuator(name).id for name in WAIST]
    assert any(abs(data.ctrl[index] - ctrl_before[index]) > 1e-6 for index in waist_actuators)
    np.testing.assert_allclose(data.qpos[base_q:base_q + 7], base_before)


def test_v4_expert_demonstration_opens_by_physical_contact():
    result = expert_physics_rollout(max_decisions=40)

    assert result["opened"]
    assert result["door_deg"] >= 30.0
    assert result["contact_steps"] > 0
    assert result["decisions"] <= 40


def test_v4_dataset_validation_and_learned_head(tmp_path: Path):
    dataset = tmp_path / "demo.npz"
    np.savez_compressed(
        dataset,
        format_version=np.array(1),
        primary=np.zeros((2, 256, 256, 3), dtype=np.uint8),
        wrist=np.zeros((2, 128, 128, 3), dtype=np.uint8),
        action=np.stack([EXPERT_ACTION, EXPERT_ACTION]).astype(np.float32),
        door_deg=np.array([0.0, 2.0], dtype=np.float32),
        episode=np.array([0, 0], dtype=np.int32),
        instruction=np.array(["open the fridge door", "open the fridge door"]),
    )
    assert _load_dataset(dataset)["action"].shape == (2, ACTION_DIM)

    checkpoint = tmp_path / "head.npz"
    np.savez_compressed(
        checkpoint,
        format_version=np.array(2),
        weights=np.zeros((FEATURE_DIM, ACTION_DIM), dtype=np.float32),
        bias=EXPERT_ACTION.astype(np.float32),
        feature_mean=np.zeros(FEATURE_DIM, dtype=np.float32),
        feature_std=np.ones(FEATURE_DIM, dtype=np.float32),
        action_prototypes=np.stack([EXPERT_ACTION, np.zeros(ACTION_DIM)]).astype(np.float32),
        training_samples=np.array(2),
        training_episodes=np.array(1),
        instruction=np.array("open the fridge door"),
    )
    head = LearnedActionHead(checkpoint)
    np.testing.assert_allclose(head.predict(np.zeros(FEATURE_DIM)), EXPERT_ACTION)


def test_v4_runner_measures_contact_for_policy_actions(monkeypatch):
    class FakeRenderer:
        def __init__(self, model, height, width):
            self.shape = (height, width, 3)

        def update_scene(self, data, camera):
            pass

        def render(self):
            return np.zeros(self.shape, dtype=np.uint8)

        def close(self):
            pass

    class FakePolicy:
        devices = ["fake"]

        def __init__(self, octo_checkpoint, action_head, instruction):
            self.instruction = instruction
            self.count = 0
            self.stop_reason = None

        def predict(self, primary, wrist):
            if self.count >= 12:
                self.stop_reason = "task_complete"
                return None
            action = _expert_action(self.count)
            self.count += 1
            return action

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    result = run_v4_trial(decisions=40, policy_factory=FakePolicy)

    assert result["opened"]
    assert result["contact_steps"] > 0
    assert result["door_deg"] >= 30.0
    assert result["goal_satisfied"]
    assert result["policy_stop_reason"] == "task_complete"


def test_v4_intent_head_selects_between_two_feature_vectors(tmp_path: Path):
    checkpoint = tmp_path / "intent.npz"
    weights = np.zeros(FEATURE_DIM, dtype=np.float32)
    weights[0] = 1.0
    np.savez_compressed(
        checkpoint,
        format_version=np.array(1),
        weights=weights,
        bias=np.array(0.0),
        feature_mean=np.zeros(FEATURE_DIM, dtype=np.float32),
        feature_std=np.ones(FEATURE_DIM, dtype=np.float32),
        threshold=np.array(0.5),
    )
    intent = LearnedIntentHead(checkpoint)
    open_feature = np.zeros(FEATURE_DIM, dtype=np.float32)
    open_feature[0] = 0.8
    assert intent.requests_opening(open_feature)
    assert not intent.requests_opening(np.zeros(FEATURE_DIM, dtype=np.float32))


def test_v4_completion_head_selects_open_visual_feature(tmp_path: Path):
    checkpoint = tmp_path / "completion.npz"
    weights = np.zeros(FEATURE_DIM, dtype=np.float32)
    weights[0] = 1.0
    np.savez_compressed(
        checkpoint,
        format_version=np.array(1),
        weights=weights,
        bias=np.array(0.0),
        feature_mean=np.zeros(FEATURE_DIM, dtype=np.float32),
        feature_std=np.ones(FEATURE_DIM, dtype=np.float32),
        threshold=np.array(0.5),
    )
    completion = LearnedCompletionHead(checkpoint)
    open_feature = np.zeros(FEATURE_DIM, dtype=np.float32)
    open_feature[0] = 0.8
    assert completion.is_complete(open_feature)
    assert not completion.is_complete(np.zeros(FEATURE_DIM, dtype=np.float32))


def test_v4_runner_honors_model_stop_without_moving_the_door(monkeypatch):
    class FakeRenderer:
        def __init__(self, model, height, width):
            self.shape = (height, width, 3)

        def update_scene(self, data, camera):
            pass

        def render(self):
            return np.zeros(self.shape, dtype=np.uint8)

        def close(self):
            pass

    class StopPolicy:
        devices = ["fake"]

        def __init__(self, octo_checkpoint, action_head, instruction):
            self.stop_reason = "intent_stop"

        def predict(self, primary, wrist):
            return None

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    result = run_v4_trial(
        instruction="leave the fridge door closed",
        expected_outcome="stop",
        policy_factory=StopPolicy,
    )

    assert result["goal_satisfied"]
    assert result["policy_stopped"]
    assert result["decisions"] == 0
    assert result["door_deg"] == 0.0
    assert result["contact_steps"] == 0
