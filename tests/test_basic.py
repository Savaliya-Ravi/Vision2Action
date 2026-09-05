from pathlib import Path

import mujoco

from vision2action.env.randomization import randomize_scene_objects
from vision2action.targets import parse_target_instruction


def test_parse_instruction_object_found():
    query = parse_target_instruction("please go to the bottle")
    assert query is not None
    assert query.category == "bottle"


def test_parse_instruction_none_for_unknown_color():
    assert parse_target_instruction("go to yellow") is None


def test_randomization_is_seeded_and_keeps_targets_separated():
    scene = Path("vision2action/env/scene.xml").resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    first = randomize_scene_objects(model, mujoco.MjData(model), seed=42)
    second = randomize_scene_objects(model, mujoco.MjData(model), seed=42)
    assert first == second
    assert len(first) == 5


def test_banana_is_placed_on_tabletop():
    scene = Path("vision2action/env/scene.xml").resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    poses = randomize_scene_objects(model, data, seed=7)
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    banana_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_banana")
    table_xy = data.xpos[table_id][:2]
    banana_xyz = data.xpos[banana_id]
    assert abs(float(banana_xyz[0] - table_xy[0])) <= 0.54
    assert abs(float(banana_xyz[1] - table_xy[1])) <= 0.29
    assert abs(float(banana_xyz[2]) - 0.795) < 1e-6
