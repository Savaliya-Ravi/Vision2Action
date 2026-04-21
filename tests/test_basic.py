from vision2action.main import parse_instruction


def test_parse_instruction_color_found():
    assert parse_instruction("please go to blue door") == "blue"


def test_parse_instruction_none_for_unknown_color():
    assert parse_instruction("go to yellow") is None
