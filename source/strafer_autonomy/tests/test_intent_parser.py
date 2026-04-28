"""Tests for planner intent parser."""

import pytest

from strafer_autonomy.planner.intent_parser import IntentParseError, parse_intent


class TestParseIntent:
    """Verify JSON extraction and validation from LLM output."""

    def test_go_to_target(self):
        raw = '{"intent_type": "go_to_target", "target_label": "door", "wait_mode": null, "requires_grounding": true}'
        intent = parse_intent(raw, "go to the door")
        assert intent.intent_type == "go_to_target"
        assert intent.target_label == "door"
        assert intent.requires_grounding is True
        assert intent.wait_mode is None

    def test_wait_by_target(self):
        raw = '{"intent_type": "wait_by_target", "target_label": "couch", "wait_mode": "until_next_command", "requires_grounding": true}'
        intent = parse_intent(raw, "wait by the couch")
        assert intent.intent_type == "wait_by_target"
        assert intent.target_label == "couch"
        assert intent.wait_mode == "until_next_command"

    def test_cancel(self):
        raw = '{"intent_type": "cancel", "target_label": null, "wait_mode": null, "requires_grounding": false}'
        intent = parse_intent(raw, "stop")
        assert intent.intent_type == "cancel"
        assert intent.target_label is None
        assert intent.requires_grounding is False

    def test_status(self):
        raw = '{"intent_type": "status", "target_label": null, "wait_mode": null, "requires_grounding": false}'
        intent = parse_intent(raw, "what are you doing")
        assert intent.intent_type == "status"

    def test_fenced_code_block(self):
        raw = 'Here is the result:\n```json\n{"intent_type": "cancel", "target_label": null, "wait_mode": null, "requires_grounding": false}\n```'
        intent = parse_intent(raw, "stop")
        assert intent.intent_type == "cancel"

    def test_fenced_block_without_json_tag(self):
        raw = '```\n{"intent_type": "status", "target_label": null, "wait_mode": null, "requires_grounding": false}\n```'
        intent = parse_intent(raw, "status")
        assert intent.intent_type == "status"

    def test_surrounding_prose(self):
        raw = 'I think the intent is: {"intent_type": "go_to_target", "target_label": "table", "wait_mode": null, "requires_grounding": true} That should work.'
        intent = parse_intent(raw, "go to the table")
        assert intent.intent_type == "go_to_target"
        assert intent.target_label == "table"

    def test_missing_json_raises(self):
        with pytest.raises(IntentParseError, match="No JSON object found"):
            parse_intent("I don't know what to do", "something")

    def test_invalid_json_raises(self):
        with pytest.raises(IntentParseError, match="Invalid JSON"):
            parse_intent("{intent_type: bad}", "something")

    def test_unknown_intent_type_raises(self):
        raw = '{"intent_type": "fly_away", "target_label": null, "wait_mode": null, "requires_grounding": false}'
        with pytest.raises(IntentParseError, match="Unknown or missing intent_type"):
            parse_intent(raw, "fly away")

    def test_missing_intent_type_raises(self):
        raw = '{"target_label": "door"}'
        with pytest.raises(IntentParseError, match="Unknown or missing intent_type"):
            parse_intent(raw, "go to the door")

    def test_go_to_target_missing_label_raises(self):
        raw = '{"intent_type": "go_to_target", "target_label": null, "wait_mode": null, "requires_grounding": true}'
        with pytest.raises(IntentParseError, match="requires a non-empty target_label"):
            parse_intent(raw, "go somewhere")

    def test_go_to_target_empty_label_raises(self):
        raw = '{"intent_type": "go_to_target", "target_label": "", "wait_mode": null, "requires_grounding": true}'
        with pytest.raises(IntentParseError, match="requires a non-empty target_label"):
            parse_intent(raw, "go somewhere")

    def test_raw_command_preserved(self):
        raw = '{"intent_type": "cancel", "target_label": null, "wait_mode": null, "requires_grounding": false}'
        intent = parse_intent(raw, "please stop now")
        assert intent.raw_command == "please stop now"


class TestRotateIntent:
    def test_numeric_orientation_mode(self):
        raw = '{"intent_type": "rotate", "orientation_mode": "90", "requires_grounding": false}'
        intent = parse_intent(raw, "turn 90 degrees")
        assert intent.intent_type == "rotate"
        assert intent.orientation_mode == "90"

    def test_cardinal_orientation_mode(self):
        raw = '{"intent_type": "rotate", "orientation_mode": "north", "requires_grounding": false}'
        intent = parse_intent(raw, "face north")
        assert intent.orientation_mode == "north"

    def test_rotate_without_orientation_mode_accepted(self):
        raw = '{"intent_type": "rotate", "requires_grounding": false}'
        intent = parse_intent(raw, "rotate")
        assert intent.intent_type == "rotate"
        assert intent.orientation_mode is None


class TestTranslateIntent:
    def test_forward_translation(self):
        raw = (
            '{"intent_type": "translate", '
            '"translation_xy": [1.0, 0.0], "requires_grounding": false}'
        )
        intent = parse_intent(raw, "move forward 1 meter")
        assert intent.intent_type == "translate"
        assert intent.translation_xy == (1.0, 0.0)
        assert intent.requires_grounding is False

    def test_backward_and_right(self):
        raw = (
            '{"intent_type": "translate", '
            '"translation_xy": [-0.5, -0.3], "requires_grounding": false}'
        )
        intent = parse_intent(raw, "back up and strafe right")
        assert intent.translation_xy == (-0.5, -0.3)

    def test_integer_values_coerce_to_float(self):
        raw = (
            '{"intent_type": "translate", '
            '"translation_xy": [2, 0], "requires_grounding": false}'
        )
        intent = parse_intent(raw, "move forward 2")
        assert intent.translation_xy == (2.0, 0.0)

    def test_missing_translation_rejected(self):
        import pytest
        from strafer_autonomy.planner.intent_parser import IntentParseError
        raw = '{"intent_type": "translate", "requires_grounding": false}'
        with pytest.raises(IntentParseError, match="translation_xy"):
            parse_intent(raw, "move forward")

    def test_wrong_length_rejected(self):
        import pytest
        from strafer_autonomy.planner.intent_parser import IntentParseError
        raw = (
            '{"intent_type": "translate", '
            '"translation_xy": [1.0], "requires_grounding": false}'
        )
        with pytest.raises(IntentParseError, match="translation_xy"):
            parse_intent(raw, "move forward")


class TestGoToTargetsIntent:
    def test_multiple_targets(self):
        raw = (
            '{"intent_type": "go_to_targets", '
            '"targets": [{"label": "cup"}, {"label": "door"}], '
            '"requires_grounding": true}'
        )
        intent = parse_intent(raw, "go to the cup then the door")
        assert intent.intent_type == "go_to_targets"
        assert intent.targets is not None
        assert len(intent.targets) == 2
        assert intent.targets[0]["label"] == "cup"
        assert intent.targets[1]["label"] == "door"

    def test_targets_with_standoff(self):
        raw = (
            '{"intent_type": "go_to_targets", '
            '"targets": [{"label": "cup", "standoff_m": 0.4}], '
            '"requires_grounding": true}'
        )
        intent = parse_intent(raw, "stop close to the cup")
        assert intent.targets[0]["standoff_m"] == 0.4

    def test_missing_targets_raises(self):
        raw = '{"intent_type": "go_to_targets", "requires_grounding": true}'
        with pytest.raises(IntentParseError, match="requires a non-empty 'targets' list"):
            parse_intent(raw, "go to things")

    def test_empty_targets_raises(self):
        raw = '{"intent_type": "go_to_targets", "targets": [], "requires_grounding": true}'
        with pytest.raises(IntentParseError, match="requires a non-empty 'targets' list"):
            parse_intent(raw, "go to things")

    def test_target_missing_label_raises(self):
        raw = (
            '{"intent_type": "go_to_targets", '
            '"targets": [{"standoff_m": 0.5}], "requires_grounding": true}'
        )
        with pytest.raises(IntentParseError, match="non-empty string 'label'"):
            parse_intent(raw, "go to things")

    def test_non_object_target_raises(self):
        raw = '{"intent_type": "go_to_targets", "targets": ["cup"], "requires_grounding": true}'
        with pytest.raises(IntentParseError, match="must be an object"):
            parse_intent(raw, "go to things")


class TestDescribeIntent:
    def test_describe(self):
        raw = '{"intent_type": "describe", "requires_grounding": false}'
        intent = parse_intent(raw, "what do you see")
        assert intent.intent_type == "describe"
        assert intent.target_label is None
        assert intent.targets is None


class TestQueryIntent:
    def test_query(self):
        raw = '{"intent_type": "query", "requires_grounding": false}'
        intent = parse_intent(raw, "where was the chair")
        assert intent.intent_type == "query"


class TestPatrolIntent:
    def test_patrol_with_targets(self):
        raw = (
            '{"intent_type": "patrol", '
            '"targets": [{"label": "kitchen"}, {"label": "living room"}], '
            '"requires_grounding": true}'
        )
        intent = parse_intent(raw, "patrol the rooms")
        assert intent.intent_type == "patrol"
        assert len(intent.targets) == 2

    def test_patrol_empty_targets_raises(self):
        raw = '{"intent_type": "patrol", "targets": [], "requires_grounding": true}'
        with pytest.raises(IntentParseError, match="requires a non-empty 'targets' list"):
            parse_intent(raw, "patrol")


class TestNestedJsonExtraction:
    def test_nested_targets_extracted(self):
        # The bare-JSON path must handle nested {...} objects inside lists.
        raw = (
            'Here: {"intent_type": "go_to_targets", '
            '"targets": [{"label": "cup"}, {"label": "door"}], '
            '"requires_grounding": true} thanks.'
        )
        intent = parse_intent(raw, "cmd")
        assert intent.intent_type == "go_to_targets"
        assert len(intent.targets) == 2
