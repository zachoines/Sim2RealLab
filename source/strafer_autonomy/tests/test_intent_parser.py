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
