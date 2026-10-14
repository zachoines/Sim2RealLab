"""Tests for planner prompt builder."""

import json
import re

from strafer_autonomy.planner.prompt_builder import SYSTEM_PROMPT, build_messages
from strafer_autonomy.schemas import PlannerRequest


class TestBuildMessages:
    """Verify prompt assembly for each intent type."""

    def test_basic_command(self):
        req = PlannerRequest(
            request_id="test_001",
            raw_command="go to the door",
        )
        messages = build_messages(req)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert "go to the door" in messages[1]["content"]

    def test_includes_available_skills(self):
        req = PlannerRequest(
            request_id="test_002",
            raw_command="wait by the couch",
            available_skills=("navigate_to_pose", "wait", "cancel_mission"),
        )
        messages = build_messages(req)
        user_msg = messages[1]["content"]
        assert "navigate_to_pose" in user_msg
        assert "wait" in user_msg
        assert "cancel_mission" in user_msg
        assert "Available skills:" in user_msg

    def test_no_skills_no_suffix(self):
        req = PlannerRequest(
            request_id="test_003",
            raw_command="stop",
            available_skills=(),
        )
        messages = build_messages(req)
        assert "Available skills:" not in messages[1]["content"]

    def test_system_prompt_contains_allowed_intents(self):
        for intent in ("go_to_target", "wait_by_target", "cancel", "status"):
            assert intent in SYSTEM_PROMPT

    def test_system_prompt_requires_json(self):
        assert "JSON" in SYSTEM_PROMPT

    def test_whitespace_stripped(self):
        req = PlannerRequest(
            request_id="test_004",
            raw_command="  go to the chair  ",
        )
        messages = build_messages(req)
        assert messages[1]["content"].startswith("go to the chair")


def _translate_examples() -> list[tuple[str, list[float]]]:
    pairs = re.findall(r'User: "([^"]+)"\n(\{[^\n]*\})', SYSTEM_PROMPT)
    examples = [(cmd, json.loads(obj)) for cmd, obj in pairs]
    return [(cmd, obj["translation_xy"]) for cmd, obj in examples if obj["intent_type"] == "translate"]


class TestTranslateExamples:
    """The translate examples carry the [forward, left] sign convention."""

    def test_signs_follow_each_example_command(self):
        for cmd, (dx, dy) in _translate_examples():
            words = set(cmd.split())
            if "right" in words:
                assert dy < 0, cmd
            if "left" in words:
                assert dy > 0, cmd
            if words & {"back", "backward"}:
                assert dx < 0, cmd
            if "forward" in words:
                assert dx > 0, cmd

    def test_two_axis_examples_cover_both_sides_and_backward(self):
        two_axis = [(dx, dy) for _, (dx, dy) in _translate_examples() if dx != 0 and dy != 0]
        assert any(dy < 0 for _, dy in two_axis)
        assert any(dy > 0 for _, dy in two_axis)
        assert any(dx < 0 for dx, _ in two_axis)
