"""Tests for planner plan compiler."""

import pytest

from strafer_autonomy.executor.mission_runner import DEFAULT_AVAILABLE_SKILLS, MissionRunner
from strafer_autonomy.planner.plan_compiler import CompilationError, compile_plan
from strafer_autonomy.schemas import MissionIntent


def _make_intent(intent_type: str, **kwargs) -> MissionIntent:
    defaults = {
        "raw_command": "test command",
        "target_label": None,
        "wait_mode": None,
        "requires_grounding": False,
    }
    defaults.update(kwargs)
    return MissionIntent(intent_type=intent_type, **defaults)


class TestGoToTarget:
    def test_step_count(self):
        intent = _make_intent("go_to_target", target_label="door", requires_grounding=True)
        plan = compile_plan(intent)
        assert len(plan.steps) == 4

    def test_skill_sequence(self):
        intent = _make_intent("go_to_target", target_label="chair", requires_grounding=True)
        plan = compile_plan(intent)
        skills = [s.skill for s in plan.steps]
        assert skills == [
            "capture_scene_observation",
            "locate_semantic_target",
            "project_detection_to_goal_pose",
            "navigate_to_pose",
        ]

    def test_target_label_in_locate(self):
        intent = _make_intent("go_to_target", target_label="table", requires_grounding=True)
        plan = compile_plan(intent)
        locate_step = [s for s in plan.steps if s.skill == "locate_semantic_target"][0]
        assert locate_step.args["label"] == "table"

    def test_navigation_backend(self):
        intent = _make_intent("go_to_target", target_label="door", requires_grounding=True)
        plan = compile_plan(intent)
        nav_step = [s for s in plan.steps if s.skill == "navigate_to_pose"][0]
        assert nav_step.args["execution_backend"] == "nav2"
        assert nav_step.args["goal_source"] == "projected_target"

    def test_mission_type(self):
        intent = _make_intent("go_to_target", target_label="door", requires_grounding=True)
        plan = compile_plan(intent)
        assert plan.mission_type == "go_to_target"

    def test_unique_step_ids(self):
        intent = _make_intent("go_to_target", target_label="door", requires_grounding=True)
        plan = compile_plan(intent)
        ids = [s.step_id for s in plan.steps]
        assert len(ids) == len(set(ids))


class TestWaitByTarget:
    def test_step_count(self):
        intent = _make_intent("wait_by_target", target_label="couch", wait_mode="until_next_command", requires_grounding=True)
        plan = compile_plan(intent)
        assert len(plan.steps) == 5

    def test_ends_with_wait(self):
        intent = _make_intent("wait_by_target", target_label="couch", wait_mode="until_next_command", requires_grounding=True)
        plan = compile_plan(intent)
        assert plan.steps[-1].skill == "wait"
        assert plan.steps[-1].args["mode"] == "until_next_command"

    def test_no_orient_skill(self):
        intent = _make_intent("wait_by_target", target_label="couch", wait_mode="until_next_command", requires_grounding=True)
        plan = compile_plan(intent)
        skills = [s.skill for s in plan.steps]
        assert "orient_relative_to_target" not in skills

    def test_unique_step_ids(self):
        intent = _make_intent("wait_by_target", target_label="couch", wait_mode="until_next_command", requires_grounding=True)
        plan = compile_plan(intent)
        ids = [s.step_id for s in plan.steps]
        assert len(ids) == len(set(ids))


class TestCancel:
    def test_single_step(self):
        intent = _make_intent("cancel")
        plan = compile_plan(intent)
        assert len(plan.steps) == 1
        assert plan.steps[0].skill == "cancel_mission"

    def test_mission_type(self):
        intent = _make_intent("cancel")
        plan = compile_plan(intent)
        assert plan.mission_type == "cancel"


class TestStatus:
    def test_single_step(self):
        intent = _make_intent("status")
        plan = compile_plan(intent)
        assert len(plan.steps) == 1
        assert plan.steps[0].skill == "report_status"


class TestUnsupported:
    def test_unknown_intent_raises(self):
        intent = _make_intent("fly_away")
        with pytest.raises(CompilationError, match="Unsupported intent_type"):
            compile_plan(intent)


class TestValidation:
    """Verify that compiled plans pass the executor's _validate_plan check."""

    def _validate(self, plan):
        """Use the real MissionRunner validator against DEFAULT_AVAILABLE_SKILLS."""
        from unittest.mock import MagicMock

        runner = MissionRunner.__new__(MissionRunner)
        runner._config = type("C", (), {"available_skills": DEFAULT_AVAILABLE_SKILLS})()
        return runner._validate_plan(plan)

    @pytest.mark.parametrize("intent_type,kwargs", [
        ("go_to_target", {"target_label": "door", "requires_grounding": True}),
        ("wait_by_target", {"target_label": "couch", "wait_mode": "until_next_command", "requires_grounding": True}),
        ("cancel", {}),
        ("status", {}),
    ])
    def test_all_plans_pass_validation(self, intent_type, kwargs):
        intent = _make_intent(intent_type, **kwargs)
        plan = compile_plan(intent)
        errors = self._validate(plan)
        assert errors == [], f"Validation errors: {errors}"

    def test_all_skills_in_available(self):
        """Every skill used in any plan template must be in DEFAULT_AVAILABLE_SKILLS."""
        allowed = set(DEFAULT_AVAILABLE_SKILLS)
        for intent_type, kwargs in [
            ("go_to_target", {"target_label": "x", "requires_grounding": True}),
            ("wait_by_target", {"target_label": "x", "wait_mode": "until_next_command", "requires_grounding": True}),
            ("cancel", {}),
            ("status", {}),
        ]:
            intent = _make_intent(intent_type, **kwargs)
            plan = compile_plan(intent)
            for step in plan.steps:
                assert step.skill in allowed, f"Skill '{step.skill}' not in DEFAULT_AVAILABLE_SKILLS"
