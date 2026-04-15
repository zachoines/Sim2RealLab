"""Tests for planner plan compiler."""

import pytest

from strafer_autonomy.executor.mission_runner import (
    DEFAULT_AVAILABLE_SKILLS,
    MissionRunner,
)
from strafer_autonomy.planner.plan_compiler import CompilationError, compile_plan
from strafer_autonomy.schemas import MissionIntent

# The planner side is allowed to reference skills the Jetson-side executor
# has not yet wired into ``DEFAULT_AVAILABLE_SKILLS``. Tests validate plans
# against the union so the compiler's assumptions are enforceable before
# the executor-side handlers land.
_PLANNED_NEW_SKILLS = (
    "verify_arrival",
    "rotate_by_degrees",
    "orient_to_direction",
    "query_environment",
)
_EXTENDED_AVAILABLE_SKILLS = tuple(
    dict.fromkeys((*DEFAULT_AVAILABLE_SKILLS, *_PLANNED_NEW_SKILLS))
)


def _make_intent(intent_type: str, **kwargs) -> MissionIntent:
    defaults = {
        "raw_command": "test command",
        "target_label": None,
        "wait_mode": None,
        "requires_grounding": False,
    }
    defaults.update(kwargs)
    return MissionIntent(intent_type=intent_type, **defaults)


def _validate(plan):
    runner = MissionRunner.__new__(MissionRunner)
    runner._config = type("C", (), {"available_skills": _EXTENDED_AVAILABLE_SKILLS})()
    return runner._validate_plan(plan)


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
            "scan_for_target",
            "project_detection_to_goal_pose",
            "navigate_to_pose",
            "verify_arrival",
        ]

    def test_verify_arrival_target_label(self):
        intent = _make_intent("go_to_target", target_label="chair", requires_grounding=True)
        plan = compile_plan(intent)
        verify = [s for s in plan.steps if s.skill == "verify_arrival"][0]
        assert verify.args["target_label"] == "chair"
        assert verify.args["goal_radius_m"] == 3.0
        assert verify.args["top_k"] == 5
        assert verify.args["majority"] == 3
        assert verify.args["fallback_on_empty_map"] == "pass"

    def test_target_label_in_scan(self):
        intent = _make_intent("go_to_target", target_label="table", requires_grounding=True)
        plan = compile_plan(intent)
        scan_step = [s for s in plan.steps if s.skill == "scan_for_target"][0]
        assert scan_step.args["label"] == "table"

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
        intent = _make_intent(
            "wait_by_target",
            target_label="couch",
            wait_mode="until_next_command",
            requires_grounding=True,
        )
        plan = compile_plan(intent)
        assert len(plan.steps) == 5

    def test_ends_with_wait(self):
        intent = _make_intent(
            "wait_by_target",
            target_label="couch",
            wait_mode="until_next_command",
            requires_grounding=True,
        )
        plan = compile_plan(intent)
        assert plan.steps[-1].skill == "wait"
        assert plan.steps[-1].args["mode"] == "until_next_command"

    def test_verify_arrival_before_wait(self):
        intent = _make_intent(
            "wait_by_target",
            target_label="couch",
            wait_mode="until_next_command",
            requires_grounding=True,
        )
        plan = compile_plan(intent)
        skills = [s.skill for s in plan.steps]
        verify_idx = skills.index("verify_arrival")
        wait_idx = skills.index("wait")
        assert verify_idx < wait_idx
        assert skills.index("navigate_to_pose") < verify_idx

    def test_unique_step_ids(self):
        intent = _make_intent(
            "wait_by_target",
            target_label="couch",
            wait_mode="until_next_command",
            requires_grounding=True,
        )
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


class TestRotate:
    def test_numeric_dispatches_to_rotate_by_degrees(self):
        intent = _make_intent("rotate", orientation_mode="90")
        plan = compile_plan(intent)
        assert len(plan.steps) == 1
        assert plan.steps[0].skill == "rotate_by_degrees"
        assert plan.steps[0].args["degrees"] == 90.0

    def test_negative_degrees(self):
        intent = _make_intent("rotate", orientation_mode="-45")
        plan = compile_plan(intent)
        assert plan.steps[0].args["degrees"] == -45.0

    def test_cardinal_dispatches_to_orient_to_direction(self):
        intent = _make_intent("rotate", orientation_mode="north")
        plan = compile_plan(intent)
        assert len(plan.steps) == 1
        assert plan.steps[0].skill == "orient_to_direction"
        assert plan.steps[0].args["direction"] == "north"

    def test_empty_orientation_falls_back_to_north(self):
        intent = _make_intent("rotate", orientation_mode=None)
        plan = compile_plan(intent)
        assert plan.steps[0].skill == "orient_to_direction"
        assert plan.steps[0].args["direction"] == "north"


class TestGoToTargets:
    def _intent(self, targets):
        return _make_intent(
            "go_to_targets",
            targets=tuple(targets),
            requires_grounding=True,
        )

    def test_four_steps_per_target(self):
        intent = self._intent([{"label": "cup"}, {"label": "door"}])
        plan = compile_plan(intent)
        assert len(plan.steps) == 8
        skills = [s.skill for s in plan.steps]
        assert skills[:4] == [
            "scan_for_target",
            "project_detection_to_goal_pose",
            "navigate_to_pose",
            "verify_arrival",
        ]
        assert skills[4:] == skills[:4]

    def test_targets_labels_wired_through(self):
        intent = self._intent([{"label": "cup"}, {"label": "door"}])
        plan = compile_plan(intent)
        scans = [s for s in plan.steps if s.skill == "scan_for_target"]
        assert [s.args["label"] for s in scans] == ["cup", "door"]
        verifies = [s for s in plan.steps if s.skill == "verify_arrival"]
        assert [s.args["target_label"] for s in verifies] == ["cup", "door"]

    def test_custom_standoff(self):
        intent = self._intent(
            [{"label": "cup", "standoff_m": 0.4}, {"label": "door", "standoff_m": 0.9}],
        )
        plan = compile_plan(intent)
        proj = [s for s in plan.steps if s.skill == "project_detection_to_goal_pose"]
        assert proj[0].args["standoff_m"] == pytest.approx(0.4)
        assert proj[1].args["standoff_m"] == pytest.approx(0.9)

    def test_unique_step_ids(self):
        intent = self._intent([{"label": "cup"}, {"label": "door"}, {"label": "chair"}])
        plan = compile_plan(intent)
        ids = [s.step_id for s in plan.steps]
        assert len(ids) == len(set(ids))
        assert ids[0] == "step_01"
        assert ids[-1] == "step_12"

    def test_empty_targets_raises(self):
        intent = _make_intent("go_to_targets", targets=())
        with pytest.raises(CompilationError):
            compile_plan(intent)


class TestDescribe:
    def test_single_describe_scene_step(self):
        intent = _make_intent("describe", raw_command="what do you see")
        plan = compile_plan(intent)
        assert len(plan.steps) == 1
        assert plan.steps[0].skill == "describe_scene"
        assert plan.steps[0].args["prompt"] == "what do you see"


class TestQuery:
    def test_single_query_environment_step(self):
        intent = _make_intent("query", raw_command="where is the chair")
        plan = compile_plan(intent)
        assert len(plan.steps) == 1
        assert plan.steps[0].skill == "query_environment"
        assert plan.steps[0].args["query"] == "where is the chair"


class TestPatrol:
    def test_four_steps_per_waypoint(self):
        intent = _make_intent(
            "patrol",
            targets=({"label": "kitchen"}, {"label": "living room"}),
            requires_grounding=True,
        )
        plan = compile_plan(intent)
        assert len(plan.steps) == 8
        assert all(
            s.skill in {
                "scan_for_target",
                "project_detection_to_goal_pose",
                "navigate_to_pose",
                "verify_arrival",
            }
            for s in plan.steps
        )

    def test_verify_arrival_per_waypoint(self):
        intent = _make_intent(
            "patrol",
            targets=({"label": "kitchen"}, {"label": "living room"}),
            requires_grounding=True,
        )
        plan = compile_plan(intent)
        verifies = [s for s in plan.steps if s.skill == "verify_arrival"]
        assert len(verifies) == 2
        assert [v.args["target_label"] for v in verifies] == ["kitchen", "living room"]

    def test_empty_targets_raises(self):
        intent = _make_intent("patrol", targets=())
        with pytest.raises(CompilationError):
            compile_plan(intent)


class TestUnsupported:
    def test_unknown_intent_raises(self):
        intent = _make_intent("fly_away")
        with pytest.raises(CompilationError, match="Unsupported intent_type"):
            compile_plan(intent)


class TestValidation:
    """Verify compiled plans pass the executor's _validate_plan against the
    extended skill set the Jetson executor will eventually wire in.
    """

    @pytest.mark.parametrize(
        "intent_type,kwargs",
        [
            ("go_to_target", {"target_label": "door", "requires_grounding": True}),
            (
                "wait_by_target",
                {
                    "target_label": "couch",
                    "wait_mode": "until_next_command",
                    "requires_grounding": True,
                },
            ),
            ("cancel", {}),
            ("status", {}),
            ("rotate", {"orientation_mode": "90"}),
            ("rotate", {"orientation_mode": "north"}),
            (
                "go_to_targets",
                {
                    "targets": ({"label": "cup"}, {"label": "door"}),
                    "requires_grounding": True,
                },
            ),
            ("describe", {}),
            ("query", {}),
            (
                "patrol",
                {
                    "targets": ({"label": "kitchen"}, {"label": "living room"}),
                    "requires_grounding": True,
                },
            ),
        ],
    )
    def test_all_plans_pass_validation(self, intent_type, kwargs):
        intent = _make_intent(intent_type, **kwargs)
        plan = compile_plan(intent)
        errors = _validate(plan)
        assert errors == [], f"Validation errors: {errors}"

    def test_all_skills_in_extended_set(self):
        allowed = set(_EXTENDED_AVAILABLE_SKILLS)
        cases = [
            ("go_to_target", {"target_label": "x", "requires_grounding": True}),
            (
                "wait_by_target",
                {
                    "target_label": "x",
                    "wait_mode": "until_next_command",
                    "requires_grounding": True,
                },
            ),
            ("cancel", {}),
            ("status", {}),
            ("rotate", {"orientation_mode": "90"}),
            ("rotate", {"orientation_mode": "east"}),
            (
                "go_to_targets",
                {"targets": ({"label": "a"},), "requires_grounding": True},
            ),
            ("describe", {}),
            ("query", {}),
            (
                "patrol",
                {"targets": ({"label": "a"},), "requires_grounding": True},
            ),
        ]
        for intent_type, kwargs in cases:
            intent = _make_intent(intent_type, **kwargs)
            plan = compile_plan(intent)
            for step in plan.steps:
                assert step.skill in allowed, (
                    f"Skill '{step.skill}' not in extended available skills"
                )
