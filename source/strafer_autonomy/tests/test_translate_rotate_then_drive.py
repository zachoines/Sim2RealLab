"""Executor decomposes diagonal ``translate`` into rotate-then-drive.

MPPI strafe is noisy on the mecanum chassis at the velocity envelope sim
missions explore, so the executor biases body-frame *diagonal* translates
toward "face the goal, then drive forward":

- Diagonal (both ``dx_m`` and ``dy_m`` outside ``translate_cardinal_epsilon_m``):
  sets the Nav2 goal yaw to the goal-bearing and, when the heading delta
  exceeds ``translate_heading_threshold_rad``, prepends ``rotate_in_place``
  to align the chassis first.
- Cardinal (one of ``dx_m`` / ``dy_m`` within epsilon of zero): preserves
  the operator's body-frame command verbatim — "strafe left 1 m" still
  strafes sideways without re-orienting, goal yaw stays at the robot's
  current yaw.

These tests pin the gating, the goal-yaw rewrite, and the failure /
cancel propagation from the pre-rotation to the translate step.
"""

from __future__ import annotations

import math
from threading import Event
from unittest.mock import MagicMock

import pytest

from strafer_autonomy.executor.mission_runner import (
    MissionRunner,
    MissionRunnerConfig,
    _MissionRuntime,
)
from strafer_autonomy.schemas import Pose3D, SkillCall, SkillResult


def _yaw_from_quat(qz: float, qw: float) -> float:
    return math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)


def _quat_from_yaw(yaw: float) -> tuple[float, float]:
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _make_runner(
    config: MissionRunnerConfig | None = None,
    *,
    current_pose: dict[str, float] | None = None,
):
    runner = MissionRunner(
        planner_client=MagicMock(),
        grounding_client=MagicMock(),
        ros_client=MagicMock(),
        config=config or MissionRunnerConfig(),
    )
    ros = runner._ros_client
    ros.get_map_pose.return_value = current_pose or {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
    }
    ros.navigate_to_pose.return_value = SkillResult(
        step_id="t1", skill="translate", status="succeeded",
    )
    ros.rotate_in_place.return_value = SkillResult(
        step_id="t1:pre_rotate", skill="rotate_in_place", status="succeeded",
    )
    return runner, ros


def _make_runtime() -> _MissionRuntime:
    return _MissionRuntime(
        mission_id="m1", request_id="r1", raw_command="t",
        source="t", started_at=0.0,
    )


# ---------------------------------------------------------------------------
# Cardinal bypass — operator's body-frame strafe is preserved verbatim.
# ---------------------------------------------------------------------------


class TestCardinalBypass:
    def test_pure_forward_does_not_rotate(self):
        runner, ros = _make_runner()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 0.0},
        )
        runner._translate(_make_runtime(), step)
        ros.rotate_in_place.assert_not_called()

    def test_pure_backward_does_not_rotate(self):
        runner, ros = _make_runner()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": -1.0, "dy_m": 0.0},
        )
        runner._translate(_make_runtime(), step)
        ros.rotate_in_place.assert_not_called()

    def test_pure_strafe_left_does_not_rotate(self):
        runner, ros = _make_runner()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 0.0, "dy_m": 1.0},
        )
        runner._translate(_make_runtime(), step)
        ros.rotate_in_place.assert_not_called()

    def test_pure_strafe_right_does_not_rotate(self):
        runner, ros = _make_runner()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 0.0, "dy_m": -1.0},
        )
        runner._translate(_make_runtime(), step)
        ros.rotate_in_place.assert_not_called()

    def test_cardinal_preserves_current_yaw_on_goal(self):
        """Operator-issued strafe right 1m: goal yaw must equal current yaw."""
        # Robot at (5, 5) facing 30° in map.
        qz, qw = _quat_from_yaw(math.radians(30))
        runner, ros = _make_runner(current_pose={
            "x": 5.0, "y": 5.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": qz, "qw": qw,
        })
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 0.0, "dy_m": -1.0},  # strafe right 1 m
        )
        runner._translate(_make_runtime(), step)
        goal: Pose3D = ros.navigate_to_pose.call_args.kwargs["goal_pose"]
        assert _yaw_from_quat(goal.qz, goal.qw) == pytest.approx(math.radians(30))

    def test_zero_distance_bypasses(self):
        """No-op translate must not call rotate_in_place."""
        runner, ros = _make_runner()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 0.0, "dy_m": 0.0},
        )
        runner._translate(_make_runtime(), step)
        ros.rotate_in_place.assert_not_called()


# ---------------------------------------------------------------------------
# Diagonal decomposition — pre-rotate above threshold, always rewrite goal yaw.
# ---------------------------------------------------------------------------


class TestDiagonalDecomposition:
    def test_45_degree_triggers_pre_rotation(self):
        """dx=1, dy=1 at 0° yaw → 45° bearing → 45° delta > 17° threshold."""
        runner, ros = _make_runner()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 1.0},
        )
        runner._translate(_make_runtime(), step)
        ros.rotate_in_place.assert_called_once()
        kwargs = ros.rotate_in_place.call_args.kwargs
        assert kwargs["step_id"] == "t1:pre_rotate"
        assert kwargs["yaw_delta_rad"] == pytest.approx(math.pi / 4)

    def test_diagonal_sets_goal_yaw_to_bearing(self):
        """Goal pose yaw equals the map-frame bearing from robot to goal."""
        runner, ros = _make_runner(current_pose={
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        })
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 2.0, "dy_m": 1.0},
        )
        runner._translate(_make_runtime(), step)
        goal: Pose3D = ros.navigate_to_pose.call_args.kwargs["goal_pose"]
        bearing = math.atan2(1.0, 2.0)
        assert _yaw_from_quat(goal.qz, goal.qw) == pytest.approx(bearing)

    def test_diagonal_below_threshold_skips_rotation_but_rewrites_yaw(self):
        """Small heading delta (5°) is below threshold — rewrite yaw, skip rotate."""
        runner, ros = _make_runner()
        # 5° bearing requires |dy| / |dx| = tan(5°) ≈ 0.0875.
        # Use dx=1, dy=0.0875 — within threshold, but not cardinal.
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 0.0875},
        )
        runner._translate(_make_runtime(), step)
        ros.rotate_in_place.assert_not_called()
        goal: Pose3D = ros.navigate_to_pose.call_args.kwargs["goal_pose"]
        assert _yaw_from_quat(goal.qz, goal.qw) == pytest.approx(
            math.atan2(0.0875, 1.0)
        )

    def test_diagonal_at_yawed_pose_rotates_into_map_correctly(self):
        """Robot facing +y, dx=1 dy=1 → goal bearing is 135° in map."""
        qz, qw = _quat_from_yaw(math.pi / 2)
        runner, ros = _make_runner(current_pose={
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": qz, "qw": qw,
        })
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 1.0},
        )
        runner._translate(_make_runtime(), step)
        # Body (1, 1) at yaw 90° → map (-1, 1).
        goal: Pose3D = ros.navigate_to_pose.call_args.kwargs["goal_pose"]
        assert goal.x == pytest.approx(-1.0)
        assert goal.y == pytest.approx(1.0)
        # Bearing in map: atan2(1, -1) = 135°.
        bearing = math.atan2(1.0, -1.0)
        assert _yaw_from_quat(goal.qz, goal.qw) == pytest.approx(bearing)
        # Heading delta from current 90° to bearing 135° = 45°.
        ros.rotate_in_place.assert_called_once()
        assert ros.rotate_in_place.call_args.kwargs["yaw_delta_rad"] == pytest.approx(
            math.pi / 4
        )

    def test_threshold_override_disables_rotation(self):
        """Raising threshold above the heading delta suppresses pre-rotation."""
        runner, ros = _make_runner(
            MissionRunnerConfig(translate_heading_threshold_rad=math.pi),
        )
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 1.0},
        )
        runner._translate(_make_runtime(), step)
        ros.rotate_in_place.assert_not_called()

    def test_pre_rotation_passes_cancel_event(self):
        runner, ros = _make_runner()
        runtime = _make_runtime()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 1.0},
        )
        runner._translate(runtime, step)
        kwargs = ros.rotate_in_place.call_args.kwargs
        assert kwargs["cancel_event"] is runtime.cancel_event

    def test_pre_rotation_uses_angular_timeout_budget(self):
        """Pre-rotation budget tracks the heading delta, not the linear distance."""
        runner, ros = _make_runner()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 1.0},
        )
        runner._translate(_make_runtime(), step)
        rotate_timeout = ros.rotate_in_place.call_args.kwargs["timeout_s"]
        nav_timeout = ros.navigate_to_pose.call_args.kwargs["timeout_s"]
        # Rotation timeout (~5s for 45° at 0.5 rad/s) shorter than the nav
        # timeout (~9s for hypot(1,1) ≈ 1.41 m at NAV_LINEAR_VEL).
        assert 0.0 < rotate_timeout < nav_timeout


# ---------------------------------------------------------------------------
# Pre-rotation failure / cancel propagation.
# ---------------------------------------------------------------------------


class TestPreRotationPropagation:
    def test_failed_rotation_propagates_as_translate_failure(self):
        runner, ros = _make_runner()
        ros.rotate_in_place.return_value = SkillResult(
            step_id="t1:pre_rotate", skill="rotate_in_place", status="failed",
            error_code="rotation_failed", message="motor stalled",
        )
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 1.0},
        )
        result = runner._translate(_make_runtime(), step)
        assert result.status == "failed"
        assert result.error_code == "rotation_failed"
        # ``navigate_to_pose`` must not run after the pre-rotation failed.
        ros.navigate_to_pose.assert_not_called()

    def test_canceled_rotation_propagates_cancel(self):
        runner, ros = _make_runner()
        ros.rotate_in_place.return_value = SkillResult(
            step_id="t1:pre_rotate", skill="rotate_in_place", status="canceled",
            error_code="rotation_canceled", message="canceled by mission",
        )
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 1.0},
        )
        result = runner._translate(_make_runtime(), step)
        assert result.status == "canceled"
        assert result.error_code == "rotation_canceled"
        ros.navigate_to_pose.assert_not_called()

    def test_rotation_exception_returns_rotation_failed(self):
        runner, ros = _make_runner()
        ros.rotate_in_place.side_effect = RuntimeError("ros down")
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 1.0},
        )
        result = runner._translate(_make_runtime(), step)
        assert result.status == "failed"
        assert result.error_code == "rotation_failed"
        ros.navigate_to_pose.assert_not_called()

    def test_pre_set_cancel_event_aborts_translate(self):
        """A cancel that arrives before _translate runs still propagates."""
        runner, ros = _make_runner()
        runtime = _make_runtime()
        runtime.cancel_event.set()
        ros.rotate_in_place.return_value = SkillResult(
            step_id="t1:pre_rotate", skill="rotate_in_place", status="canceled",
            error_code="rotation_canceled", message="canceled by mission",
        )
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 1.0},
        )
        result = runner._translate(runtime, step)
        assert result.status == "canceled"
        ros.navigate_to_pose.assert_not_called()


# ---------------------------------------------------------------------------
# Acceptance scenario from the brief.
# ---------------------------------------------------------------------------


class TestBriefAcceptance:
    def test_offset_45deg_at_2m_rotates_then_drives_forward(self):
        """Brief AC: a 45°-offset 2m goal rotates to face, then drives forward.

        Robot at origin facing east issues body-frame ``translate(1.41, 1.41)``
        (= 2 m at 45° body-frame). Executor must (a) call rotate_in_place to
        align with the 45° bearing, (b) dispatch a Nav2 goal whose yaw equals
        the goal-bearing — once the chassis is aligned, MPPI's commanded vy
        stays small because the path is along the body x-axis.
        """
        runner, ros = _make_runner()
        d = 2.0 * math.cos(math.pi / 4)
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": d, "dy_m": d},
        )
        runner._translate(_make_runtime(), step)
        # (a) pre-rotation aligned chassis with bearing.
        ros.rotate_in_place.assert_called_once()
        rot_kwargs = ros.rotate_in_place.call_args.kwargs
        assert rot_kwargs["yaw_delta_rad"] == pytest.approx(math.pi / 4)
        # (b) Nav2 goal yaw equals bearing so MPPI drives forward into the goal.
        goal: Pose3D = ros.navigate_to_pose.call_args.kwargs["goal_pose"]
        assert _yaw_from_quat(goal.qz, goal.qw) == pytest.approx(math.pi / 4)
        # Goal pose is the expected 2 m at 45°.
        assert math.hypot(goal.x, goal.y) == pytest.approx(2.0)
