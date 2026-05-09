"""Progress-aware motion timeouts (executor side) + Nav2 stall watchdog.

Covers the behavior introduced by ``progress-aware-nav-timeouts``:

- ``compute_motion_budget_s`` (pure function): formula, ceiling cap,
  setup-overhead floor.
- ``MissionRunner._motion_timeout_s``: explicit override wins, progress-
  aware path uses NAV_LINEAR_VEL / NAV_ANGULAR_VEL, escape hatch falls
  back to ``default_navigation_timeout_s``.
- ``_translate`` / ``_rotate_by_degrees`` / ``_orient_to_direction`` /
  ``_dispatch_nav_goal`` end-to-end: each motion handler computes a
  per-step budget and forwards it to the ROS client.
- Stall watchdog stub kwargs flow from MissionRunner to ros_client.
- ``_ProgressTracker`` standalone unit: stall detection on synthetic
  feedback ticks.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from strafer_autonomy.clients.ros_client import _ProgressTracker
from strafer_autonomy.executor.mission_runner import (
    MissionRunner,
    MissionRunnerConfig,
    _MissionRuntime,
    compute_motion_budget_s,
)
from strafer_autonomy.schemas import (
    GoalPoseCandidate,
    Pose3D,
    SkillCall,
    SkillResult,
)
from strafer_shared.constants import NAV_ANGULAR_VEL, NAV_LINEAR_VEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(config: MissionRunnerConfig | None = None):
    runner = MissionRunner(
        planner_client=MagicMock(),
        grounding_client=MagicMock(),
        ros_client=MagicMock(),
        config=config or MissionRunnerConfig(),
    )
    ros = runner._ros_client
    ros.navigate_to_pose.return_value = SkillResult(
        step_id="x", skill="navigate_to_pose", status="succeeded",
    )
    ros.rotate_in_place.return_value = SkillResult(
        step_id="x", skill="rotate_in_place", status="succeeded",
    )
    ros.get_map_pose.return_value = {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
    }
    return runner, ros


def _make_runtime() -> _MissionRuntime:
    return _MissionRuntime(
        mission_id="m1", request_id="r1", raw_command="t",
        source="t", started_at=0.0,
    )


def _expected_budget(
    *, magnitude: float, nominal: float,
    safety_factor: float = 2.0, setup_overhead_s: float = 5.0,
    ceiling_s: float = 90.0,
) -> float:
    return min(
        ceiling_s,
        max(setup_overhead_s, abs(magnitude) / nominal * safety_factor + setup_overhead_s),
    )


# ---------------------------------------------------------------------------
# compute_motion_budget_s (pure function)
# ---------------------------------------------------------------------------


class TestComputeMotionBudget:
    def test_short_distance_just_above_setup_overhead(self):
        b = compute_motion_budget_s(
            magnitude=0.5, nominal_speed=NAV_LINEAR_VEL,
            safety_factor=2.0, setup_overhead_s=5.0, ceiling_s=90.0,
        )
        # 0.5 / 0.78 * 2 + 5 ≈ 6.28 s
        assert 5.5 < b < 7.5

    def test_long_distance_grows_with_magnitude(self):
        short = compute_motion_budget_s(
            magnitude=0.5, nominal_speed=NAV_LINEAR_VEL,
            safety_factor=2.0, setup_overhead_s=5.0, ceiling_s=90.0,
        )
        long = compute_motion_budget_s(
            magnitude=10.0, nominal_speed=NAV_LINEAR_VEL,
            safety_factor=2.0, setup_overhead_s=5.0, ceiling_s=90.0,
        )
        assert long > short
        # 10 / 0.78 * 2 + 5 ≈ 30.6 s
        assert 25.0 < long < 35.0

    def test_capped_at_ceiling(self):
        b = compute_motion_budget_s(
            magnitude=1000.0, nominal_speed=NAV_LINEAR_VEL,
            safety_factor=2.0, setup_overhead_s=5.0, ceiling_s=90.0,
        )
        assert b == 90.0

    def test_setup_overhead_floor_applies(self):
        b = compute_motion_budget_s(
            magnitude=0.0, nominal_speed=NAV_LINEAR_VEL,
            safety_factor=2.0, setup_overhead_s=5.0, ceiling_s=90.0,
        )
        assert b == 5.0

    def test_zero_speed_falls_through_to_ceiling(self):
        b = compute_motion_budget_s(
            magnitude=10.0, nominal_speed=0.0,
            safety_factor=2.0, setup_overhead_s=5.0, ceiling_s=90.0,
        )
        assert b == 90.0

    def test_negative_magnitude_treated_as_unsigned(self):
        a = compute_motion_budget_s(
            magnitude=-3.0, nominal_speed=NAV_LINEAR_VEL,
            safety_factor=2.0, setup_overhead_s=5.0, ceiling_s=90.0,
        )
        b = compute_motion_budget_s(
            magnitude=3.0, nominal_speed=NAV_LINEAR_VEL,
            safety_factor=2.0, setup_overhead_s=5.0, ceiling_s=90.0,
        )
        assert a == b


# ---------------------------------------------------------------------------
# _motion_timeout_s — resolution order
# ---------------------------------------------------------------------------


class TestMotionTimeoutResolution:
    def test_explicit_step_timeout_wins(self):
        runner, _ = _make_runner()
        step = SkillCall(skill="translate", step_id="t", timeout_s=42.5)
        assert runner._motion_timeout_s(step=step, magnitude=10.0, kind="linear") == 42.5

    def test_progress_aware_path_uses_linear_speed(self):
        runner, _ = _make_runner()
        step = SkillCall(skill="translate", step_id="t", timeout_s=None)
        got = runner._motion_timeout_s(step=step, magnitude=3.0, kind="linear")
        expected = _expected_budget(magnitude=3.0, nominal=NAV_LINEAR_VEL)
        assert got == pytest.approx(expected)

    def test_progress_aware_path_uses_angular_speed(self):
        runner, _ = _make_runner()
        step = SkillCall(skill="rotate_by_degrees", step_id="r", timeout_s=None)
        got = runner._motion_timeout_s(step=step, magnitude=math.pi, kind="angular")
        expected = _expected_budget(magnitude=math.pi, nominal=NAV_ANGULAR_VEL)
        assert got == pytest.approx(expected)

    def test_escape_hatch_returns_default_navigation_timeout(self):
        runner, _ = _make_runner(MissionRunnerConfig(nav_progress_aware=False))
        step = SkillCall(skill="translate", step_id="t", timeout_s=None)
        # Magnitude doesn't matter when progress-aware is off.
        got = runner._motion_timeout_s(step=step, magnitude=10.0, kind="linear")
        assert got == MissionRunnerConfig().default_navigation_timeout_s

    def test_unknown_kind_raises(self):
        runner, _ = _make_runner()
        step = SkillCall(skill="x", step_id="x", timeout_s=None)
        with pytest.raises(ValueError):
            runner._motion_timeout_s(step=step, magnitude=1.0, kind="bogus")


# ---------------------------------------------------------------------------
# Per-handler integration: translate / rotate_by_degrees / orient_to_direction
# ---------------------------------------------------------------------------


class TestTranslateProgressAware:
    def test_short_translate_gets_tight_budget(self):
        runner, ros = _make_runner()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 0.5, "dy_m": 0.0}, timeout_s=None,
        )
        runner._translate(_make_runtime(), step)
        got = ros.navigate_to_pose.call_args.kwargs["timeout_s"]
        expected = _expected_budget(magnitude=0.5, nominal=NAV_LINEAR_VEL)
        assert got == pytest.approx(expected)

    def test_long_translate_gets_larger_budget(self):
        runner, ros = _make_runner()
        for dx, label in [(0.5, "short"), (10.0, "long")]:
            step = SkillCall(
                skill="translate", step_id=f"t_{label}",
                args={"dx_m": dx, "dy_m": 0.0}, timeout_s=None,
            )
            runner._translate(_make_runtime(), step)
        # Two calls; second (10 m) timeout > first (0.5 m).
        first_to = ros.navigate_to_pose.call_args_list[0].kwargs["timeout_s"]
        second_to = ros.navigate_to_pose.call_args_list[1].kwargs["timeout_s"]
        assert second_to > first_to

    def test_translate_diagonal_uses_hypot(self):
        runner, ros = _make_runner()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 3.0, "dy_m": 4.0}, timeout_s=None,
        )
        runner._translate(_make_runtime(), step)
        got = ros.navigate_to_pose.call_args.kwargs["timeout_s"]
        expected = _expected_budget(magnitude=5.0, nominal=NAV_LINEAR_VEL)  # hypot(3,4)=5
        assert got == pytest.approx(expected)

    def test_translate_capped_at_env_knob_ceiling(self):
        runner, ros = _make_runner(
            MissionRunnerConfig(default_navigation_timeout_s=10.0),
        )
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 100.0, "dy_m": 0.0}, timeout_s=None,
        )
        runner._translate(_make_runtime(), step)
        # 100m / 0.78 * 2 + 5 ≈ 261s, but ceiling is 10.
        assert ros.navigate_to_pose.call_args.kwargs["timeout_s"] == 10.0

    def test_translate_passes_stall_watchdog_kwargs(self):
        runner, ros = _make_runner()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 0.0}, timeout_s=None,
        )
        runner._translate(_make_runtime(), step)
        kwargs = ros.navigate_to_pose.call_args.kwargs
        assert kwargs["stall_progress_m"] == MissionRunnerConfig().nav_stall_progress_m
        assert kwargs["stall_window_s"] == MissionRunnerConfig().nav_stall_window_s

    def test_translate_omits_stall_kwargs_in_legacy_mode(self):
        runner, ros = _make_runner(MissionRunnerConfig(nav_progress_aware=False))
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 0.0}, timeout_s=None,
        )
        runner._translate(_make_runtime(), step)
        kwargs = ros.navigate_to_pose.call_args.kwargs
        assert "stall_progress_m" not in kwargs
        assert "stall_window_s" not in kwargs


class TestRotateByDegreesProgressAware:
    @pytest.mark.parametrize("degrees", [30.0, 90.0, 180.0, 720.0])
    def test_budget_scales_with_angle(self, degrees):
        runner, ros = _make_runner()
        step = SkillCall(
            skill="rotate_by_degrees", step_id="r1",
            args={"degrees": degrees}, timeout_s=None,
        )
        runner._rotate_by_degrees(_make_runtime(), step)
        got = ros.rotate_in_place.call_args.kwargs["timeout_s"]
        expected = _expected_budget(
            magnitude=math.radians(degrees), nominal=NAV_ANGULAR_VEL,
        )
        assert got == pytest.approx(expected)

    def test_negative_degrees_uses_unsigned_magnitude(self):
        runner, ros = _make_runner()
        step_neg = SkillCall(
            skill="rotate_by_degrees", step_id="r-",
            args={"degrees": -90.0}, timeout_s=None,
        )
        step_pos = SkillCall(
            skill="rotate_by_degrees", step_id="r+",
            args={"degrees": 90.0}, timeout_s=None,
        )
        runner._rotate_by_degrees(_make_runtime(), step_neg)
        neg_to = ros.rotate_in_place.call_args.kwargs["timeout_s"]
        runner._rotate_by_degrees(_make_runtime(), step_pos)
        pos_to = ros.rotate_in_place.call_args.kwargs["timeout_s"]
        assert neg_to == pos_to


class TestOrientToDirectionProgressAware:
    def test_east_to_west_uses_pi_rad_delta(self):
        runner, ros = _make_runner()
        ros.get_robot_state.return_value = {"pose": {"qz": 0.0, "qw": 1.0}}  # east
        step = SkillCall(
            skill="orient_to_direction", step_id="o1",
            args={"direction": "west"}, timeout_s=None,
        )
        runner._orient_to_direction(_make_runtime(), step)
        got = ros.rotate_in_place.call_args.kwargs["timeout_s"]
        expected = _expected_budget(magnitude=math.pi, nominal=NAV_ANGULAR_VEL)
        assert got == pytest.approx(expected)

    def test_east_to_north_uses_half_pi_delta(self):
        runner, ros = _make_runner()
        ros.get_robot_state.return_value = {"pose": {"qz": 0.0, "qw": 1.0}}  # east
        step = SkillCall(
            skill="orient_to_direction", step_id="o1",
            args={"direction": "north"}, timeout_s=None,
        )
        runner._orient_to_direction(_make_runtime(), step)
        got = ros.rotate_in_place.call_args.kwargs["timeout_s"]
        expected = _expected_budget(
            magnitude=math.pi / 2, nominal=NAV_ANGULAR_VEL,
        )
        assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _dispatch_nav_goal — straight-line distance budget
# ---------------------------------------------------------------------------


class TestDispatchNavGoalProgressAware:
    def test_uses_robot_to_goal_straight_line(self):
        runner, ros = _make_runner()
        ros.get_map_pose.return_value = {
            "x": 1.0, "y": 1.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        }
        ros.navigate_to_pose.return_value = SkillResult(
            step_id="step_03", skill="navigate_to_pose", status="succeeded",
        )
        step = SkillCall(
            step_id="step_03", skill="navigate_to_pose",
            args={"goal_source": "explicit", "execution_backend": "nav2"},
            timeout_s=None,
        )
        # Goal at (4, 5) from robot at (1, 1) → distance = hypot(3, 4) = 5
        runner._dispatch_nav_goal(step, Pose3D(x=4.0, y=5.0), 0.0)
        got = ros.navigate_to_pose.call_args.kwargs["timeout_s"]
        expected = _expected_budget(magnitude=5.0, nominal=NAV_LINEAR_VEL)
        assert got == pytest.approx(expected)

    def test_falls_back_to_setup_overhead_when_pose_unavailable(self):
        runner, ros = _make_runner()
        ros.get_map_pose.return_value = None
        ros.navigate_to_pose.return_value = SkillResult(
            step_id="step_03", skill="navigate_to_pose", status="succeeded",
        )
        step = SkillCall(
            step_id="step_03", skill="navigate_to_pose",
            args={"goal_source": "explicit"}, timeout_s=None,
        )
        runner._dispatch_nav_goal(step, Pose3D(x=10.0, y=10.0), 0.0)
        # Distance unknown → magnitude=0 → budget = setup_overhead_s = 5.0
        assert ros.navigate_to_pose.call_args.kwargs["timeout_s"] == 5.0

    def test_passes_stall_watchdog_kwargs(self):
        runner, ros = _make_runner()
        ros.navigate_to_pose.return_value = SkillResult(
            step_id="x", skill="navigate_to_pose", status="succeeded",
        )
        step = SkillCall(
            step_id="x", skill="navigate_to_pose",
            args={"goal_source": "explicit"}, timeout_s=None,
        )
        runner._dispatch_nav_goal(step, Pose3D(x=2.0, y=1.0), 0.0)
        kwargs = ros.navigate_to_pose.call_args.kwargs
        assert kwargs["stall_progress_m"] == MissionRunnerConfig().nav_stall_progress_m
        assert kwargs["stall_window_s"] == MissionRunnerConfig().nav_stall_window_s


# ---------------------------------------------------------------------------
# _ProgressTracker — pure unit
# ---------------------------------------------------------------------------


class TestProgressTracker:
    def test_empty_tracker_not_stalled(self):
        t = _ProgressTracker(stall_progress_m=0.1, stall_window_s=20.0)
        assert not t.is_stalled()

    def test_insufficient_history_not_stalled(self):
        t = _ProgressTracker(stall_progress_m=0.1, stall_window_s=20.0)
        # Only 5 s of history, window is 20 s.
        for i in range(6):
            t.record(sim_time_s=float(i), distance_remaining_m=10.0 - 0.05 * i)
        assert not t.is_stalled()

    def test_steady_progress_not_stalled(self):
        t = _ProgressTracker(stall_progress_m=0.1, stall_window_s=10.0)
        # Distance drops 1 m per second over 15 s — far more than 0.1 m/window.
        for i in range(16):
            t.record(sim_time_s=float(i), distance_remaining_m=20.0 - float(i))
        assert not t.is_stalled()

    def test_flat_distance_over_window_is_stalled(self):
        t = _ProgressTracker(stall_progress_m=0.1, stall_window_s=10.0)
        # Distance pinned at 5.0 for 15 s.
        for i in range(16):
            t.record(sim_time_s=float(i), distance_remaining_m=5.0)
        assert t.is_stalled()

    def test_progress_then_plateau_eventually_stalls(self):
        t = _ProgressTracker(stall_progress_m=0.1, stall_window_s=10.0)
        # First 5 s of fast progress, then 15 s plateau.
        for i in range(6):
            t.record(sim_time_s=float(i), distance_remaining_m=10.0 - float(i))
        # Plateau at 5.0 from t=5 to t=20.
        for i in range(6, 21):
            t.record(sim_time_s=float(i), distance_remaining_m=5.0)
        assert t.is_stalled()

    def test_at_or_above_threshold_not_stalled(self):
        # Exact 1.0 m progress against a 1.0 m threshold (binary-exact values
        # to dodge IEEE-754 boundary surprises). Condition is
        # ``< stall_progress_m``, so equality is NOT stalled.
        t = _ProgressTracker(stall_progress_m=1.0, stall_window_s=10.0)
        t.record(sim_time_s=0.0, distance_remaining_m=10.0)
        t.record(sim_time_s=11.0, distance_remaining_m=9.0)
        assert not t.is_stalled()

    def test_below_threshold_is_stalled(self):
        t = _ProgressTracker(stall_progress_m=1.0, stall_window_s=10.0)
        t.record(sim_time_s=0.0, distance_remaining_m=10.0)
        t.record(sim_time_s=11.0, distance_remaining_m=9.5)  # 0.5 m < 1.0 m threshold
        assert t.is_stalled()

    def test_invalid_positive_args_rejected(self):
        with pytest.raises(ValueError):
            _ProgressTracker(stall_progress_m=0.0, stall_window_s=10.0)
        with pytest.raises(ValueError):
            _ProgressTracker(stall_progress_m=0.1, stall_window_s=0.0)
        with pytest.raises(ValueError):
            _ProgressTracker(stall_progress_m=-1.0, stall_window_s=10.0)


# ---------------------------------------------------------------------------
# Cross-cut: explicit step.timeout_s overrides everything
# ---------------------------------------------------------------------------


class TestExplicitTimeoutOverride:
    """A non-zero ``step.timeout_s`` (from a future advanced compiler or a
    test) bypasses progress-aware computation entirely."""

    def test_explicit_translate_override(self):
        runner, ros = _make_runner()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 100.0, "dy_m": 0.0}, timeout_s=42.0,
        )
        runner._translate(_make_runtime(), step)
        assert ros.navigate_to_pose.call_args.kwargs["timeout_s"] == 42.0

    def test_explicit_rotate_override(self):
        runner, ros = _make_runner()
        step = SkillCall(
            skill="rotate_by_degrees", step_id="r1",
            args={"degrees": 720.0}, timeout_s=42.0,
        )
        runner._rotate_by_degrees(_make_runtime(), step)
        assert ros.rotate_in_place.call_args.kwargs["timeout_s"] == 42.0
