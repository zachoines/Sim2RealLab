"""Tests for the costmap-aware staging path in `_navigate_to_pose`.

Covers:
- ``clamp_goal_to_costmap_bounds`` geometry across a set of representative
  target positions (inside, just-outside, far-outside, diagonal).
- ``CostmapBounds`` query helper, including the not-yet-available state.
- The mission runner's no-regression behavior for in-costmap goals.
- The staging loop's budget exhaustion path.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from strafer_autonomy.clients.ros_client import (
    CostmapBounds,
    JetsonRosClient,
    RosClientConfig,
)
from strafer_autonomy.executor.mission_runner import (
    MissionRunner,
    MissionRunnerConfig,
    _MissionRuntime,
    clamp_goal_to_costmap_bounds,
)
from strafer_autonomy.schemas import (
    GoalPoseCandidate,
    GroundingResult,
    Pose3D,
    SceneObservation,
    SkillCall,
    SkillResult,
)


# ----------------------------------------------------------------------
# clamp_goal_to_costmap_bounds — pure geometry
# ----------------------------------------------------------------------


def _bounds(min_x=-5.0, min_y=-5.0, max_x=5.0, max_y=5.0, resolution=0.05) -> CostmapBounds:
    return CostmapBounds(
        min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, resolution=resolution,
    )


class TestClampGoalToCostmapBounds:

    def test_goal_inside_returns_unchanged(self):
        clamped, was_clamped = clamp_goal_to_costmap_bounds(
            robot_xy=(0.0, 0.0),
            goal_xy=(2.0, 1.5),
            bounds=_bounds(),
            footprint_radius_m=0.3,
        )
        assert was_clamped is False
        assert clamped == (2.0, 1.5)

    def test_goal_just_outside_clamped_to_safe_edge(self):
        # Bounds [-5, 5] in both axes; safe rect [-4.7, 4.7] with margin 0.3.
        clamped, was_clamped = clamp_goal_to_costmap_bounds(
            robot_xy=(0.0, 0.0),
            goal_xy=(5.5, 0.0),
            bounds=_bounds(),
            footprint_radius_m=0.3,
        )
        assert was_clamped is True
        assert clamped[0] == pytest.approx(4.7, abs=1e-6)
        assert clamped[1] == pytest.approx(0.0, abs=1e-6)

    def test_goal_far_outside_clamped_to_intersection(self):
        # Goal at (50, 50), robot at origin → ray y=x; first hits x=4.7 at y=4.7.
        clamped, was_clamped = clamp_goal_to_costmap_bounds(
            robot_xy=(0.0, 0.0),
            goal_xy=(50.0, 50.0),
            bounds=_bounds(),
            footprint_radius_m=0.3,
        )
        assert was_clamped is True
        assert clamped[0] == pytest.approx(4.7, abs=1e-6)
        assert clamped[1] == pytest.approx(4.7, abs=1e-6)

    def test_goal_diagonal_outside_clamps_to_first_boundary(self):
        # Goal far in +x but only mildly in +y → clamps on x first, with
        # the proportional y component preserved.
        clamped, was_clamped = clamp_goal_to_costmap_bounds(
            robot_xy=(0.0, 0.0),
            goal_xy=(10.0, 2.0),
            bounds=_bounds(),
            footprint_radius_m=0.3,
        )
        assert was_clamped is True
        # t_max = 4.7 / 10.0 = 0.47, so y = 0.47 * 2.0 = 0.94.
        assert clamped[0] == pytest.approx(4.7, abs=1e-6)
        assert clamped[1] == pytest.approx(0.94, abs=1e-6)

    def test_goal_outside_with_offset_robot(self):
        # Robot at (3, 3); goal at (10, 10). The +x edge (4.7) and +y edge
        # (4.7) are both 1.7 m away, so t_max = 1.7 / 7.0 ≈ 0.2429.
        clamped, was_clamped = clamp_goal_to_costmap_bounds(
            robot_xy=(3.0, 3.0),
            goal_xy=(10.0, 10.0),
            bounds=_bounds(),
            footprint_radius_m=0.3,
        )
        assert was_clamped is True
        assert clamped[0] == pytest.approx(4.7, abs=1e-6)
        assert clamped[1] == pytest.approx(4.7, abs=1e-6)

    def test_negative_axis_clamps_to_min_edge(self):
        clamped, was_clamped = clamp_goal_to_costmap_bounds(
            robot_xy=(0.0, 0.0),
            goal_xy=(-10.0, -2.0),
            bounds=_bounds(),
            footprint_radius_m=0.3,
        )
        assert was_clamped is True
        # Safe -x edge at -4.7; t_max = 4.7 / 10.0 = 0.47, y = -0.94.
        assert clamped[0] == pytest.approx(-4.7, abs=1e-6)
        assert clamped[1] == pytest.approx(-0.94, abs=1e-6)

    def test_degenerate_safe_rect_returns_unchanged(self):
        # Footprint > half rect → safe rect is empty.
        clamped, was_clamped = clamp_goal_to_costmap_bounds(
            robot_xy=(0.0, 0.0),
            goal_xy=(10.0, 0.0),
            bounds=_bounds(min_x=-0.1, min_y=-0.1, max_x=0.1, max_y=0.1),
            footprint_radius_m=0.5,
        )
        assert was_clamped is False
        assert clamped == (10.0, 0.0)

    def test_robot_at_goal_returns_robot_position(self):
        clamped, was_clamped = clamp_goal_to_costmap_bounds(
            robot_xy=(10.0, 10.0),
            goal_xy=(10.0, 10.0),
            bounds=_bounds(),
            footprint_radius_m=0.3,
        )
        # Goal == robot, both outside — returns robot position with the
        # was-clamped flag set so the caller knows no progress is possible.
        assert was_clamped is True
        assert clamped == (10.0, 10.0)


# ----------------------------------------------------------------------
# CostmapBounds query helper
# ----------------------------------------------------------------------


def _make_client_for_bounds() -> JetsonRosClient:
    """Construct JetsonRosClient without starting rclpy (mirrors test_ros_client)."""
    client = object.__new__(JetsonRosClient)
    client._config = RosClientConfig()
    client._cache_lock = threading.Lock()
    client._latest_costmap = None
    return client


def _occupancy_grid(
    *, origin_x=-5.0, origin_y=-5.0, width=200, height=200, resolution=0.05,
) -> SimpleNamespace:
    info = SimpleNamespace(
        resolution=resolution,
        width=width,
        height=height,
        origin=SimpleNamespace(
            position=SimpleNamespace(x=origin_x, y=origin_y, z=0.0),
        ),
    )
    return SimpleNamespace(info=info, data=[0] * (width * height))


class TestGetGlobalCostmapBounds:

    def test_returns_none_when_costmap_not_received(self):
        client = _make_client_for_bounds()
        assert client.get_global_costmap_bounds() is None

    def test_returns_bounds_from_latest_costmap(self):
        client = _make_client_for_bounds()
        # 200×200 cells × 0.05 m → 10 m on each side, origin at (-5, -5).
        client._latest_costmap = _occupancy_grid()
        bounds = client.get_global_costmap_bounds()
        assert bounds is not None
        assert bounds.min_x == pytest.approx(-5.0)
        assert bounds.min_y == pytest.approx(-5.0)
        assert bounds.max_x == pytest.approx(5.0)
        assert bounds.max_y == pytest.approx(5.0)
        assert bounds.resolution == pytest.approx(0.05)

    def test_handles_offset_origin(self):
        client = _make_client_for_bounds()
        client._latest_costmap = _occupancy_grid(
            origin_x=2.0, origin_y=-3.0, width=100, height=80, resolution=0.1,
        )
        bounds = client.get_global_costmap_bounds()
        assert bounds.min_x == pytest.approx(2.0)
        assert bounds.min_y == pytest.approx(-3.0)
        assert bounds.max_x == pytest.approx(12.0)
        assert bounds.max_y == pytest.approx(5.0)


# ----------------------------------------------------------------------
# Mission runner — _navigate_via_staging
# ----------------------------------------------------------------------


def _make_observation(stamp: float = 1.0) -> SceneObservation:
    return SceneObservation(
        observation_id="obs",
        stamp_sec=stamp,
        color_image_bgr=np.zeros((360, 640, 3), dtype=np.uint8),
        aligned_depth_m=np.zeros((360, 640), dtype=np.float32),
        camera_frame="d555_color_optical_frame",
    )


def _make_grounding(found: bool, label: str = "door") -> GroundingResult:
    return GroundingResult(
        request_id="g",
        found=found,
        bbox_2d=(100, 200, 300, 400) if found else None,
        label=label if found else None,
        confidence=0.9 if found else None,
        latency_s=0.05,
    )


def _make_goal_candidate(x: float, y: float, label: str = "door") -> GoalPoseCandidate:
    return GoalPoseCandidate(
        request_id="proj",
        found=True,
        goal_frame="map",
        goal_pose=Pose3D(x=x, y=y, z=0.0, qw=1.0),
        target_pose=Pose3D(x=x, y=y, z=0.0, qw=1.0),
        standoff_m=0.7,
        depth_valid=True,
        quality_flags=(),
    )


def _make_runner(
    *, staging_budget: int = 4,
) -> tuple[MissionRunner, MagicMock, MagicMock, MagicMock]:
    planner = MagicMock()
    grounding = MagicMock()
    ros = MagicMock()
    config = MissionRunnerConfig(staging_budget=staging_budget)
    runner = MissionRunner(
        planner_client=planner,
        grounding_client=grounding,
        ros_client=ros,
        config=config,
    )
    return runner, planner, grounding, ros


def _make_runtime() -> _MissionRuntime:
    return _MissionRuntime(
        mission_id="m1",
        request_id="m1",
        raw_command="navigate to the door",
        source="test",
        started_at=time.time(),
    )


def _make_nav_step() -> SkillCall:
    return SkillCall(
        step_id="step_03",
        skill="navigate_to_pose",
        args={"goal_source": "projected_target", "execution_backend": "nav2"},
        timeout_s=90.0,
        retry_limit=0,
    )


def _success_nav_result(step_id: str = "step_03") -> SkillResult:
    return SkillResult(
        step_id=step_id,
        skill="navigate_to_pose",
        status="succeeded",
        outputs={},
        message="Navigation completed successfully.",
        started_at=0.0,
        finished_at=0.0,
    )


class TestNavigateViaStagingNoRegression:
    """In-costmap goal: behaves like pre-staging code, no extra Nav2 / VLM calls."""

    def test_in_costmap_goal_dispatches_single_nav(self):
        runner, _, grounding, ros = _make_runner()
        ros.get_global_costmap_bounds.return_value = _bounds()  # ±5 m
        ros.get_map_pose.return_value = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        }
        ros.navigate_to_pose.return_value = _success_nav_result()

        runtime = _make_runtime()
        runtime.latest_grounding = _make_grounding(found=True)
        runtime.latest_goal_pose = _make_goal_candidate(2.0, 1.0)
        result = runner._navigate_to_pose(runtime, _make_nav_step())

        assert result.status == "succeeded"
        assert ros.navigate_to_pose.call_count == 1
        # No re-grounding for in-costmap goals.
        grounding.locate_semantic_target.assert_not_called()
        ros.capture_scene_observation.assert_not_called()
        # No staging metadata added when no clamped legs ran.
        assert "staging_legs" not in result.outputs

    def test_no_costmap_dispatches_single_nav(self):
        runner, _, grounding, ros = _make_runner()
        ros.get_global_costmap_bounds.return_value = None  # not yet available
        ros.get_map_pose.return_value = None
        ros.navigate_to_pose.return_value = _success_nav_result()

        runtime = _make_runtime()
        runtime.latest_grounding = _make_grounding(found=True)
        runtime.latest_goal_pose = _make_goal_candidate(8.0, 8.0)
        result = runner._navigate_to_pose(runtime, _make_nav_step())

        assert result.status == "succeeded"
        assert ros.navigate_to_pose.call_count == 1
        grounding.locate_semantic_target.assert_not_called()


class TestNavigateViaStagingMultiLeg:
    """Off-costmap goal: at least one intermediate Nav2 goal + re-ground + re-project."""

    def test_off_costmap_triggers_intermediate_leg(self):
        runner, _, grounding, ros = _make_runner()
        # Bounds ±5; goal at (10, 0) is off-costmap on x.
        ros.get_global_costmap_bounds.return_value = _bounds()
        ros.get_map_pose.return_value = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        }
        ros.navigate_to_pose.return_value = _success_nav_result()

        # After the leg arrives, robot is at (4.7, 0) and re-grounding finds
        # the same target. Re-projection produces a new goal at (8, 0) —
        # still off-costmap relative to robot at 4.7? bounds ±5, so 8 > 4.7,
        # so still off. Then the next iteration: robot at 8 is outside
        # safe range; we'll stop short of that complication by having the
        # second projection land at (4.0, 0) — comfortably in-costmap.
        ros.capture_scene_observation.return_value = _make_observation()
        grounding.locate_semantic_target.return_value = _make_grounding(found=True)
        ros.project_detection_to_goal_pose.return_value = _make_goal_candidate(4.0, 0.0)

        runtime = _make_runtime()
        runtime.latest_grounding = _make_grounding(found=True)
        runtime.latest_goal_pose = _make_goal_candidate(10.0, 0.0)
        result = runner._navigate_to_pose(runtime, _make_nav_step())

        assert result.status == "succeeded"
        # One clamped leg + one final = 2 Nav2 dispatches.
        assert ros.navigate_to_pose.call_count == 2
        # Re-grounding fired exactly once (after the single clamped leg).
        assert grounding.locate_semantic_target.call_count == 1
        assert ros.project_detection_to_goal_pose.call_count == 1
        # Staging metadata records the leg.
        assert result.outputs.get("stages_used") == 1
        assert len(result.outputs.get("staging_legs", [])) == 1
        leg = result.outputs["staging_legs"][0]
        assert leg["stage"] == 1
        assert leg["leg_status"] == "succeeded"
        assert leg["regrounding_status"] == "ok"

    def test_off_costmap_two_legs_before_in_costmap(self):
        runner, _, grounding, ros = _make_runner()
        # Bounds expand each leg as SLAM observes more area — mirrors the
        # production behavior where each clamped leg moves the robot to
        # the SLAM horizon and lets RTAB-Map fill in the next ring of
        # cells before the next bounds query.
        ros.get_global_costmap_bounds.side_effect = [
            _bounds(min_x=-5.0, max_x=5.0, min_y=-5.0, max_y=5.0),
            _bounds(min_x=-10.0, max_x=10.0, min_y=-10.0, max_y=10.0),
            _bounds(min_x=-15.0, max_x=15.0, min_y=-15.0, max_y=15.0),
        ]
        ros.get_map_pose.side_effect = [
            {"x": 0.0, "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
            {"x": 4.7, "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
            {"x": 9.7, "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
        ]
        ros.navigate_to_pose.return_value = _success_nav_result()
        ros.capture_scene_observation.return_value = _make_observation()
        grounding.locate_semantic_target.return_value = _make_grounding(found=True)
        ros.project_detection_to_goal_pose.side_effect = [
            # After leg 1 (robot=4.7), bounds=±10 → still off-costmap.
            _make_goal_candidate(15.0, 0.0),
            # After leg 2 (robot=9.7), bounds=±15 → in-costmap on next iter.
            _make_goal_candidate(14.0, 0.0),
        ]

        runtime = _make_runtime()
        runtime.latest_grounding = _make_grounding(found=True)
        runtime.latest_goal_pose = _make_goal_candidate(15.0, 0.0)
        result = runner._navigate_to_pose(runtime, _make_nav_step())

        assert result.status == "succeeded"
        # 2 clamped staging legs + 1 final = 3 Nav2 dispatches.
        assert ros.navigate_to_pose.call_count == 3
        assert grounding.locate_semantic_target.call_count == 2
        assert ros.project_detection_to_goal_pose.call_count == 2
        assert result.outputs["stages_used"] == 2
        assert len(result.outputs["staging_legs"]) == 2
        for stage in result.outputs["staging_legs"]:
            assert stage["leg_status"] == "succeeded"
            assert stage["regrounding_status"] == "ok"


class TestNavigateViaStagingExhaustion:

    def test_budget_exhausted_returns_distinct_error(self):
        runner, _, grounding, ros = _make_runner(staging_budget=2)
        ros.get_global_costmap_bounds.return_value = _bounds()
        ros.get_map_pose.return_value = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        }
        ros.navigate_to_pose.return_value = _success_nav_result()
        ros.capture_scene_observation.return_value = _make_observation()
        grounding.locate_semantic_target.return_value = _make_grounding(found=True)
        # Re-projection always returns a still-off-costmap goal.
        ros.project_detection_to_goal_pose.return_value = _make_goal_candidate(20.0, 0.0)

        runtime = _make_runtime()
        runtime.latest_grounding = _make_grounding(found=True)
        runtime.latest_goal_pose = _make_goal_candidate(20.0, 0.0)
        result = runner._navigate_to_pose(runtime, _make_nav_step())

        assert result.status == "failed"
        assert result.error_code == "navigate_via_staging_exhausted"
        assert result.outputs["stages_used"] == 2
        assert len(result.outputs["staging_legs"]) == 2
        # Distinct from goal_projection_failed and navigation_timeout.
        assert result.error_code != "goal_projection_failed"
        assert result.error_code != "navigation_timeout"

    def test_leg_failure_aborts_with_staging_metadata(self):
        runner, _, grounding, ros = _make_runner()
        ros.get_global_costmap_bounds.return_value = _bounds()
        ros.get_map_pose.return_value = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        }
        # First leg fails.
        ros.navigate_to_pose.return_value = SkillResult(
            step_id="step_03:stage_1",
            skill="navigate_to_pose",
            status="failed",
            outputs={},
            error_code="navigation_failed",
            message="Nav2 reported failure.",
            started_at=0.0,
            finished_at=0.0,
        )

        runtime = _make_runtime()
        runtime.latest_grounding = _make_grounding(found=True)
        runtime.latest_goal_pose = _make_goal_candidate(20.0, 0.0)
        result = runner._navigate_to_pose(runtime, _make_nav_step())

        assert result.status == "failed"
        assert result.error_code == "navigation_failed"
        assert result.outputs["failed_stage"] == 1
        # Re-grounding never fired because the leg failed first.
        grounding.locate_semantic_target.assert_not_called()

    def test_target_lost_after_leg(self):
        runner, _, grounding, ros = _make_runner()
        ros.get_global_costmap_bounds.return_value = _bounds()
        ros.get_map_pose.return_value = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        }
        ros.navigate_to_pose.return_value = _success_nav_result()
        ros.capture_scene_observation.return_value = _make_observation()
        # Re-grounding fails to find the target.
        grounding.locate_semantic_target.return_value = _make_grounding(found=False)

        runtime = _make_runtime()
        runtime.latest_grounding = _make_grounding(found=True)
        runtime.latest_goal_pose = _make_goal_candidate(20.0, 0.0)
        result = runner._navigate_to_pose(runtime, _make_nav_step())

        assert result.status == "failed"
        assert result.error_code == "navigate_via_staging_target_lost"
        ros.project_detection_to_goal_pose.assert_not_called()
