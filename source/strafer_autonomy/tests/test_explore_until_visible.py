"""Tests for the explore_until_visible executor skill handler."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from strafer_autonomy.clients.ros_client import CostmapBounds, CostmapSnapshot
from strafer_autonomy.executor.frontier import DEFAULT_OCCUPIED_THRESHOLD
from strafer_autonomy.executor.mission_runner import (
    MissionRunner,
    _MissionRuntime,
)
from strafer_autonomy.schemas import (
    GroundingResult,
    SceneObservation,
    SkillCall,
    SkillResult,
)


def _make_observation() -> SceneObservation:
    return SceneObservation(
        observation_id="obs_1",
        stamp_sec=1.0,
        color_image_bgr=np.zeros((64, 64, 3), dtype=np.uint8),
        aligned_depth_m=np.zeros((64, 64), dtype=np.float32),
        camera_frame="d555_color_optical_frame",
    )


def _make_grounding(found: bool, label: str = "door") -> GroundingResult:
    return GroundingResult(
        request_id="test",
        found=found,
        bbox_2d=(100, 200, 300, 400) if found else None,
        label=label if found else None,
        confidence=0.9 if found else None,
        latency_s=0.05,
    )


def _make_nav_succeeded(step_id: str) -> SkillResult:
    return SkillResult(
        step_id=step_id,
        skill="navigate_to_pose",
        status="succeeded",
        outputs={},
        started_at=0.0,
        finished_at=0.0,
    )


def _make_nav_failed(step_id: str) -> SkillResult:
    return SkillResult(
        step_id=step_id,
        skill="navigate_to_pose",
        status="failed",
        outputs={},
        error_code="navigation_failed",
        message="Nav2 failed.",
        started_at=0.0,
        finished_at=0.0,
    )


def _half_unknown_snapshot(*, width: int = 30, height: int = 30) -> CostmapSnapshot:
    """A 30x30 costmap with the right half unknown — one large frontier
    along the vertical midline."""
    data = np.zeros((height, width), dtype=np.int8)
    data[:, width // 2:] = -1
    bounds = CostmapBounds(
        min_x=0.0, min_y=0.0,
        max_x=width * 0.1, max_y=height * 0.1,
        resolution=0.1,
    )
    return CostmapSnapshot(bounds=bounds, width=width, height=height, data=data)


def _all_free_snapshot() -> CostmapSnapshot:
    data = np.zeros((20, 20), dtype=np.int8)
    bounds = CostmapBounds(
        min_x=0.0, min_y=0.0, max_x=2.0, max_y=2.0, resolution=0.1,
    )
    return CostmapSnapshot(bounds=bounds, width=20, height=20, data=data)


def _make_runner() -> tuple[MissionRunner, MagicMock, MagicMock, MagicMock]:
    planner = MagicMock()
    grounding = MagicMock()
    ros = MagicMock()
    ros.get_map_pose.return_value = {"x": 0.0, "y": 1.5, "qz": 0.0, "qw": 1.0}
    runner = MissionRunner(
        planner_client=planner,
        grounding_client=grounding,
        ros_client=ros,
    )
    return runner, planner, grounding, ros


def _make_runtime(mission_id: str = "m1") -> _MissionRuntime:
    return _MissionRuntime(
        mission_id=mission_id,
        request_id=mission_id,
        raw_command="find the door",
        source="test",
        started_at=0.0,
    )


def _make_step(**kwargs) -> SkillCall:
    args = {
        "label": "door",
        "max_frontiers": 3,
        "max_distance_m": 10.0,
        "timeout_s": 30.0,
        "min_frontier_cells": 4,
        "max_scan_steps": 2,
        "scan_arc_deg": 360,
    }
    args.update(kwargs)
    return SkillCall(
        step_id="step_explore",
        skill="explore_until_visible",
        args=args,
        timeout_s=60.0,
        retry_limit=0,
    )


class TestExploreUntilVisible:
    def test_missing_label_returns_invalid_args(self):
        runner, _, _, _ = _make_runner()
        runtime = _make_runtime()
        step = _make_step(label="")
        result = runner._explore_until_visible(runtime, step)
        assert result.status == "failed"
        assert result.error_code == "invalid_args"

    def test_no_costmap_yet(self):
        runner, _, _, ros = _make_runner()
        ros.get_global_costmap_snapshot.return_value = None
        runtime = _make_runtime()
        result = runner._explore_until_visible(runtime, _make_step())
        assert result.status == "failed"
        assert result.error_code == "no_costmap"

    def test_no_robot_pose(self):
        runner, _, _, ros = _make_runner()
        ros.get_global_costmap_snapshot.return_value = _half_unknown_snapshot()
        ros.get_map_pose.return_value = None
        runtime = _make_runtime()
        result = runner._explore_until_visible(runtime, _make_step())
        assert result.status == "failed"
        assert result.error_code == "no_robot_pose"

    def test_no_frontiers_returns_exhausted(self):
        runner, _, _, ros = _make_runner()
        # All-free map → no unknown cells → no frontiers.
        ros.get_global_costmap_snapshot.return_value = _all_free_snapshot()
        runtime = _make_runtime()
        result = runner._explore_until_visible(runtime, _make_step())
        assert result.status == "failed"
        assert result.error_code == "frontier_set_exhausted"
        assert result.outputs["reason"] == "no_candidates"

    def test_finds_target_on_first_frontier(self):
        runner, _, grounding, ros = _make_runner()
        ros.get_global_costmap_snapshot.return_value = _half_unknown_snapshot()
        ros.capture_scene_observation.return_value = _make_observation()
        ros.navigate_to_pose.return_value = _make_nav_succeeded("nav_0")
        ros.rotate_in_place.return_value = SkillResult(
            step_id="rot", skill="rotate_in_place", status="succeeded",
            outputs={}, started_at=0.0, finished_at=0.0,
        )
        # First scan attempt at the arrival pose grounds the target.
        grounding.locate_semantic_target.return_value = _make_grounding(
            found=True, label="door",
        )

        runtime = _make_runtime()
        result = runner._explore_until_visible(runtime, _make_step())

        assert result.status == "succeeded"
        assert result.outputs["frontier_index"] == 0
        assert result.outputs["found"] is True
        assert len(result.outputs["attempts"]) == 1
        assert ros.navigate_to_pose.call_count == 1

    def test_advances_to_next_frontier_when_scan_fails(self):
        runner, _, grounding, ros = _make_runner()
        ros.get_global_costmap_snapshot.return_value = _half_unknown_snapshot()
        ros.capture_scene_observation.return_value = _make_observation()
        ros.navigate_to_pose.return_value = _make_nav_succeeded("nav")
        ros.rotate_in_place.return_value = SkillResult(
            step_id="rot", skill="rotate_in_place", status="succeeded",
            outputs={}, started_at=0.0, finished_at=0.0,
        )
        # Scan never grounds — every locate_semantic_target call returns
        # found=False. The skill should exhaust max_frontiers attempts
        # without erroring.
        grounding.locate_semantic_target.return_value = _make_grounding(found=False)
        grounding.describe_scene.side_effect = ConnectionError("no describe")

        runtime = _make_runtime()
        # With a single frontier in this map, the ranking de-dupes the
        # cluster after each attempt → second iteration finds nothing
        # new. Use max_frontiers=3 to confirm the loop terminates with
        # frontier_set_exhausted/reason=no_candidates not max_frontiers.
        result = runner._explore_until_visible(runtime, _make_step(max_frontiers=3))
        assert result.status == "failed"
        assert result.error_code == "frontier_set_exhausted"
        # The attempt list records the first frontier visit.
        assert len(result.outputs["attempts"]) >= 1

    def test_max_frontiers_budget(self):
        """With multiple distinct frontiers and never-grounding scans,
        the loop terminates after max_frontiers attempts."""
        runner, _, grounding, ros = _make_runner()

        # Costmap with multiple separate frontier strips.
        height, width = 40, 40
        data = np.zeros((height, width), dtype=np.int8)
        # Three vertical bands of unknown, each isolated by a wall.
        data[:, 8:12] = -1
        data[:, 20:24] = -1
        data[:, 32:36] = -1
        # Walls bracketing each unknown band so frontier clusters don't merge.
        data[:, 12] = DEFAULT_OCCUPIED_THRESHOLD
        data[:, 24] = DEFAULT_OCCUPIED_THRESHOLD
        snapshot = CostmapSnapshot(
            bounds=CostmapBounds(
                min_x=0.0, min_y=0.0, max_x=width * 0.1, max_y=height * 0.1,
                resolution=0.1,
            ),
            width=width, height=height, data=data,
        )
        ros.get_global_costmap_snapshot.return_value = snapshot
        ros.capture_scene_observation.return_value = _make_observation()
        ros.navigate_to_pose.return_value = _make_nav_succeeded("nav")
        ros.rotate_in_place.return_value = SkillResult(
            step_id="rot", skill="rotate_in_place", status="succeeded",
            outputs={}, started_at=0.0, finished_at=0.0,
        )
        grounding.locate_semantic_target.return_value = _make_grounding(found=False)
        grounding.describe_scene.side_effect = ConnectionError("no describe")

        runtime = _make_runtime()
        result = runner._explore_until_visible(runtime, _make_step(max_frontiers=2))

        assert result.status == "failed"
        assert result.error_code == "frontier_set_exhausted"
        assert result.outputs["reason"] == "max_frontiers"
        assert len(result.outputs["attempts"]) == 2

    def test_nav_failure_skips_to_next_frontier(self):
        runner, _, grounding, ros = _make_runner()

        # Two distinct frontier bands.
        height, width = 30, 30
        data = np.zeros((height, width), dtype=np.int8)
        data[:, 6:10] = -1
        data[:, 20:24] = -1
        data[:, 14] = DEFAULT_OCCUPIED_THRESHOLD
        snapshot = CostmapSnapshot(
            bounds=CostmapBounds(
                min_x=0.0, min_y=0.0, max_x=width * 0.1, max_y=height * 0.1,
                resolution=0.1,
            ),
            width=width, height=height, data=data,
        )
        ros.get_global_costmap_snapshot.return_value = snapshot
        ros.capture_scene_observation.return_value = _make_observation()
        # First nav fails, second succeeds.
        ros.navigate_to_pose.side_effect = [
            _make_nav_failed("nav_a"),
            _make_nav_succeeded("nav_b"),
        ]
        ros.rotate_in_place.return_value = SkillResult(
            step_id="rot", skill="rotate_in_place", status="succeeded",
            outputs={}, started_at=0.0, finished_at=0.0,
        )
        grounding.locate_semantic_target.return_value = _make_grounding(
            found=True, label="door",
        )

        runtime = _make_runtime()
        result = runner._explore_until_visible(runtime, _make_step(max_frontiers=3))

        assert result.status == "succeeded"
        # First attempt had nav failure, second succeeded.
        assert len(result.outputs["attempts"]) == 2
        assert result.outputs["attempts"][0]["nav_status"] == "failed"
        assert result.outputs["attempts"][1]["nav_status"] == "succeeded"

    def test_cancel_event_stops_exploration(self):
        runner, _, _, ros = _make_runner()
        ros.get_global_costmap_snapshot.return_value = _half_unknown_snapshot()
        runtime = _make_runtime()
        runtime.cancel_event.set()  # Cancel before the loop begins.
        result = runner._explore_until_visible(runtime, _make_step())
        assert result.status == "canceled"
        assert result.error_code == "mission_canceled"

    def test_timeout_short_circuits(self):
        runner, _, _, ros = _make_runner()
        ros.get_global_costmap_snapshot.return_value = _half_unknown_snapshot()
        runtime = _make_runtime()
        # timeout_s=0 forces the deadline check to fire immediately.
        result = runner._explore_until_visible(runtime, _make_step(timeout_s=0.0))
        assert result.status == "failed"
        assert result.error_code == "frontier_set_exhausted"
        assert result.outputs["reason"] == "timeout"

    def test_skill_is_registered(self):
        from strafer_autonomy.executor.mission_runner import DEFAULT_AVAILABLE_SKILLS
        assert "explore_until_visible" in DEFAULT_AVAILABLE_SKILLS

    def test_skill_dispatches_through_execute_step(self):
        runner, _, grounding, ros = _make_runner()
        ros.get_global_costmap_snapshot.return_value = _half_unknown_snapshot()
        ros.capture_scene_observation.return_value = _make_observation()
        ros.navigate_to_pose.return_value = _make_nav_succeeded("nav")
        ros.rotate_in_place.return_value = SkillResult(
            step_id="rot", skill="rotate_in_place", status="succeeded",
            outputs={}, started_at=0.0, finished_at=0.0,
        )
        grounding.locate_semantic_target.return_value = _make_grounding(
            found=True, label="door",
        )
        runtime = _make_runtime()
        # Routes through the public dispatch path rather than calling
        # the private method directly.
        result = runner._execute_step(runtime, _make_step())
        assert result.status == "succeeded"
        assert result.skill == "explore_until_visible"
