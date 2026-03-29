"""Tests for the scan_for_target executor skill handler."""

from __future__ import annotations

import math
from threading import Event
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from strafer_autonomy.executor.mission_runner import MissionRunner, MissionRunnerConfig, _MissionRuntime
from strafer_autonomy.schemas import GroundingRequest, GroundingResult, SceneObservation, SkillCall, SkillResult


def _make_observation(observation_id: str = "obs_1") -> SceneObservation:
    return SceneObservation(
        observation_id=observation_id,
        stamp_sec=1.0,
        color_image_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
        aligned_depth_m=np.zeros((480, 640), dtype=np.float32),
        camera_frame="camera_color_optical_frame",
    )


def _make_grounding(found: bool, label: str = "door") -> GroundingResult:
    return GroundingResult(
        request_id="test",
        found=found,
        bbox_2d=(100, 200, 300, 400) if found else None,
        label=label if found else None,
        confidence=0.9 if found else None,
        latency_s=0.1,
    )


def _make_rotate_result(step_id: str = "rotate") -> SkillResult:
    return SkillResult(
        step_id=step_id,
        skill="rotate_in_place",
        status="succeeded",
        outputs={},
        started_at=0.0,
        finished_at=0.0,
    )


def _make_runner() -> tuple[MissionRunner, MagicMock, MagicMock, MagicMock]:
    planner = MagicMock()
    grounding = MagicMock()
    ros = MagicMock()
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


def _make_scan_step(**kwargs) -> SkillCall:
    args = {"label": "door", "max_scan_steps": 6, "scan_arc_deg": 360}
    args.update(kwargs)
    return SkillCall(
        step_id="step_scan",
        skill="scan_for_target",
        args=args,
        timeout_s=60.0,
        retry_limit=0,
    )


class TestScanForTarget:
    def test_found_first_heading(self):
        runner, _, grounding, ros = _make_runner()
        ros.capture_scene_observation.return_value = _make_observation()
        grounding.locate_semantic_target.return_value = _make_grounding(found=True)

        runtime = _make_runtime()
        step = _make_scan_step()
        result = runner._scan_for_target(runtime, step)

        assert result.status == "succeeded"
        assert result.outputs["heading_index"] == 0
        assert result.outputs["found"] is True
        ros.rotate_in_place.assert_not_called()

    def test_found_third_heading(self):
        runner, _, grounding, ros = _make_runner()
        ros.capture_scene_observation.return_value = _make_observation()
        ros.rotate_in_place.return_value = _make_rotate_result()
        grounding.locate_semantic_target.side_effect = [
            _make_grounding(found=False),
            _make_grounding(found=False),
            _make_grounding(found=True, label="door"),
        ]

        runtime = _make_runtime()
        step = _make_scan_step()
        result = runner._scan_for_target(runtime, step)

        assert result.status == "succeeded"
        assert result.outputs["heading_index"] == 2
        assert ros.rotate_in_place.call_count == 2

    def test_not_found_after_full_rotation(self):
        runner, _, grounding, ros = _make_runner()
        ros.capture_scene_observation.return_value = _make_observation()
        ros.rotate_in_place.return_value = _make_rotate_result()
        grounding.locate_semantic_target.return_value = _make_grounding(found=False)
        grounding.describe_scene.side_effect = ConnectionError("no describe")

        runtime = _make_runtime()
        step = _make_scan_step(max_scan_steps=3, scan_arc_deg=360)
        result = runner._scan_for_target(runtime, step)

        assert result.status == "failed"
        assert result.error_code == "target_not_found_after_scan"
        assert result.outputs["headings_checked"] == 3
        assert ros.rotate_in_place.call_count == 2  # skips rotation after last heading

    def test_scan_failure_includes_scene_description(self):
        from strafer_autonomy.schemas import SceneDescription

        runner, _, grounding, ros = _make_runner()
        ros.capture_scene_observation.return_value = _make_observation()
        ros.rotate_in_place.return_value = _make_rotate_result()
        grounding.locate_semantic_target.return_value = _make_grounding(found=False)
        grounding.describe_scene.return_value = SceneDescription(
            request_id="test",
            description="I see a hallway with a closed door and a bookshelf.",
            latency_s=0.1,
        )

        runtime = _make_runtime()
        step = _make_scan_step(max_scan_steps=2, scan_arc_deg=360)
        result = runner._scan_for_target(runtime, step)

        assert result.status == "failed"
        assert "hallway" in result.message
        assert result.outputs["last_scene_description"] == "I see a hallway with a closed door and a bookshelf."

    def test_cancel_during_scan(self):
        runner, _, grounding, ros = _make_runner()
        ros.capture_scene_observation.return_value = _make_observation()
        ros.rotate_in_place.return_value = _make_rotate_result()
        grounding.locate_semantic_target.return_value = _make_grounding(found=False)

        runtime = _make_runtime()
        # Set cancel after first grounding attempt
        original_locate = grounding.locate_semantic_target.side_effect

        def cancel_after_first(*args, **kwargs):
            result = _make_grounding(found=False)
            runtime.cancel_event.set()
            return result

        grounding.locate_semantic_target.side_effect = cancel_after_first

        step = _make_scan_step()
        result = runner._scan_for_target(runtime, step)

        assert result.status == "canceled"
        assert result.error_code == "mission_canceled"

    def test_grounding_service_unavailable(self):
        runner, _, grounding, ros = _make_runner()
        ros.capture_scene_observation.return_value = _make_observation()
        grounding.locate_semantic_target.side_effect = ConnectionError("VLM unreachable")

        runtime = _make_runtime()
        step = _make_scan_step()
        result = runner._scan_for_target(runtime, step)

        assert result.status == "failed"
        assert result.error_code == "grounding_failed"

    def test_missing_label_returns_invalid_args(self):
        runner, _, _, _ = _make_runner()
        runtime = _make_runtime()
        step = _make_scan_step(label="")
        result = runner._scan_for_target(runtime, step)

        assert result.status == "failed"
        assert result.error_code == "invalid_args"

    def test_rotation_failure_aborts_scan(self):
        runner, _, grounding, ros = _make_runner()
        ros.capture_scene_observation.return_value = _make_observation()
        grounding.locate_semantic_target.return_value = _make_grounding(found=False)
        ros.rotate_in_place.return_value = SkillResult(
            step_id="rotate",
            skill="rotate_in_place",
            status="failed",
            outputs={},
            error_code="rotation_failed",
            message="Motor error",
            started_at=0.0,
            finished_at=0.0,
        )

        runtime = _make_runtime()
        step = _make_scan_step()
        result = runner._scan_for_target(runtime, step)

        assert result.status == "failed"
        assert result.error_code == "rotation_failed"

    def test_step_angle_calculation(self):
        runner, _, grounding, ros = _make_runner()
        ros.capture_scene_observation.return_value = _make_observation()
        ros.rotate_in_place.return_value = _make_rotate_result()
        grounding.locate_semantic_target.return_value = _make_grounding(found=False)

        runtime = _make_runtime()
        step = _make_scan_step(max_scan_steps=4, scan_arc_deg=360)
        runner._scan_for_target(runtime, step)

        # 360° / 4 steps = 90° per step = π/2 rad
        expected_angle = math.pi / 2
        for call in ros.rotate_in_place.call_args_list:
            actual_angle = call[1]["yaw_delta_rad"]
            assert abs(actual_angle - expected_angle) < 1e-9
