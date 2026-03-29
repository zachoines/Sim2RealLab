"""Tests for the describe_scene executor skill handler."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from strafer_autonomy.executor.mission_runner import MissionRunner, MissionRunnerConfig, _MissionRuntime
from strafer_autonomy.schemas import SceneDescription, SceneObservation, SkillCall


def _make_observation(observation_id: str = "obs_1") -> SceneObservation:
    return SceneObservation(
        observation_id=observation_id,
        stamp_sec=1.0,
        color_image_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
        aligned_depth_m=np.zeros((480, 640), dtype=np.float32),
        camera_frame="camera_color_optical_frame",
    )


def _make_description(description: str = "A table and two chairs.") -> SceneDescription:
    return SceneDescription(
        request_id="test",
        description=description,
        latency_s=0.2,
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
        raw_command="describe what you see",
        source="test",
        started_at=0.0,
    )


def _make_describe_step(**kwargs) -> SkillCall:
    args = dict(kwargs)
    return SkillCall(
        step_id="step_desc",
        skill="describe_scene",
        args=args,
        timeout_s=10.0,
        retry_limit=0,
    )


class TestDescribeScene:
    def test_describe_with_existing_observation(self):
        runner, _, grounding, ros = _make_runner()
        grounding.describe_scene.return_value = _make_description("A hallway with a door.")

        runtime = _make_runtime()
        runtime.latest_observation = _make_observation()
        step = _make_describe_step()
        result = runner._describe_scene(runtime, step)

        assert result.status == "succeeded"
        assert result.outputs["description"] == "A hallway with a door."
        ros.capture_scene_observation.assert_not_called()

    def test_describe_captures_if_no_observation(self):
        runner, _, grounding, ros = _make_runner()
        ros.capture_scene_observation.return_value = _make_observation()
        grounding.describe_scene.return_value = _make_description()

        runtime = _make_runtime()
        step = _make_describe_step()
        result = runner._describe_scene(runtime, step)

        assert result.status == "succeeded"
        ros.capture_scene_observation.assert_called_once()

    def test_describe_returns_description_in_outputs(self):
        runner, _, grounding, _ = _make_runner()
        grounding.describe_scene.return_value = _make_description("Two chairs and a table.")

        runtime = _make_runtime()
        runtime.latest_observation = _make_observation()
        step = _make_describe_step()
        result = runner._describe_scene(runtime, step)

        assert result.outputs["description"] == "Two chairs and a table."
        assert "latency_s" in result.outputs

    def test_describe_service_failure(self):
        runner, _, grounding, _ = _make_runner()
        grounding.describe_scene.side_effect = ConnectionError("VLM unreachable")

        runtime = _make_runtime()
        runtime.latest_observation = _make_observation()
        step = _make_describe_step()
        result = runner._describe_scene(runtime, step)

        assert result.status == "failed"
        assert result.error_code == "describe_failed"

    def test_describe_capture_failure(self):
        runner, _, _, ros = _make_runner()
        ros.capture_scene_observation.side_effect = RuntimeError("Camera error")

        runtime = _make_runtime()
        step = _make_describe_step()
        result = runner._describe_scene(runtime, step)

        assert result.status == "failed"
        assert result.error_code == "capture_failed"

    def test_custom_prompt_forwarded(self):
        runner, _, grounding, _ = _make_runner()
        grounding.describe_scene.return_value = _make_description()

        runtime = _make_runtime()
        runtime.latest_observation = _make_observation()
        step = _make_describe_step(prompt="List all toys.")
        runner._describe_scene(runtime, step)

        call_kwargs = grounding.describe_scene.call_args[1]
        assert call_kwargs["prompt"] == "List all toys."
