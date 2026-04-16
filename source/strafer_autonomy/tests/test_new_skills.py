"""Tests for verify_arrival, rotate_by_degrees, orient_to_direction, query_environment."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from strafer_autonomy.executor.mission_runner import (
    MissionRunner,
    MissionRunnerConfig,
    _MissionRuntime,
)
from strafer_autonomy.schemas import (
    GoalPoseCandidate,
    GroundingResult,
    Pose3D,
    SceneObservation,
    SkillCall,
    SkillResult,
)
from strafer_autonomy.semantic_map.models import Pose2D


class _FakeNode:
    def __init__(self, x: float, y: float):
        self.pose = Pose2D(x=x, y=y, yaw=0.0)
        self.node_id = f"n_{x}_{y}"


def _random_embedding(dim: int = 512) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _make_observation() -> SceneObservation:
    return SceneObservation(
        observation_id="obs_1",
        stamp_sec=1.0,
        color_image_bgr=np.zeros((64, 64, 3), dtype=np.uint8),
        aligned_depth_m=np.zeros((64, 64), dtype=np.float32),
        camera_frame="d555_color_optical_frame",
        robot_pose_map={"x": 0.0, "y": 0.0, "qz": 0.0, "qw": 1.0},
    )


def _make_semantic_map(top_results: list) -> MagicMock:
    mock_map = MagicMock()
    mock_map.clip_encoder = MagicMock()
    mock_map.clip_encoder.encode_image.return_value = _random_embedding()
    mock_map.clip_encoder.encode_text.return_value = _random_embedding()
    mock_map.query_by_embedding.return_value = top_results
    mock_map.query_by_text.return_value = [
        {"node_id": "n1", "text_description": "a chair"},
    ]
    return mock_map


def _make_runner(*, semantic_map=None, background_mapper=None):
    planner = MagicMock()
    grounding = MagicMock()
    ros = MagicMock()
    runner = MissionRunner(
        planner_client=planner,
        grounding_client=grounding,
        ros_client=ros,
        semantic_map=semantic_map,
        background_mapper=background_mapper,
    )
    return runner, ros


def _make_runtime() -> _MissionRuntime:
    return _MissionRuntime(
        mission_id="m1",
        request_id="r1",
        raw_command="test",
        source="test",
        started_at=0.0,
    )


# ---------------------------------------------------------------------------
# verify_arrival
# ---------------------------------------------------------------------------


class TestVerifyArrival:
    def test_pass_when_semantic_map_disabled(self):
        runner, ros = _make_runner(semantic_map=None)
        runtime = _make_runtime()
        step = SkillCall(skill="verify_arrival", step_id="v1")
        result = runner._verify_arrival(runtime, step)
        assert result.status == "succeeded"
        assert result.outputs["verified"] is False

    def test_fails_without_goal_pose(self):
        top_results = [(_FakeNode(0.0, 0.0), 0.1)]
        mock_map = _make_semantic_map(top_results)
        runner, ros = _make_runner(semantic_map=mock_map)
        ros.capture_scene_observation.return_value = _make_observation()
        runtime = _make_runtime()  # no latest_goal_pose
        step = SkillCall(skill="verify_arrival", step_id="v1")
        result = runner._verify_arrival(runtime, step)
        assert result.status == "failed"
        assert result.error_code == "no_goal_pose"

    def test_verified_when_majority_near_goal(self):
        top_results = [
            (_FakeNode(5.0, 5.0), 0.1),
            (_FakeNode(5.2, 5.1), 0.15),
            (_FakeNode(4.8, 5.3), 0.2),
            (_FakeNode(20.0, 20.0), 0.6),
            (_FakeNode(0.0, 0.0), 0.8),
        ]
        mock_map = _make_semantic_map(top_results)
        runner, ros = _make_runner(semantic_map=mock_map)
        ros.capture_scene_observation.return_value = _make_observation()
        runtime = _make_runtime()
        runtime.latest_goal_pose = GoalPoseCandidate(
            request_id="g", found=True, goal_frame="map",
            goal_pose=Pose3D(x=5.0, y=5.0),
            standoff_m=0.0, depth_valid=True,
        )
        step = SkillCall(
            skill="verify_arrival", step_id="v1",
            args={"target_label": "table", "top_k": 5, "majority": 3, "goal_radius_m": 1.0},
        )
        result = runner._verify_arrival(runtime, step)
        assert result.status == "succeeded"
        assert result.outputs["verified"] is True
        assert result.outputs["near_goal_count"] == 3

    def test_fails_when_majority_far_from_goal(self):
        top_results = [
            (_FakeNode(20.0, 20.0), 0.1),
            (_FakeNode(21.0, 19.5), 0.2),
            (_FakeNode(19.0, 20.5), 0.3),
            (_FakeNode(5.0, 5.0), 0.6),
            (_FakeNode(0.0, 0.0), 0.8),
        ]
        mock_map = _make_semantic_map(top_results)
        runner, ros = _make_runner(semantic_map=mock_map)
        ros.capture_scene_observation.return_value = _make_observation()
        runtime = _make_runtime()
        runtime.latest_goal_pose = GoalPoseCandidate(
            request_id="g", found=True, goal_frame="map",
            goal_pose=Pose3D(x=5.0, y=5.0),
            standoff_m=0.0, depth_valid=True,
        )
        step = SkillCall(
            skill="verify_arrival", step_id="v1",
            args={"target_label": "table", "top_k": 5, "majority": 3, "goal_radius_m": 2.0},
        )
        result = runner._verify_arrival(runtime, step)
        assert result.status == "failed"
        assert result.error_code == "arrival_verification_failed"

    def test_passes_on_empty_map_by_default(self):
        mock_map = _make_semantic_map(top_results=[])
        runner, ros = _make_runner(semantic_map=mock_map)
        ros.capture_scene_observation.return_value = _make_observation()
        runtime = _make_runtime()
        runtime.latest_goal_pose = GoalPoseCandidate(
            request_id="g", found=True, goal_frame="map",
            goal_pose=Pose3D(x=5.0, y=5.0),
            standoff_m=0.0, depth_valid=True,
        )
        step = SkillCall(skill="verify_arrival", step_id="v1")
        result = runner._verify_arrival(runtime, step)
        assert result.status == "succeeded"
        assert result.outputs["reason"] == "no_map_data"

    def test_fails_on_empty_map_when_configured(self):
        mock_map = _make_semantic_map(top_results=[])
        runner, ros = _make_runner(semantic_map=mock_map)
        ros.capture_scene_observation.return_value = _make_observation()
        runtime = _make_runtime()
        runtime.latest_goal_pose = GoalPoseCandidate(
            request_id="g", found=True, goal_frame="map",
            goal_pose=Pose3D(x=5.0, y=5.0),
            standoff_m=0.0, depth_valid=True,
        )
        step = SkillCall(
            skill="verify_arrival", step_id="v1",
            args={"fallback_on_empty_map": "fail"},
        )
        result = runner._verify_arrival(runtime, step)
        assert result.status == "failed"


# ---------------------------------------------------------------------------
# rotate_by_degrees
# ---------------------------------------------------------------------------


class TestRotateByDegrees:
    def test_rotation_radians_computed(self):
        runner, ros = _make_runner()
        ros.rotate_in_place.return_value = SkillResult(
            step_id="r1", skill="rotate_in_place", status="succeeded",
        )
        runtime = _make_runtime()
        step = SkillCall(skill="rotate_by_degrees", step_id="r1", args={"degrees": 90})
        result = runner._rotate_by_degrees(runtime, step)
        assert result.status == "succeeded"
        args = ros.rotate_in_place.call_args
        assert abs(args.kwargs["yaw_delta_rad"] - math.pi / 2) < 1e-6

    def test_invalid_degrees_returns_error(self):
        runner, ros = _make_runner()
        runtime = _make_runtime()
        step = SkillCall(
            skill="rotate_by_degrees", step_id="r1", args={"degrees": "bad"},
        )
        result = runner._rotate_by_degrees(runtime, step)
        assert result.status == "failed"

    def test_negative_degrees(self):
        runner, ros = _make_runner()
        ros.rotate_in_place.return_value = SkillResult(
            step_id="r1", skill="rotate_in_place", status="succeeded",
        )
        runtime = _make_runtime()
        step = SkillCall(skill="rotate_by_degrees", step_id="r1", args={"degrees": -45})
        runner._rotate_by_degrees(runtime, step)
        assert abs(ros.rotate_in_place.call_args.kwargs["yaw_delta_rad"] + math.pi / 4) < 1e-6


# ---------------------------------------------------------------------------
# orient_to_direction
# ---------------------------------------------------------------------------


class TestOrientToDirection:
    def test_invalid_direction(self):
        runner, ros = _make_runner()
        runtime = _make_runtime()
        step = SkillCall(
            skill="orient_to_direction", step_id="o1",
            args={"direction": "up"},
        )
        result = runner._orient_to_direction(runtime, step)
        assert result.status == "failed"
        assert result.error_code == "invalid_args"

    def test_no_pose(self):
        runner, ros = _make_runner()
        ros.get_robot_state.return_value = {"pose": None}
        runtime = _make_runtime()
        step = SkillCall(
            skill="orient_to_direction", step_id="o1",
            args={"direction": "north"},
        )
        result = runner._orient_to_direction(runtime, step)
        assert result.status == "failed"
        assert result.error_code == "no_pose"

    def test_north_from_east(self):
        runner, ros = _make_runner()
        # Robot currently facing east (yaw=0)
        ros.get_robot_state.return_value = {
            "pose": {"qz": 0.0, "qw": 1.0},
        }
        ros.rotate_in_place.return_value = SkillResult(
            step_id="o1", skill="rotate_in_place", status="succeeded",
        )
        runtime = _make_runtime()
        step = SkillCall(
            skill="orient_to_direction", step_id="o1",
            args={"direction": "north"},
        )
        runner._orient_to_direction(runtime, step)
        delta = ros.rotate_in_place.call_args.kwargs["yaw_delta_rad"]
        assert abs(delta - math.pi / 2) < 1e-6


# ---------------------------------------------------------------------------
# query_environment
# ---------------------------------------------------------------------------


class TestQueryEnvironment:
    def test_missing_query(self):
        runner, ros = _make_runner()
        runtime = _make_runtime()
        step = SkillCall(skill="query_environment", step_id="q1", args={})
        result = runner._query_environment(runtime, step)
        assert result.status == "failed"

    def test_disabled_returns_empty(self):
        runner, ros = _make_runner(semantic_map=None)
        runtime = _make_runtime()
        step = SkillCall(
            skill="query_environment", step_id="q1",
            args={"query": "red cup"},
        )
        result = runner._query_environment(runtime, step)
        assert result.status == "succeeded"
        assert result.outputs["results"] == []

    def test_returns_results(self):
        mock_map = _make_semantic_map(top_results=[])
        mock_map.query_by_text.return_value = [
            {"node_id": "n1", "text_description": "a wooden door"},
        ]
        runner, ros = _make_runner(semantic_map=mock_map)
        runtime = _make_runtime()
        step = SkillCall(
            skill="query_environment", step_id="q1",
            args={"query": "door"},
        )
        result = runner._query_environment(runtime, step)
        assert result.status == "succeeded"
        assert len(result.outputs["results"]) == 1


# ---------------------------------------------------------------------------
# Dispatch coverage
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_default_skills_include_new(self):
        from strafer_autonomy.executor.mission_runner import DEFAULT_AVAILABLE_SKILLS
        assert "verify_arrival" in DEFAULT_AVAILABLE_SKILLS
        assert "rotate_by_degrees" in DEFAULT_AVAILABLE_SKILLS
        assert "orient_to_direction" in DEFAULT_AVAILABLE_SKILLS
        assert "query_environment" in DEFAULT_AVAILABLE_SKILLS

    def test_execute_step_dispatches_verify_arrival(self):
        mock_map = _make_semantic_map(top_results=[])
        runner, ros = _make_runner(semantic_map=mock_map)
        ros.capture_scene_observation.return_value = _make_observation()
        runtime = _make_runtime()
        runtime.latest_goal_pose = GoalPoseCandidate(
            request_id="g", found=True, goal_frame="map",
            goal_pose=Pose3D(),
        )
        step = SkillCall(skill="verify_arrival", step_id="v1")
        result = runner._execute_step(runtime, step)
        # Empty map by default passes (fallback_on_empty_map=pass)
        assert result.status == "succeeded"
