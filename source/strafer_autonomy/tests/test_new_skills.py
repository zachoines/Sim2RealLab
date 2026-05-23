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
    # Diagonal translates decompose into rotate_in_place + navigate_to_pose,
    # so any translate test must hand back a real SkillResult here or the
    # executor will propagate the bare MagicMock as a "failed" rotation.
    ros.rotate_in_place.return_value = SkillResult(
        step_id="rotate", skill="rotate_in_place", status="succeeded",
    )
    # align_to_goal_yaw queries the planner first; default to None so the
    # existing tests get the goal-pose-yaw fallback verbatim.
    ros.compute_path_to_pose.return_value = None
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
# translate
# ---------------------------------------------------------------------------


class TestTranslate:
    def test_forward_from_origin(self):
        runner, ros = _make_runner()
        ros.get_map_pose.return_value = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        }
        ros.navigate_to_pose.return_value = SkillResult(
            step_id="t1", skill="translate", status="succeeded",
        )
        runtime = _make_runtime()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 0.0},
        )
        result = runner._translate(runtime, step)
        assert result.status == "succeeded"
        goal = ros.navigate_to_pose.call_args.kwargs["goal_pose"]
        assert abs(goal.x - 1.0) < 1e-6
        assert abs(goal.y - 0.0) < 1e-6

    def test_lateral_left_rotates_into_map(self):
        runner, ros = _make_runner()
        # Robot at (2, 3) facing +y (yaw=90°, qz=sqrt(2)/2, qw=sqrt(2)/2).
        half_sqrt2 = math.sqrt(2) / 2
        ros.get_map_pose.return_value = {
            "x": 2.0, "y": 3.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": half_sqrt2, "qw": half_sqrt2,
        }
        ros.navigate_to_pose.return_value = SkillResult(
            step_id="t1", skill="translate", status="succeeded",
        )
        runtime = _make_runtime()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 0.5},
        )
        runner._translate(runtime, step)
        goal = ros.navigate_to_pose.call_args.kwargs["goal_pose"]
        # +1m forward (along +y in map) + 0.5m left (along -x in map).
        assert abs(goal.x - (2.0 - 0.5)) < 1e-6
        assert abs(goal.y - (3.0 + 1.0)) < 1e-6

    def test_no_map_pose_returns_error(self):
        runner, ros = _make_runner()
        ros.get_map_pose.return_value = None
        runtime = _make_runtime()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 0.0},
        )
        result = runner._translate(runtime, step)
        assert result.status == "failed"
        assert result.error_code == "no_map_pose"

    def test_invalid_args_returns_error(self):
        runner, ros = _make_runner()
        runtime = _make_runtime()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": "bad", "dy_m": 0.0},
        )
        result = runner._translate(runtime, step)
        assert result.status == "failed"
        assert result.error_code == "invalid_args"

    def test_none_timeout_uses_default_in_legacy_mode(self):
        """With progress-aware disabled, timeout falls back to default_navigation_timeout_s."""
        runner = MissionRunner(
            planner_client=MagicMock(),
            grounding_client=MagicMock(),
            ros_client=MagicMock(),
            config=MissionRunnerConfig(nav_progress_aware=False),
        )
        ros = runner._ros_client
        ros.get_map_pose.return_value = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        }
        ros.navigate_to_pose.return_value = SkillResult(
            step_id="t1", skill="translate", status="succeeded",
        )
        runtime = _make_runtime()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 0.0},
            timeout_s=None,
        )
        runner._translate(runtime, step)
        assert (
            ros.navigate_to_pose.call_args.kwargs["timeout_s"]
            == MissionRunnerConfig().default_navigation_timeout_s
        )

    def test_none_timeout_picks_up_env_override_in_legacy_mode(self):
        runner = MissionRunner(
            planner_client=MagicMock(),
            grounding_client=MagicMock(),
            ros_client=MagicMock(),
            config=MissionRunnerConfig(
                default_navigation_timeout_s=180.0,
                nav_progress_aware=False,
            ),
        )
        ros = runner._ros_client
        ros.get_map_pose.return_value = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        }
        ros.navigate_to_pose.return_value = SkillResult(
            step_id="t1", skill="translate", status="succeeded",
        )
        runtime = _make_runtime()
        step = SkillCall(
            skill="translate", step_id="t1",
            args={"dx_m": 1.0, "dy_m": 0.0},
            timeout_s=None,
        )
        runner._translate(runtime, step)
        assert ros.navigate_to_pose.call_args.kwargs["timeout_s"] == 180.0


# ---------------------------------------------------------------------------
# navigate_to_pose timeout fallback (_dispatch_nav_goal path)
# ---------------------------------------------------------------------------


class TestNavigateToPoseTimeoutFallback:
    """A SkillCall with ``timeout_s=None`` no longer carries a compiler-side
    hardcode. Under the legacy (escape-hatch) mode the executor falls back
    to ``default_navigation_timeout_s`` (sourced from
    ``STRAFER_NAVIGATION_TIMEOUT_S``); progress-aware mode is covered in
    ``test_progress_aware_timeouts``.
    """

    def test_none_timeout_uses_config_default_in_legacy_mode(self):
        runner = MissionRunner(
            planner_client=MagicMock(),
            grounding_client=MagicMock(),
            ros_client=MagicMock(),
            config=MissionRunnerConfig(nav_progress_aware=False),
        )
        ros = runner._ros_client
        ros.navigate_to_pose.return_value = SkillResult(
            step_id="step_03", skill="navigate_to_pose", status="succeeded",
        )
        runtime = _make_runtime()
        runtime.latest_goal_pose = GoalPoseCandidate(
            request_id="g", found=True, goal_frame="map",
            goal_pose=Pose3D(x=2.0, y=1.0),
            standoff_m=0.0, depth_valid=True,
        )
        step = SkillCall(
            step_id="step_03",
            skill="navigate_to_pose",
            args={"goal_source": "explicit", "execution_backend": "nav2"},
            timeout_s=None,
        )
        runner._dispatch_nav_goal(step, runtime.latest_goal_pose.goal_pose, 0.0)
        assert (
            ros.navigate_to_pose.call_args.kwargs["timeout_s"]
            == MissionRunnerConfig().default_navigation_timeout_s
        )

    def test_none_timeout_picks_up_env_override_in_legacy_mode(self):
        runner = MissionRunner(
            planner_client=MagicMock(),
            grounding_client=MagicMock(),
            ros_client=MagicMock(),
            config=MissionRunnerConfig(
                default_navigation_timeout_s=180.0,
                nav_progress_aware=False,
            ),
        )
        ros = runner._ros_client
        ros.navigate_to_pose.return_value = SkillResult(
            step_id="step_03", skill="navigate_to_pose", status="succeeded",
        )
        step = SkillCall(
            step_id="step_03",
            skill="navigate_to_pose",
            args={"goal_source": "explicit", "execution_backend": "nav2"},
            timeout_s=None,
        )
        runner._dispatch_nav_goal(step, Pose3D(x=2.0, y=1.0), 0.0)
        assert ros.navigate_to_pose.call_args.kwargs["timeout_s"] == 180.0


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
# align_to_goal_yaw
# ---------------------------------------------------------------------------


def _goal_pose_with_yaw(yaw: float, x: float = 1.0, y: float = 0.0) -> Pose3D:
    return Pose3D(
        x=x, y=y, z=0.0,
        qx=0.0, qy=0.0,
        qz=math.sin(yaw / 2.0),
        qw=math.cos(yaw / 2.0),
    )


def _goal_pose_candidate(yaw: float) -> GoalPoseCandidate:
    return GoalPoseCandidate(
        request_id="g1", found=True, goal_frame="map",
        goal_pose=_goal_pose_with_yaw(yaw),
    )


class TestAlignToGoalYaw:
    def test_missing_goal_pose_fails_with_prereq_code(self):
        runner, ros = _make_runner()
        runtime = _make_runtime()
        step = SkillCall(skill="align_to_goal_yaw", step_id="a1")
        result = runner._align_to_goal_yaw(runtime, step)
        assert result.status == "failed"
        assert result.error_code == "align_prereq_missing"

    def test_candidate_without_goal_pose_fails_with_prereq_code(self):
        runner, ros = _make_runner()
        runtime = _make_runtime()
        runtime.latest_goal_pose = GoalPoseCandidate(
            request_id="g1", found=False, goal_frame="map", goal_pose=None,
        )
        step = SkillCall(skill="align_to_goal_yaw", step_id="a1")
        result = runner._align_to_goal_yaw(runtime, step)
        assert result.status == "failed"
        assert result.error_code == "align_prereq_missing"

    def test_no_robot_pose_fails(self):
        runner, ros = _make_runner()
        ros.get_robot_state.return_value = {"pose": None}
        runtime = _make_runtime()
        runtime.latest_goal_pose = _goal_pose_candidate(yaw=0.0)
        step = SkillCall(skill="align_to_goal_yaw", step_id="a1")
        result = runner._align_to_goal_yaw(runtime, step)
        assert result.status == "failed"
        assert result.error_code == "no_pose"

    def test_off_axis_delta_passed_to_rotate(self):
        runner, ros = _make_runner()
        ros.get_robot_state.return_value = {"pose": {"qz": 0.0, "qw": 1.0}}
        ros.rotate_in_place.return_value = SkillResult(
            step_id="a1", skill="rotate_in_place", status="succeeded",
        )
        runtime = _make_runtime()
        runtime.latest_goal_pose = _goal_pose_candidate(yaw=math.radians(30.0))
        step = SkillCall(
            skill="align_to_goal_yaw", step_id="a1",
            args={"tolerance_rad": 0.087},
        )
        runner._align_to_goal_yaw(runtime, step)
        kwargs = ros.rotate_in_place.call_args.kwargs
        assert abs(kwargs["yaw_delta_rad"] - math.radians(30.0)) < 1e-6
        assert kwargs["tolerance_rad"] == pytest.approx(0.087)
        assert kwargs["step_id"] == "a1"

    def test_shortest_arc_across_pi(self):
        runner, ros = _make_runner()
        # Robot at +170°, goal at -170° → shortest path is +20° (not -340°).
        robot_yaw = math.radians(170.0)
        ros.get_robot_state.return_value = {
            "pose": {
                "qz": math.sin(robot_yaw / 2.0),
                "qw": math.cos(robot_yaw / 2.0),
            },
        }
        ros.rotate_in_place.return_value = SkillResult(
            step_id="a1", skill="rotate_in_place", status="succeeded",
        )
        runtime = _make_runtime()
        runtime.latest_goal_pose = _goal_pose_candidate(yaw=math.radians(-170.0))
        step = SkillCall(skill="align_to_goal_yaw", step_id="a1")
        runner._align_to_goal_yaw(runtime, step)
        delta = ros.rotate_in_place.call_args.kwargs["yaw_delta_rad"]
        assert abs(delta - math.radians(20.0)) < 1e-6

    def test_already_aligned_still_calls_rotate_for_pass_through(self):
        runner, ros = _make_runner()
        # Robot and goal both at yaw=0 → delta ~ 0.
        ros.get_robot_state.return_value = {"pose": {"qz": 0.0, "qw": 1.0}}
        ros.rotate_in_place.return_value = SkillResult(
            step_id="a1", skill="rotate_in_place", status="succeeded",
        )
        runtime = _make_runtime()
        runtime.latest_goal_pose = _goal_pose_candidate(yaw=0.0)
        step = SkillCall(skill="align_to_goal_yaw", step_id="a1")
        result = runner._align_to_goal_yaw(runtime, step)
        assert result.status == "succeeded"
        # rotate_in_place is invoked even when within tolerance — it short-
        # circuits internally, keeping cmd_vel/log telemetry consistent.
        ros.rotate_in_place.assert_called_once()
        assert abs(ros.rotate_in_place.call_args.kwargs["yaw_delta_rad"]) < 1e-6

    def test_align_uses_first_waypoint_past_lookahead(self):
        """Curved path: align to the bearing of the first waypoint
        beyond ``lookahead_m``, not the goal pose's yaw.
        """
        runner, ros = _make_runner()
        ros.get_robot_state.return_value = {
            "pose": {"x": 0.0, "y": 0.0, "qz": 0.0, "qw": 1.0},
        }
        ros.rotate_in_place.return_value = SkillResult(
            step_id="a1", skill="rotate_in_place", status="succeeded",
        )
        # Planned path heads north for ~1 m, then east. Robot is facing
        # east (yaw=0). At lookahead_m=1.0 the first waypoint past
        # 1 m sits at (0, 1.0) → bearing = +90°.
        ros.compute_path_to_pose.return_value = [
            (0.0, 0.2), (0.0, 0.6), (0.0, 1.2), (1.0, 1.2), (2.0, 1.2),
        ]
        runtime = _make_runtime()
        # Goal yaw is east (0); lookahead bearing should override it.
        runtime.latest_goal_pose = _goal_pose_candidate(yaw=0.0)
        step = SkillCall(
            skill="align_to_goal_yaw", step_id="a1",
            args={"lookahead_m": 1.0},
        )
        runner._align_to_goal_yaw(runtime, step)
        delta = ros.rotate_in_place.call_args.kwargs["yaw_delta_rad"]
        assert delta == pytest.approx(math.pi / 2, abs=1e-6)

    def test_align_falls_back_to_goal_yaw_when_path_too_short(self):
        """All waypoints within lookahead_m → use the last waypoint's
        bearing (closest available approximation).
        """
        runner, ros = _make_runner()
        ros.get_robot_state.return_value = {
            "pose": {"x": 0.0, "y": 0.0, "qz": 0.0, "qw": 1.0},
        }
        ros.rotate_in_place.return_value = SkillResult(
            step_id="a1", skill="rotate_in_place", status="succeeded",
        )
        # All path points are within 0.3 m of the robot.
        ros.compute_path_to_pose.return_value = [(0.1, 0.0), (0.2, 0.1)]
        runtime = _make_runtime()
        runtime.latest_goal_pose = _goal_pose_candidate(yaw=math.radians(90))
        step = SkillCall(
            skill="align_to_goal_yaw", step_id="a1",
            args={"lookahead_m": 1.0},
        )
        runner._align_to_goal_yaw(runtime, step)
        delta = ros.rotate_in_place.call_args.kwargs["yaw_delta_rad"]
        # Last waypoint (0.2, 0.1) → bearing atan2(0.1, 0.2) ≈ 0.4636 rad.
        assert delta == pytest.approx(math.atan2(0.1, 0.2), abs=1e-6)

    def test_align_falls_back_when_planner_returns_none(self):
        runner, ros = _make_runner()
        ros.get_robot_state.return_value = {
            "pose": {"x": 0.0, "y": 0.0, "qz": 0.0, "qw": 1.0},
        }
        ros.rotate_in_place.return_value = SkillResult(
            step_id="a1", skill="rotate_in_place", status="succeeded",
        )
        ros.compute_path_to_pose.return_value = None
        runtime = _make_runtime()
        # Goal yaw is 30° — fallback must rotate to that.
        runtime.latest_goal_pose = _goal_pose_candidate(yaw=math.radians(30))
        step = SkillCall(skill="align_to_goal_yaw", step_id="a1")
        runner._align_to_goal_yaw(runtime, step)
        delta = ros.rotate_in_place.call_args.kwargs["yaw_delta_rad"]
        assert delta == pytest.approx(math.radians(30), abs=1e-6)

    def test_align_falls_back_when_planner_raises(self):
        runner, ros = _make_runner()
        ros.get_robot_state.return_value = {
            "pose": {"x": 0.0, "y": 0.0, "qz": 0.0, "qw": 1.0},
        }
        ros.rotate_in_place.return_value = SkillResult(
            step_id="a1", skill="rotate_in_place", status="succeeded",
        )
        ros.compute_path_to_pose.side_effect = RuntimeError("planner down")
        runtime = _make_runtime()
        runtime.latest_goal_pose = _goal_pose_candidate(yaw=math.radians(45))
        step = SkillCall(skill="align_to_goal_yaw", step_id="a1")
        runner._align_to_goal_yaw(runtime, step)
        delta = ros.rotate_in_place.call_args.kwargs["yaw_delta_rad"]
        assert delta == pytest.approx(math.radians(45), abs=1e-6)


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
        assert "translate" in DEFAULT_AVAILABLE_SKILLS
        assert "query_environment" in DEFAULT_AVAILABLE_SKILLS
        assert "align_to_goal_yaw" in DEFAULT_AVAILABLE_SKILLS

    def test_execute_step_dispatches_align_to_goal_yaw(self):
        runner, ros = _make_runner()
        ros.get_robot_state.return_value = {"pose": {"qz": 0.0, "qw": 1.0}}
        ros.rotate_in_place.return_value = SkillResult(
            step_id="a1", skill="rotate_in_place", status="succeeded",
        )
        runtime = _make_runtime()
        runtime.latest_goal_pose = _goal_pose_candidate(yaw=math.radians(10.0))
        step = SkillCall(skill="align_to_goal_yaw", step_id="a1")
        result = runner._execute_step(runtime, step)
        assert result.status == "succeeded"
        ros.rotate_in_place.assert_called_once()

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


# ---------------------------------------------------------------------------
# cancel_event forwarding from executor rotate sites + _cancel_robot_actions
# ---------------------------------------------------------------------------


class TestRotateCancelEventForwarding:
    """Each direct ``rotate_in_place`` caller in the executor passes
    ``runtime.cancel_event`` so a mid-rotation cancel actually stops
    the chassis. Without this, rotates ignore cancel until tolerance
    or timeout — a real-robot safety bug.
    """

    def test_rotate_by_degrees_forwards_cancel_event(self):
        runner, ros = _make_runner()
        ros.rotate_in_place.return_value = SkillResult(
            step_id="r1", skill="rotate_in_place", status="succeeded",
        )
        runtime = _make_runtime()
        step = SkillCall(skill="rotate_by_degrees", step_id="r1", args={"degrees": 30})
        runner._rotate_by_degrees(runtime, step)
        assert ros.rotate_in_place.call_args.kwargs["cancel_event"] is runtime.cancel_event

    def test_align_to_goal_yaw_forwards_cancel_event(self):
        runner, ros = _make_runner()
        ros.get_robot_state.return_value = {"pose": {"qz": 0.0, "qw": 1.0}}
        ros.rotate_in_place.return_value = SkillResult(
            step_id="a1", skill="rotate_in_place", status="succeeded",
        )
        runtime = _make_runtime()
        runtime.latest_goal_pose = _goal_pose_candidate(yaw=math.radians(15.0))
        step = SkillCall(skill="align_to_goal_yaw", step_id="a1")
        runner._align_to_goal_yaw(runtime, step)
        assert ros.rotate_in_place.call_args.kwargs["cancel_event"] is runtime.cancel_event

    def test_orient_to_direction_forwards_cancel_event(self):
        runner, ros = _make_runner()
        ros.get_robot_state.return_value = {"pose": {"qz": 0.0, "qw": 1.0}}
        ros.rotate_in_place.return_value = SkillResult(
            step_id="o1", skill="rotate_in_place", status="succeeded",
        )
        runtime = _make_runtime()
        step = SkillCall(
            skill="orient_to_direction", step_id="o1",
            args={"direction": "north"},
        )
        runner._orient_to_direction(runtime, step)
        assert ros.rotate_in_place.call_args.kwargs["cancel_event"] is runtime.cancel_event


class TestCancelRobotActions:
    """``_cancel_robot_actions`` zeroes ``/cmd_vel`` after canceling
    Nav2 — belt-and-braces fail-safe for direct-publish skills
    (rotate_in_place) that bypass Nav2 entirely, and to catch the
    race where cancel arrives before a rotate enters its loop.
    """

    def test_publishes_zero_cmd_vel_after_canceling_nav(self):
        runner, ros = _make_runner()
        runner._cancel_robot_actions()
        ros.cancel_active_navigation.assert_called_once()
        ros.publish_zero_cmd_vel.assert_called_once()

    def test_zero_publish_runs_even_when_nav_cancel_raises(self):
        runner, ros = _make_runner()
        ros.cancel_active_navigation.side_effect = RuntimeError("boom")
        runner._cancel_robot_actions()
        ros.publish_zero_cmd_vel.assert_called_once()

    def test_no_publish_when_client_lacks_method(self):
        """Older RosClient impls (Protocol-only stubs) without
        ``publish_zero_cmd_vel`` must not raise — the Nav2 cancel
        stays the primary mechanism.
        """
        runner, _ = _make_runner()
        # ``spec=[...]`` makes getattr return None for the missing
        # method instead of auto-creating a MagicMock attribute.
        runner._ros_client = MagicMock(spec=["cancel_active_navigation"])
        runner._cancel_robot_actions()  # must not raise
        runner._ros_client.cancel_active_navigation.assert_called_once()
