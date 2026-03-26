"""Jetson-local ROS client abstractions and stub implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from strafer_autonomy.schemas import GoalPoseCandidate, SceneObservation, SkillResult


@runtime_checkable
class RosClient(Protocol):
    """Executor-facing interface for robot-local observation and skill execution."""

    def capture_scene_observation(self) -> SceneObservation:
        """Return the latest synchronized robot observation bundle."""

    def get_robot_state(self) -> dict[str, Any]:
        """Return the latest robot state snapshot for planning and status."""

    def project_detection_to_goal_pose(
        self,
        *,
        request_id: str,
        image_stamp_sec: float,
        bbox_2d: tuple[int, int, int, int],
        standoff_m: float,
        target_label: str | None = None,
    ) -> GoalPoseCandidate:
        """Project a 2D detection into a reachable robot goal pose."""

    def navigate_to_pose(
        self,
        *,
        step_id: str,
        goal_pose: dict[str, Any],
        execution_backend: str = "nav2",
        behavior_tree: str | None = None,
        timeout_s: float | None = None,
    ) -> SkillResult:
        """Execute goal-directed motion through the selected local backend."""

    def cancel_active_navigation(self) -> bool:
        """Cancel the currently active motion backend if one exists."""

    def orient_relative_to_target(
        self,
        *,
        step_id: str,
        target_pose: dict[str, Any],
        mode: str,
        yaw_offset_rad: float = 0.0,
        tolerance_rad: float = 0.1,
        timeout_s: float | None = None,
    ) -> SkillResult:
        """Rotate the robot relative to a projected target pose."""


@dataclass(frozen=True)
class RosClientConfig:
    """Jetson-local runtime settings for the ROS adapter."""

    observation_max_age_s: float = 0.5
    default_goal_frame: str = "map"
    default_nav_timeout_s: float = 90.0
    default_navigation_backend: str = "nav2"
    default_orientation_tolerance_rad: float = 0.1


class JetsonRosClient:
    """Stub ROS adapter for the Jetson-resident mission executor."""

    def __init__(self, config: RosClientConfig | None = None) -> None:
        self._config = config or RosClientConfig()

    @property
    def config(self) -> RosClientConfig:
        """Return the immutable ROS client configuration."""

        return self._config

    def capture_scene_observation(self) -> SceneObservation:
        """Read the latest cached RGB, depth, camera info, and robot pose data."""

        raise NotImplementedError(
            "ROS observation capture is not implemented yet. "
            "The first implementation should cache RGB, depth, camera info, odom, and TF locally on the Jetson."
        )

    def get_robot_state(self) -> dict[str, Any]:
        """Return the latest high-level robot state for planning and status."""

        raise NotImplementedError(
            "Robot state retrieval is not implemented yet. "
            "The first implementation should compose state from local odom, TF, and action status."
        )

    def project_detection_to_goal_pose(
        self,
        *,
        request_id: str,
        image_stamp_sec: float,
        bbox_2d: tuple[int, int, int, int],
        standoff_m: float,
        target_label: str | None = None,
    ) -> GoalPoseCandidate:
        """Call the robot-local projection service and normalize its response."""

        raise NotImplementedError(
            "Projection service integration is not implemented yet. "
            "Wire this method to strafer_msgs/srv/ProjectDetectionToGoalPose.srv on the Jetson."
        )

    def navigate_to_pose(
        self,
        *,
        step_id: str,
        goal_pose: dict[str, Any],
        execution_backend: str = "nav2",
        behavior_tree: str | None = None,
        timeout_s: float | None = None,
    ) -> SkillResult:
        """Dispatch goal execution to the selected local backend and normalize the result."""

        raise NotImplementedError(
            "Navigation action integration is not implemented yet. "
            "Dispatch to either the Nav2 backend or the strafer_inference backend "
            "and translate outcomes into SkillResult."
        )

    def cancel_active_navigation(self) -> bool:
        """Cancel the active local motion backend, if present."""

        raise NotImplementedError(
            "Navigation cancelation is not implemented yet. "
            "Track the active local motion backend inside the Jetson executor "
            "so Nav2 or strafer_inference can be canceled uniformly."
        )

    def orient_relative_to_target(
        self,
        *,
        step_id: str,
        target_pose: dict[str, Any],
        mode: str,
        yaw_offset_rad: float = 0.0,
        tolerance_rad: float = 0.1,
        timeout_s: float | None = None,
    ) -> SkillResult:
        """Execute the future robot-specific orientation behavior."""

        raise NotImplementedError(
            "Target-relative orientation is not implemented yet. "
            "Add a Strafer-specific action once the navigation and projection path is working."
        )
