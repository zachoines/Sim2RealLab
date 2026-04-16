"""ROS action/service ingress scaffold for natural-language mission commands."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Protocol, runtime_checkable


DEFAULT_EXECUTE_MISSION_ACTION = "execute_mission"
DEFAULT_STATUS_SERVICE = "get_mission_status"

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MissionSubmission:
    """Immediate acceptance response returned when a mission is submitted."""

    accepted: bool
    mission_id: str = ""
    final_state: str = ""
    error_code: str = ""
    message: str = ""


@dataclass(frozen=True)
class MissionStatusSnapshot:
    """Current executor-facing mission state used for feedback and status queries."""

    active: bool
    mission_id: str = ""
    state: str = "idle"
    raw_command: str = ""
    current_step_id: str = ""
    current_skill: str = ""
    message: str = ""
    error_code: str = ""
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class CommandServerConfig:
    """ROS-side names and timing for the Jetson-local command ingress node."""

    node_name: str = "strafer_autonomy_command_server"
    action_name: str = DEFAULT_EXECUTE_MISSION_ACTION
    status_service_name: str = DEFAULT_STATUS_SERVICE
    feedback_period_s: float = 0.5


@runtime_checkable
class MissionCommandHandler(Protocol):
    """Mission execution adapter owned by the future mission runner."""

    def start_mission(
        self,
        *,
        request_id: str,
        raw_command: str,
        source: str,
        replace_active_mission: bool,
    ) -> MissionSubmission:
        """Accept or reject a new mission request."""

    def get_status(self) -> MissionStatusSnapshot:
        """Return the current active mission snapshot, or idle state if none."""

    def cancel_active_mission(self) -> MissionStatusSnapshot:
        """Request cancellation of the current mission and return the new status."""


class AutonomyCommandServer:
    """Jetson-local ROS ingress node for mission submit, status, and cancel flow."""

    def __init__(
        self,
        handler: MissionCommandHandler,
        config: CommandServerConfig | None = None,
    ) -> None:
        self._handler = handler
        self._config = config or CommandServerConfig()
        self._node = None
        self._action_server = None
        self._status_service = None
        self._rclpy = None
        self._execute_mission = None
        self._goal_response = None
        self._cancel_response = None

    @property
    def config(self) -> CommandServerConfig:
        """Return immutable runtime settings for the ingress node."""

        return self._config

    def spin(self) -> None:
        """Create ROS entities and block until shutdown."""

        self._ensure_ros_entities()
        assert self._rclpy is not None
        assert self._node is not None
        self._rclpy.spin(self._node)

    def destroy(self) -> None:
        """Destroy ROS entities created by this command server."""

        if self._action_server is not None:
            self._action_server.destroy()
            self._action_server = None
        if self._status_service is not None:
            self._status_service.destroy()
            self._status_service = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

    def _ensure_ros_entities(self) -> None:
        if self._node is not None:
            return

        try:
            import rclpy
            from rclpy.action import ActionServer, CancelResponse, GoalResponse
            from rclpy.node import Node
            from strafer_msgs.action import ExecuteMission
            from strafer_msgs.srv import GetMissionStatus
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 Python dependencies are not available. "
                "Run this on the Jetson inside the ROS environment with strafer_msgs built."
            ) from exc

        self._rclpy = rclpy
        self._execute_mission = ExecuteMission
        self._goal_response = GoalResponse
        self._cancel_response = CancelResponse

        self._node = Node(self._config.node_name)
        self._action_server = ActionServer(
            self._node,
            ExecuteMission,
            self._config.action_name,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )
        self._status_service = self._node.create_service(
            GetMissionStatus,
            self._config.status_service_name,
            self._handle_status_request,
        )

    def _goal_callback(self, goal_request):  # type: ignore[no-untyped-def]
        del goal_request
        assert self._goal_response is not None
        return self._goal_response.ACCEPT

    def _cancel_callback(self, goal_handle):  # type: ignore[no-untyped-def]
        del goal_handle
        assert self._cancel_response is not None
        return self._cancel_response.ACCEPT

    def _execute_callback(self, goal_handle):  # type: ignore[no-untyped-def]
        assert self._execute_mission is not None
        assert self._rclpy is not None

        submission = self._handler.start_mission(
            request_id=goal_handle.request.request_id,
            raw_command=goal_handle.request.raw_command,
            source=goal_handle.request.source,
            replace_active_mission=goal_handle.request.replace_active_mission,
        )
        if not submission.accepted:
            goal_handle.abort()
            return self._make_result_from_submission(submission)

        last_snapshot = self._handler.get_status()
        while self._rclpy.ok():
            if goal_handle.is_cancel_requested:
                last_snapshot = self._handler.cancel_active_mission()
                goal_handle.canceled()
                return self._make_result_from_snapshot(last_snapshot, accepted=True)

            last_snapshot = self._handler.get_status()
            goal_handle.publish_feedback(self._make_feedback(last_snapshot))
            if not last_snapshot.active:
                if last_snapshot.state == "succeeded":
                    goal_handle.succeed()
                elif last_snapshot.state == "canceled":
                    goal_handle.canceled()
                else:
                    goal_handle.abort()
                return self._make_result_from_snapshot(last_snapshot, accepted=True)
            time.sleep(self._config.feedback_period_s)

        goal_handle.abort()
        return self._make_result_from_snapshot(last_snapshot, accepted=True)

    def _handle_status_request(self, request, response):  # type: ignore[no-untyped-def]
        del request
        snapshot = self._handler.get_status()
        response.active = snapshot.active
        response.mission_id = snapshot.mission_id
        response.state = snapshot.state
        response.raw_command = snapshot.raw_command
        response.current_step_id = snapshot.current_step_id
        response.current_skill = snapshot.current_skill
        response.message = snapshot.message
        response.elapsed_s = float(snapshot.elapsed_s)
        return response

    def _make_feedback(self, snapshot: MissionStatusSnapshot):
        assert self._execute_mission is not None
        feedback = self._execute_mission.Feedback()
        feedback.mission_id = snapshot.mission_id
        feedback.state = snapshot.state
        feedback.current_step_id = snapshot.current_step_id
        feedback.current_skill = snapshot.current_skill
        feedback.message = snapshot.message
        feedback.elapsed_s = float(snapshot.elapsed_s)
        return feedback

    def _make_result_from_submission(self, submission: MissionSubmission):
        assert self._execute_mission is not None
        result = self._execute_mission.Result()
        result.accepted = submission.accepted
        result.mission_id = submission.mission_id
        result.final_state = submission.final_state
        result.error_code = submission.error_code
        result.message = submission.message
        return result

    def _make_result_from_snapshot(self, snapshot: MissionStatusSnapshot, *, accepted: bool):
        assert self._execute_mission is not None
        result = self._execute_mission.Result()
        result.accepted = accepted
        result.mission_id = snapshot.mission_id
        result.final_state = snapshot.state
        result.error_code = snapshot.error_code
        result.message = snapshot.message
        return result


def build_command_server(
    *,
    planner_client,
    grounding_client,
    ros_client,
    runner_config=None,
    server_config: CommandServerConfig | None = None,
    check_vlm_health: bool = True,
    semantic_map=None,
    background_mapper=None,
):
    """Construct a mission runner and wrap it in the Jetson-local command server.

    When *check_vlm_health* is ``True`` (the default), the grounding client and
    planner client are health-checked in parallel. If either is reachable but
    reports ``model_loaded: false``, a ``RuntimeError`` is raised. Unreachable
    services propagate their original exception.
    """

    if check_vlm_health:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        probes: dict[str, Any] = {}
        if hasattr(grounding_client, "health"):
            probes["VLM"] = grounding_client.health
        if hasattr(planner_client, "health"):
            probes["Planner"] = planner_client.health

        if probes:
            _logger.info("Running parallel health checks for: %s", list(probes))
            with ThreadPoolExecutor(max_workers=max(2, len(probes))) as pool:
                futures = {pool.submit(fn): name for name, fn in probes.items()}
                for future in as_completed(futures):
                    name = futures[future]
                    health = future.result(timeout=10.0)
                    if not health.get("model_loaded", False):
                        if name == "VLM":
                            from strafer_autonomy.clients.vlm_client import (
                                GroundingServiceUnavailable,
                            )
                            raise GroundingServiceUnavailable(
                                f"VLM service is reachable but model is not loaded: {health}"
                            )
                        raise RuntimeError(
                            f"{name} service is reachable but model is not loaded: {health}"
                        )
                    _logger.info("%s service healthy: %s", name, health)

    from .mission_runner import MissionRunner

    runner = MissionRunner(
        planner_client=planner_client,
        grounding_client=grounding_client,
        ros_client=ros_client,
        config=runner_config,
        semantic_map=semantic_map,
        background_mapper=background_mapper,
    )
    return AutonomyCommandServer(handler=runner, config=server_config), runner
