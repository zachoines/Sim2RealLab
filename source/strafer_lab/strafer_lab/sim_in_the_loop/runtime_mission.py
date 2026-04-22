"""Runtime ``MissionApi`` backed by the Jetson's ``execute_mission`` action.

Wraps an ``rclpy.action.ActionClient`` for ``execute_mission`` plus a
service client for ``get_mission_status``, exposing the
:class:`strafer_lab.sim_in_the_loop.harness.MissionApi` protocol.

Topology assumed at construction time:
  - The Jetson autonomy executor is running and advertises both
    ``execute_mission`` (action) and ``get_mission_status`` (service)
    on the same ROS_DOMAIN_ID this process is configured with.
  - ``RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`` (set by ``env_setup.sh``)
    so cross-host discovery works over LAN.

This module imports ``rclpy`` lazily inside its constructor so it can
be import-checked from a plain Python environment for syntax / type
checking without a ROS install. Tests of the harness use a fake
``MissionApi`` rather than mocking ``rclpy``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from strafer_lab.sim_in_the_loop.harness import MissionApi, MissionStatus

logger = logging.getLogger(__name__)


# Terminal states the executor reports for ``execute_mission``. Matches
# the ``MissionStatusSnapshot.state`` strings produced by
# ``strafer_autonomy.executor.command_server`` /
# ``strafer_autonomy.executor.mission_runner``. Anything not in this set
# is treated as in-flight.
_TERMINAL_STATES = frozenset(
    {"succeeded", "success", "failed", "cancelled", "canceled", "aborted", "timeout"}
)


class Ros2MissionApi(MissionApi):
    """``MissionApi`` driven by a live rclpy ``ActionClient`` + service client.

    Parameters
    ----------
    node_name :
        ROS2 node name used by this client. Defaults to a Strafer-prefixed
        name so concurrent harness runs can be told apart.
    action_name :
        Action name to call. Defaults to ``execute_mission`` to match
        ``DEFAULT_EXECUTE_MISSION_ACTION`` in
        ``strafer_autonomy.executor.command_server``.
    status_service_name :
        Service name for status queries. Defaults to ``get_mission_status``.
    source :
        ``goal.source`` string forwarded to the executor for telemetry —
        identifies who submitted the mission.
    wait_for_server_timeout_s :
        Hard cap on action-server discovery on construction.
    submit_accept_timeout_s :
        Hard cap on the action server's accept response after
        ``send_goal_async``. The executor's accept callback is fast in
        practice; a long wait here usually means the executor is stuck.
    """

    def __init__(
        self,
        *,
        node_name: str = "strafer_sim_in_the_loop_harness",
        action_name: str = "execute_mission",
        status_service_name: str = "get_mission_status",
        source: str = "sim_in_the_loop_harness",
        wait_for_server_timeout_s: float = 10.0,
        submit_accept_timeout_s: float = 5.0,
    ) -> None:
        self._action_name = action_name
        self._status_service_name = status_service_name
        self._source = source
        self._wait_for_server_timeout_s = float(wait_for_server_timeout_s)
        self._submit_accept_timeout_s = float(submit_accept_timeout_s)

        # All rclpy imports are deferred to here so the module is
        # importable for syntax checks without ROS installed.
        import rclpy  # noqa: PLC0415
        from rclpy.action import ActionClient  # noqa: PLC0415
        from rclpy.node import Node  # noqa: PLC0415
        from rclpy.parameter import Parameter  # noqa: PLC0415

        from strafer_msgs.action import ExecuteMission  # noqa: PLC0415
        from strafer_msgs.srv import GetMissionStatus  # noqa: PLC0415

        self._rclpy = rclpy
        self._ExecuteMission = ExecuteMission
        self._GetMissionStatus = GetMissionStatus

        self._owns_rclpy_init = not rclpy.ok()
        if self._owns_rclpy_init:
            rclpy.init(args=None)

        # use_sim_time=True so any stamp this client reads off node.get_clock()
        # matches the bridge's /clock publisher and the Jetson nodes (which
        # also run with use_sim_time:=True in the sim-in-the-loop bringup).
        # Without this the harness would compare wall-time stamps against
        # sim-time stamps coming back from the executor.
        self._node = Node(
            node_name,
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self._action_client = ActionClient(self._node, ExecuteMission, action_name)
        self._status_client = self._node.create_client(
            GetMissionStatus, status_service_name,
        )

        if not self._action_client.wait_for_server(
            timeout_sec=self._wait_for_server_timeout_s,
        ):
            self.shutdown()
            raise RuntimeError(
                f"Action server {action_name!r} did not become available within "
                f"{self._wait_for_server_timeout_s:.1f}s. Confirm the Jetson "
                f"executor is running on the same ROS_DOMAIN_ID."
            )
        if not self._status_client.wait_for_service(
            timeout_sec=self._wait_for_server_timeout_s,
        ):
            self.shutdown()
            raise RuntimeError(
                f"Service {status_service_name!r} did not become available "
                f"within {self._wait_for_server_timeout_s:.1f}s."
            )

        self._goal_handle: Any = None

    # ------------------------------------------------------------------
    # MissionApi protocol
    # ------------------------------------------------------------------

    def submit(self, *, raw_command: str, request_id: str) -> str:
        goal = self._ExecuteMission.Goal()
        goal.request_id = request_id
        goal.raw_command = raw_command
        goal.source = self._source
        goal.replace_active_mission = False

        send_future = self._action_client.send_goal_async(goal)
        self._spin_until_complete(send_future, self._submit_accept_timeout_s)

        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(
                f"Executor rejected mission {request_id!r}: "
                f"{getattr(handle, 'status', 'no_handle')}"
            )

        self._goal_handle = handle
        # The ``execute_mission`` action's mission_id is assigned by the
        # executor and surfaced via the status service — return the
        # request_id here as the externally-visible identifier and let
        # the harness reconcile via status() if it needs the executor's
        # internal id.
        return request_id

    def status(self) -> MissionStatus:
        request = self._GetMissionStatus.Request()
        future = self._status_client.call_async(request)
        self._spin_until_complete(future, timeout_s=2.0)
        response = future.result()
        if response is None:
            return MissionStatus(
                terminal=False, state="unknown",
                error_code="status_no_response",
                message="get_mission_status returned no response",
            )

        state = str(getattr(response, "state", "")).lower()
        terminal = state in _TERMINAL_STATES
        return MissionStatus(
            terminal=terminal,
            state=state,
            error_code="",
            elapsed_s=float(getattr(response, "elapsed_s", 0.0)),
            message=str(getattr(response, "message", "")),
        )

    def cancel(self) -> None:
        if self._goal_handle is None:
            return
        try:
            future = self._goal_handle.cancel_goal_async()
            self._spin_until_complete(future, timeout_s=2.0)
        except Exception:
            logger.exception("cancel_goal_async raised")

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Tear down the rclpy node and (if we owned it) the rclpy context."""

        try:
            if self._node is not None:
                self._node.destroy_node()
        finally:
            self._node = None  # type: ignore[assignment]
            if self._owns_rclpy_init and self._rclpy.ok():
                self._rclpy.shutdown()

    def __enter__(self) -> "Ros2MissionApi":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spin_until_complete(self, future: Any, timeout_s: float) -> None:
        """Spin the node until ``future`` resolves or ``timeout_s`` elapses.

        Uses ``rclpy.spin_until_future_complete`` rather than a manual
        spin_once loop so we get the executor's internal scheduling.
        """

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while not future.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._rclpy.spin_once(self._node, timeout_sec=min(remaining, 0.1))
