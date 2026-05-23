"""Tests for build_command_server health-check-at-startup behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strafer_autonomy.clients.vlm_client import GroundingServiceUnavailable
from strafer_autonomy.executor.command_server import (
    AutonomyCommandServer,
    CommandServerConfig,
    build_command_server,
)


def _make_grounding_client(*, health_response=None, health_raises=None):
    """Return a mock grounding client with optional health() behaviour."""
    client = MagicMock()
    if health_raises is not None:
        client.health.side_effect = health_raises
    elif health_response is not None:
        client.health.return_value = health_response
    else:
        client.health.return_value = {
            "status": "ok",
            "model_loaded": True,
            "model_name": "test-model",
        }
    return client


class TestBuildCommandServerHealth:
    def test_healthy_service_creates_server(self):
        client = _make_grounding_client()
        server, runner = build_command_server(
            planner_client=MagicMock(),
            grounding_client=client,
            ros_client=MagicMock(),
        )
        assert server is not None
        assert runner is not None
        client.health.assert_called_once()

    def test_model_not_loaded_raises(self):
        client = _make_grounding_client(
            health_response={"status": "loading", "model_loaded": False, "model_name": None},
        )
        with pytest.raises(GroundingServiceUnavailable, match="model is not loaded"):
            build_command_server(
                planner_client=MagicMock(),
                grounding_client=client,
                ros_client=MagicMock(),
            )

    def test_unreachable_service_propagates(self):
        client = _make_grounding_client(
            health_raises=GroundingServiceUnavailable("connection refused"),
        )
        with pytest.raises(GroundingServiceUnavailable):
            build_command_server(
                planner_client=MagicMock(),
                grounding_client=client,
                ros_client=MagicMock(),
            )

    def test_skip_health_check(self):
        client = _make_grounding_client(
            health_raises=GroundingServiceUnavailable("should not be called"),
        )
        server, runner = build_command_server(
            planner_client=MagicMock(),
            grounding_client=client,
            ros_client=MagicMock(),
            check_vlm_health=False,
        )
        assert server is not None
        client.health.assert_not_called()

    def test_no_health_method_skips_check(self):
        """Clients without a health() method (e.g. test mocks) pass through."""
        client = MagicMock(spec=["locate_semantic_target"])
        server, runner = build_command_server(
            planner_client=MagicMock(spec=["plan_mission"]),
            grounding_client=client,
            ros_client=MagicMock(),
        )
        assert server is not None

    def test_parallel_planner_health_check(self):
        vlm = _make_grounding_client()
        planner = MagicMock()
        planner.health.return_value = {"status": "ok", "model_loaded": True}
        server, runner = build_command_server(
            planner_client=planner,
            grounding_client=vlm,
            ros_client=MagicMock(),
        )
        vlm.health.assert_called_once()
        planner.health.assert_called_once()

    def test_planner_not_loaded_raises(self):
        vlm = _make_grounding_client()
        planner = MagicMock()
        planner.health.return_value = {"status": "loading", "model_loaded": False}
        with pytest.raises(RuntimeError, match="Planner"):
            build_command_server(
                planner_client=planner,
                grounding_client=vlm,
                ros_client=MagicMock(),
            )


# ------------------------------------------------------------------
# Tests — concurrent callback scheduling for cancel during execute
# ------------------------------------------------------------------


class TestCommandServerConcurrency:
    """The action server must keep the cancel + status callbacks
    responsive while ``execute_callback`` is in its mission-feedback
    loop. Single-threaded ``rclpy.spin`` + the default
    ``MutuallyExclusiveCallbackGroup`` lets the execute callback hold
    the only callback thread for the entire mission — so the cancel
    service never fires and ``strafer-autonomy-cli cancel`` hangs.
    """

    def test_config_default_executor_threads_supports_concurrent_cancel(self):
        """At least two executor threads are required: one for the
        long-running execute_callback, one for cancel + status."""
        assert CommandServerConfig().executor_num_threads >= 2

    @pytest.mark.requires_ros
    def test_ensure_ros_entities_wires_multithreaded_executor(self):
        """``_ensure_ros_entities`` constructs a ``MultiThreadedExecutor``
        and a ``ReentrantCallbackGroup``, shared by the action server
        and the status service. Without this wiring, cancel during an
        active mission hangs because execute_callback blocks the only
        callback thread.
        """
        import rclpy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor

        rclpy.init()
        try:
            handler = MagicMock()
            server = AutonomyCommandServer(handler=handler)
            try:
                server._ensure_ros_entities()
                assert isinstance(server._executor, MultiThreadedExecutor)
                assert (
                    server._action_server.callback_group.__class__
                    is ReentrantCallbackGroup
                )
                # Status service shares the same reentrant group so a
                # status query during execute_callback also resolves.
                assert (
                    server._status_service.callback_group
                    is server._action_server.callback_group
                )
            finally:
                server.destroy()
        finally:
            rclpy.shutdown()
