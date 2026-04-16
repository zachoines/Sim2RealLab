"""Tests for build_command_server health-check-at-startup behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strafer_autonomy.clients.vlm_client import GroundingServiceUnavailable
from strafer_autonomy.executor.command_server import build_command_server


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
