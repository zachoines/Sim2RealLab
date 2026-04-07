"""Tests for HttpPlannerClient with mocked HTTP responses."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from strafer_autonomy.clients.planner_client import (
    HttpPlannerClient,
    HttpPlannerClientConfig,
    PlannerServiceUnavailable,
    mission_plan_from_payload,
    planner_request_to_payload,
)
from strafer_autonomy.schemas import PlannerRequest


def _make_config(**kwargs) -> HttpPlannerClientConfig:
    defaults = {"base_url": "http://localhost:8200"}
    defaults.update(kwargs)
    return HttpPlannerClientConfig(**defaults)


def _make_request(**kwargs) -> PlannerRequest:
    defaults = {
        "request_id": "plan_001",
        "raw_command": "go to the door",
        "available_skills": ("navigate_to_pose", "wait"),
    }
    defaults.update(kwargs)
    return PlannerRequest(**defaults)


_PLAN_RESPONSE = {
    "mission_id": "mission_001",
    "mission_type": "go_to_target",
    "raw_command": "go to the door",
    "steps": [
        {"step_id": "step_01", "skill": "capture_scene_observation", "args": {}, "timeout_s": 5.0, "retry_limit": 0},
        {"step_id": "step_02", "skill": "locate_semantic_target", "args": {"label": "door"}, "timeout_s": 8.0, "retry_limit": 1},
    ],
    "created_at": 1710000000.0,
}


class TestPlannerRequestToPayload:
    def test_basic(self):
        req = _make_request()
        payload = planner_request_to_payload(req)
        assert payload["request_id"] == "plan_001"
        assert payload["raw_command"] == "go to the door"
        assert isinstance(payload["available_skills"], list)

    def test_none_fields(self):
        req = _make_request(robot_state=None, active_mission_summary=None)
        payload = planner_request_to_payload(req)
        assert payload["robot_state"] is None
        assert payload["active_mission_summary"] is None


class TestMissionPlanFromPayload:
    def test_basic(self):
        plan = mission_plan_from_payload(_PLAN_RESPONSE)
        assert plan.mission_id == "mission_001"
        assert plan.mission_type == "go_to_target"
        assert len(plan.steps) == 2
        assert plan.steps[0].skill == "capture_scene_observation"
        assert plan.steps[1].args["label"] == "door"


class TestHttpPlannerClient:
    def test_plan_mission_success(self):
        config = _make_config()
        client = HttpPlannerClient(config)
        mock_response = MagicMock()
        mock_response.json.return_value = _PLAN_RESPONSE
        mock_response.raise_for_status = MagicMock()
        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            plan = client.plan_mission(_make_request())
        assert plan.mission_id == "mission_001"
        mock_post.assert_called_once()

    def test_connection_error_retries(self):
        config = _make_config(max_retries=2, retry_backoff_s=0.01)
        client = HttpPlannerClient(config)
        with patch.object(
            client._session, "post",
            side_effect=requests.ConnectionError("refused"),
        ):
            with pytest.raises(PlannerServiceUnavailable, match="unreachable after 3 attempts"):
                client.plan_mission(_make_request())

    def test_timeout_retries(self):
        config = _make_config(max_retries=1, retry_backoff_s=0.01)
        client = HttpPlannerClient(config)
        with patch.object(
            client._session, "post",
            side_effect=requests.Timeout("timed out"),
        ):
            with pytest.raises(PlannerServiceUnavailable):
                client.plan_mission(_make_request())

    def test_4xx_does_not_retry(self):
        config = _make_config(max_retries=2, retry_backoff_s=0.01)
        client = HttpPlannerClient(config)
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Unprocessable"
        exc = requests.HTTPError(response=mock_response)
        with patch.object(client._session, "post", side_effect=exc):
            with pytest.raises(PlannerServiceUnavailable, match="422"):
                client.plan_mission(_make_request())

    def test_5xx_retries(self):
        config = _make_config(max_retries=1, retry_backoff_s=0.01)
        client = HttpPlannerClient(config)
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        exc = requests.HTTPError(response=mock_response)
        with patch.object(client._session, "post", side_effect=exc):
            with pytest.raises(PlannerServiceUnavailable, match="unreachable after 2 attempts"):
                client.plan_mission(_make_request())

    def test_health_success(self):
        config = _make_config()
        client = HttpPlannerClient(config)
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "model_loaded": True}
        mock_response.raise_for_status = MagicMock()
        with patch.object(client._session, "get", return_value=mock_response):
            result = client.health()
        assert result["status"] == "ok"

    def test_health_connection_error(self):
        config = _make_config()
        client = HttpPlannerClient(config)
        with patch.object(
            client._session, "get",
            side_effect=requests.ConnectionError("refused"),
        ):
            with pytest.raises(PlannerServiceUnavailable, match="health check failed"):
                client.health()
