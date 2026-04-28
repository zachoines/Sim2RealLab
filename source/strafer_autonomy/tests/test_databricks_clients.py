"""Tests for Databricks Model Serving client implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from strafer_autonomy.clients.databricks_planner_client import (
    DatabricksServingPlannerClient,
    DatabricksServingPlannerClientConfig,
)
from strafer_autonomy.clients.databricks_vlm_client import (
    DatabricksServingGroundingClient,
    DatabricksServingGroundingClientConfig,
)
from strafer_autonomy.clients.planner_client import PlannerServiceUnavailable
from strafer_autonomy.clients.vlm_client import GroundingServiceUnavailable
from strafer_autonomy.schemas import GroundingRequest, PlannerRequest


def _make_planner_config(**overrides) -> DatabricksServingPlannerClientConfig:
    defaults = dict(
        endpoint_name="strafer-planner",
        workspace_url="https://example.databricks.net",
        token="dapi-fake-token",
        timeout_s=5.0,
        max_retries=1,
        retry_backoff_s=0.0,
    )
    defaults.update(overrides)
    return DatabricksServingPlannerClientConfig(**defaults)


def _make_vlm_config(**overrides) -> DatabricksServingGroundingClientConfig:
    defaults = dict(
        endpoint_name="strafer-vlm",
        workspace_url="https://example.databricks.net",
        token="dapi-fake-token",
        timeout_s=5.0,
        max_retries=1,
        retry_backoff_s=0.0,
    )
    defaults.update(overrides)
    return DatabricksServingGroundingClientConfig(**defaults)


def _mock_response(*, status_code: int = 200, json_body=None, text: str = ""):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_body or {}
    if status_code >= 400:
        exc = requests.HTTPError(response=resp)
        resp.raise_for_status.side_effect = exc
    else:
        resp.raise_for_status.return_value = None
    return resp


def _rgb_image(width: int = 16, height: int = 16) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Planner client
# ---------------------------------------------------------------------------


class TestDatabricksPlannerClient:
    def test_invocations_url(self):
        cfg = _make_planner_config()
        assert cfg.invocations_url() == (
            "https://example.databricks.net/serving-endpoints/strafer-planner/invocations"
        )

    def test_plan_mission_happy_path(self):
        cfg = _make_planner_config()
        client = DatabricksServingPlannerClient(cfg)
        prediction = {
            "mission_id": "mission_abc",
            "mission_type": "cancel",
            "raw_command": "stop",
            "steps": [
                {"step_id": "step_01", "skill": "cancel_mission", "args": {}, "timeout_s": 5.0},
            ],
            "created_at": 12345.0,
        }
        with patch.object(
            client._session,
            "post",
            return_value=_mock_response(json_body={"predictions": [prediction]}),
        ) as mock_post:
            plan = client.plan_mission(
                PlannerRequest(request_id="r1", raw_command="stop"),
            )
        assert plan.mission_id == "mission_abc"
        assert plan.mission_type == "cancel"
        assert len(plan.steps) == 1
        # URL and auth header wired through
        call_url = mock_post.call_args.args[0]
        assert "/serving-endpoints/strafer-planner/invocations" in call_url
        assert client._session.headers["Authorization"] == "Bearer dapi-fake-token"

    def test_plan_mission_sends_inputs_row(self):
        cfg = _make_planner_config()
        client = DatabricksServingPlannerClient(cfg)
        prediction = {
            "mission_id": "m",
            "mission_type": "cancel",
            "raw_command": "stop",
            "steps": [],
            "created_at": 0.0,
        }
        with patch.object(
            client._session,
            "post",
            return_value=_mock_response(json_body={"predictions": [prediction]}),
        ) as mock_post:
            client.plan_mission(
                PlannerRequest(
                    request_id="r1",
                    raw_command="stop",
                    available_skills=("cancel_mission",),
                ),
            )
        body = mock_post.call_args.kwargs["json"]
        assert "inputs" in body and len(body["inputs"]) == 1
        row = body["inputs"][0]
        assert row["raw_command"] == "stop"
        assert row["request_id"] == "r1"
        assert row["available_skills"] == ["cancel_mission"]

    def test_client_error_not_retried(self):
        cfg = _make_planner_config(max_retries=3)
        client = DatabricksServingPlannerClient(cfg)
        with patch.object(
            client._session,
            "post",
            return_value=_mock_response(status_code=400, text="bad request"),
        ) as mock_post:
            with pytest.raises(PlannerServiceUnavailable, match="400"):
                client.plan_mission(PlannerRequest(request_id="r1", raw_command="stop"))
        assert mock_post.call_count == 1

    def test_server_error_retried(self):
        cfg = _make_planner_config(max_retries=2, retry_backoff_s=0.0)
        client = DatabricksServingPlannerClient(cfg)
        with patch.object(
            client._session,
            "post",
            return_value=_mock_response(status_code=503, text="nope"),
        ) as mock_post:
            with pytest.raises(PlannerServiceUnavailable):
                client.plan_mission(PlannerRequest(request_id="r1", raw_command="stop"))
        assert mock_post.call_count == 3  # 1 + max_retries

    def test_connection_error_retried(self):
        cfg = _make_planner_config(max_retries=2, retry_backoff_s=0.0)
        client = DatabricksServingPlannerClient(cfg)
        with patch.object(
            client._session,
            "post",
            side_effect=requests.ConnectionError("dns"),
        ) as mock_post:
            with pytest.raises(PlannerServiceUnavailable):
                client.plan_mission(PlannerRequest(request_id="r1", raw_command="stop"))
        assert mock_post.call_count == 3

    def test_empty_predictions_raises(self):
        cfg = _make_planner_config()
        client = DatabricksServingPlannerClient(cfg)
        with patch.object(
            client._session,
            "post",
            return_value=_mock_response(json_body={"predictions": []}),
        ):
            with pytest.raises(PlannerServiceUnavailable, match="missing predictions"):
                client.plan_mission(PlannerRequest(request_id="r1", raw_command="stop"))

    def test_health_ready(self):
        cfg = _make_planner_config()
        client = DatabricksServingPlannerClient(cfg)
        with patch.object(
            client._session,
            "get",
            return_value=_mock_response(
                json_body={"state": {"ready": "READY", "config_update": "NOT_UPDATING"}},
            ),
        ) as mock_get:
            health = client.health()
        assert health["status"] == "ok"
        assert health["model_loaded"] is True
        assert health["backend"] == "databricks"
        assert mock_get.call_args.args[0].endswith("/api/2.0/serving-endpoints/strafer-planner")

    def test_health_not_ready(self):
        cfg = _make_planner_config()
        client = DatabricksServingPlannerClient(cfg)
        with patch.object(
            client._session,
            "get",
            return_value=_mock_response(
                json_body={"state": {"ready": "NOT_READY", "config_update": "UPDATING"}},
            ),
        ):
            health = client.health()
        assert health["status"] == "loading"
        assert health["model_loaded"] is False


# ---------------------------------------------------------------------------
# VLM client
# ---------------------------------------------------------------------------


class TestDatabricksVlmClient:
    def test_locate_semantic_target(self):
        cfg = _make_vlm_config()
        client = DatabricksServingGroundingClient(cfg)
        prediction = {
            "request_id": "r1",
            "found": True,
            "bbox_2d": [100, 200, 300, 400],
            "label": "door",
            "confidence": 0.9,
            "raw_output": None,
            "latency_s": 0.25,
        }
        with patch.object(
            client._session,
            "post",
            return_value=_mock_response(json_body={"predictions": [prediction]}),
        ) as mock_post:
            result = client.locate_semantic_target(
                GroundingRequest(
                    request_id="r1",
                    prompt="door",
                    image_rgb_u8=_rgb_image(),
                    image_stamp_sec=0.0,
                ),
            )
        assert result.found is True
        assert result.bbox_2d == (100, 200, 300, 400)
        assert result.label == "door"
        row = mock_post.call_args.kwargs["json"]["inputs"][0]
        assert row["mode"] == "ground"
        assert row["prompt"] == "door"
        assert isinstance(row["image_b64"], str) and len(row["image_b64"]) > 0

    def test_describe_scene(self):
        cfg = _make_vlm_config()
        client = DatabricksServingGroundingClient(cfg)
        prediction = {
            "request_id": "r2",
            "description": "A bright kitchen with a wooden table.",
            "latency_s": 1.1,
        }
        with patch.object(
            client._session,
            "post",
            return_value=_mock_response(json_body={"predictions": [prediction]}),
        ) as mock_post:
            scene = client.describe_scene(
                request_id="r2",
                image_rgb_u8=_rgb_image(),
                prompt="What is in this room?",
            )
        assert scene.description.startswith("A bright")
        assert scene.latency_s == pytest.approx(1.1)
        row = mock_post.call_args.kwargs["json"]["inputs"][0]
        assert row["mode"] == "describe"
        assert row["prompt"] == "What is in this room?"

    def test_client_error_not_retried(self):
        cfg = _make_vlm_config(max_retries=3)
        client = DatabricksServingGroundingClient(cfg)
        with patch.object(
            client._session,
            "post",
            return_value=_mock_response(status_code=404, text="missing"),
        ) as mock_post:
            with pytest.raises(GroundingServiceUnavailable, match="404"):
                client.locate_semantic_target(
                    GroundingRequest(
                        request_id="r1",
                        prompt="door",
                        image_rgb_u8=_rgb_image(),
                        image_stamp_sec=0.0,
                    ),
                )
        assert mock_post.call_count == 1

    def test_health_ok(self):
        cfg = _make_vlm_config()
        client = DatabricksServingGroundingClient(cfg)
        with patch.object(
            client._session,
            "get",
            return_value=_mock_response(
                json_body={"state": {"ready": "READY", "config_update": "NOT_UPDATING"}},
            ),
        ):
            health = client.health()
        assert health["status"] == "ok"
        assert health["backend"] == "databricks"
