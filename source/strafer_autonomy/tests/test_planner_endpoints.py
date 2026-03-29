"""Tests for planner FastAPI endpoints with mocked LLM."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from strafer_autonomy.planner.app import _state, create_app


@pytest.fixture(autouse=True)
def _block_model_download(monkeypatch):
    """Prevent lifespan from downloading a real model during tests."""
    monkeypatch.setenv("PLANNER_MODEL", "/nonexistent")


@pytest.fixture()
def client():
    """Create a TestClient without triggering lifespan model loading."""
    app = create_app()
    # Manually mark the LLM as ready with a mock
    _state.llm.ready = True
    _state.llm.model_name = "test-model"
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    _state.llm.ready = False
    _state.llm.model_name = ""


@pytest.fixture()
def unloaded_client():
    """Client where the LLM is NOT loaded."""
    app = create_app()
    _state.llm.ready = False
    _state.llm.model_name = ""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_name"] == "test-model"

    def test_health_loading(self, unloaded_client):
        resp = unloaded_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "loading"
        assert body["model_loaded"] is False


class TestPlanEndpoint:
    _GO_TO_DOOR_LLM_OUTPUT = '{"intent_type": "go_to_target", "target_label": "door", "wait_mode": null, "requires_grounding": true}'
    _CANCEL_LLM_OUTPUT = '{"intent_type": "cancel", "target_label": null, "wait_mode": null, "requires_grounding": false}'
    _STATUS_LLM_OUTPUT = '{"intent_type": "status", "target_label": null, "wait_mode": null, "requires_grounding": false}'
    _WAIT_BY_COUCH_LLM_OUTPUT = '{"intent_type": "wait_by_target", "target_label": "couch", "wait_mode": "until_next_command", "requires_grounding": true}'

    def _post_plan(self, client, raw_command="go to the door", llm_output=None):
        llm_output = llm_output or self._GO_TO_DOOR_LLM_OUTPUT
        with patch.object(_state.llm, "generate", return_value=llm_output):
            return client.post("/plan", json={
                "request_id": "test_001",
                "raw_command": raw_command,
                "available_skills": [
                    "capture_scene_observation", "locate_semantic_target",
                    "project_detection_to_goal_pose", "navigate_to_pose",
                    "wait", "cancel_mission", "report_status",
                ],
            })

    def test_go_to_target(self, client):
        resp = self._post_plan(client, "go to the door", self._GO_TO_DOOR_LLM_OUTPUT)
        assert resp.status_code == 200
        body = resp.json()
        assert body["mission_type"] == "go_to_target"
        assert body["raw_command"] == "go to the door"
        assert len(body["steps"]) == 3
        skills = [s["skill"] for s in body["steps"]]
        assert skills == [
            "scan_for_target",
            "project_detection_to_goal_pose",
            "navigate_to_pose",
        ]

    def test_cancel(self, client):
        resp = self._post_plan(client, "stop", self._CANCEL_LLM_OUTPUT)
        assert resp.status_code == 200
        body = resp.json()
        assert body["mission_type"] == "cancel"
        assert len(body["steps"]) == 1
        assert body["steps"][0]["skill"] == "cancel_mission"

    def test_status(self, client):
        resp = self._post_plan(client, "what are you doing", self._STATUS_LLM_OUTPUT)
        assert resp.status_code == 200
        assert resp.json()["mission_type"] == "status"

    def test_wait_by_target(self, client):
        resp = self._post_plan(client, "wait by the couch", self._WAIT_BY_COUCH_LLM_OUTPUT)
        assert resp.status_code == 200
        body = resp.json()
        assert body["mission_type"] == "wait_by_target"
        assert len(body["steps"]) == 4
        assert body["steps"][-1]["skill"] == "wait"

    def test_model_not_loaded_returns_503(self, unloaded_client):
        resp = unloaded_client.post("/plan", json={
            "request_id": "test_002",
            "raw_command": "go to the door",
        })
        assert resp.status_code == 503

    def test_empty_command_returns_400(self, client):
        with patch.object(_state.llm, "generate", return_value="{}"):
            resp = client.post("/plan", json={
                "request_id": "test_003",
                "raw_command": "",
            })
        assert resp.status_code == 400

    def test_malformed_llm_output_returns_422(self, client):
        resp = self._post_plan(client, "go to the door", "I cannot understand this request")
        assert resp.status_code == 422

    def test_response_has_created_at(self, client):
        resp = self._post_plan(client)
        assert resp.status_code == 200
        assert "created_at" in resp.json()
        assert isinstance(resp.json()["created_at"], float)
