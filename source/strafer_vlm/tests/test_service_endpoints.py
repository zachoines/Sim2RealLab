"""Integration tests for the FastAPI grounding service endpoints.

These tests mock the heavy model to avoid GPU dependency.
"""

from __future__ import annotations

import base64
import io
import time
from unittest.mock import patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_b64(width: int = 64, height: int = 64) -> str:
    img = Image.new("RGB", (width, height), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_payload(
    *,
    prompt: str = "Locate: chair",
    request_id: str = "test-001",
    image_jpeg_b64: str | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "prompt": prompt,
        "image_jpeg_b64": image_jpeg_b64 or _make_jpeg_b64(),
    }


# ---------------------------------------------------------------------------
# Alias shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(mock_vlm_client):
    """Alias the shared conftest fixture for readability."""
    return mock_vlm_client


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model_name"] == "mock-model"


# ---------------------------------------------------------------------------
# /ground endpoint — validation
# ---------------------------------------------------------------------------


class TestGroundValidation:
    def test_empty_prompt_returns_400(self, client):
        payload = _make_payload(prompt="")
        resp = client.post("/ground", json=payload)
        assert resp.status_code == 400
        assert "non-empty" in resp.json()["detail"]

    def test_whitespace_prompt_returns_400(self, client):
        payload = _make_payload(prompt="   ")
        resp = client.post("/ground", json=payload)
        assert resp.status_code == 400

    def test_invalid_image_returns_400(self, client):
        payload = _make_payload(image_jpeg_b64="not-valid-base64!!!")
        resp = client.post("/ground", json=payload)
        assert resp.status_code == 400
        assert "image" in resp.json()["detail"].lower()

    def test_missing_fields_returns_422(self, client):
        resp = client.post("/ground", json={})
        assert resp.status_code == 422

    def test_oversized_image_returns_400(self, client):
        # Create a 5000x5000 image (25 MP, exceeds default 20 MP limit)
        payload = _make_payload(image_jpeg_b64=_make_jpeg_b64(5000, 5000))
        resp = client.post("/ground", json=payload)
        assert resp.status_code == 400
        assert "MP" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /ground endpoint — successful grounding (mocked inference)
# ---------------------------------------------------------------------------


class TestGroundInference:
    @patch("strafer_vlm.inference.qwen_runtime.run_grounding_generation")
    def test_found_object(self, mock_gen, client):
        mock_gen.return_value = '{"found": true, "bbox_2d": [100, 200, 500, 800], "label": "chair"}'

        resp = client.post("/ground", json=_make_payload())

        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["bbox_2d"] is not None
        assert data["request_id"] == "test-001"
        assert data["latency_s"] >= 0

    @patch("strafer_vlm.inference.qwen_runtime.run_grounding_generation")
    def test_not_found(self, mock_gen, client):
        mock_gen.return_value = '{"found": false}'

        resp = client.post("/ground", json=_make_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is False
        assert data["bbox_2d"] is None

    @patch("strafer_vlm.inference.qwen_runtime.run_grounding_generation")
    def test_unparseable_output(self, mock_gen, client):
        mock_gen.return_value = "I don't understand the question."

        resp = client.post("/ground", json=_make_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is False
        assert data["raw_output"] == "I don't understand the question."


# ---------------------------------------------------------------------------
# /ground endpoint — inference timeout
# ---------------------------------------------------------------------------


class TestGroundTimeout:
    @patch("strafer_vlm.inference.qwen_runtime.run_grounding_generation")
    def test_inference_timeout_returns_504(self, mock_gen):
        """When inference exceeds the timeout, the service returns 504."""

        def slow_inference(**kwargs):
            time.sleep(5)
            return '{"found": true, "bbox_2d": [0, 0, 500, 500]}'

        mock_gen.side_effect = slow_inference

        # Build app with a very short timeout
        import os
        from contextlib import asynccontextmanager
        from unittest.mock import MagicMock

        from starlette.testclient import TestClient

        from strafer_vlm.service.app import _state, create_app

        _state.model = MagicMock()
        _state.processor = MagicMock()
        _state.model_name = "mock-model"
        _state.ready = True

        @asynccontextmanager
        async def _noop_lifespan(app):
            yield

        orig = os.environ.get("GROUNDING_INFERENCE_TIMEOUT")
        os.environ["GROUNDING_INFERENCE_TIMEOUT"] = "0.5"
        try:
            app = create_app()
            app.router.lifespan_context = _noop_lifespan
            with TestClient(app) as c:
                resp = c.post("/ground", json=_make_payload())
                assert resp.status_code == 504
                assert "timed out" in resp.json()["detail"].lower()
        finally:
            if orig is None:
                os.environ.pop("GROUNDING_INFERENCE_TIMEOUT", None)
            else:
                os.environ["GROUNDING_INFERENCE_TIMEOUT"] = orig
            _state.ready = False
            _state.model = None
            _state.processor = None


# ---------------------------------------------------------------------------
# /ground endpoint — debug overlay
# ---------------------------------------------------------------------------


class TestGroundDebugOverlay:
    @patch("strafer_vlm.inference.qwen_runtime.run_grounding_generation")
    def test_overlay_returned_when_found(self, mock_gen, client):
        mock_gen.return_value = '{"found": true, "bbox_2d": [100, 200, 500, 800], "label": "chair"}'

        payload = _make_payload()
        payload["return_debug_overlay"] = True
        resp = client.post("/ground", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["debug_overlay_jpeg_b64"] is not None

        # Verify it's valid JPEG base64
        overlay_bytes = base64.b64decode(data["debug_overlay_jpeg_b64"])
        img = Image.open(io.BytesIO(overlay_bytes))
        assert img.format == "JPEG"

    @patch("strafer_vlm.inference.qwen_runtime.run_grounding_generation")
    def test_overlay_null_when_not_found(self, mock_gen, client):
        mock_gen.return_value = '{"found": false}'

        payload = _make_payload()
        payload["return_debug_overlay"] = True
        resp = client.post("/ground", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is False
        assert data["debug_overlay_jpeg_b64"] is None

    @patch("strafer_vlm.inference.qwen_runtime.run_grounding_generation")
    def test_overlay_null_when_not_requested(self, mock_gen, client):
        mock_gen.return_value = '{"found": true, "bbox_2d": [100, 200, 500, 800], "label": "chair"}'

        payload = _make_payload()
        # return_debug_overlay defaults to False
        resp = client.post("/ground", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["debug_overlay_jpeg_b64"] is None
