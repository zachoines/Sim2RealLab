"""Tests for the POST /describe VLM service endpoint."""

from __future__ import annotations

import base64
import io
from unittest.mock import patch

import pytest
from PIL import Image


def _make_jpeg_b64(width: int = 64, height: int = 64) -> str:
    img = Image.new("RGB", (width, height), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_describe_payload(
    *,
    request_id: str = "desc-001",
    image_jpeg_b64: str | None = None,
    prompt: str | None = None,
) -> dict:
    payload: dict = {
        "request_id": request_id,
        "image_jpeg_b64": image_jpeg_b64 or _make_jpeg_b64(),
    }
    if prompt is not None:
        payload["prompt"] = prompt
    return payload


@pytest.fixture()
def client(mock_vlm_client):
    return mock_vlm_client


class TestDescribeEndpoint:
    def test_describe_returns_text(self, client):
        with patch(
            "strafer_vlm.inference.qwen_runtime.run_grounding_generation",
            return_value="A hallway with a closed door and a bookshelf.",
        ):
            resp = client.post("/describe", json=_make_describe_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["request_id"] == "desc-001"
        assert body["description"] == "A hallway with a closed door and a bookshelf."
        assert body["latency_s"] >= 0

    def test_describe_custom_prompt(self, client):
        with patch(
            "strafer_vlm.inference.qwen_runtime.run_grounding_generation",
            return_value="I see a red ball.",
        ) as mock_gen:
            resp = client.post(
                "/describe",
                json=_make_describe_payload(prompt="List all toys."),
            )
        assert resp.status_code == 200
        # Verify the custom prompt was forwarded
        call_kwargs = mock_gen.call_args
        assert call_kwargs[1]["prompt"] == "List all toys."

    def test_describe_image_size_guard(self, client):
        payload = _make_describe_payload(image_jpeg_b64=_make_jpeg_b64(5000, 5000))
        resp = client.post("/describe", json=payload)
        assert resp.status_code == 400
        assert "MP" in resp.json()["detail"]

    def test_describe_invalid_image(self, client):
        payload = _make_describe_payload(image_jpeg_b64="not-valid-base64!!!")
        resp = client.post("/describe", json=payload)
        assert resp.status_code == 400
        assert "image" in resp.json()["detail"].lower()

    def test_describe_503_when_model_not_loaded(self, client):
        from strafer_vlm.service.app import _state
        _state.ready = False
        try:
            resp = client.post("/describe", json=_make_describe_payload())
            assert resp.status_code == 503
        finally:
            _state.ready = True

    def test_describe_missing_fields_returns_422(self, client):
        resp = client.post("/describe", json={})
        assert resp.status_code == 422
