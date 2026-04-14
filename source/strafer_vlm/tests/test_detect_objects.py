"""Tests for the POST /detect_objects VLM service endpoint."""

from __future__ import annotations

import base64
import io
from unittest.mock import patch

import pytest
from PIL import Image

from strafer_vlm.inference.parsing import parse_ref_box_detections


def _make_jpeg_b64(width: int = 64, height: int = 64) -> str:
    img = Image.new("RGB", (width, height), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_payload(
    *,
    request_id: str = "det-001",
    image_jpeg_b64: str | None = None,
    max_objects: int = 20,
    min_confidence: float = 0.3,
    max_image_side: int = 1024,
) -> dict:
    return {
        "request_id": request_id,
        "image_jpeg_b64": image_jpeg_b64 or _make_jpeg_b64(),
        "max_image_side": max_image_side,
        "max_objects": max_objects,
        "min_confidence": min_confidence,
    }


@pytest.fixture()
def client(mock_vlm_client):
    return mock_vlm_client


# ---------------------------------------------------------------------------
# Parser unit tests
# ---------------------------------------------------------------------------


class TestParseRefBoxDetections:
    def test_empty_text(self):
        assert parse_ref_box_detections("") == []

    def test_single_object(self):
        text = "<ref>table</ref><box>(120,340),(450,720)</box>"
        parsed = parse_ref_box_detections(text)
        assert len(parsed) == 1
        assert parsed[0].label == "table"
        assert parsed[0].bbox_2d == (120, 340, 450, 720)
        assert parsed[0].confidence == 1.0

    def test_multiple_objects(self):
        text = (
            "<ref>table</ref><box>(120,340),(450,720)</box>"
            "<ref>chair</ref><box>(500,280),(680,710)</box>"
            "<ref>door</ref><box>(780,50),(990,980)</box>"
        )
        parsed = parse_ref_box_detections(text)
        assert [d.label for d in parsed] == ["table", "chair", "door"]
        assert parsed[0].bbox_2d == (120, 340, 450, 720)
        assert parsed[2].bbox_2d == (780, 50, 990, 980)

    def test_multiple_boxes_per_ref(self):
        text = (
            "<ref>box</ref><box>(100,200),(250,400)</box>"
            "<box>(600,300),(780,520)</box>"
        )
        parsed = parse_ref_box_detections(text)
        assert len(parsed) == 2
        assert all(d.label == "box" for d in parsed)
        assert parsed[0].bbox_2d == (100, 200, 250, 400)
        assert parsed[1].bbox_2d == (600, 300, 780, 520)

    def test_deduplicates(self):
        text = (
            "<ref>table</ref><box>(10,20),(30,40)</box>"
            "<ref>table</ref><box>(10,20),(30,40)</box>"
        )
        parsed = parse_ref_box_detections(text)
        assert len(parsed) == 1

    def test_skips_degenerate_box(self):
        text = "<ref>bad</ref><box>(100,100),(100,100)</box>"
        assert parse_ref_box_detections(text) == []

    def test_float_coords_rounded(self):
        text = "<ref>x</ref><box>(10.4,20.6),(30.1,40.9)</box>"
        parsed = parse_ref_box_detections(text)
        assert parsed[0].bbox_2d == (10, 21, 30, 41)

    def test_surrounding_prose_ignored(self):
        text = "Here are the objects I see: <ref>lamp</ref><box>(5,5),(15,15)</box> done."
        parsed = parse_ref_box_detections(text)
        assert len(parsed) == 1
        assert parsed[0].label == "lamp"

    def test_malformed_ref_skipped(self):
        text = "<ref>ok</ref><box>(1,2),(3,4)</box> <ref></ref><box>(5,6),(7,8)</box>"
        parsed = parse_ref_box_detections(text)
        assert len(parsed) == 1
        assert parsed[0].label == "ok"


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestDetectObjectsEndpoint:
    def test_basic_detection(self, client):
        raw = (
            "<ref>table</ref><box>(100,200),(500,800)</box>"
            "<ref>chair</ref><box>(600,300),(900,700)</box>"
        )
        with patch(
            "strafer_vlm.inference.qwen_runtime.run_grounding_generation",
            return_value=raw,
        ):
            resp = client.post("/detect_objects", json=_make_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["request_id"] == "det-001"
        assert len(body["objects"]) == 2
        labels = [o["label"] for o in body["objects"]]
        assert labels == ["table", "chair"]
        # 64x64 image, normalized coords 100..500 → pixel 6..32
        first = body["objects"][0]
        assert first["bbox_2d"][0] == pytest.approx(6, abs=1)
        assert first["confidence"] == 1.0

    def test_max_objects_cap(self, client):
        raw = "".join(
            f"<ref>obj{i}</ref><box>({i*10},{i*10}),({i*10+50},{i*10+50})</box>"
            for i in range(10)
        )
        with patch(
            "strafer_vlm.inference.qwen_runtime.run_grounding_generation",
            return_value=raw,
        ):
            resp = client.post(
                "/detect_objects", json=_make_payload(max_objects=3),
            )
        assert resp.status_code == 200
        assert len(resp.json()["objects"]) == 3

    def test_empty_output(self, client):
        with patch(
            "strafer_vlm.inference.qwen_runtime.run_grounding_generation",
            return_value="",
        ):
            resp = client.post("/detect_objects", json=_make_payload())
        assert resp.status_code == 200
        assert resp.json()["objects"] == []

    def test_invalid_image(self, client):
        payload = _make_payload(image_jpeg_b64="not-base64!!!")
        resp = client.post("/detect_objects", json=payload)
        assert resp.status_code == 400
        assert "image" in resp.json()["detail"].lower()

    def test_image_size_guard(self, client):
        payload = _make_payload(image_jpeg_b64=_make_jpeg_b64(5000, 5000))
        resp = client.post("/detect_objects", json=payload)
        assert resp.status_code == 400

    def test_invalid_max_objects(self, client):
        resp = client.post("/detect_objects", json=_make_payload(max_objects=0))
        assert resp.status_code == 400

    def test_invalid_min_confidence(self, client):
        resp = client.post(
            "/detect_objects", json=_make_payload(min_confidence=1.5),
        )
        assert resp.status_code == 400

    def test_503_when_model_not_loaded(self, client):
        from strafer_vlm.service.app import _state
        _state.ready = False
        try:
            resp = client.post("/detect_objects", json=_make_payload())
            assert resp.status_code == 503
        finally:
            _state.ready = True

    def test_missing_fields_returns_422(self, client):
        resp = client.post("/detect_objects", json={})
        assert resp.status_code == 422
