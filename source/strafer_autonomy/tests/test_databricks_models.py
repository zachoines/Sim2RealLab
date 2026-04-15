"""Unit tests for the MLflow pyfunc wrappers.

These tests exercise only the wrapper logic (row iteration, runtime
plumbing, error handling). They do NOT load a real Qwen model — the
planner runtime and qwen inference functions are mocked so the tests run
in CPU-only CI environments.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strafer_autonomy.databricks.planner_model import StraferPlannerModel
from strafer_autonomy.databricks.vlm_model import StraferVLMModel


class _DummyContext:
    def __init__(self, artifacts: dict | None = None):
        self.artifacts = artifacts or {}


class TestStraferPlannerModel:
    def test_predict_requires_load(self):
        model = StraferPlannerModel(model_path="/fake")
        with pytest.raises(RuntimeError, match="before load_context"):
            model.predict(None, [{"request_id": "r", "raw_command": "stop"}])

    def test_predict_happy_path(self):
        model = StraferPlannerModel(model_path="/fake")

        # Fake runtime that returns a valid cancel JSON.
        mock_runtime = MagicMock()
        mock_runtime.ready = True
        mock_runtime.generate.return_value = (
            '{"intent_type": "cancel", "target_label": null, '
            '"wait_mode": null, "requires_grounding": false}'
        )

        with patch(
            "strafer_autonomy.planner.llm_runtime.LLMRuntime",
            return_value=mock_runtime,
        ):
            model.load_context(_DummyContext({"model": "/fake"}))

        preds = model.predict(
            None,
            [{"request_id": "r1", "raw_command": "stop"}],
        )
        assert len(preds) == 1
        row = preds[0]
        assert row["mission_type"] == "cancel"
        assert row["raw_command"] == "stop"
        assert len(row["steps"]) == 1
        assert row["steps"][0]["skill"] == "cancel_mission"

    def test_predict_multiple_rows(self):
        model = StraferPlannerModel(model_path="/fake")
        mock_runtime = MagicMock()
        mock_runtime.ready = True
        mock_runtime.generate.side_effect = [
            '{"intent_type": "cancel"}',
            '{"intent_type": "status"}',
        ]
        with patch(
            "strafer_autonomy.planner.llm_runtime.LLMRuntime",
            return_value=mock_runtime,
        ):
            model.load_context(_DummyContext({"model": "/fake"}))
        preds = model.predict(
            None,
            [
                {"request_id": "a", "raw_command": "stop"},
                {"request_id": "b", "raw_command": "status"},
            ],
        )
        assert [p["mission_type"] for p in preds] == ["cancel", "status"]

    def test_predict_rejects_empty_command(self):
        model = StraferPlannerModel(model_path="/fake")
        mock_runtime = MagicMock()
        mock_runtime.ready = True
        with patch(
            "strafer_autonomy.planner.llm_runtime.LLMRuntime",
            return_value=mock_runtime,
        ):
            model.load_context(_DummyContext({"model": "/fake"}))
        with pytest.raises(ValueError, match="raw_command"):
            model.predict(None, [{"request_id": "r", "raw_command": ""}])

    def test_load_context_requires_model_path(self):
        model = StraferPlannerModel()
        with pytest.raises(RuntimeError, match="requires a 'model' artifact path"):
            model.load_context(_DummyContext())


class TestStraferVLMModel:
    def test_predict_requires_load(self):
        model = StraferVLMModel(model_path="/fake")
        with pytest.raises(RuntimeError, match="before load_context"):
            model.predict(None, [{"request_id": "r", "image_b64": "x", "mode": "ground"}])

    def test_predict_ground_mode(self):
        model = StraferVLMModel(model_path="/fake")
        model._model = MagicMock()
        model._processor = MagicMock()

        # Fabricate a 4x4 red JPEG so PIL decodes successfully.
        import base64
        import io

        from PIL import Image

        img = Image.new("RGB", (4, 4), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        with patch(
            "strafer_vlm.inference.qwen_runtime.run_grounding_generation",
            return_value='{"found": true, "bbox_2d": [100, 200, 400, 700], "label": "door", "confidence": 0.9}',
        ):
            preds = model.predict(
                None,
                [
                    {
                        "request_id": "r1",
                        "image_b64": image_b64,
                        "mode": "ground",
                        "prompt": "door",
                    }
                ],
            )
        assert len(preds) == 1
        assert preds[0]["found"] is True
        assert preds[0]["label"] == "door"
        assert preds[0]["bbox_2d"] == [100, 200, 400, 700]

    def test_predict_describe_mode(self):
        model = StraferVLMModel(model_path="/fake")
        model._model = MagicMock()
        model._processor = MagicMock()

        import base64
        import io

        from PIL import Image

        img = Image.new("RGB", (4, 4), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        with patch(
            "strafer_vlm.inference.qwen_runtime.run_grounding_generation",
            return_value="A small blue square.",
        ):
            preds = model.predict(
                None,
                [
                    {
                        "request_id": "r2",
                        "image_b64": image_b64,
                        "mode": "describe",
                    }
                ],
            )
        assert preds[0]["description"] == "A small blue square."
        assert preds[0]["request_id"] == "r2"

    def test_unknown_mode_raises(self):
        model = StraferVLMModel(model_path="/fake")
        model._model = MagicMock()
        model._processor = MagicMock()
        with pytest.raises(ValueError, match="unknown mode"):
            model.predict(
                None,
                [{"request_id": "r", "image_b64": "abc", "mode": "explode"}],
            )

    def test_missing_image_raises(self):
        model = StraferVLMModel(model_path="/fake")
        model._model = MagicMock()
        model._processor = MagicMock()
        with pytest.raises(ValueError, match="missing image_b64"):
            model.predict(None, [{"request_id": "r", "mode": "ground"}])
