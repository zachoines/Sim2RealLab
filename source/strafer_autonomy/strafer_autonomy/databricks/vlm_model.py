"""MLflow ``pyfunc`` wrapper for the Strafer VLM service.

Replicates ``POST /ground`` and ``POST /describe`` behavior over the same
Qwen2.5-VL-3B pipeline used by the LAN HTTP service, so a Databricks
Model Serving endpoint can replace the LAN service transparently.

Accepted input columns per row:
- ``request_id`` (str, required)
- ``image_b64`` (str, required) — JPEG base64
- ``mode`` (str, required) — ``"ground"`` or ``"describe"``
- ``prompt`` (str, optional)
- ``max_image_side`` (int, optional, default 1024)
- ``image_stamp_sec`` (float, optional, default 0.0)
- ``return_debug_overlay`` (bool, optional, default False)

Per-row output shape matches the JSON payloads from
``strafer_vlm.service.app``.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def _rows_from_model_input(model_input: Any) -> Iterable[dict[str, Any]]:
    if model_input is None:
        return []
    if isinstance(model_input, dict):
        return [model_input]
    if isinstance(model_input, list):
        return model_input
    if hasattr(model_input, "to_dict"):
        return model_input.to_dict(orient="records")
    raise TypeError(f"Unsupported model_input type: {type(model_input).__name__}")


def _decode_image(image_b64: str) -> Any:
    from PIL import Image

    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _resize(image: Any, max_side: int) -> Any:
    from PIL import Image

    if max_side <= 0:
        return image
    if max(image.width, image.height) <= max_side:
        return image
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


class StraferVLMModel:
    """MLflow pyfunc wrapper around the Qwen2.5-VL service pipeline.

    Like :class:`StraferPlannerModel`, this class does NOT subclass
    ``mlflow.pyfunc.PythonModel`` at import time — keep ``mlflow`` optional
    for Jetson-side imports. ``register.py`` composes it with
    ``mlflow.pyfunc.PythonModel`` at log time.
    """

    DEFAULT_DESCRIBE_SYSTEM_PROMPT = (
        "You are a robot vision system. Describe the scene in the image "
        "concisely. List the main objects, surfaces, and spatial layout "
        "visible. Keep your response to 1-3 sentences."
    )

    def __init__(
        self,
        *,
        model_path: str | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None

    def load_context(self, context: Any) -> None:
        from strafer_vlm.inference.qwen_runtime import load_qwen_model_and_processor

        artifacts = getattr(context, "artifacts", None) or {}
        model_path = artifacts.get("model") or self._model_path
        if not model_path:
            raise RuntimeError(
                "StraferVLMModel requires a 'model' artifact path or an "
                "explicit model_path in __init__."
            )
        model, processor = load_qwen_model_and_processor(
            model_name_or_path=model_path,
        )
        model.eval()
        self._model = model
        self._processor = processor
        logger.info("StraferVLMModel loaded runtime from %s", model_path)

    def predict(self, context: Any, model_input: Any, params: Any = None) -> list[dict[str, Any]]:
        if self._model is None or self._processor is None:
            raise RuntimeError(
                "StraferVLMModel.predict called before load_context()."
            )

        from strafer_vlm.inference.parsing import (
            SYSTEM_PROMPT_DEFAULT,
            normalize_prediction_bbox_to_1000,
            parse_grounding_prediction,
        )
        from strafer_vlm.inference.qwen_runtime import run_grounding_generation

        results: list[dict[str, Any]] = []
        for row in _rows_from_model_input(model_input):
            request_id = str(row.get("request_id") or "")
            mode = str(row.get("mode") or "").strip().lower()
            image_b64 = row.get("image_b64")
            if not image_b64:
                raise ValueError(f"row {request_id} missing image_b64")
            if mode not in {"ground", "describe"}:
                raise ValueError(
                    f"row {request_id} has unknown mode {mode!r}; must be 'ground' or 'describe'"
                )

            image = _decode_image(str(image_b64))
            max_image_side = int(row.get("max_image_side", 1024))
            image = _resize(image, max_image_side)

            started = time.perf_counter()
            if mode == "ground":
                raw_output = run_grounding_generation(
                    model=self._model,
                    processor=self._processor,
                    image=image,
                    prompt=str(row.get("prompt", "")),
                    system_prompt=SYSTEM_PROMPT_DEFAULT,
                    max_new_tokens=self._max_new_tokens,
                    temperature=0.0,
                )
                latency_s = time.perf_counter() - started

                prediction = parse_grounding_prediction(raw_output)
                if prediction is None:
                    results.append(
                        {
                            "request_id": request_id,
                            "found": False,
                            "bbox_2d": None,
                            "label": None,
                            "confidence": None,
                            "raw_output": raw_output,
                            "latency_s": round(latency_s, 4),
                        }
                    )
                    continue
                normalized = normalize_prediction_bbox_to_1000(
                    prediction,
                    image_width=image.width,
                    image_height=image.height,
                )
                results.append(
                    {
                        "request_id": request_id,
                        "found": normalized.found,
                        "bbox_2d": list(normalized.bbox_2d) if normalized.bbox_2d else None,
                        "label": normalized.label,
                        "confidence": (
                            round(normalized.confidence, 4)
                            if normalized.confidence is not None
                            else None
                        ),
                        "raw_output": raw_output,
                        "latency_s": round(latency_s, 4),
                    }
                )
            else:  # describe
                prompt = str(
                    row.get("prompt")
                    or "Describe the objects and layout visible in this image in one or two sentences."
                )
                raw_output = run_grounding_generation(
                    model=self._model,
                    processor=self._processor,
                    image=image,
                    prompt=prompt,
                    system_prompt=self.DEFAULT_DESCRIBE_SYSTEM_PROMPT,
                    max_new_tokens=self._max_new_tokens,
                    temperature=0.0,
                )
                latency_s = time.perf_counter() - started
                results.append(
                    {
                        "request_id": request_id,
                        "description": raw_output,
                        "latency_s": round(latency_s, 4),
                    }
                )

        return results

    @staticmethod
    def input_schema_columns() -> tuple[str, ...]:
        return (
            "request_id",
            "image_b64",
            "mode",
            "prompt",
            "max_image_side",
            "image_stamp_sec",
        )
