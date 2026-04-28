"""Databricks Model Serving implementation of ``GroundingClient``.

The serving endpoint is expected to wrap :class:`strafer_vlm.inference`
runtime via an MLflow ``pyfunc`` model (see ``databricks/vlm_model.py``)
that accepts rows of
``{"request_id", "image_b64", "prompt", "mode": "ground" | "describe"}``
and returns one prediction per row matching the structure of the
existing HTTP service responses.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from strafer_autonomy.clients.vlm_client import (
    GroundingServiceUnavailable,
    _encode_image_to_jpeg_b64,
    grounding_result_from_payload,
)
from strafer_autonomy.schemas import GroundingRequest, GroundingResult, SceneDescription

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabricksServingGroundingClientConfig:
    """Connection settings for a Databricks Model Serving grounding endpoint."""

    endpoint_name: str
    workspace_url: str
    token: str
    timeout_s: float = 30.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    jpeg_quality: int = 90
    extra_headers: dict[str, str] = field(default_factory=dict)

    def invocations_url(self) -> str:
        base = self.workspace_url.rstrip("/")
        return f"{base}/serving-endpoints/{self.endpoint_name}/invocations"

    def state_url(self) -> str:
        base = self.workspace_url.rstrip("/")
        return f"{base}/api/2.0/serving-endpoints/{self.endpoint_name}"


class DatabricksServingGroundingClient:
    """``GroundingClient`` implementation backed by Databricks Model Serving."""

    def __init__(self, config: DatabricksServingGroundingClientConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {config.token}",
                "Content-Type": "application/json",
            }
        )
        if config.extra_headers:
            self._session.headers.update(config.extra_headers)

    @property
    def config(self) -> DatabricksServingGroundingClientConfig:
        return self._config

    def locate_semantic_target(self, request: GroundingRequest) -> GroundingResult:
        image_b64 = _encode_image_to_jpeg_b64(
            request.image_rgb_u8, quality=self._config.jpeg_quality,
        )
        row = {
            "request_id": request.request_id,
            "image_b64": image_b64,
            "prompt": request.prompt,
            "mode": "ground",
            "image_stamp_sec": request.image_stamp_sec,
            "max_image_side": request.max_image_side,
            "return_debug_overlay": request.return_debug_overlay,
        }
        prediction = self._invoke(row, tag=f"locate {request.request_id}")
        return grounding_result_from_payload(prediction)

    def describe_scene(
        self,
        *,
        request_id: str,
        image_rgb_u8: Any,
        prompt: str | None = None,
        max_image_side: int = 1024,
    ) -> SceneDescription:
        image_b64 = _encode_image_to_jpeg_b64(
            image_rgb_u8, quality=self._config.jpeg_quality,
        )
        row = {
            "request_id": request_id,
            "image_b64": image_b64,
            "mode": "describe",
            "max_image_side": max_image_side,
        }
        if prompt is not None:
            row["prompt"] = prompt

        prediction = self._invoke(row, tag=f"describe {request_id}")
        return SceneDescription(
            request_id=str(prediction.get("request_id", request_id)),
            description=str(prediction["description"]),
            latency_s=float(prediction.get("latency_s", 0.0)),
        )

    def health(self) -> dict[str, Any]:
        url = self._config.state_url()
        try:
            resp = self._session.get(url, timeout=self._config.timeout_s)
            resp.raise_for_status()
            body = resp.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            raise GroundingServiceUnavailable(
                f"Databricks VLM health check failed: {exc}"
            ) from exc
        except ValueError as exc:
            raise GroundingServiceUnavailable(
                f"Databricks VLM health response was not valid JSON: {exc}"
            ) from exc

        state = (body.get("state") or {}) if isinstance(body, dict) else {}
        ready_state = str(state.get("ready", "")).upper()
        config_update = str(state.get("config_update", "")).upper()
        is_ready = ready_state == "READY" and config_update in {"NOT_UPDATING", ""}
        return {
            "status": "ok" if is_ready else "loading",
            "model_loaded": is_ready,
            "model_name": self._config.endpoint_name,
            "backend": "databricks",
            "endpoint_state": ready_state or None,
            "config_update": config_update or None,
        }

    def _invoke(self, row: dict[str, Any], *, tag: str) -> dict[str, Any]:
        payload = {"inputs": [row]}
        url = self._config.invocations_url()

        last_exc: Exception | None = None
        for attempt in range(1 + self._config.max_retries):
            try:
                resp = self._session.post(
                    url, json=payload, timeout=self._config.timeout_s,
                )
                resp.raise_for_status()
                return _extract_prediction(resp.json())
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                logger.warning(
                    "Databricks VLM %s attempt %d/%d failed: %s",
                    tag, attempt + 1, 1 + self._config.max_retries, exc,
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code < 500:
                    raise GroundingServiceUnavailable(
                        "Databricks VLM endpoint returned "
                        f"{exc.response.status_code}: {exc.response.text}"
                    ) from exc
                last_exc = exc
                logger.warning(
                    "Databricks VLM %s attempt %d/%d server error: %s",
                    tag, attempt + 1, 1 + self._config.max_retries, exc,
                )
            except ValueError as exc:
                raise GroundingServiceUnavailable(
                    f"Databricks VLM response was not valid JSON: {exc}"
                ) from exc

            if attempt < self._config.max_retries:
                backoff = self._config.retry_backoff_s * (2 ** attempt)
                time.sleep(backoff)

        raise GroundingServiceUnavailable(
            "Databricks VLM endpoint unreachable after "
            f"{1 + self._config.max_retries} attempts."
        ) from last_exc


def _extract_prediction(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise GroundingServiceUnavailable(
            f"Databricks VLM response was not an object: {type(body).__name__}"
        )
    predictions = body.get("predictions")
    if predictions is None:
        predictions = body.get("outputs") or body
    if isinstance(predictions, dict):
        predictions = [predictions]
    if not isinstance(predictions, list) or not predictions:
        raise GroundingServiceUnavailable(
            f"Databricks VLM response missing predictions: {body!r}"
        )
    first = predictions[0]
    if not isinstance(first, dict):
        raise GroundingServiceUnavailable(
            f"Databricks VLM prediction was not a dict: {first!r}"
        )
    return first
