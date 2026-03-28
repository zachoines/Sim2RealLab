"""Grounding client abstractions and HTTP transport."""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import requests

from strafer_autonomy.schemas import GroundingRequest, GroundingResult

logger = logging.getLogger(__name__)


class GroundingServiceUnavailable(Exception):
    """Raised when the VLM grounding service cannot be reached or returns a server error."""


@runtime_checkable
class GroundingClient(Protocol):
    """Executor-facing interface for semantic target grounding."""

    def locate_semantic_target(self, request: GroundingRequest) -> GroundingResult:
        """Return a normalized grounding result for one prompt-image pair."""


@dataclass(frozen=True)
class HttpGroundingClientConfig:
    """Connection settings for the workstation-hosted grounding service."""

    base_url: str
    timeout_s: float = 15.0
    ground_path: str = "/ground"
    health_path: str = "/health"
    headers: dict[str, str] | None = None
    max_retries: int = 2
    retry_backoff_s: float = 0.5


class HttpGroundingClient:
    """LAN HTTP grounding client for the workstation VLM service."""

    def __init__(self, config: HttpGroundingClientConfig) -> None:
        self._config = config
        self._session = requests.Session()
        if config.headers:
            self._session.headers.update(config.headers)

    @property
    def config(self) -> HttpGroundingClientConfig:
        """Return the immutable grounding client configuration."""

        return self._config

    def locate_semantic_target(self, request: GroundingRequest) -> GroundingResult:
        """Send a grounding request to the remote VLM service.

        Retries up to ``config.max_retries`` times on connection and server
        errors with exponential back-off.  Raises ``GroundingServiceUnavailable``
        if all attempts fail.
        """

        image_jpeg_b64 = _encode_image_to_jpeg_b64(request.image_rgb_u8)
        payload = grounding_request_to_payload(request, image_jpeg_b64)
        url = self._config.base_url.rstrip("/") + self._config.ground_path

        last_exc: Exception | None = None
        for attempt in range(1 + self._config.max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=self._config.timeout_s)
                resp.raise_for_status()
                return grounding_result_from_payload(resp.json())
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                logger.warning(
                    "Grounding request %s attempt %d/%d failed: %s",
                    request.request_id, attempt + 1, 1 + self._config.max_retries, exc,
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code < 500:
                    raise GroundingServiceUnavailable(
                        f"Grounding service returned {exc.response.status_code}: {exc.response.text}"
                    ) from exc
                last_exc = exc
                logger.warning(
                    "Grounding request %s attempt %d/%d server error: %s",
                    request.request_id, attempt + 1, 1 + self._config.max_retries, exc,
                )

            if attempt < self._config.max_retries:
                backoff = self._config.retry_backoff_s * (2 ** attempt)
                time.sleep(backoff)

        raise GroundingServiceUnavailable(
            f"Grounding service unreachable after {1 + self._config.max_retries} attempts."
        ) from last_exc

    def health(self) -> dict[str, Any]:
        """Check the remote service health.

        Raises ``GroundingServiceUnavailable`` on connection or HTTP errors.
        """

        url = self._config.base_url.rstrip("/") + self._config.health_path
        try:
            resp = self._session.get(url, timeout=5.0)
            resp.raise_for_status()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            raise GroundingServiceUnavailable(
                f"VLM health check failed: {exc}"
            ) from exc
        return resp.json()


def _encode_image_to_jpeg_b64(image_rgb_u8: Any, *, quality: int = 90) -> str:
    """Encode a numpy uint8 HxWx3 RGB array to a base64 JPEG string."""

    from PIL import Image

    image = Image.fromarray(image_rgb_u8, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def grounding_request_to_payload(request: GroundingRequest, image_jpeg_b64: str) -> dict[str, Any]:
    """Convert a grounding request schema into the JSON payload expected by POST /ground."""

    return {
        "request_id": request.request_id,
        "prompt": request.prompt,
        "image_jpeg_b64": image_jpeg_b64,
        "image_stamp_sec": request.image_stamp_sec,
        "max_image_side": request.max_image_side,
        "return_debug_overlay": request.return_debug_overlay,
    }


def grounding_result_from_payload(payload: dict[str, Any]) -> GroundingResult:
    """Parse a grounding JSON response into the shared result schema."""

    bbox_value = payload.get("bbox_2d")
    bbox_2d = tuple(int(value) for value in bbox_value) if bbox_value is not None else None
    return GroundingResult(
        request_id=str(payload["request_id"]),
        found=bool(payload["found"]),
        bbox_2d=bbox_2d,
        label=str(payload["label"]) if payload.get("label") is not None else None,
        confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
        raw_output=str(payload["raw_output"]) if payload.get("raw_output") is not None else None,
        latency_s=float(payload.get("latency_s", 0.0)),
        debug_overlay_jpeg_b64=(
            str(payload["debug_overlay_jpeg_b64"])
            if payload.get("debug_overlay_jpeg_b64") is not None
            else None
        ),
    )
