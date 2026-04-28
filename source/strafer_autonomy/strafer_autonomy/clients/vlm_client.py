"""Grounding client abstractions and HTTP transport."""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import requests

from strafer_autonomy.schemas import GroundingRequest, GroundingResult, SceneDescription

logger = logging.getLogger(__name__)


class GroundingServiceUnavailable(Exception):
    """Raised when the VLM grounding service cannot be reached or returns a server error."""


@runtime_checkable
class GroundingClient(Protocol):
    """Executor-facing interface for semantic target grounding."""

    def locate_semantic_target(self, request: GroundingRequest) -> GroundingResult:
        """Return a normalized grounding result for one prompt-image pair."""

    def describe_scene(
        self,
        *,
        request_id: str,
        image_rgb_u8: Any,
        prompt: str | None = None,
        max_image_side: int = 1024,
    ) -> SceneDescription:
        """Return a free-text scene description for one image."""


@dataclass(frozen=True)
class DetectedObjectResult:
    """One detected object returned by POST /detect_objects."""

    label: str
    bbox_2d: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class DetectObjectsResult:
    """Parsed response from POST /detect_objects."""

    request_id: str
    objects: tuple[DetectedObjectResult, ...]
    raw_output: str | None
    latency_s: float


@dataclass(frozen=True)
class HttpGroundingClientConfig:
    """Connection settings for the workstation-hosted grounding service."""

    base_url: str
    timeout_s: float = 15.0
    ground_path: str = "/ground"
    describe_path: str = "/describe"
    detect_objects_path: str = "/detect_objects"
    health_path: str = "/health"
    headers: dict[str, str] | None = None
    max_retries: int = 2
    retry_backoff_s: float = 0.5
    jpeg_quality: int = 90


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

        image_jpeg_b64 = _encode_image_to_jpeg_b64(
            request.image_rgb_u8,
            quality=self._config.jpeg_quality,
        )
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

    def describe_scene(
        self,
        *,
        request_id: str,
        image_rgb_u8: Any,
        prompt: str | None = None,
        max_image_side: int = 1024,
    ) -> SceneDescription:
        """Send a scene description request to the remote VLM service.

        Uses the same retry logic as ``locate_semantic_target``.
        """
        image_jpeg_b64 = _encode_image_to_jpeg_b64(
            image_rgb_u8,
            quality=self._config.jpeg_quality,
        )
        payload: dict[str, Any] = {
            "request_id": request_id,
            "image_jpeg_b64": image_jpeg_b64,
            "max_image_side": max_image_side,
        }
        if prompt is not None:
            payload["prompt"] = prompt

        url = self._config.base_url.rstrip("/") + self._config.describe_path

        last_exc: Exception | None = None
        for attempt in range(1 + self._config.max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=self._config.timeout_s)
                resp.raise_for_status()
                body = resp.json()
                return SceneDescription(
                    request_id=str(body["request_id"]),
                    description=str(body["description"]),
                    latency_s=float(body.get("latency_s", 0.0)),
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                logger.warning(
                    "Describe request %s attempt %d/%d failed: %s",
                    request_id, attempt + 1, 1 + self._config.max_retries, exc,
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code < 500:
                    raise GroundingServiceUnavailable(
                        f"Describe service returned {exc.response.status_code}: {exc.response.text}"
                    ) from exc
                last_exc = exc
                logger.warning(
                    "Describe request %s attempt %d/%d server error: %s",
                    request_id, attempt + 1, 1 + self._config.max_retries, exc,
                )

            if attempt < self._config.max_retries:
                backoff = self._config.retry_backoff_s * (2 ** attempt)
                time.sleep(backoff)

        raise GroundingServiceUnavailable(
            f"Describe service unreachable after {1 + self._config.max_retries} attempts."
        ) from last_exc

    def detect_objects(
        self,
        *,
        request_id: str,
        image_rgb_u8: Any,
        max_image_side: int = 1024,
        max_objects: int = 20,
        min_confidence: float = 0.3,
    ) -> DetectObjectsResult:
        """Send a multi-object detection request to the remote VLM service.

        This method is additive — it is intentionally not part of the
        ``GroundingClient`` protocol so existing implementations remain valid.
        Callers that rely on it should type-check at the call site.
        """
        image_jpeg_b64 = _encode_image_to_jpeg_b64(
            image_rgb_u8,
            quality=self._config.jpeg_quality,
        )
        payload = detect_objects_request_to_payload(
            request_id=request_id,
            image_jpeg_b64=image_jpeg_b64,
            max_image_side=max_image_side,
            max_objects=max_objects,
            min_confidence=min_confidence,
        )
        url = self._config.base_url.rstrip("/") + self._config.detect_objects_path

        last_exc: Exception | None = None
        for attempt in range(1 + self._config.max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=self._config.timeout_s)
                resp.raise_for_status()
                return detect_objects_result_from_payload(resp.json())
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                logger.warning(
                    "Detect request %s attempt %d/%d failed: %s",
                    request_id, attempt + 1, 1 + self._config.max_retries, exc,
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code < 500:
                    raise GroundingServiceUnavailable(
                        f"Detect service returned {exc.response.status_code}: {exc.response.text}"
                    ) from exc
                last_exc = exc
                logger.warning(
                    "Detect request %s attempt %d/%d server error: %s",
                    request_id, attempt + 1, 1 + self._config.max_retries, exc,
                )

            if attempt < self._config.max_retries:
                backoff = self._config.retry_backoff_s * (2 ** attempt)
                time.sleep(backoff)

        raise GroundingServiceUnavailable(
            f"Detect service unreachable after {1 + self._config.max_retries} attempts."
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


def detect_objects_request_to_payload(
    *,
    request_id: str,
    image_jpeg_b64: str,
    max_image_side: int,
    max_objects: int,
    min_confidence: float,
) -> dict[str, Any]:
    """Build the JSON payload expected by POST /detect_objects."""

    return {
        "request_id": request_id,
        "image_jpeg_b64": image_jpeg_b64,
        "max_image_side": max_image_side,
        "max_objects": max_objects,
        "min_confidence": min_confidence,
    }


def detect_objects_result_from_payload(payload: dict[str, Any]) -> DetectObjectsResult:
    """Parse a /detect_objects JSON response into a ``DetectObjectsResult``."""

    objects: list[DetectedObjectResult] = []
    for entry in payload.get("objects", ()) or ():
        bbox_value = entry.get("bbox_2d")
        if bbox_value is None or len(bbox_value) != 4:
            continue
        bbox = tuple(int(v) for v in bbox_value)
        objects.append(
            DetectedObjectResult(
                label=str(entry["label"]),
                bbox_2d=bbox,  # type: ignore[arg-type]
                confidence=float(entry.get("confidence", 0.0)),
            )
        )
    return DetectObjectsResult(
        request_id=str(payload["request_id"]),
        objects=tuple(objects),
        raw_output=(
            str(payload["raw_output"]) if payload.get("raw_output") is not None else None
        ),
        latency_s=float(payload.get("latency_s", 0.0)),
    )


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
