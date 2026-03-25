"""Grounding client abstractions and HTTP transport stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from strafer_autonomy.schemas import GroundingRequest, GroundingResult


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


class HttpGroundingClient:
    """LAN HTTP grounding client stub for the MVP workstation service."""

    def __init__(self, config: HttpGroundingClientConfig) -> None:
        self._config = config

    @property
    def config(self) -> HttpGroundingClientConfig:
        """Return the immutable grounding client configuration."""

        return self._config

    def locate_semantic_target(self, request: GroundingRequest) -> GroundingResult:
        """Send a grounding request to the remote VLM service."""

        raise NotImplementedError(
            "HTTP grounding transport is not implemented yet. "
            "Use grounding_request_to_payload() and grounding_result_from_payload() when wiring the first LAN client."
        )


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
        debug_artifact_path=(
            str(payload["debug_artifact_path"])
            if payload.get("debug_artifact_path") is not None
            else None
        ),
    )
