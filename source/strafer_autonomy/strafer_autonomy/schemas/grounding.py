"""Grounding and goal projection schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroundingRequest:
    """Logical request sent from the executor to the grounding service."""

    request_id: str
    prompt: str
    image_rgb_u8: Any
    image_stamp_sec: float
    max_image_side: int = 1024
    return_debug_overlay: bool = False


@dataclass(frozen=True)
class GroundingResult:
    """Normalized result returned by the grounding service."""

    request_id: str
    found: bool
    bbox_2d: tuple[int, int, int, int] | None = None
    label: str | None = None
    confidence: float | None = None
    raw_output: str | None = None
    latency_s: float = 0.0
    debug_artifact_path: str | None = None


@dataclass(frozen=True)
class GoalPoseCandidate:
    """Robot-side projection output from a 2D grounding result."""

    request_id: str
    found: bool
    goal_frame: str
    goal_pose: dict[str, Any] | None = None
    target_pose: dict[str, Any] | None = None
    standoff_m: float = 0.0
    depth_valid: bool = False
    quality_flags: tuple[str, ...] = ()
    message: str | None = None
