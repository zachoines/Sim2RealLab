"""Grounding and goal projection schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Pose3D:
    """Typed 3D pose matching ``geometry_msgs/Pose``.

    position (x, y, z) in metres, orientation as unit quaternion (qx, qy, qz, qw).
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0


@dataclass(frozen=True)
class GroundingRequest:
    """Logical request sent from the executor to the grounding service.

    This is the executor-side logical type.  ``image_rgb_u8`` holds a
    raw numpy uint8 HxWx3 RGB array.  The ``HttpGroundingClient`` is
    responsible for JPEG-encoding and base64-encoding the image before
    sending it over the network via ``grounding_request_to_payload()``.
    """

    request_id: str
    prompt: str
    image_rgb_u8: Any  # numpy uint8 HxWx3 RGB array
    image_stamp_sec: float
    max_image_side: int = 1024
    return_debug_overlay: bool = False


@dataclass(frozen=True)
class GroundingResult:
    """Normalized result returned by the grounding service.

    Coordinate convention:
        ``bbox_2d`` values are in the Qwen normalized [0, 1000] coordinate
        space, **not** pixel coordinates.  The projection service is
        responsible for converting to pixel space using camera intrinsics.
    """

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
    goal_pose: Pose3D | None = None
    target_pose: Pose3D | None = None
    standoff_m: float = 0.0
    depth_valid: bool = False
    quality_flags: tuple[str, ...] = ()
    message: str | None = None
