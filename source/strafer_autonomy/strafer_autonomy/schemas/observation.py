"""Robot observation schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SceneObservation:
    """Jetson-local synchronized robot observation bundle.

    This is a Jetson-local runtime object, not a network-portable schema.
    ``color_image_bgr`` and ``aligned_depth_m`` hold raw numpy arrays
    cached from ROS topic subscriptions.  Do not attempt to serialize
    this dataclass directly for network transport.
    """

    observation_id: str
    stamp_sec: float
    color_image_bgr: Any  # numpy uint8 HxWx3 BGR array from ROS Image
    aligned_depth_m: Any  # numpy float32 HxW depth in meters from aligned depth
    camera_frame: str
    camera_info: dict[str, Any] = field(default_factory=dict)
    robot_pose_map: dict[str, Any] | None = None
    tf_snapshot_ready: bool = False


@dataclass(frozen=True)
class SceneDescription:
    """VLM-generated free-text description of a scene image."""

    request_id: str
    description: str
    stamp_sec: float = 0.0
    latency_s: float = 0.0
