"""Robot observation schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SceneObservation:
    """Jetson-local synchronized robot observation bundle."""

    observation_id: str
    stamp_sec: float
    color_image_bgr: Any
    aligned_depth_m: Any
    camera_frame: str
    camera_info: dict[str, Any] = field(default_factory=dict)
    robot_pose_map: dict[str, Any] | None = None
    tf_snapshot_ready: bool = False
