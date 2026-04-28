"""Movement-gated background observation capture thread."""

from __future__ import annotations

import logging
import math
from threading import Event, Thread
from typing import Any

import numpy as np

from .models import Pose2D
from .transit_monitor import TransitMonitor

_logger = logging.getLogger(__name__)


def _bgr_to_rgb(image: Any) -> Any:
    try:
        return image[..., ::-1]
    except Exception:
        return image


class BackgroundMapper:
    """Captures observations whenever the robot has moved far enough.

    Movement-gated (spatial or angular threshold). Runs in a daemon thread
    and writes to the shared SemanticMapManager. Optionally feeds a
    TransitMonitor during active navigation legs.
    """

    def __init__(
        self,
        *,
        ros_client: Any,
        semantic_map: Any,
        transit_monitor: TransitMonitor | None = None,
        min_translation_m: float = 0.5,
        min_rotation_deg: float = 30.0,
        poll_interval_s: float = 2.0,
    ) -> None:
        self._ros_client = ros_client
        self._semantic_map = semantic_map
        self._transit_monitor = transit_monitor
        self._min_translation_m = min_translation_m
        self._min_rotation_rad = math.radians(min_rotation_deg)
        self._poll_interval_s = poll_interval_s
        self._last_capture_pose: Pose2D | None = None
        self._stop_event = Event()
        self._divergence_flag = Event()
        self._thread: Thread | None = None

    @property
    def transit_monitor(self) -> TransitMonitor | None:
        return self._transit_monitor

    def divergence_detected(self) -> bool:
        return self._divergence_flag.is_set()

    def clear_divergence(self) -> None:
        self._divergence_flag.clear()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="background-mapper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _should_capture(self, current: Pose2D) -> bool:
        if self._last_capture_pose is None:
            return True
        dx = current.x - self._last_capture_pose.x
        dy = current.y - self._last_capture_pose.y
        dist = math.sqrt(dx * dx + dy * dy)
        dyaw = abs(math.atan2(
            math.sin(current.yaw - self._last_capture_pose.yaw),
            math.cos(current.yaw - self._last_capture_pose.yaw),
        ))
        return dist > self._min_translation_m or dyaw > self._min_rotation_rad

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick_once()
            except Exception:
                _logger.debug("BackgroundMapper tick failed", exc_info=True)
            self._stop_event.wait(self._poll_interval_s)

    def _tick_once(self) -> None:
        robot_state = self._ros_client.get_robot_state()
        pose_dict = robot_state.get("pose") if robot_state else None
        if pose_dict is None:
            return

        current_pose = Pose2D.from_pose_map_dict(pose_dict)
        if not self._should_capture(current_pose):
            return

        observation = self._ros_client.capture_scene_observation()
        image_rgb = _bgr_to_rgb(observation.color_image_bgr)
        clip_emb = self._semantic_map.clip_encoder.encode_image(image_rgb)

        self._semantic_map.add_observation(
            pose=current_pose,
            timestamp=observation.stamp_sec,
            clip_embedding=clip_emb,
            source="background",
        )
        self._last_capture_pose = current_pose

        if self._transit_monitor is not None and self._transit_monitor.is_active:
            robot_xy = np.array([current_pose.x, current_pose.y])
            status = self._transit_monitor.check(clip_emb, robot_xy)
            if status.get("abort"):
                _logger.warning(
                    "Transit divergence detected: %s", status.get("message", "")
                )
                self._divergence_flag.set()
