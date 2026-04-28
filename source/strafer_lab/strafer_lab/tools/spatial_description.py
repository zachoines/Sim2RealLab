"""Stage 1 of the scene description pipeline: programmatic spatial analysis.

Given a frame's perception record (robot pose, camera pose, visible
bboxes) and the scene's ``scene_metadata.json``, the
:class:`SpatialDescriptionBuilder` computes **factual** spatial relations
straight from simulation ground truth:

- Which room the robot is in (point-in-polygon against room footprints).
- Each visible object's distance, bearing, and near/mid/far region.
- The room containing each visible object.
- Relations (``SupportedBy`` / ``StableAgainst`` / ...) attached during
  Infinigen generation.

The output JSON is fed to Stage 2 (VLM) together with the raw RGB frame
so the VLM's descriptions are spatially accurate (from GT facts) and
visually grounded (from the image).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from strafer_lab.tools.scene_labels import (
    ObjectEntry,
    RoomEntry,
    iter_objects,
    iter_rooms,
)


# Region bucket boundaries in metres (near/mid/far relative to robot).
_NEAR_MAX_M = 1.5
_MID_MAX_M = 4.0


def quat_to_yaw(quat: Sequence[float]) -> float:
    """Return the yaw (rotation about Z) in radians for a ``[qx, qy, qz, qw]``.

    Accepts either a four-element sequence ``[qx, qy, qz, qw]`` or a
    three-element sequence interpreted as the imaginary part (``qw = 1``).
    """
    if len(quat) == 4:
        qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    elif len(quat) == 3:
        qx, qy, qz = (float(v) for v in quat)
        qw = 1.0
    else:
        raise ValueError(f"Unexpected quaternion length: {len(quat)}")
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def classify_region(distance_m: float) -> str:
    """Bucket a distance into near / midway / far."""
    if distance_m < _NEAR_MAX_M:
        return "near"
    if distance_m < _MID_MAX_M:
        return "midway"
    return "far"


def classify_bearing(bearing_rad: float) -> str:
    """Turn a signed bearing (radians) into a human-readable direction.

    Buckets: ahead / ahead-left / left / behind-left / behind /
    behind-right / right / ahead-right. Positive bearing is to the
    robot's left (CCW), matching the standard right-handed frame.
    """
    bearing_rad = _wrap_angle(bearing_rad)
    deg = math.degrees(bearing_rad)
    if -22.5 <= deg < 22.5:
        return "ahead"
    if 22.5 <= deg < 67.5:
        return "ahead-left"
    if 67.5 <= deg < 112.5:
        return "left"
    if 112.5 <= deg < 157.5:
        return "behind-left"
    if deg >= 157.5 or deg < -157.5:
        return "behind"
    if -157.5 <= deg < -112.5:
        return "behind-right"
    if -112.5 <= deg < -67.5:
        return "right"
    return "ahead-right"


@dataclass(frozen=True)
class DescribedObject:
    """Spatial description for one visible object."""

    label: str
    semantic_tags: tuple[str, ...]
    distance_m: float
    bearing_rad: float
    bearing: str
    region: str
    room_type: str | None
    relations: tuple[str, ...]
    materials: tuple[str, ...]
    bbox_2d: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "label": self.label,
            "semantic_tags": list(self.semantic_tags),
            "distance_m": round(self.distance_m, 2),
            "bearing_rad": round(self.bearing_rad, 4),
            "bearing": self.bearing,
            "region": self.region,
            "room_type": self.room_type,
            "relations": list(self.relations),
            "materials": list(self.materials),
        }
        if self.bbox_2d is not None:
            data["bbox_2d"] = list(self.bbox_2d)
        return data


class SpatialDescriptionBuilder:
    """Computes factual spatial descriptions from scene ground truth."""

    def __init__(self, scene_metadata: dict[str, Any]):
        from shapely.geometry import Polygon

        self._rooms: list[tuple[RoomEntry, Polygon]] = []
        for room in iter_rooms(scene_metadata):
            if len(room.footprint_xy) < 3:
                continue
            polygon = Polygon(room.footprint_xy)
            if polygon.is_valid:
                self._rooms.append((room, polygon))

        self._objects_by_instance: dict[int, ObjectEntry] = {
            obj.instance_id: obj for obj in iter_objects(scene_metadata)
        }
        self._objects_by_label: dict[str, list[ObjectEntry]] = {}
        for obj in self._objects_by_instance.values():
            self._objects_by_label.setdefault(obj.label, []).append(obj)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        """Compute spatial facts for one frame.

        Required keys in ``frame_data``:
        - ``robot_pos`` : sequence of ``[x, y, z]`` in metres.
        - ``robot_quat`` : ``[qx, qy, qz, qw]`` (or 3-element imaginary part).

        Optional keys:
        - ``bboxes`` : list of dicts with ``instance_id`` or ``label`` plus
          ``bbox_2d``. These identify which objects are visible in this
          frame and (optionally) where they project to in image space.
        """
        robot_pos = np.array(frame_data["robot_pos"], dtype=float)
        robot_quat = frame_data.get("robot_quat", [0.0, 0.0, 0.0, 1.0])
        robot_yaw = quat_to_yaw(robot_quat)
        robot_xy = robot_pos[:2]

        current_room = self._room_at(robot_xy)
        visible = self._resolve_visible_objects(frame_data.get("bboxes") or [])

        described: list[DescribedObject] = []
        for obj, bbox in visible:
            pos_xy = np.array(obj.position_3d[:2], dtype=float)
            delta = pos_xy - robot_xy
            distance = float(np.linalg.norm(delta))
            bearing = self._compute_bearing(delta, robot_yaw)
            region = classify_region(distance)
            relations_str: list[str] = []
            for rel in obj.relations:
                target = rel.get("target")
                rel_type = rel.get("type", "")
                if target:
                    relations_str.append(f"{rel_type} {target}")
            room_type = None
            if obj.room_idx is not None and 0 <= obj.room_idx < len(self._rooms):
                room_type = self._rooms[obj.room_idx][0].room_type
            described.append(
                DescribedObject(
                    label=obj.label,
                    semantic_tags=obj.semantic_tags,
                    distance_m=distance,
                    bearing_rad=bearing,
                    bearing=classify_bearing(bearing),
                    region=region,
                    room_type=room_type,
                    relations=tuple(relations_str),
                    materials=obj.materials,
                    bbox_2d=bbox,
                )
            )

        described.sort(key=lambda d: d.distance_m)
        return {
            "robot_position": [float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])],
            "robot_yaw_rad": round(robot_yaw, 4),
            "robot_room_type": current_room.room_type if current_room else None,
            "visible_objects": [d.to_dict() for d in described],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _room_at(self, xy: np.ndarray) -> RoomEntry | None:
        from shapely.geometry import Point

        pt = Point(float(xy[0]), float(xy[1]))
        for room, polygon in self._rooms:
            if polygon.contains(pt):
                return room
        return None

    @staticmethod
    def _compute_bearing(delta_xy: np.ndarray, robot_yaw: float) -> float:
        """Return the signed bearing from robot to object in its body frame.

        Positive bearing is to the left of the robot (CCW).
        """
        world_angle = math.atan2(float(delta_xy[1]), float(delta_xy[0]))
        return _wrap_angle(world_angle - robot_yaw)

    def _resolve_visible_objects(
        self, bboxes: Iterable[dict[str, Any]],
    ) -> list[tuple[ObjectEntry, tuple[int, int, int, int] | None]]:
        """Map each bbox entry back to its :class:`ObjectEntry`.

        Bbox entries may carry either an ``instance_id`` (preferred) or a
        ``label``. When resolving by label and multiple objects share the
        label, all matching objects are returned so every copy gets
        described.
        """
        out: list[tuple[ObjectEntry, tuple[int, int, int, int] | None]] = []
        seen: set[int] = set()
        for entry in bboxes:
            if not isinstance(entry, dict):
                continue
            bbox_raw = entry.get("bbox_2d")
            bbox: tuple[int, int, int, int] | None
            if bbox_raw is not None and len(bbox_raw) == 4:
                bbox = tuple(int(v) for v in bbox_raw)  # type: ignore[assignment]
            else:
                bbox = None

            instance_id = entry.get("instance_id")
            if instance_id is not None:
                obj = self._objects_by_instance.get(int(instance_id))
                if obj is not None and obj.instance_id not in seen:
                    out.append((obj, bbox))
                    seen.add(obj.instance_id)
                continue

            label = entry.get("label")
            if isinstance(label, str):
                for obj in self._objects_by_label.get(label, []):
                    if obj.instance_id in seen:
                        continue
                    out.append((obj, bbox))
                    seen.add(obj.instance_id)
        return out
