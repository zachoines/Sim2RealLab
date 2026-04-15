"""Runtime accessor for per-scene metadata produced by Infinigen generation.

The metadata file (``scene_metadata.json``) is written alongside each
generated scene USD and holds the richer data that Infinigen's in-memory
``State`` exposes but does not survive to USD by default: room types,
room polygons, semantic tags, object relations, materials, etc.

Downstream tools (CLIP training data export, VLM SFT data export,
description pipeline) call these helpers instead of re-reading raw USD.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class SceneMetadataError(Exception):
    """Raised when a scene metadata file is missing or malformed."""


@dataclass(frozen=True)
class RoomEntry:
    """One room entry as read from ``scene_metadata.json``."""

    index: int
    room_type: str
    footprint_xy: tuple[tuple[float, float], ...]
    area_m2: float
    story: int
    raw: dict[str, Any]


@dataclass(frozen=True)
class ObjectEntry:
    """One object entry as read from ``scene_metadata.json``."""

    instance_id: int
    label: str
    semantic_tags: tuple[str, ...]
    prim_path: str | None
    position_3d: tuple[float, float, float]
    bbox_3d_min: tuple[float, float, float]
    bbox_3d_max: tuple[float, float, float]
    room_idx: int | None
    relations: tuple[dict[str, Any], ...]
    materials: tuple[str, ...]
    raw: dict[str, Any]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scene_metadata_path(metadata_root: Path, scene_name: str) -> Path:
    """Return the ``scene_metadata.json`` path for a given scene."""
    metadata_root = Path(metadata_root)
    candidate = metadata_root / scene_name / "scene_metadata.json"
    if candidate.exists():
        return candidate
    # Alternate layout: flat file per scene.
    flat = metadata_root / f"{scene_name}.json"
    if flat.exists():
        return flat
    return candidate  # return the primary path even if missing so caller can surface it


def get_scene_metadata(metadata_root: Path, scene_name: str) -> dict[str, Any]:
    """Load and return the raw ``scene_metadata.json`` dict for a scene.

    Raises ``SceneMetadataError`` if the file is missing or invalid.
    """
    path = scene_metadata_path(metadata_root, scene_name)
    if not path.exists():
        raise SceneMetadataError(f"scene_metadata.json not found at {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise SceneMetadataError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SceneMetadataError(f"Expected object at top level of {path}")
    data.setdefault("rooms", [])
    data.setdefault("objects", [])
    data.setdefault("room_adjacency", [])
    return data


def iter_rooms(metadata: dict[str, Any]) -> Iterable[RoomEntry]:
    """Yield typed :class:`RoomEntry` instances from a metadata dict."""
    for idx, raw in enumerate(metadata.get("rooms", []) or []):
        if not isinstance(raw, dict):
            continue
        yield RoomEntry(
            index=idx,
            room_type=str(raw.get("room_type", "")),
            footprint_xy=tuple(
                (float(x), float(y)) for x, y in (raw.get("footprint_xy") or ())
            ),
            area_m2=float(raw.get("area_m2", 0.0)),
            story=int(raw.get("story", 0)),
            raw=raw,
        )


def iter_objects(metadata: dict[str, Any]) -> Iterable[ObjectEntry]:
    """Yield typed :class:`ObjectEntry` instances from a metadata dict."""
    for raw in metadata.get("objects", []) or []:
        if not isinstance(raw, dict):
            continue
        position = raw.get("position_3d") or (0.0, 0.0, 0.0)
        bbox = raw.get("bbox_3d") or {}
        bbox_min = bbox.get("min") or (0.0, 0.0, 0.0)
        bbox_max = bbox.get("max") or (0.0, 0.0, 0.0)
        yield ObjectEntry(
            instance_id=int(raw.get("instance_id", -1)),
            label=str(raw.get("label", "")),
            semantic_tags=tuple(str(t) for t in (raw.get("semantic_tags") or ())),
            prim_path=(str(raw["prim_path"]) if raw.get("prim_path") is not None else None),
            position_3d=(float(position[0]), float(position[1]), float(position[2])),
            bbox_3d_min=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
            bbox_3d_max=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
            room_idx=(int(raw["room_idx"]) if raw.get("room_idx") is not None else None),
            relations=tuple(
                dict(rel) for rel in (raw.get("relations") or ()) if isinstance(rel, dict)
            ),
            materials=tuple(str(m) for m in (raw.get("materials") or ())),
            raw=raw,
        )


def get_scene_label_set(metadata_root: Path, scene_name: str) -> set[str]:
    """Return the unique object labels present in a scene.

    Empty labels are filtered out. The return type is :class:`set` so
    callers can cheaply test membership for ground-truth validation of
    VLM output.
    """
    metadata = get_scene_metadata(metadata_root, scene_name)
    labels = {obj.label for obj in iter_objects(metadata) if obj.label}
    return labels


def get_room_at_position(
    metadata: dict[str, Any],
    xy: tuple[float, float],
) -> RoomEntry | None:
    """Point-in-polygon lookup: return the room containing ``xy`` or None.

    Uses ``shapely`` for the containment test. ``shapely`` is an explicit
    dependency of the batch pipeline (see PHASE_15_DGX.md).
    """
    from shapely.geometry import Point, Polygon

    point = Point(float(xy[0]), float(xy[1]))
    for room in iter_rooms(metadata):
        if len(room.footprint_xy) < 3:
            continue
        polygon = Polygon(room.footprint_xy)
        if polygon.is_valid and polygon.contains(point):
            return room
    return None


def get_objects_in_room(
    metadata: dict[str, Any], room_idx: int,
) -> list[ObjectEntry]:
    """Return all objects whose ``room_idx`` matches the given index."""
    return [obj for obj in iter_objects(metadata) if obj.room_idx == room_idx]
