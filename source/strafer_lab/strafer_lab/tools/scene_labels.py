"""Typed accessors for per-scene metadata embedded in a scene USD.

The metadata (labeled ``objects[]``, room polygons, semantic tags,
relations, materials) travels inside the scene USD's root-prim
``customData`` — see :mod:`strafer_lab.tools.scene_metadata_reader`,
the single ``pxr`` touch-point that loads it. The helpers here turn the
loaded dict into typed records.

The pure-data accessors (:func:`iter_rooms`, :func:`iter_objects`,
:func:`get_scene_label_set_from_data`, :func:`get_room_at_position`,
:func:`get_objects_in_room`) operate on an already-loaded dict and never
touch ``pxr``, so the pxr-free autonomy test suite exercises them
against in-memory dicts. Only :func:`get_scene_metadata`
opens a USD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from strafer_lab.tools.scene_metadata_reader import (
    SceneMetadataError,
    load as _load_scene_metadata,
)

__all__ = [
    "SceneMetadataError",
    "RoomEntry",
    "ObjectEntry",
    "get_scene_metadata",
    "iter_rooms",
    "iter_objects",
    "get_scene_label_set",
    "get_scene_label_set_from_data",
    "get_room_at_position",
    "get_objects_in_room",
]


@dataclass(frozen=True)
class RoomEntry:
    """One room entry as read from a scene's embedded metadata."""

    index: int
    room_type: str
    footprint_xy: tuple[tuple[float, float], ...]
    area_m2: float
    story: int
    raw: dict[str, Any]


@dataclass(frozen=True)
class ObjectEntry:
    """One object entry as read from a scene's embedded metadata."""

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


def get_scene_metadata(scene_usd_path: Path | str) -> dict[str, Any]:
    """Load and return the embedded metadata dict for a scene USD.

    Reads the root-prim ``customData`` via
    :func:`scene_metadata_reader.load`. Raises ``SceneMetadataError`` when
    the USD is missing or carries no embedded metadata (no sidecar
    fallback). Needs ``pxr``; pure-data consumers use the ``*_from_data``
    accessors instead.
    """
    return _load_scene_metadata(scene_usd_path)


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


def get_scene_label_set_from_data(metadata: dict[str, Any]) -> set[str]:
    """Return the unique object labels present in a metadata dict.

    Empty labels are filtered out. Pure-data — no ``pxr``, no disk.
    """
    return {obj.label for obj in iter_objects(metadata) if obj.label}


def get_scene_label_set(scene_usd_path: Path | str) -> set[str]:
    """Return the unique object labels present in a scene USD.

    Convenience wrapper that loads the embedded metadata and delegates to
    :func:`get_scene_label_set_from_data`. Needs ``pxr``.
    """
    return get_scene_label_set_from_data(get_scene_metadata(scene_usd_path))


def get_room_at_position(
    metadata: dict[str, Any],
    xy: tuple[float, float],
) -> RoomEntry | None:
    """Point-in-polygon lookup: return the room containing ``xy`` or None.

    Uses ``shapely`` for the containment test — an explicit dependency
    of the batch data pipeline.
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
