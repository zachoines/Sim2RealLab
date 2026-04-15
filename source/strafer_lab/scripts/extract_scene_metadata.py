"""Serialize Infinigen scene metadata into ``scene_metadata.json``.

Infinigen stores the richest scene knowledge (room types, room polygons,
object semantic tags, spatial relations) in the Blender Python ``State``
object during generation. This data is NOT written to USD or
``saved_mesh.json`` by default.

This script provides two entry points:

1. ``extract_from_state(state, output_dir)`` — called from Infinigen's
   Blender generation pipeline (in-process). Serializes ``state`` to
   ``scene_metadata.json`` while the rich Python state is still alive.

2. ``extract_from_blend(blend_path, output_dir)`` — post-process a saved
   ``.blend`` file by re-opening it in Blender headless mode
   (``blender --background --python``). Best-effort: walks the object
   collection and reconstructs the metadata from custom properties /
   object names, but necessarily misses anything the generator did not
   persist to the ``.blend``.

It also handles the USD prim labelling pass — walking a scene's USD
stage and writing ``semanticLabel`` and ``instanceId`` prim attributes
so Replicator annotators produce labeled bboxes.

Usage:

    # Post-process an already-generated scene
    python scripts/extract_scene_metadata.py \\
        --blend Assets/generated/scenes/kitchen_01.blend \\
        --usd Assets/generated/scenes/kitchen_01/kitchen_01.usd \\
        --output Assets/generated/scenes/kitchen_01

    # Label an already-generated USD (no metadata extraction)
    python scripts/extract_scene_metadata.py \\
        --usd Assets/generated/scenes/kitchen_01/kitchen_01.usd \\
        --metadata Assets/generated/scenes/kitchen_01/scene_metadata.json \\
        --label-usd-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("extract_scene_metadata")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RoomRecord:
    room_type: str
    footprint_xy: list[list[float]]
    area_m2: float
    story: int = 0


@dataclass
class ObjectRecord:
    instance_id: int
    label: str
    semantic_tags: list[str]
    prim_path: str | None
    position_3d: list[float]
    bbox_3d_min: list[float]
    bbox_3d_max: list[float]
    room_idx: int | None
    relations: list[dict[str, Any]]
    materials: list[str]


# ---------------------------------------------------------------------------
# In-process extraction (called from Infinigen generation pipeline)
# ---------------------------------------------------------------------------


def extract_from_state(state: Any, output_dir: Path) -> Path:
    """Serialize Infinigen's in-memory ``State`` to ``scene_metadata.json``.

    Parameters
    ----------
    state :
        Infinigen ``State`` object (imported from ``infinigen.core.tags``
        or similar). Duck-typed so this works even when Infinigen's exact
        module path evolves.
    output_dir :
        Directory into which ``scene_metadata.json`` will be written. The
        directory is created if it does not exist.

    Returns
    -------
    Path
        Absolute path of the written metadata JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rooms: list[RoomRecord] = []
    for room in getattr(state, "rooms", None) or []:
        footprint = _polygon_footprint(getattr(room, "polygon", None))
        area = float(getattr(room, "polygon", None).area) if getattr(room, "polygon", None) else 0.0
        rooms.append(
            RoomRecord(
                room_type=_semantics_name(getattr(room, "semantics", None)),
                footprint_xy=footprint,
                area_m2=area,
                story=int(getattr(room, "story", 0)),
            )
        )

    objects: list[ObjectRecord] = []
    for obj in getattr(state, "objects", None) or []:
        objects.append(_record_object(obj))

    room_adjacency = _serialize_room_graph(getattr(state, "room_graph", None))

    metadata = {
        "rooms": [r.__dict__ for r in rooms],
        "objects": [o.__dict__ for o in objects],
        "room_adjacency": room_adjacency,
    }

    output_path = output_dir / "scene_metadata.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(
        "Wrote %s with %d rooms, %d objects",
        output_path, len(rooms), len(objects),
    )
    return output_path


def _polygon_footprint(polygon: Any) -> list[list[float]]:
    if polygon is None:
        return []
    try:
        coords = list(polygon.exterior.coords)
    except AttributeError:
        return []
    out: list[list[float]] = []
    for x, *rest in coords:
        y = rest[0] if rest else 0.0
        out.append([float(x), float(y)])
    # Drop the duplicated closing vertex if shapely returned it.
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _semantics_name(semantics: Any) -> str:
    if semantics is None:
        return ""
    name_attr = getattr(semantics, "name", None)
    if isinstance(name_attr, str):
        return name_attr
    return str(semantics)


def _record_object(obj: Any) -> ObjectRecord:
    tags = []
    for tag in getattr(obj, "tags", None) or []:
        name = _semantics_name(tag)
        if name:
            tags.append(name)

    relations = []
    for rel in getattr(obj, "relations", None) or []:
        target_obj = getattr(rel, "target", None)
        relations.append(
            {
                "type": type(rel).__name__,
                "target": (
                    getattr(target_obj, "name", None) if target_obj is not None else None
                ),
            }
        )

    position = _vec3(getattr(obj, "position", None))
    bbox_min = _vec3(getattr(obj, "bbox_min", None))
    bbox_max = _vec3(getattr(obj, "bbox_max", None))

    materials: list[str] = []
    for mat in getattr(obj, "materials", None) or []:
        mat_name = getattr(mat, "name", None)
        if mat_name:
            materials.append(str(mat_name))

    return ObjectRecord(
        instance_id=int(getattr(obj, "instance_id", -1)),
        label=_infer_label(obj, tags),
        semantic_tags=tags,
        prim_path=_infer_prim_path(obj),
        position_3d=position,
        bbox_3d_min=bbox_min,
        bbox_3d_max=bbox_max,
        room_idx=(
            int(getattr(obj, "room_index"))
            if getattr(obj, "room_index", None) is not None
            else None
        ),
        relations=relations,
        materials=materials,
    )


def _vec3(value: Any) -> list[float]:
    if value is None:
        return [0.0, 0.0, 0.0]
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, IndexError):
        return [0.0, 0.0, 0.0]


def _infer_label(obj: Any, tags: list[str]) -> str:
    """Turn the Infinigen tag set into a stable, VLM-friendly label.

    Prefers the most specific concrete-noun tag (e.g. ``"Chair"``) over
    broader category tags (``"Furniture"``, ``"Seating"``). Falls back to
    the object's ``name`` attribute when no useful tag is present.
    """
    _category_priority = (
        "Table",
        "Chair",
        "Bed",
        "Door",
        "Window",
        "Lamp",
        "Sink",
        "Couch",
    )
    preferred: set[str] = {c.lower() for c in _category_priority}
    for tag in tags:
        if tag.lower() in preferred:
            return tag.lower()
    # Next-best: use the most specific tag (longest string).
    if tags:
        return max(tags, key=len).lower()
    name = getattr(obj, "name", "") or ""
    return str(name).split(".")[0].lower()


def _infer_prim_path(obj: Any) -> str | None:
    name = getattr(obj, "name", None)
    if not name:
        return None
    return f"/World/Room/{name}"


def _serialize_room_graph(room_graph: Any) -> list[list[int]]:
    """Flatten a RoomGraph's adjacency into an ``[[i, j], ...]`` edge list."""
    if room_graph is None:
        return []
    adjacency = getattr(room_graph, "adjacency", None)
    if adjacency is None:
        return []
    edges: list[list[int]] = []
    if hasattr(adjacency, "items"):
        for src, neighbors in adjacency.items():
            for dst in neighbors or ():
                edges.append([int(src), int(dst)])
        return edges
    # Assume an n×n matrix fallback.
    try:
        rows = list(adjacency)
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                if cell:
                    edges.append([i, j])
    except TypeError:
        pass
    return edges


# ---------------------------------------------------------------------------
# USD prim labelling
# ---------------------------------------------------------------------------


def label_usd_prims(usd_path: Path, metadata_path: Path) -> int:
    """Write ``semanticLabel`` / ``instanceId`` attributes on USD prims.

    Reads ``scene_metadata.json`` to build a name → (label, instance_id)
    lookup, then walks the USD stage and stamps any matching prim. The
    function is resilient to missing ``pxr`` (not available when running
    outside Isaac Sim / USD tooling) — it logs a warning and returns 0.
    """
    metadata = _load_metadata_for_labelling(metadata_path)
    lookup: dict[str, tuple[str, int]] = {}
    for entry in metadata.get("objects", []) or []:
        prim_path = entry.get("prim_path")
        if prim_path:
            lookup[Path(prim_path).name] = (
                str(entry.get("label", "")),
                int(entry.get("instance_id", -1)),
            )

    try:
        from pxr import Sdf, Usd  # type: ignore
    except ImportError:
        logger.warning(
            "pxr (USD) is not installed; skipping USD prim labelling. "
            "Run this script inside an Isaac Sim environment to label prims."
        )
        return 0

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    labeled = 0
    for prim in stage.Traverse():
        info = lookup.get(prim.GetName())
        if not info:
            continue
        label, instance_id = info
        if not label:
            continue
        attr = prim.CreateAttribute("semanticLabel", Sdf.ValueTypeNames.String)
        attr.Set(label)
        if instance_id >= 0:
            iid_attr = prim.CreateAttribute("instanceId", Sdf.ValueTypeNames.Int)
            iid_attr.Set(instance_id)
        labeled += 1
    stage.Save()
    logger.info("Labeled %d USD prims in %s", labeled, usd_path)
    return labeled


def _load_metadata_for_labelling(metadata_path: Path) -> dict[str, Any]:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"scene_metadata.json not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# .blend post-processing path (best-effort, for scenes generated upstream)
# ---------------------------------------------------------------------------


def extract_from_blend(blend_path: Path, output_dir: Path) -> Path:
    """Open a ``.blend`` in headless Blender and dump scene metadata.

    Walks ``bpy.data.objects`` and reconstructs a best-effort metadata
    JSON from custom properties (``['semantic_label']``, ``['room_idx']``,
    etc.) and bounding boxes. Requires ``bpy`` to be importable — usually
    this means the script is run inside ``blender --background --python``.
    """
    try:
        import bpy  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "bpy is not importable. Run this function under "
            "`blender --background --python extract_scene_metadata.py`."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    objects: list[ObjectRecord] = []
    for obj in bpy.data.objects:
        if obj.type not in {"MESH", "EMPTY"}:
            continue
        label = str(obj.get("semantic_label", obj.name.split(".")[0])).lower()
        room_idx = obj.get("room_idx")
        position = list(obj.matrix_world.translation)
        bb = [obj.matrix_world @ v for v in (obj.bound_box or ())]
        if bb:
            xs = [p[0] for p in bb]
            ys = [p[1] for p in bb]
            zs = [p[2] for p in bb]
            bbox_min = [min(xs), min(ys), min(zs)]
            bbox_max = [max(xs), max(ys), max(zs)]
        else:
            bbox_min = position[:3]
            bbox_max = position[:3]
        objects.append(
            ObjectRecord(
                instance_id=obj.get("instance_id", -1),
                label=label,
                semantic_tags=list(obj.get("semantic_tags", []) or []),
                prim_path=f"/World/Room/{obj.name}",
                position_3d=[float(position[0]), float(position[1]), float(position[2])],
                bbox_3d_min=[float(v) for v in bbox_min],
                bbox_3d_max=[float(v) for v in bbox_max],
                room_idx=int(room_idx) if room_idx is not None else None,
                relations=list(obj.get("relations", []) or []),
                materials=[
                    str(slot.material.name)
                    for slot in obj.material_slots
                    if slot.material is not None
                ],
            )
        )

    metadata = {
        "rooms": [],  # Blender-only fallback cannot recover room polygons.
        "objects": [o.__dict__ for o in objects],
        "room_adjacency": [],
    }
    output_path = output_dir / "scene_metadata.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(
        "Wrote %s from %s with %d objects (rooms NOT recoverable from blend)",
        output_path, blend_path, len(objects),
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, default=None, help="Blend file to extract metadata from.")
    parser.add_argument("--usd", type=Path, default=None, help="USD stage to label.")
    parser.add_argument("--metadata", type=Path, default=None, help="Path to an existing scene_metadata.json.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for scene_metadata.json.")
    parser.add_argument("--label-usd-only", action="store_true", help="Skip metadata extraction; only label USD.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    metadata_path: Path | None = args.metadata
    if not args.label_usd_only:
        if args.blend is None:
            parser.error("--blend is required unless --label-usd-only is set")
        metadata_path = extract_from_blend(args.blend, args.output)

    if args.usd is not None:
        if metadata_path is None:
            parser.error("--metadata is required when labelling USD without extraction")
        label_usd_prims(args.usd, metadata_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli_main())
