"""Author Infinigen scene metadata into the scene USD's ``customData``.

Infinigen stores the richest scene knowledge (room types, room polygons,
object semantic tags, spatial relations) in the Blender Python ``State``
object during generation. This data is NOT written to USD or
``saved_mesh.json`` by default.

This script builds the per-scene metadata dict and embeds it in the
scene USD's root-prim ``customData`` (key ``strafer_scene_metadata``) —
the metadata travels *with* the geometry, read back by
:mod:`strafer_lab.tools.scene_metadata_reader`. It writes **no** sidecar
JSON.

Dict builders (each returns the metadata dict, no I/O):

1. ``extract_from_state(state)`` — called from Infinigen's Blender
   generation pipeline (in-process), while the rich Python state is alive.
2. ``extract_from_blend(blend_path)`` — re-open a saved ``.blend`` in
   headless Blender and reconstruct best-effort metadata.
3. ``extract_from_usd(usd_path)`` — parse an Infinigen-exported USDC's
   prim names; this path also *authors* the result into the USD in one
   pass (it already holds the stage).

Authoring (needs ``pxr``; semantics need the Isaac Sim Kit runtime):

- ``author_scene_metadata(usd, metadata)`` writes ``customData`` and, on
  each labeled object prim, both the legacy ``semanticLabel`` /
  ``instanceId`` provenance attrs **and** the ``UsdSemantics.LabelsAPI``
  (``instance_name="class"``) that Replicator's ``bounding_box_2d_tight``
  annotator boxes on. Structural classes are excluded from the semantics
  pass (see ``--label-denylist``).

The customData payload authored here is the ``objects[]`` / ``rooms[]`` /
``room_adjacency`` shape (see ``docs/SCENE_PROVIDER_CONTRACT.md``). One more
block is layered onto the same payload *after* this pass, by the sibling
``validate_scene_connectivity.py`` (kept separate so this Infinigen-state
walker stays pure): a **verified** ``connectivity[]`` graph (per room-pair
``reachable`` / ``via_doorway_xy`` / ``path_length_m`` / ``door_state``,
indices into ``rooms[]``) plus a top-level ``multi_story`` flag. That step
generates the scene's occupancy grid, plans every candidate room pair with
the shared grid planner, and forces doors open — it does not re-walk the
Infinigen state, it merges into the customData this script wrote.

Usage::

    # Build + author from an exported USDC (the chained corpus path),
    # under the Kit launcher so UsdSemantics labels can be applied:
    $ISAACLAB -p source/strafer_lab/scripts/extract_scene_metadata.py \\
        --from-usd --usd Assets/generated/scenes/<scene>.usdc

    # Richer (rooms + relations) from a .blend: build the dict in Blender,
    # then author it into the USD under Kit.
    $STRAFER_BLENDER_BIN --background --python <this script> -- \\
        --blend Assets/generated/scenes/<scene>/coarse/scene.blend \\
        --metadata-out /tmp/<scene>_metadata.json
    $ISAACLAB -p source/strafer_lab/scripts/extract_scene_metadata.py \\
        --author-from-json /tmp/<scene>_metadata.json \\
        --usd Assets/generated/scenes/<scene>.usdc
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from strafer_lab.tools import scene_metadata_reader
from strafer_lab.tools.scene_classes import STRUCTURAL_CLASSES

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


# Objects whose recorded position is the world origin are unplaceable
# fallbacks: Infinigen seeds creature prims (carnivore / herbivore /
# fish / etc.) at (0,0,0) before final placement, and the three USD /
# Blender / State extractors all fall back to (0,0,0) when a prim has
# no resolvable bbox. Both classes carry no useful spatial information
# for downstream pickers, so drop them here at metadata-author time
# rather than re-filtering at every consumer.
_ORIGIN_EPS = 1e-3


def _is_at_origin(record: ObjectRecord) -> bool:
    return all(abs(c) < _ORIGIN_EPS for c in record.position_3d[:3])


def _drop_origin_records(
    records: list[ObjectRecord],
) -> tuple[list[ObjectRecord], int]:
    """Return ``(kept_records, dropped_count)``."""
    kept = [r for r in records if not _is_at_origin(r)]
    return kept, len(records) - len(kept)


# ---------------------------------------------------------------------------
# In-process extraction (called from Infinigen generation pipeline)
# ---------------------------------------------------------------------------


def extract_from_state(state: Any) -> dict[str, Any]:
    """Build the metadata dict from Infinigen's in-memory ``State``.

    Returns the dict (rooms + objects + room_adjacency); does not write
    anything. The caller embeds it into the scene USD via
    :func:`author_scene_metadata` once the USD exists.
    """
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
    objects, dropped = _drop_origin_records(objects)

    room_adjacency = _serialize_room_graph(getattr(state, "room_graph", None))

    metadata = {
        "rooms": [r.__dict__ for r in rooms],
        "objects": [o.__dict__ for o in objects],
        "room_adjacency": room_adjacency,
    }
    logger.info(
        "Built metadata: %d rooms, %d objects (dropped %d at origin)",
        len(rooms), len(objects), dropped,
    )
    return metadata


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


# Ordered preference of concrete-noun categories for label inference. When an
# Infinigen object carries one of these tags, it wins over broader category tags
# (``Furniture``, ``Seating``). Externalized from ``_infer_label`` (mirroring the
# shared vocabulary in ``scene_classes.STRUCTURAL_CLASSES``) so the label
# taxonomy lives in one place. This is a small, hand-picked seed list, NOT a
# comprehensive Infinigen object→label map; expanding it to full coverage (which
# directly affects detection vocab + grounding quality) is tracked as a separate
# follow-up.
LABEL_CATEGORY_PRIORITY: tuple[str, ...] = (
    "Table",
    "Chair",
    "Bed",
    "Door",
    "Window",
    "Lamp",
    "Sink",
    "Couch",
)


def _infer_label(obj: Any, tags: list[str]) -> str:
    """Turn the Infinigen tag set into a stable, VLM-friendly label.

    Prefers the most specific concrete-noun tag (e.g. ``"Chair"``) over
    broader category tags (``"Furniture"``, ``"Seating"``). Falls back to
    the object's ``name`` attribute when no useful tag is present.
    """
    preferred = {c.lower() for c in LABEL_CATEGORY_PRIORITY}
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
# USD-only extraction (no .blend, no in-process State)
# ---------------------------------------------------------------------------
#
# When Infinigen has already produced a ``.usdc`` via ``infinigen.tools.export``
# but the original ``.blend`` and the in-process generation ``State`` are no
# longer available, the only semantic signal that survives is the prim name
# itself, which encodes the factory class that produced each asset (e.g.
# ``GlassPanelDoorFactory_430087__spawn_asset_5_``). ``extract_from_usd``
# recovers labels from those prim names alone — no Blender, no in-process
# State, no custom .blend properties needed — and authors the result back
# into the same USD.


def extract_from_usd(
    usd_path: Path,
    *,
    label_denylist: Iterable[str] = STRUCTURAL_CLASSES,
    apply_semantics: bool = True,
) -> dict[str, Any]:
    """Parse an Infinigen ``.usdc``'s prim names and author its metadata.

    Walks the stage, turns every parseable factory prim into one
    ``objects[]`` entry (label + instance id + 3D bbox from the USD), then
    embeds the dict in the stage's ``customData`` and applies the
    detection-label semantics in one pass before saving. ``rooms`` is
    empty — USD geometry alone can't recover the constraint solver's room
    polygons (``extract_from_state`` / ``extract_from_blend`` do).
    """

    from strafer_lab.tools.infinigen_label_parser import (
        is_skippable_prim,
        parse_factory_label,
    )

    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pxr is required for extract_from_usd. Run inside an Isaac Sim "
            "Python environment where the USD runtime is available."
        ) from exc

    usd_path = Path(usd_path)
    stage, save = _open_stage_for_authoring(usd_path, kit=apply_semantics)

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True,
    )

    objects: list[ObjectRecord] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        prim_name = prim.GetName()
        if is_skippable_prim(prim_name):
            continue
        parsed = parse_factory_label(prim_name)
        if parsed is None:
            continue
        # Only count parent Xform / Mesh prims, not material children.
        type_name = str(prim.GetTypeName())
        if type_name not in {"Xform", "Mesh", ""}:
            continue
        # If this Xform has a same-named Mesh child, skip the child to
        # avoid duplicate records — the Xform is the canonical handle
        # Infinigen uses for placement.
        parent = prim.GetParent()
        if parent is not None and parent.IsValid() and parent.GetName() == prim_name:
            continue

        try:
            bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if bbox.IsEmpty():
                position = [0.0, 0.0, 0.0]
                bbox_min = [0.0, 0.0, 0.0]
                bbox_max = [0.0, 0.0, 0.0]
            else:
                bbox_min = [float(v) for v in bbox.GetMin()]
                bbox_max = [float(v) for v in bbox.GetMax()]
                position = [
                    0.5 * (bbox_min[0] + bbox_max[0]),
                    0.5 * (bbox_min[1] + bbox_max[1]),
                    0.5 * (bbox_min[2] + bbox_max[2]),
                ]
        except Exception:  # noqa: BLE001  USD bbox can throw on malformed prims
            position = [0.0, 0.0, 0.0]
            bbox_min = [0.0, 0.0, 0.0]
            bbox_max = [0.0, 0.0, 0.0]

        objects.append(
            ObjectRecord(
                instance_id=int(parsed.instance_id),
                label=parsed.label,
                semantic_tags=[parsed.factory_class] if parsed.factory_class else [],
                prim_path=str(prim.GetPath()),
                position_3d=position,
                bbox_3d_min=bbox_min,
                bbox_3d_max=bbox_max,
                room_idx=None,
                relations=[],
                materials=[],
            )
        )

    objects, dropped = _drop_origin_records(objects)
    metadata = {
        "rooms": [],
        "objects": [o.__dict__ for o in objects],
        "room_adjacency": [],
        "source": "usd_prim_names",
    }
    logger.info(
        "Parsed %s: %d objects (dropped %d at origin; rooms NOT recoverable "
        "from USD alone)",
        usd_path, len(objects), dropped,
    )

    labeled = _author_into_stage(
        stage, metadata, label_denylist=label_denylist, apply_semantics=apply_semantics,
    )
    save()
    logger.info("Authored metadata + labeled %d prims in %s", labeled, usd_path)
    return metadata


# ---------------------------------------------------------------------------
# .blend post-processing path (best-effort, for scenes generated upstream)
# ---------------------------------------------------------------------------


def extract_from_blend(blend_path: Path) -> dict[str, Any]:
    """Open a ``.blend`` in headless Blender and build the metadata dict.

    Walks ``bpy.data.objects`` and reconstructs a best-effort metadata
    dict from custom properties (``['semantic_label']``, ``['room_idx']``,
    etc.) and bounding boxes. Requires ``bpy`` to be importable — usually
    this means the script is run inside ``blender --background --python``.
    Returns the dict; authoring it into a USD is a separate ``pxr`` pass.
    """
    try:
        import bpy  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "bpy is not importable. Run this function under "
            "`blender --background --python extract_scene_metadata.py`."
        ) from exc

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

    objects, dropped = _drop_origin_records(objects)
    metadata = {
        "rooms": [],  # Blender-only fallback cannot recover room polygons.
        "objects": [o.__dict__ for o in objects],
        "room_adjacency": [],
    }
    logger.info(
        "Built metadata from %s: %d objects (dropped %d at origin; rooms NOT "
        "recoverable from blend)",
        blend_path, len(objects), dropped,
    )
    return metadata


# ---------------------------------------------------------------------------
# USD authoring — customData + detection-label semantics
# ---------------------------------------------------------------------------


def _open_stage_for_authoring(usd_path: Path | str, *, kit: bool):
    """Return ``(stage, save_fn)`` for authoring into ``usd_path``.

    When ``kit`` (semantics pass), open via the Isaac Sim USD context so
    ``add_labels`` — which routes through ``omni.replicator`` and operates
    on the *managed* stage — actually applies the ``UsdSemantics`` schema;
    a raw ``Usd.Stage.Open`` stage silently no-ops the label write. The
    plain path (customData only) uses raw ``pxr`` and ``Stage.Save``.
    """
    if kit:
        import omni.usd  # noqa: WPS433 — Kit-only, lazy by design

        ctx = omni.usd.get_context()
        if not ctx.open_stage(str(usd_path)):
            raise RuntimeError(f"omni.usd failed to open stage: {usd_path}")
        return ctx.get_stage(), ctx.save_stage

    from pxr import Usd  # type: ignore

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    return stage, stage.Save


def author_scene_metadata(
    usd_path: Path | str,
    metadata: dict[str, Any],
    *,
    label_denylist: Iterable[str] = STRUCTURAL_CLASSES,
    apply_semantics: bool = True,
) -> int:
    """Embed ``metadata`` in a USD's ``customData`` and label its prims.

    Opens ``usd_path``, writes the metadata payload, applies the prim
    labels, and saves. Returns the number of prims labelled. Needs
    ``pxr``; ``apply_semantics`` additionally needs the Isaac Sim Kit
    runtime (run under ``$ISAACLAB -p``).
    """
    try:
        import pxr  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pxr is required to author scene metadata into USD. Run inside "
            "an Isaac Sim Python environment."
        ) from exc

    stage, save = _open_stage_for_authoring(usd_path, kit=apply_semantics)
    labeled = _author_into_stage(
        stage, metadata, label_denylist=label_denylist, apply_semantics=apply_semantics,
    )
    save()
    logger.info("Authored metadata + labeled %d prims in %s", labeled, usd_path)
    return labeled


def _author_into_stage(
    stage: Any,
    metadata: dict[str, Any],
    *,
    label_denylist: Iterable[str],
    apply_semantics: bool,
) -> int:
    """Write customData + per-object prim labels on an already-open stage.

    Does not save. Two label kinds per object prim:

    - ``semanticLabel`` / ``instanceId`` string/int attrs — the provenance
      attrs the metadata is built from (kept unchanged).
    - ``UsdSemantics.LabelsAPI`` (``instance_name="class"``) — what the
      Replicator ``bounding_box_2d_tight`` annotator boxes on. Applied
      only to non-structural classes (``label_denylist``), because
      truncation keeps the largest-area boxes and structure would evict
      furniture from the detections column.
    """
    from pxr import Sdf  # type: ignore

    scene_metadata_reader.write_custom_data(stage, metadata)

    denylist = {str(c).strip().lower() for c in label_denylist}
    add_labels = _resolve_add_labels() if apply_semantics else None

    labeled = 0
    for entry in metadata.get("objects", []) or []:
        prim_path = entry.get("prim_path")
        label = str(entry.get("label", "")).strip()
        if not prim_path or not label:
            continue
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue

        attr = prim.CreateAttribute("semanticLabel", Sdf.ValueTypeNames.String)
        attr.Set(label)
        instance_id = int(entry.get("instance_id", -1))
        if instance_id >= 0:
            iid_attr = prim.CreateAttribute("instanceId", Sdf.ValueTypeNames.Int)
            iid_attr.Set(instance_id)

        if add_labels is not None and label.lower() not in denylist:
            add_labels(prim, [label], instance_name="class")
        labeled += 1
    return labeled


def _resolve_add_labels() -> Any:
    """Return ``isaacsim.core.utils.semantics.add_labels`` or raise.

    The ``UsdSemantics.LabelsAPI`` schema + this helper are provided by the
    Isaac Sim Kit runtime, not plain ``pxr``; applying labels requires
    running under ``$ISAACLAB -p``.
    """
    try:
        from isaacsim.core.utils.semantics import add_labels  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Applying UsdSemantics detection labels needs the Isaac Sim Kit "
            "runtime (isaacsim.core.utils.semantics). Run this under "
            "`$ISAACLAB -p`, or pass apply_semantics=False to author "
            "customData only (the scene will not be detections-ready)."
        ) from exc
    return add_labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_denylist(spec: str) -> set[str]:
    return {tok.strip().lower() for tok in spec.split(",") if tok.strip()}


def _boot_kit_for_semantics() -> Any:
    """Boot a headless Isaac Sim Kit app and return the SimulationApp.

    ``isaacsim.core.utils.semantics`` (the ``UsdSemantics.LabelsAPI`` helper)
    and the ``UsdSemantics`` schema plugin are only importable once Kit has
    initialized — being under ``$ISAACLAB -p`` alone is not enough. The CLI
    boots Kit for the semantics-authoring paths; the caller closes the app.
    """
    from isaaclab.app import AppLauncher  # noqa: WPS433 — Kit-only, lazy by design

    launcher_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launcher_parser)
    kit_args = launcher_parser.parse_args([])
    kit_args.headless = True
    # add_labels routes through omni.replicator.core, which only loads when
    # the camera/Replicator extensions are enabled.
    kit_args.enable_cameras = True
    return AppLauncher(kit_args).app


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, default=None, help="Blend file to build metadata from.")
    parser.add_argument("--usd", type=Path, default=None, help="Scene USD to author metadata into.")
    parser.add_argument(
        "--from-usd",
        action="store_true",
        help="Build metadata from --usd by parsing prim names, then author it "
             "(customData + semantics) back into the same USD.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="With --blend (run in Blender): write the built metadata dict to "
             "this JSON for a later --author-from-json pass. A transient "
             "handoff, not a discovered sidecar.",
    )
    parser.add_argument(
        "--author-from-json",
        type=Path,
        default=None,
        help="Author the metadata dict in this JSON into --usd (customData + "
             "semantics). Pairs with a prior --blend --metadata-out build.",
    )
    parser.add_argument(
        "--label-denylist",
        type=str,
        default=",".join(sorted(STRUCTURAL_CLASSES)),
        help="Comma-separated object classes excluded from the UsdSemantics "
             "detection-label pass (still kept in objects[]). Default: the "
             "structural classes. Empty string labels everything.",
    )
    parser.add_argument(
        "--no-semantics",
        action="store_true",
        help="Author customData only; skip the UsdSemantics label pass (the "
             "scene will not be detections-ready). Use when the Kit runtime "
             "is unavailable.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    denylist = _parse_denylist(args.label_denylist)
    apply_semantics = not args.no_semantics

    # The USD-authoring paths apply UsdSemantics labels, which need the Kit
    # runtime — boot it headless before the lazy isaacsim.core import. The
    # blend builder runs in Blender (no Kit) and must not boot it.
    needs_kit = apply_semantics and (args.from_usd or args.author_from_json is not None)
    kit_app = _boot_kit_for_semantics() if needs_kit else None

    try:
        if args.from_usd:
            if args.usd is None:
                parser.error("--usd is required with --from-usd")
            extract_from_usd(
                args.usd, label_denylist=denylist, apply_semantics=apply_semantics,
            )
            return 0

        if args.author_from_json is not None:
            if args.usd is None:
                parser.error("--usd is required with --author-from-json")
            metadata = json.loads(Path(args.author_from_json).read_text(encoding="utf-8"))
            author_scene_metadata(
                args.usd, metadata, label_denylist=denylist, apply_semantics=apply_semantics,
            )
            return 0

        if args.blend is not None:
            metadata = extract_from_blend(args.blend)
            if args.metadata_out is not None:
                Path(args.metadata_out).write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8",
                )
                logger.info("Wrote transient metadata handoff %s", args.metadata_out)
            else:
                json.dump(metadata, sys.stdout, indent=2)
            return 0

        parser.error(
            "nothing to do: pass --from-usd, --author-from-json, or --blend.",
        )
        return 2
    finally:
        if kit_app is not None:
            kit_app.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli_main())
