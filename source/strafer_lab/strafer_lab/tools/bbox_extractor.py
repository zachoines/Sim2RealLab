"""Replicator-backed 2D detection extraction for Isaac Sim perception data.

Wraps two Omniverse Replicator annotators into typed records that downstream
code (the detections writer, the description / finetune scripts, the Isaac Sim
ROS2 bridge) consumes without any Replicator knowledge:

- ``bounding_box_2d_tight`` -> :class:`DetectedBbox` (per-class boxes +
  occlusion ratio) via the pure :func:`parse_bbox_data`. Labels live in a
  separate ``info["idToLabels"]`` dict and may be comma-separated when a prim
  carries several semantic types; the full list is kept and the primary class
  exposed as ``.label``.
- ``instance_id_segmentation`` -> :class:`InstanceSegmentation` (per-pixel
  instance ids) via :func:`parse_instance_seg_data`, used to pin a KNOWN target
  by its USD prim path (:func:`segment_ids_for_prim_path`). The bbox annotator
  boxes per class, so it cannot tell same-label siblings apart; the instance
  map can.

The parsers take no omni / Isaac Sim import, so they unit-test from plain
Python; the ``Replicator*Extractor`` classes defer the ``omni.replicator.core``
import to ``__init__`` and accept an injected ``annotator=`` mock for tests.
Scene objects need ``semanticLabel`` USD attrs (Infinigen scenes get them from
``extract_scene_metadata``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np


UNKNOWN_LABEL = "unknown"

# Field names + column order of the ``bounding_box_2d_tight`` annotator's
# structured array. ``parse_bbox_data`` reads by name with a positional
# fallback, so a renamed/reordered schema would silently corrupt the
# detections columns; matching against this tuple turns that into a loud
# :class:`BboxSchemaError`.
BBOX_2D_TIGHT_FIELDS: tuple[str, ...] = (
    "semanticId", "x_min", "y_min", "x_max", "y_max", "occlusionRatio",
)


class BboxSchemaError(RuntimeError):
    """Raised when the ``bounding_box_2d_tight`` schema drifts from :data:`BBOX_2D_TIGHT_FIELDS`."""


@dataclass(frozen=True)
class DetectedBbox:
    """One 2D bounding box with its semantic label(s) and occlusion ratio.

    Coordinates are pixel values in the camera's render-product resolution
    (top-left origin, +x right, +y down). The ``bbox_2d`` tuple is
    ``(x_min, y_min, x_max, y_max)``.

    ``label`` is the first (primary) class name from
    ``info["idToLabels"][str(semantic_id)]["class"]``. When a prim has
    multiple semantic types the full list is preserved in :attr:`labels`;
    single-label prims produce a one-element tuple.

    ``occlusion_ratio`` comes directly from Replicator:
    ``0.0 == fully visible``, ``1.0 == fully occluded``. Consumers that
    prefer a visibility fraction can do ``1.0 - bbox.occlusion_ratio``.
    """

    semantic_id: int
    label: str
    labels: tuple[str, ...]
    bbox_2d: tuple[int, int, int, int]
    occlusion_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the ``bboxes: list[dict]`` shape consumers expect on ``frame_data``."""
        return {
            "label": self.label,
            "labels": list(self.labels),
            "bbox_2d": list(self.bbox_2d),
            "semantic_id": int(self.semantic_id),
            "occlusion_ratio": float(self.occlusion_ratio),
        }

    @property
    def is_degenerate(self) -> bool:
        """``True`` if the bbox has zero-or-negative area."""
        x1, y1, x2, y2 = self.bbox_2d
        return x2 <= x1 or y2 <= y1


# ---------------------------------------------------------------------------
# Pure-Python parser — no omni / numpy / Isaac Sim imports
# ---------------------------------------------------------------------------


def _coerce_int(value: Any) -> int:
    """Coerce a numpy scalar or Python number to a Python int."""
    return int(value)


def _coerce_float(value: Any) -> float:
    return float(value)


def _row_field(row: Any, name: str, index: int, default: Any = None) -> Any:
    """Read ``name`` from a structured-array row with positional fallback.

    Numpy structured array rows (``np.void``) support ``row["field"]``.
    Plain tuples from test fixtures fall back to ``row[index]``. Dicts
    built by tests fall through the first branch via ``__getitem__``.
    """
    try:
        return row[name]
    except (KeyError, ValueError, TypeError, IndexError):
        try:
            return row[index]
        except (IndexError, TypeError):
            return default


def _resolve_labels(
    semantic_id: int,
    id_to_labels: Mapping[str, Any] | None,
) -> tuple[str, tuple[str, ...]]:
    """Map a ``semanticId`` to ``(primary_label, all_labels)``.

    ``info["idToLabels"]`` keys are strings (even when the semantic id is
    numeric), and values are ``{"class": "label1,label2"}`` with a
    comma-separated ``class`` field when the prim has multiple semantic
    types. Missing ids return ``UNKNOWN_LABEL``.
    """
    if id_to_labels is None:
        return UNKNOWN_LABEL, (UNKNOWN_LABEL,)

    entry = id_to_labels.get(str(semantic_id))
    if entry is None:
        return UNKNOWN_LABEL, (UNKNOWN_LABEL,)

    if isinstance(entry, Mapping):
        class_str = entry.get("class")
    else:
        # Some Replicator writer variants put the string directly under the id,
        # not behind a "class" key.
        class_str = entry

    if not isinstance(class_str, str) or not class_str.strip():
        return UNKNOWN_LABEL, (UNKNOWN_LABEL,)

    parts = tuple(part.strip() for part in class_str.split(",") if part.strip())
    if not parts:
        return UNKNOWN_LABEL, (UNKNOWN_LABEL,)
    return parts[0], parts


def parse_bbox_data(
    raw: Mapping[str, Any] | None,
    *,
    drop_degenerate: bool = True,
    min_occlusion_visible: float | None = None,
) -> list[DetectedBbox]:
    """Parse a Replicator ``bounding_box_2d_tight`` output into typed records.

    ``raw`` is the annotator's ``get_data()`` dict — ``{"data": <structured
    array>, "info": {"idToLabels": {"0": {"class": "table"}, ...}}}`` — or
    ``None`` before the first frame (returns ``[]``). Real annotator output (a
    numpy structured array) has its field names checked against
    :data:`BBOX_2D_TIGHT_FIELDS`; a mismatch raises :class:`BboxSchemaError`
    rather than reading the wrong columns. Test fixtures (lists of dicts /
    tuples, no ``dtype``) skip the check.

    ``drop_degenerate`` drops zero-or-negative-area rows. ``min_occlusion_visible``,
    if set, drops rows whose visible fraction (``1 - occlusion_ratio``) is below it.
    """
    if raw is None:
        return []

    data = raw.get("data")
    if data is None:
        return []

    # Real annotator output is a numpy structured array (``dtype.names``); test
    # fixtures (lists of dicts / tuples) have no dtype and skip the schema check.
    field_names = getattr(getattr(data, "dtype", None), "names", None)
    if field_names is not None and tuple(field_names) != BBOX_2D_TIGHT_FIELDS:
        raise BboxSchemaError(
            "bounding_box_2d_tight annotator schema changed: expected fields "
            f"{BBOX_2D_TIGHT_FIELDS}, got {tuple(field_names)}. parse_bbox_data "
            "reads these by name with a positional fallback, so a reorder or "
            "rename would corrupt observation.detections.*; review the schema "
            "and update BBOX_2D_TIGHT_FIELDS (+ the field reads) deliberately.",
        )

    info = raw.get("info") or {}
    id_to_labels = info.get("idToLabels")

    out: list[DetectedBbox] = []
    for row in _iter_rows(data):
        semantic_id = _coerce_int(_row_field(row, "semanticId", 0, default=-1))
        x_min = _coerce_int(_row_field(row, "x_min", 1, default=0))
        y_min = _coerce_int(_row_field(row, "y_min", 2, default=0))
        x_max = _coerce_int(_row_field(row, "x_max", 3, default=0))
        y_max = _coerce_int(_row_field(row, "y_max", 4, default=0))
        occlusion = _coerce_float(
            _row_field(row, "occlusionRatio", 5, default=0.0)
        )

        label, labels = _resolve_labels(semantic_id, id_to_labels)
        bbox = DetectedBbox(
            semantic_id=semantic_id,
            label=label,
            labels=labels,
            bbox_2d=(x_min, y_min, x_max, y_max),
            occlusion_ratio=occlusion,
        )

        if drop_degenerate and bbox.is_degenerate:
            continue
        if (
            min_occlusion_visible is not None
            and (1.0 - occlusion) < min_occlusion_visible
        ):
            continue
        out.append(bbox)

    return out


def _iter_rows(data: Any) -> Iterable[Any]:
    """Yield one row per bbox from a numpy structured array or a list fixture."""
    if data is None:
        return []
    return iter(data)


def resolve_render_product_path(camera_sensor: Any) -> str:
    """Return the render-product path backing an Isaac Lab camera sensor.

    Two attribute layouts exist: older camera classes expose
    ``render_product_paths`` directly, while the renderer-abstraction
    cameras keep them on the backend render-data object
    (``camera._render_data.render_product_paths``). Raises
    :class:`RuntimeError` when neither is populated — e.g. before the
    sensor is initialized, or under a renderer backend that doesn't
    create Replicator render products.
    """
    for holder in (camera_sensor, getattr(camera_sensor, "_render_data", None)):
        paths = getattr(holder, "render_product_paths", None)
        if paths:
            return str(paths[0])
    raise RuntimeError(
        f"Could not resolve a render-product path from {type(camera_sensor).__name__}; "
        "the sensor is not initialized or its renderer backend exposes no "
        "Replicator render product to attach annotators to.",
    )


# ---------------------------------------------------------------------------
# Replicator-backed extractor
# ---------------------------------------------------------------------------


@dataclass
class ReplicatorBboxExtractor:
    """Attach a ``bounding_box_2d_tight`` annotator to a camera render product.

    Constructed once per camera per env. Call :meth:`extract` after each
    simulation step to read the current frame's bboxes. Use this from
    Isaac Sim runtime code only — importing this module is safe outside
    Isaac Sim, but instantiating this class without supplying an
    ``annotator`` triggers an ``omni.replicator.core`` import that will
    fail in a plain Python env.

    Parameters
    ----------
    camera_render_product_path:
        The render product path of the camera to attach to. For Isaac Lab
        ``TiledCameraCfg`` instances you can obtain this via
        ``env.scene["d555_camera_perception"].render_product_paths[0]``.
    semantic_types:
        Optional list of semantic type filters forwarded to the annotator
        (for example ``["class"]``). When ``None`` the annotator returns
        bboxes for all semantic types known to the stage.
    annotator:
        Optional pre-built annotator object. Used by tests to inject a
        mock without initialising Omniverse. When ``None`` the extractor
        builds a real ``omni.replicator.core`` annotator and attaches it
        to ``camera_render_product_path``.
    """

    camera_render_product_path: str
    semantic_types: tuple[str, ...] | None = None
    annotator: Any = None
    drop_degenerate: bool = True
    min_occlusion_visible: float | None = None

    _attached: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.annotator is None:
            self.annotator = self._build_real_annotator()
        self._attached = True

    def _build_real_annotator(self) -> Any:
        """Build a real ``omni.replicator.core`` annotator (lazy import so tests stay omni-free)."""
        import omni.replicator.core as rep  # noqa: WPS433 — lazy by design

        init_params: dict[str, Any] = {}
        if self.semantic_types is not None:
            init_params["semanticTypes"] = list(self.semantic_types)

        annotator = rep.AnnotatorRegistry.get_annotator(
            "bounding_box_2d_tight",
            init_params=init_params or None,
        )
        annotator.attach([self.camera_render_product_path])
        return annotator

    def extract(self) -> list[DetectedBbox]:
        """Return the current frame's parsed bboxes (schema handling in :func:`parse_bbox_data`)."""
        if not self._attached or self.annotator is None:
            raise RuntimeError(
                "ReplicatorBboxExtractor has no annotator — construct with "
                "a camera_render_product_path or pass annotator=..."
            )
        raw = self.annotator.get_data()
        return parse_bbox_data(
            raw,
            drop_degenerate=self.drop_degenerate,
            min_occlusion_visible=self.min_occlusion_visible,
        )

    def extract_as_dicts(self) -> list[dict[str, Any]]:
        """Return the extracted bboxes as plain dicts for frame-data / JSON payloads."""
        return [bbox.to_dict() for bbox in self.extract()]


# ---------------------------------------------------------------------------
# Instance segmentation — per-instance target identity
# ---------------------------------------------------------------------------
# bounding_box_2d_tight boxes per semantic CLASS (scenes label with
# instance_name="class"), so same-label siblings can merge into one box and a
# label match cannot tell them apart. instance_id_segmentation gives a per-pixel
# instance id plus info["idToLabels"] mapping that id -> USD prim path, so a
# target is matched by its prim_path. (The authored instanceId USD attr is not
# visible to the annotator.)

# Required info keys; a missing key raises InstanceSegSchemaError rather than
# matching nothing. The annotator data is an (H, W) or (H, W, 1) uint32 id mask;
# its idToLabels values are USD prim paths (renderer ids -> drawn mesh prim, plus
# an "INVALID" sentinel for unlabelled pixels).
INSTANCE_SEG_FIELDS: tuple[str, ...] = ("idToLabels",)


class InstanceSegSchemaError(RuntimeError):
    """Raised when instance_id_segmentation output is missing a required info key."""


@dataclass(frozen=True)
class InstanceSegmentation:
    """One frame of ``instance_id_segmentation``: a per-pixel id mask + id map.

    ``mask`` is the ``(H, W)`` array of renderer-assigned instance ids (a numpy
    array in production, a list-of-lists in fixtures). ``info`` is the raw
    annotator info dict (carries ``idToLabels``). ``frame_w`` / ``frame_h`` are
    the mask dimensions. Pure data — no Omniverse types.
    """

    mask: Any
    info: Mapping[str, Any]
    frame_w: int
    frame_h: int


def _mask_dims(data: Any) -> tuple[int, int]:
    """Return ``(height, width)`` of a 2D id mask (numpy array or list-of-lists)."""
    shape = getattr(data, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    try:
        height = len(data)
        width = len(data[0]) if height else 0
        return int(height), int(width)
    except (TypeError, IndexError):
        return 0, 0


def parse_instance_seg_data(
    raw: Mapping[str, Any] | None,
) -> "InstanceSegmentation | None":
    """Parse a Replicator ``instance_id_segmentation`` output into typed data.

    Accepts the dict shape ``{"data": <(H, W) id mask>, "info": {"idToLabels":
    {...}}}``. Returns ``None`` when the annotator has not produced a frame
    (``raw`` or ``raw["data"]`` is ``None``) — the caller treats that as a
    skip. Raises :class:`InstanceSegSchemaError` when the ``info`` dict is
    missing a required key (see :data:`INSTANCE_SEG_FIELDS`).
    """
    if raw is None:
        return None
    data = raw.get("data")
    if data is None:
        return None
    info = raw.get("info") or {}
    missing = [key for key in INSTANCE_SEG_FIELDS if key not in info]
    if missing:
        raise InstanceSegSchemaError(
            f"instance_id_segmentation missing required info key(s) {missing}; "
            f"expected {INSTANCE_SEG_FIELDS}",
        )
    height, width = _mask_dims(data)
    return InstanceSegmentation(mask=data, info=info, frame_w=width, frame_h=height)


def segment_ids_for_prim_path(
    info: Mapping[str, Any] | None, prim_path: str | None
) -> list[int]:
    """Renderer instance ids belonging to a target object's ``prim_path``.

    ``info["idToLabels"]`` maps each id to the prim path it was drawn from (the
    value), so this reverse-scans the values — do not index ``idToLabels`` by
    ``prim_path``. The render-time path differs from the one the metadata
    recorded: the env references the scene under a different root and the mesh
    leaf repeats the object name (e.g. recorded ``/World/Foo`` renders as
    ``/World/Room/Foo/Foo``). So a target also matches when its object-name
    segment is the rendered prim's leaf or its immediate parent (the last two
    path segments). Requiring it in the last two — not anywhere — keeps a target
    that is a PARENT of nested child objects (``.../Shelf/Trinket/Trinket``) from
    claiming the children's ids. One object can own several ids (multi-mesh).
    Returns an empty list when the target was not rendered.
    """
    if not prim_path or not info:
        return []
    id_to_labels = info.get("idToLabels")
    if not id_to_labels:
        return []
    prim_path = str(prim_path).rstrip("/")
    prefix = prim_path + "/"
    leaf = prim_path.rsplit("/", 1)[-1]
    ids: list[int] = []
    for seg_id, value in id_to_labels.items():
        candidate: Any = value
        if isinstance(value, Mapping):
            candidate = value.get("prim_path") or value.get("class") or value.get("name")
        if not isinstance(candidate, str):
            continue
        if (
            candidate == prim_path
            or candidate.startswith(prefix)
            or (leaf and leaf in candidate.split("/")[-2:])
        ):
            try:
                ids.append(int(seg_id))
            except (TypeError, ValueError):
                continue
    return ids


def segment_pixel_extent(
    mask: Any, segment_ids: int | Iterable[int]
) -> tuple[int, tuple[int, int, int, int]] | None:
    """Pixel count + bounding box of all ``mask`` pixels matching ``segment_ids``.

    ``segment_ids`` is one id or an iterable of ids (an object split across
    several renderer instances is unioned). Returns ``(pixel_count, (x_min,
    y_min, x_max, y_max))`` with an EXCLUSIVE max (a single pixel -> area 1,
    matching the ``bounding_box_2d_tight`` area convention), or ``None`` when no
    pixel matches. A trailing channel axis (the annotator's ``(H, W, 1)`` shape)
    is reduced to 2D. Uses numpy for array masks; falls back to a pure scan for
    list-of-lists fixtures.
    """
    ids = [segment_ids] if isinstance(segment_ids, int) else list(segment_ids)
    if not ids:
        return None
    if getattr(mask, "shape", None) is not None:
        arr = np.asarray(mask)
        if arr.ndim >= 3:
            arr = arr[..., 0]
        if arr.ndim == 2:
            ys, xs = np.where(np.isin(arr, ids))
            if xs.size == 0:
                return None
            return int(xs.size), (
                int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1,
            )
    id_set = set(ids)
    count = 0
    x_min: int | None = None
    y_min: int | None = None
    x_max = 0
    y_max = 0
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value in id_set:
                count += 1
                if x_min is None or x < x_min:
                    x_min = x
                if y_min is None or y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
    if count == 0 or x_min is None or y_min is None:
        return None
    return count, (x_min, y_min, x_max + 1, y_max + 1)


def _rect_overlap_area(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    """Intersection area of two ``(x_min, y_min, x_max, y_max)`` pixel boxes."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0
    return ix * iy


def bbox_row_for_segment(
    bboxes: Iterable[DetectedBbox],
    segment_bbox: tuple[int, int, int, int],
    label: str | None = None,
) -> DetectedBbox | None:
    """Select the ``bounding_box_2d_tight`` row overlapping a segment's mask box.

    The target's per-instance identity comes from the segmentation mask; its
    occlusion ratio is only available per-class from ``bounding_box_2d_tight``.
    This picks the class row with the largest pixel overlap with the segment's
    mask bbox (ties -> first), or ``None`` when no row overlaps the segment.

    When ``label`` is given, a row whose class matches it is preferred over a
    larger-overlap row of a different class — so a foreground occluder's box
    cannot supply the target's occlusion. Falls back to the best overlap over
    all rows when no matching-class row overlaps (never worse than no filter).
    """
    best: DetectedBbox | None = None
    best_overlap = 0
    best_labeled: DetectedBbox | None = None
    best_labeled_overlap = 0
    for bbox in bboxes:
        overlap = _rect_overlap_area(bbox.bbox_2d, segment_bbox)
        if overlap <= 0:
            continue
        if overlap > best_overlap:
            best_overlap = overlap
            best = bbox
        if (
            label is not None
            and overlap > best_labeled_overlap
            and (bbox.label == label or label in bbox.labels)
        ):
            best_labeled_overlap = overlap
            best_labeled = bbox
    return best_labeled if best_labeled is not None else best


@dataclass
class ReplicatorInstanceSegExtractor:
    """Attach an ``instance_id_segmentation`` annotator to a camera render product.

    Sibling of :class:`ReplicatorBboxExtractor`: same lazy
    ``omni.replicator.core`` import + ``annotator=`` mock hook for tests.
    :meth:`extract` returns the current frame's :class:`InstanceSegmentation`
    (a per-pixel id mask + the id->prim map), or ``None`` before the annotator
    has produced a frame. Use this from Isaac Sim runtime code only —
    instantiating without an injected ``annotator`` triggers the Omniverse
    import.

    Parameters
    ----------
    camera_render_product_path:
        The render product path to attach to (the same one
        :class:`ReplicatorBboxExtractor` uses, via
        :func:`resolve_render_product_path`).
    annotator:
        Optional pre-built annotator. Tests inject a mock to exercise
        :meth:`extract` without Omniverse.
    """

    camera_render_product_path: str
    annotator: Any = None

    _attached: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.annotator is None:
            self.annotator = self._build_real_annotator()
        self._attached = True

    def _build_real_annotator(self) -> Any:
        """Build a real instance_id_segmentation annotator (colorize=False -> raw ids)."""
        import omni.replicator.core as rep  # noqa: WPS433 — lazy by design

        annotator = rep.AnnotatorRegistry.get_annotator(
            "instance_id_segmentation",
            init_params={"colorize": False},
        )
        annotator.attach([self.camera_render_product_path])
        return annotator

    def extract(self) -> InstanceSegmentation | None:
        """Return the current frame's parsed instance segmentation (or ``None``)."""
        if not self._attached or self.annotator is None:
            raise RuntimeError(
                "ReplicatorInstanceSegExtractor has no annotator — construct "
                "with a camera_render_product_path or pass annotator=..."
            )
        return parse_instance_seg_data(self.annotator.get_data())
