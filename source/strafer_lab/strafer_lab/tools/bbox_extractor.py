"""Replicator-backed 2D bounding-box extraction for Isaac Sim perception data.

Wraps the ``bounding_box_2d_tight`` Omniverse Replicator annotator and turns
its raw output — a structured numpy array of semantic ids + pixel bounds
plus a separate ``info["idToLabels"]`` dict — into a list of typed
:class:`DetectedBbox` records that downstream code (``generate_descriptions``,
``prepare_vlm_finetune_data``, the Isaac Sim ROS2 bridge) can consume without
any knowledge of Replicator internals.

Design notes:

- The ``bounding_box_2d_tight`` annotator returns rows with fields
  ``(semanticId, x_min, y_min, x_max, y_max, occlusionRatio)`` (confirmed
  against Isaac Sim 5.1's ``data_visualization_writer.py`` schema comment).
  Labels live in a separate ``info["idToLabels"]`` dict keyed by
  ``str(semantic_id)`` and can be a comma-separated list when a prim has
  multiple semantic types — we preserve the full list AND expose the
  primary class as ``.label`` for simple consumers.

- ``instanceId`` is NOT part of ``bounding_box_2d_tight`` — that field
  only exists on the ``instance_id_segmentation`` annotator. We leave
  :class:`DetectedBbox` with a ``semantic_id`` field instead; callers
  that need per-instance IDs attach :class:`ReplicatorInstanceSegExtractor`
  (the sibling wrapper below) and join a known target by its USD prim path
  via :func:`segment_ids_for_prim_path`.

- :func:`parse_bbox_data` is a pure function with NO Isaac Sim / numpy /
  Omniverse dependency — it accepts any iterable-of-row-objects that behaves
  like a structured numpy array (supports ``row["field"]`` access). This
  keeps the parser unit-testable from plain Python envs that do not have
  Isaac Sim installed.

- :class:`ReplicatorBboxExtractor` defers the ``omni.replicator.core`` import
  to ``__init__`` so merely importing this module from a plain Python env
  does not trigger an Omniverse load. Tests inject a mock annotator via the
  ``annotator`` keyword to exercise :meth:`extract` without Isaac Sim.

- Requires ``semanticLabel`` USD prim attributes on the scene objects. On
  Infinigen scenes these are populated by
  :mod:`scripts.extract_scene_metadata`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


UNKNOWN_LABEL = "unknown"

# The structured-array field names + order the ``bounding_box_2d_tight``
# annotator emits, in column order. Verified against the live Isaac Sim
# 6.0.0 annotator output and its ``data_visualization_writer`` schema.
# ``parse_bbox_data`` reads fields by name with a positional fallback, so a
# silent rename or reorder in a future Isaac Sim would corrupt the
# detections columns rather than error — this tuple is the guard that turns
# that into a loud failure (see :class:`BboxSchemaError`).
BBOX_2D_TIGHT_FIELDS: tuple[str, ...] = (
    "semanticId", "x_min", "y_min", "x_max", "y_max", "occlusionRatio",
)


class BboxSchemaError(RuntimeError):
    """Raised when an annotator's structured-array schema drifts.

    The detections producer relies on :data:`BBOX_2D_TIGHT_FIELDS`; if the
    installed Isaac Sim ships a renamed / reordered ``bounding_box_2d_tight``
    schema, parsing it silently would write mislabelled boxes into the
    dataset's ``observation.detections.*`` columns. Failing here forces a
    deliberate schema review instead.
    """


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
        """Serialize to the plain-dict shape consumed by the batch scripts.

        Mirrors the ``bboxes: list[dict]`` key shape the Stage-1 spatial
        description builder (:class:`strafer_lab.tools.spatial_description.
        SpatialDescriptionBuilder`) and the VLM data-prep scripts under
        :mod:`strafer_lab.scripts` expect on ``frame_data``.
        """
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
        # Some Replicator writer variants put the string directly under
        # the id, not behind a "class" key. Handle both for robustness.
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

    Accepts the dict shape returned by
    ``rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight").get_data()``:

    .. code-block:: python

        {
            "data": numpy.ndarray,  # structured, fields listed below
            "info": {
                "idToLabels": {"0": {"class": "table"}, "1": {"class": "chair,seat"}, ...},
                # other keys ignored by this parser
            },
        }

    Row fields, in column order (verified against the live Isaac Sim 6.0.0
    ``bounding_box_2d_tight`` annotator — see :data:`BBOX_2D_TIGHT_FIELDS`):
    ``semanticId (uint32)``, ``x_min (int32)``, ``y_min (int32)``,
    ``x_max (int32)``, ``y_max (int32)``, ``occlusionRatio (float32)``.
    Real annotator output (a numpy structured array) has its schema checked
    against that tuple; a mismatch raises :class:`BboxSchemaError` rather
    than reading the wrong columns by positional fallback. Test fixtures
    (lists of dicts / tuples, no ``dtype``) skip the check.

    Parameters
    ----------
    raw:
        The annotator output, or ``None`` when the annotator has not yet
        produced a frame. Returns an empty list in that case.
    drop_degenerate:
        Drop rows with zero-or-negative area before returning. Default True.
    min_occlusion_visible:
        If set, drop rows whose visible fraction (``1 - occlusion_ratio``)
        is below this threshold. Useful for filtering out heavily occluded
        objects that would produce noisy labels. Default ``None`` (keep
        everything).
    """
    if raw is None:
        return []

    data = raw.get("data")
    if data is None:
        return []

    # Guard against a silent annotator-schema drift across Isaac Sim
    # versions. Real annotator output is a numpy structured array carrying
    # ``dtype.names``; test fixtures (lists of dicts / tuples) have no
    # ``dtype`` and skip the check.
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
    """Yield rows from a numpy structured array, list-of-dicts, or ndarray.

    Replicator returns a numpy structured array in production. Tests pass
    a list of dicts (or list of tuples). Both iterate to yield one row
    per bbox; this helper hides the difference from the parser body.
    """
    if data is None:
        return []
    # numpy structured arrays, plain lists, and tuples all support iter().
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

    # Internal sentinel so we only build / attach once even if the caller
    # keeps a long-lived reference and __post_init__ races with attach.
    _attached: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.annotator is None:
            self.annotator = self._build_real_annotator()
        self._attached = True

    def _build_real_annotator(self) -> Any:
        """Create a real ``omni.replicator.core`` annotator. Lazy-imports.

        Separated so unit tests that pass an injected ``annotator`` never
        touch Omniverse. Only runtime users pay for the import.
        """
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
        """Return the current frame's parsed bboxes.

        Delegates all schema handling to :func:`parse_bbox_data`. Raises
        :class:`RuntimeError` if the extractor was never attached (should
        never happen under normal use — ``__post_init__`` handles it).
        """
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
        """Convenience: return the extracted bboxes as plain dicts.

        Use this when passing the bboxes straight into a frame-data
        payload or any JSON serialization. Both the run_sim_in_the_loop
        harness mode and any consumer that ingests bbox JSON take the
        flat-dict shape.
        """
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


def _maybe_numpy() -> Any:
    """Return the ``numpy`` module if importable, else ``None`` (pure fallback)."""
    try:
        import numpy as np  # noqa: PLC0415 — optional fast path; pure scan fallback exists
    except Exception:
        return None
    return np


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
    ``/World/Room/Foo/Foo``). So a target matches on its unique object-name
    segment as well as exact / descendant paths; the name is per-instance, so a
    same-label sibling still cannot match. One object can own several ids
    (multi-mesh). Returns an empty list when the target was not rendered.
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
            or (leaf and leaf in candidate.split("/"))
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
    is reduced to 2D. Uses numpy when available; falls back to a pure scan for
    list-of-lists fixtures.
    """
    ids = [segment_ids] if isinstance(segment_ids, int) else list(segment_ids)
    if not ids:
        return None
    np = _maybe_numpy()
    if np is not None and getattr(mask, "shape", None) is not None:
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
    bboxes: Iterable[DetectedBbox], segment_bbox: tuple[int, int, int, int]
) -> DetectedBbox | None:
    """Select the ``bounding_box_2d_tight`` row overlapping a segment's mask box.

    The target's per-instance identity comes from the segmentation mask; its
    occlusion ratio is only available per-class from ``bounding_box_2d_tight``.
    This picks the class row with the largest pixel overlap with the segment's
    mask bbox (ties -> first), or ``None`` when no row overlaps the segment.
    """
    best: DetectedBbox | None = None
    best_overlap = 0
    for bbox in bboxes:
        overlap = _rect_overlap_area(bbox.bbox_2d, segment_bbox)
        if overlap > best_overlap:
            best_overlap = overlap
            best = bbox
    return best


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
