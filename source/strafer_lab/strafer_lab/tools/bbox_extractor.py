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

- ``instanceId`` is NOT part of ``bounding_box_2d_tight``. The task spec in
  ``PHASE_15_WINDOWS.md`` asked for it, but that field only exists on the
  ``instance_id_segmentation`` annotator. We leave :class:`DetectedBbox`
  with a ``semantic_id`` field instead; callers that need per-instance IDs
  should attach a second annotator.

- :func:`parse_bbox_data` is a pure function with NO Isaac Sim / numpy /
  Omniverse dependency — it accepts any iterable-of-row-objects that behaves
  like a structured numpy array (supports ``row["field"]`` access). This
  keeps the parser unit-testable from ``.venv_vlm`` with synthetic dicts.

- :class:`ReplicatorBboxExtractor` defers the ``omni.replicator.core`` import
  to ``__init__`` so merely importing this module from a plain Python env
  (the namespace-stub path used by the DGX batch-processing tests) does not
  trigger an Omniverse load. Tests inject a mock annotator via the
  ``annotator`` keyword to exercise :meth:`extract` without Isaac Sim.

- Requires ``semanticLabel`` USD prim attributes on the scene objects. On
  Infinigen scenes these are populated by
  ``scripts/extract_scene_metadata.py`` (DGX Task 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


UNKNOWN_LABEL = "unknown"


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
        description builder and Task 9 / Task 12 data prep scripts expect
        on ``frame_data``.
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

    Row fields (in order, from Isaac Sim 5.1's
    ``data_visualization_writer.py`` schema comment):
    ``semanticId (uint32)``, ``x_min (int32)``, ``y_min (int32)``,
    ``x_max (int32)``, ``y_max (int32)``, ``occlusionRatio (float32)``.

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

        Use this when passing the bboxes straight into
        ``frame_data["bboxes"]`` for Task 2's ``collect_perception_data.py``
        HDF5 serialization, or for any JSON output.
        """
        return [bbox.to_dict() for bbox in self.extract()]
