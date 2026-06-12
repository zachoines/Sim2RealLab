"""First-class per-frame detection columns for strafer LeRobot v3 datasets.

Detections ride as padded parquet columns inside the LeRobot schema —
unlike depth (see :mod:`lerobot_depth`), a frame's detections are a few
hundred bytes of numbers, exactly what a parquet column stores well. The
four columns are declared together, padded to a fixed ``detections_max``
slot count:

- ``observation.detections.bbox``      float32 ``(detections_max, 4)``
  — pixel ``(x_min, y_min, x_max, y_max)``, top-left origin, in the
  perception camera's render-product resolution. Zero in padding rows.
- ``observation.detections.label_id``  int64 ``(detections_max,)``
  — index into ``meta/detection_labels.json``'s ``labels[]``; ``-1`` in
  padding rows so an accidental vocab lookup fails loudly.
- ``observation.detections.occlusion`` float32 ``(detections_max,)``
  — Replicator ``occlusionRatio``: ``0.0`` fully visible → ``1.0`` fully
  occluded. ``0.0`` in padding rows.
- ``observation.detections.valid``     bool ``(detections_max,)``
  — padding mask; the only authority on which rows are real detections.

The id↔string vocab lives at ``<root>/meta/detection_labels.json`` as
``{"labels": [...]}``, accumulated in first-seen order at capture time.
Ids are dataset-local: stable within one dataset, not across datasets —
consumers merging corpora must join through the label strings.

Box / label / occlusion conventions match
:class:`strafer_lab.tools.bbox_extractor.DetectedBbox` (the Replicator
``bbox_2d_tight`` parser) so the sim producer maps in without
translation.

This module is pure-Python — no torch, no LeRobot. Unit-testable
against synthetic detections. Callers (writer + consumers) import from
here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bbox_extractor import DetectedBbox


DETECTIONS_BBOX = "observation.detections.bbox"
DETECTIONS_LABEL_ID = "observation.detections.label_id"
DETECTIONS_OCCLUSION = "observation.detections.occlusion"
DETECTIONS_VALID = "observation.detections.valid"

DETECTIONS_MAX_DEFAULT = 32
PAD_LABEL_ID = -1

_VOCAB_FILENAME = "detection_labels.json"


def detections_features(detections_max: int) -> dict[str, dict[str, Any]]:
    """Return the four detections entries for a LeRobot v3 features dict."""
    n = int(detections_max)
    if n < 1:
        raise ValueError(f"detections_max must be >= 1; got {detections_max}")
    return {
        DETECTIONS_BBOX: {
            "dtype": "float32",
            "shape": (n, 4),
            "names": ["x_min", "y_min", "x_max", "y_max"],
        },
        DETECTIONS_LABEL_ID: {
            "dtype": "int64",
            "shape": (n,),
            "names": None,
        },
        DETECTIONS_OCCLUSION: {
            "dtype": "float32",
            "shape": (n,),
            "names": None,
        },
        DETECTIONS_VALID: {
            "dtype": "bool",
            "shape": (n,),
            "names": None,
        },
    }


class DetectionLabelVocab:
    """Accumulating label→id mapping persisted to ``meta/detection_labels.json``.

    Ids are assigned in first-seen order and never reassigned, so a
    ``label_id`` written early in a capture stays valid as the vocab
    grows. The writer owns one instance per dataset and persists it at
    finalize; consumers read it back with :func:`read_detection_labels`.
    """

    def __init__(self, labels: Sequence[str] = ()) -> None:
        self._labels: list[str] = []
        self._ids: dict[str, int] = {}
        for label in labels:
            self.id_for(label)

    def id_for(self, label: str) -> int:
        """Return the id for ``label``, assigning the next id if unseen."""
        label = str(label)
        existing = self._ids.get(label)
        if existing is not None:
            return existing
        new_id = len(self._labels)
        self._labels.append(label)
        self._ids[label] = new_id
        return new_id

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self._labels)

    def __len__(self) -> int:
        return len(self._labels)

    def write(self, dataset_root: Path | str) -> Path:
        """Persist to ``<root>/meta/detection_labels.json``; returns the path."""
        path = vocab_path(dataset_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"labels": self._labels}, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def vocab_path(dataset_root: Path | str) -> Path:
    """Return ``<root>/meta/detection_labels.json``."""
    return Path(dataset_root) / "meta" / _VOCAB_FILENAME


def read_detection_labels(dataset_root: Path | str) -> tuple[str, ...]:
    """Read the label vocab back; ``()`` when the dataset has none."""
    path = vocab_path(dataset_root)
    if not path.is_file():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(str(label) for label in payload["labels"])


def _bbox_area(bbox: DetectedBbox) -> int:
    x1, y1, x2, y2 = bbox.bbox_2d
    return (x2 - x1) * (y2 - y1)


def pack_detections(
    detections: Sequence[DetectedBbox],
    detections_max: int,
    vocab: DetectionLabelVocab,
) -> dict[str, np.ndarray]:
    """Pack one frame's detections into the four padded column arrays.

    Degenerate boxes (zero-or-negative area) are dropped. When more than
    ``detections_max`` remain, the largest-pixel-area boxes are kept,
    with a deterministic tie-break on ``(label, bbox_2d)`` so the same
    detection set always packs identically. ``vocab`` accumulates any
    unseen labels as a side effect.
    """
    n = int(detections_max)
    kept = [d for d in detections if not d.is_degenerate]
    if len(kept) > n:
        kept.sort(key=lambda d: (-_bbox_area(d), d.label, d.bbox_2d))
        kept = kept[:n]

    bbox = np.zeros((n, 4), dtype=np.float32)
    label_id = np.full((n,), PAD_LABEL_ID, dtype=np.int64)
    occlusion = np.zeros((n,), dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)
    for i, det in enumerate(kept):
        bbox[i] = det.bbox_2d
        label_id[i] = vocab.id_for(det.label)
        occlusion[i] = det.occlusion_ratio
        valid[i] = True

    return {
        DETECTIONS_BBOX: bbox,
        DETECTIONS_LABEL_ID: label_id,
        DETECTIONS_OCCLUSION: occlusion,
        DETECTIONS_VALID: valid,
    }
