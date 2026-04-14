"""JSON extraction, bbox coercion, coordinate conversion, and grounding parsing.

This module contains all pure-function helpers that do **not** require
``torch`` or a loaded model at import time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

SYSTEM_PROMPT_DEFAULT = (
    "You are a robot navigation assistant. Given one camera image and an object "
    "description, return JSON only with keys: found (bool), bbox_2d "
    "([x1,y1,x2,y2] in 0..1000), label (string), optional confidence (0..1). "
    'If object is not visible, return {"found": false}.'
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundingTarget:
    """Ground-truth or model-predicted grounding target."""

    found: bool
    bbox_2d: tuple[int, int, int, int] | None = None
    label: str | None = None
    confidence: float | None = None
    bbox_coordinate_mode: str | None = None


@dataclass(frozen=True)
class GroundingExample:
    """One grounding sample backed by a local image path."""

    image_path: "Path"  # noqa: F821 — resolved at runtime via pathlib
    prompt: str
    target: GroundingTarget


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt into the expected instruction form."""
    cleaned = str(prompt).strip()
    if not cleaned:
        raise ValueError("Prompt cannot be empty.")
    if cleaned.lower().startswith("locate:"):
        return cleaned
    return f"Locate: {cleaned}"


def serialize_target(target: GroundingTarget) -> str:
    """Serialize a grounding target to a stable JSON string."""
    payload: dict[str, Any] = {"found": bool(target.found)}
    if target.found and target.bbox_2d is not None:
        payload["bbox_2d"] = list(target.bbox_2d)
    if target.label:
        payload["label"] = target.label
    if target.confidence is not None:
        payload["confidence"] = round(float(target.confidence), 4)
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object found in arbitrary model text."""
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        try:
            candidate = json.loads(fenced_match.group(1))
            if isinstance(candidate, dict):
                return candidate
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, numeric))


def _coerce_bbox(
    value: Any,
    *,
    clamp_range: tuple[int, int] | None = (0, 1000),
) -> tuple[int, int, int, int] | None:
    if value is None or not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    coords: list[int] = []
    for item in value:
        try:
            numeric = int(round(float(item)))
        except (TypeError, ValueError):
            return None
        if clamp_range is not None:
            low, high = clamp_range
            numeric = max(low, min(high, numeric))
        coords.append(numeric)
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Grounding target parsing
# ---------------------------------------------------------------------------


def parse_grounding_target(value: Any, *, strict_bbox_when_found: bool) -> GroundingTarget:
    """Parse a grounding target from dict or JSON string."""
    payload = value
    if isinstance(value, str):
        payload = extract_first_json_object(value) or {}
    if not isinstance(payload, dict):
        raise ValueError("Grounding target must be a JSON object.")

    bbox = _coerce_bbox(payload.get("bbox_2d"))
    if strict_bbox_when_found and "found" not in payload and bbox is None:
        raise ValueError("Target JSON must include found and/or bbox_2d.")
    found = _coerce_bool(payload.get("found"), default=bbox is not None)

    if found and strict_bbox_when_found and bbox is None:
        raise ValueError("Target has found=true but missing/invalid bbox_2d.")

    return GroundingTarget(
        found=found,
        bbox_2d=bbox,
        label=_coerce_optional_text(payload.get("label")),
        confidence=_coerce_confidence(payload.get("confidence")),
        bbox_coordinate_mode="normalized_1000",
    )


_REF_BOX_RE = re.compile(
    r"<ref>(?P<label>.*?)</ref>\s*(?P<boxes>(?:<box>\([^)]+\),\([^)]+\)</box>\s*)+)",
    re.DOTALL,
)
_SINGLE_BOX_RE = re.compile(
    r"<box>\((?P<x1>-?\d+(?:\.\d+)?),\s*(?P<y1>-?\d+(?:\.\d+)?)\),\s*"
    r"\((?P<x2>-?\d+(?:\.\d+)?),\s*(?P<y2>-?\d+(?:\.\d+)?)\)</box>"
)


@dataclass(frozen=True)
class DetectedObjectParse:
    """One parsed ``<ref>/<box>`` detection from multi-object VLM output."""

    label: str
    bbox_2d: tuple[int, int, int, int]
    confidence: float


def parse_ref_box_detections(
    text: str,
    *,
    default_confidence: float = 1.0,
) -> list[DetectedObjectParse]:
    """Parse Qwen2.5-VL ``<ref>label</ref><box>(x1,y1),(x2,y2)</box>`` output.

    Each ``<ref>`` can be followed by one or more ``<box>`` tags. Coordinates
    are preserved exactly as emitted by the VLM (Qwen normalized 0..1000
    space); the caller is responsible for converting to pixel coordinates.
    Malformed entries are skipped rather than raising. Detections are
    deduplicated by (label, bbox) while preserving insertion order.
    """

    if not text:
        return []

    out: list[DetectedObjectParse] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for ref_match in _REF_BOX_RE.finditer(text):
        label = ref_match.group("label").strip()
        if not label:
            continue
        box_blob = ref_match.group("boxes")
        for box_match in _SINGLE_BOX_RE.finditer(box_blob):
            try:
                x1 = int(round(float(box_match.group("x1"))))
                y1 = int(round(float(box_match.group("y1"))))
                x2 = int(round(float(box_match.group("x2"))))
                y2 = int(round(float(box_match.group("y2"))))
            except (TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            bbox = (x1, y1, x2, y2)
            key = (label, bbox)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                DetectedObjectParse(
                    label=label,
                    bbox_2d=bbox,
                    confidence=max(0.0, min(1.0, float(default_confidence))),
                )
            )
    return out


def parse_grounding_prediction(text: str) -> GroundingTarget | None:
    """Parse model output text into a grounding target, if possible."""
    payload = extract_first_json_object(text)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return None

    bbox = _coerce_bbox(payload.get("bbox_2d"), clamp_range=None)
    found = _coerce_bool(payload.get("found"), default=bbox is not None)
    return GroundingTarget(
        found=found,
        bbox_2d=bbox,
        label=_coerce_optional_text(payload.get("label")),
        confidence=_coerce_confidence(payload.get("confidence")),
        bbox_coordinate_mode=None,
    )


# ---------------------------------------------------------------------------
# Coordinate conversion & bbox utilities
# ---------------------------------------------------------------------------


def infer_prediction_bbox_coordinate_mode(
    bbox_2d: tuple[int, int, int, int] | None,
    *,
    image_width: int,
    image_height: int,
) -> str | None:
    """Infer whether a prediction bbox is pixel-space or normalized [0,1000]."""
    if bbox_2d is None:
        return None

    x1, y1, x2, y2 = bbox_2d
    if image_width > 0 and image_height > 0:
        fits_image = x1 >= 0 and y1 >= 0 and x2 <= image_width and y2 <= image_height
        if fits_image:
            return "pixel"
    if max(x1, y1, x2, y2) > 1000:
        return "pixel"
    return "normalized_1000"


def normalize_prediction_bbox_to_1000(
    prediction: GroundingTarget,
    *,
    image_width: int,
    image_height: int,
) -> GroundingTarget:
    """Normalize a model prediction bbox to [0,1000]."""
    if prediction.bbox_2d is None:
        return prediction

    coordinate_mode = prediction.bbox_coordinate_mode or infer_prediction_bbox_coordinate_mode(
        prediction.bbox_2d,
        image_width=image_width,
        image_height=image_height,
    )
    if coordinate_mode != "pixel":
        return GroundingTarget(
            found=prediction.found,
            bbox_2d=prediction.bbox_2d,
            label=prediction.label,
            confidence=prediction.confidence,
            bbox_coordinate_mode="normalized_1000",
        )

    if image_width <= 0 or image_height <= 0:
        return prediction

    x1, y1, x2, y2 = prediction.bbox_2d
    nx1 = int(round(x1 * 1000.0 / image_width))
    nx2 = int(round(x2 * 1000.0 / image_width))
    ny1 = int(round(y1 * 1000.0 / image_height))
    ny2 = int(round(y2 * 1000.0 / image_height))
    normalized_bbox = _coerce_bbox((nx1, ny1, nx2, ny2), clamp_range=(0, 1000))
    return GroundingTarget(
        found=prediction.found,
        bbox_2d=normalized_bbox,
        label=prediction.label,
        confidence=prediction.confidence,
        bbox_coordinate_mode="normalized_1000",
    )


def clamp_pixel_bbox(
    bbox_2d: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Clamp pixel-space bbox coordinates to image bounds."""
    x1, y1, x2, y2 = bbox_2d
    px1 = max(0, min(image_width - 1, x1))
    px2 = max(0, min(image_width - 1, x2))
    py1 = max(0, min(image_height - 1, y1))
    py2 = max(0, min(image_height - 1, y2))
    if px2 <= px1:
        px2 = min(image_width - 1, px1 + 1)
    if py2 <= py1:
        py2 = min(image_height - 1, py1 + 1)
    return (px1, py1, px2, py2)


def bbox_to_pixel_coords(
    bbox_2d: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    coordinate_mode: str | None,
) -> tuple[int, int, int, int]:
    """Convert a bbox in either pixel or normalized coordinates to pixel coords."""
    if coordinate_mode == "pixel":
        return clamp_pixel_bbox(
            bbox_2d,
            image_width=image_width,
            image_height=image_height,
        )
    return denormalize_bbox_1000(bbox_2d, image_width, image_height)


def denormalize_bbox_1000(
    bbox_2d: tuple[int, int, int, int], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Convert [0,1000] bbox to pixel coordinates for a given image shape."""
    x1, y1, x2, y2 = bbox_2d
    px1 = int(round((x1 / 1000.0) * image_width))
    py1 = int(round((y1 / 1000.0) * image_height))
    px2 = int(round((x2 / 1000.0) * image_width))
    py2 = int(round((y2 / 1000.0) * image_height))
    px1 = max(0, min(image_width - 1, px1))
    px2 = max(0, min(image_width - 1, px2))
    py1 = max(0, min(image_height - 1, py1))
    py2 = max(0, min(image_height - 1, py2))
    if px2 <= px1:
        px2 = min(image_width - 1, px1 + 1)
    if py2 <= py1:
        py2 = min(image_height - 1, py1 + 1)
    return (px1, py1, px2, py2)


def bbox_iou(
    bbox_a: tuple[int, int, int, int] | None,
    bbox_b: tuple[int, int, int, int] | None,
) -> float | None:
    """Compute IoU for two normalized bboxes."""
    if bbox_a is None or bbox_b is None:
        return None

    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def overlay_bbox(
    image: Image.Image,
    bbox_2d: tuple[int, int, int, int],
    *,
    label: str | None = None,
    color: str = "lime",
    width: int = 3,
    coordinate_mode: str | None = "normalized_1000",
) -> Image.Image:
    """Return a copy of the image with bbox overlaid."""
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    px_bbox = bbox_to_pixel_coords(
        bbox_2d,
        image_width=output.width,
        image_height=output.height,
        coordinate_mode=coordinate_mode,
    )
    draw.rectangle(px_bbox, outline=color, width=width)
    if label:
        draw.text((px_bbox[0], max(0, px_bbox[1] - 16)), label, fill=color)
    return output
