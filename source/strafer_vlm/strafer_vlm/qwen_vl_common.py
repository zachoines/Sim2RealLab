#!/usr/bin/env python3
"""Shared helpers for local Qwen2.5-VL grounding workflows."""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

SYSTEM_PROMPT_DEFAULT = (
    "You are a robot navigation assistant. Given one camera image and an object "
    "description, return JSON only with keys: found (bool), bbox_2d "
    "([x1,y1,x2,y2] in 0..1000), label (string), optional confidence (0..1). "
    "If object is not visible, return {\"found\": false}."
)


@dataclass(frozen=True)
class GroundingTarget:
    """Ground-truth or model-predicted grounding target."""

    found: bool
    bbox_2d: tuple[int, int, int, int] | None = None
    label: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class GroundingExample:
    """One grounding sample backed by a local image path."""

    image_path: Path
    prompt: str
    target: GroundingTarget


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
    )


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
    )


def normalize_prediction_bbox_to_1000(
    prediction: GroundingTarget,
    *,
    image_width: int,
    image_height: int,
) -> GroundingTarget:
    """Normalize model bbox to [0,1000] if it appears to be in pixel coordinates."""
    if prediction.bbox_2d is None:
        return prediction

    x1, y1, x2, y2 = prediction.bbox_2d
    if max(x1, y1, x2, y2) <= 1000:
        return prediction

    if image_width <= 0 or image_height <= 0:
        return prediction

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
    )


def _resolve_path(raw_value: str, dataset_root: Path) -> Path:
    raw = raw_value.strip()
    if raw.startswith("file://"):
        parsed = urllib.parse.urlparse(raw)
        raw = urllib.parse.unquote(parsed.path)
        if len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (dataset_root / candidate).resolve()


def _parse_chat_record(record: dict[str, Any], dataset_root: Path) -> GroundingExample:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list.")

    user_message = next(
        (m for m in messages if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if user_message is None:
        raise ValueError("Could not find user message in chat record.")

    content = user_message.get("content")
    image_ref: str | None = None
    prompt: str | None = None

    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image" and image_ref is None:
                image_ref = _coerce_optional_text(item.get("image"))
            if item_type == "text" and prompt is None:
                prompt = _coerce_optional_text(item.get("text"))
    elif isinstance(content, str):
        prompt = content
    else:
        raise ValueError("Unsupported user content format.")

    if image_ref is None:
        image_ref = _coerce_optional_text(record.get("image") or record.get("image_path"))
    if prompt is None:
        raise ValueError("Could not parse prompt text from user message.")

    assistant_message = next(
        (
            m
            for m in reversed(messages)
            if isinstance(m, dict) and m.get("role") == "assistant"
        ),
        None,
    )
    if assistant_message is None:
        raise ValueError("Could not find assistant message in chat record.")

    target = parse_grounding_target(
        assistant_message.get("content"),
        strict_bbox_when_found=True,
    )
    if image_ref is None:
        raise ValueError("Could not parse image path from user message.")

    image_path = _resolve_path(image_ref, dataset_root)
    return GroundingExample(image_path=image_path, prompt=normalize_prompt(prompt), target=target)


def _parse_flat_record(record: dict[str, Any], dataset_root: Path) -> GroundingExample:
    image_ref = (
        _coerce_optional_text(record.get("image"))
        or _coerce_optional_text(record.get("image_path"))
        or _coerce_optional_text(record.get("img"))
    )
    if image_ref is None:
        raise ValueError("Missing image or image_path field.")

    prompt = (
        _coerce_optional_text(record.get("prompt"))
        or _coerce_optional_text(record.get("query"))
        or _coerce_optional_text(record.get("text"))
    )
    if prompt is None:
        raise ValueError("Missing prompt/query/text field.")

    target_payload: Any
    if "target" in record:
        target_payload = record["target"]
    elif "assistant" in record:
        target_payload = record["assistant"]
    elif "response" in record:
        target_payload = record["response"]
    else:
        target_payload = {
            "found": record.get("found"),
            "bbox_2d": record.get("bbox_2d"),
            "label": record.get("label"),
            "confidence": record.get("confidence"),
        }

    target = parse_grounding_target(target_payload, strict_bbox_when_found=True)
    image_path = _resolve_path(image_ref, dataset_root)
    return GroundingExample(image_path=image_path, prompt=normalize_prompt(prompt), target=target)


def load_grounding_dataset(dataset_path: str) -> list[GroundingExample]:
    """Load a grounding dataset from JSON or JSONL."""
    path = Path(dataset_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    dataset_root = path.parent
    records: list[dict[str, Any]] = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip().lstrip("\ufeff")
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON.") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object.")
                records.append(payload)
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            records = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict) and isinstance(payload.get("examples"), list):
            records = [item for item in payload["examples"] if isinstance(item, dict)]
        else:
            raise ValueError("Dataset JSON must be a list or an object with an 'examples' list.")

    examples: list[GroundingExample] = []
    for index, record in enumerate(records, start=1):
        try:
            if isinstance(record.get("messages"), list):
                example = _parse_chat_record(record, dataset_root)
            else:
                example = _parse_flat_record(record, dataset_root)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to parse dataset record #{index}: {exc}") from exc

        if not example.image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {example.image_path}")
        examples.append(example)

    if not examples:
        raise ValueError(f"No valid examples found in dataset: {path}")
    return examples


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
) -> Image.Image:
    """Return a copy of the image with denormalized bbox overlaid."""
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    px_bbox = denormalize_bbox_1000(bbox_2d, output.width, output.height)
    draw.rectangle(px_bbox, outline=color, width=width)
    if label:
        draw.text((px_bbox[0], max(0, px_bbox[1] - 16)), label, fill=color)
    return output


def _resolve_torch_dtype(dtype_name: str) -> Any:
    import torch

    normalized = dtype_name.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def get_model_device(model: Any) -> Any:
    """Best-effort retrieval of the model device."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_qwen_model_and_processor(
    *,
    model_name_or_path: str,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: str | None = None,
    load_in_4bit: bool = False,
    bnb_4bit_compute_dtype: str = "bfloat16",
) -> tuple[Any, Any]:
    """Load Qwen2.5-VL model + processor with optional 4-bit quantization."""
    import torch

    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Missing transformers dependency. Install with: "
            "pip install transformers accelerate"
        ) from exc

    resolved_dtype = _resolve_torch_dtype(torch_dtype)
    if resolved_dtype == "auto" and torch.cuda.is_available():
        # Prefer FP16 on CUDA to avoid defaulting to FP32 and OOM.
        resolved_dtype = torch.float16

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolved_dtype,
        "device_map": device_map,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "4-bit loading requested but bitsandbytes support is missing."
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_resolve_torch_dtype(bnb_4bit_compute_dtype),
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    return model, processor


def run_grounding_generation(
    *,
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT_DEFAULT,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:
    """Run one grounding inference and return decoded assistant text."""
    user_prompt = normalize_prompt(prompt)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "local_image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = processor(
        text=[chat_text],
        images=[image.convert("RGB")],
        return_tensors="pt",
    )
    device = get_model_device(model)
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }

    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    generated = model.generate(**model_inputs, **generation_kwargs)
    prompt_length = model_inputs["input_ids"].shape[1]
    completion_ids = generated[:, prompt_length:]
    decoded = processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip()
