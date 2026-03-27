"""Dataset loading and serialisation helpers for grounding datasets."""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any

from strafer_vlm.inference.parsing import (
    GroundingExample,
    _coerce_optional_text,
    normalize_prompt,
    parse_grounding_target,
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
