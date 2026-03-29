"""Parse raw LLM text output into a validated MissionIntent."""

from __future__ import annotations

import json
import re

from strafer_autonomy.schemas import MissionIntent

_VALID_INTENTS = frozenset({"go_to_target", "wait_by_target", "cancel", "status"})

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


class IntentParseError(Exception):
    """Raised when LLM output cannot be parsed into a valid MissionIntent."""


def parse_intent(raw_output: str, raw_command: str) -> MissionIntent:
    """Extract a ``MissionIntent`` from the LLM's raw text output.

    Raises ``IntentParseError`` on malformed or invalid output.
    """
    json_str = _extract_json(raw_output)
    if json_str is None:
        raise IntentParseError(f"No JSON object found in LLM output: {raw_output!r}")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise IntentParseError(f"Invalid JSON in LLM output: {exc}") from exc

    if not isinstance(data, dict):
        raise IntentParseError(f"Expected a JSON object, got {type(data).__name__}")

    intent_type = data.get("intent_type")
    if intent_type not in _VALID_INTENTS:
        raise IntentParseError(
            f"Unknown or missing intent_type: {intent_type!r}. "
            f"Must be one of {sorted(_VALID_INTENTS)}."
        )

    target_label = data.get("target_label")
    if intent_type in ("go_to_target", "wait_by_target") and not target_label:
        raise IntentParseError(
            f"intent_type '{intent_type}' requires a non-empty target_label."
        )

    return MissionIntent(
        intent_type=intent_type,
        raw_command=raw_command,
        target_label=str(target_label) if target_label is not None else None,
        wait_mode=str(data["wait_mode"]) if data.get("wait_mode") is not None else None,
        requires_grounding=bool(data.get("requires_grounding", False)),
    )


def _extract_json(text: str) -> str | None:
    """Try to extract a JSON object string from LLM output.

    Handles fenced code blocks and bare JSON objects.
    """
    # Try fenced code block first
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()

    # Fall back to bare JSON object
    match = _JSON_OBJECT_RE.search(text)
    if match:
        return match.group(0).strip()

    return None
