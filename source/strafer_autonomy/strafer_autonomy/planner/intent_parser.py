"""Parse raw LLM text output into a validated MissionIntent."""

from __future__ import annotations

import json
import re

from strafer_autonomy.schemas import MissionIntent

_VALID_INTENTS = frozenset({
    "go_to_target",
    "wait_by_target",
    "cancel",
    "status",
    "rotate",
    "go_to_targets",
    "describe",
    "query",
    "patrol",
})

_INTENTS_REQUIRING_TARGET_LABEL = frozenset({"go_to_target", "wait_by_target"})
_INTENTS_REQUIRING_TARGETS_LIST = frozenset({"go_to_targets", "patrol"})

_FENCE_OPEN_RE = re.compile(r"```(?:json)?\s*", re.IGNORECASE)


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
    if intent_type in _INTENTS_REQUIRING_TARGET_LABEL and not target_label:
        raise IntentParseError(
            f"intent_type '{intent_type}' requires a non-empty target_label."
        )

    targets_raw = data.get("targets")
    targets: tuple[dict, ...] | None = None
    if intent_type in _INTENTS_REQUIRING_TARGETS_LIST:
        if not isinstance(targets_raw, list) or len(targets_raw) == 0:
            raise IntentParseError(
                f"intent_type '{intent_type}' requires a non-empty 'targets' list."
            )
        validated: list[dict] = []
        for i, entry in enumerate(targets_raw):
            if not isinstance(entry, dict):
                raise IntentParseError(
                    f"targets[{i}] must be an object; got {type(entry).__name__}."
                )
            label = entry.get("label")
            if not isinstance(label, str) or not label.strip():
                raise IntentParseError(
                    f"targets[{i}] must have a non-empty string 'label'."
                )
            validated.append(dict(entry))
        targets = tuple(validated)

    return MissionIntent(
        intent_type=intent_type,
        raw_command=raw_command,
        target_label=str(target_label) if target_label is not None else None,
        orientation_mode=(
            str(data["orientation_mode"])
            if data.get("orientation_mode") is not None
            else None
        ),
        wait_mode=str(data["wait_mode"]) if data.get("wait_mode") is not None else None,
        requires_grounding=bool(data.get("requires_grounding", False)),
        targets=targets,
    )


def _extract_json(text: str) -> str | None:
    """Try to extract a JSON object string from LLM output.

    Supports fenced code blocks, prose surrounding a JSON object, and nested
    objects (``{"targets": [{"label": "cup"}]}``) by scanning for balanced
    braces rather than relying on a regex.
    """
    if not text:
        return None

    # If the text is wrapped in a fenced code block, strip the fence so
    # the brace scanner doesn't confuse backticks with content.
    fence_match = _FENCE_OPEN_RE.search(text)
    if fence_match:
        after_fence = text[fence_match.end():]
        end_fence = after_fence.find("```")
        if end_fence != -1:
            text = after_fence[:end_fence]

    # Scan for the first balanced JSON object respecting strings/escapes.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1].strip()
        start = text.find("{", start + 1)

    return None
