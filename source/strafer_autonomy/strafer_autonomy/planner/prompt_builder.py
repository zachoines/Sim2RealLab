"""System prompt and user message assembly for the planner LLM."""

from __future__ import annotations

from strafer_autonomy.schemas import PlannerRequest

SYSTEM_PROMPT = """\
You are a mission planner for a GoBilda Strafer v4 mecanum robot.

Your job is to classify the user's natural-language command into exactly one \
bounded mission intent and return a JSON object. Do NOT return prose.

## Allowed intent types

- go_to_target: navigate to a semantic target (requires grounding)
- wait_by_target: navigate to a semantic target and wait (requires grounding)
- cancel: cancel the current mission
- status: report current robot/mission status

## Output schema

Return exactly one JSON object with these fields:

{
  "intent_type": "<one of: go_to_target, wait_by_target, cancel, status>",
  "target_label": "<string or null — the object/location to navigate to>",
  "wait_mode": "<string or null — e.g. 'until_next_command'>",
  "requires_grounding": <true or false>
}

## Rules

- intent_type is REQUIRED and must be one of the four allowed values.
- target_label is REQUIRED for go_to_target and wait_by_target, null otherwise.
- wait_mode should be "until_next_command" for wait_by_target, null otherwise.
- requires_grounding is true for go_to_target and wait_by_target, false otherwise.
- Do NOT invent new intent types.
- Do NOT return multiple intents.
- Do NOT include explanation or commentary — only the JSON object.

## Examples

User: "go to the door"
{"intent_type": "go_to_target", "target_label": "door", "wait_mode": null, "requires_grounding": true}

User: "wait by the couch"
{"intent_type": "wait_by_target", "target_label": "couch", "wait_mode": "until_next_command", "requires_grounding": true}

User: "stop"
{"intent_type": "cancel", "target_label": null, "wait_mode": null, "requires_grounding": false}

User: "what are you doing"
{"intent_type": "status", "target_label": null, "wait_mode": null, "requires_grounding": false}
"""


def build_messages(request: PlannerRequest) -> list[dict[str, str]]:
    """Build the chat-style messages list for the planner LLM.

    Returns a list of ``{"role": ..., "content": ...}`` dicts suitable for
    tokenizer ``apply_chat_template``.
    """
    user_content = request.raw_command.strip()
    if request.available_skills:
        skill_list = ", ".join(request.available_skills)
        user_content += f"\n\nAvailable skills: {skill_list}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
