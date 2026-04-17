# Strafer Autonomy LLM Planner

This document defines the LLM planner service for `strafer_autonomy`.

It is intended to be specific enough that an implementation agent can begin building the planner without re-deciding the system boundary.

## Scope

This document covers:

- the planner's role in the autonomy stack
- the planner service contract
- bounded planner behavior for the MVP
- recommended project structure under `source/strafer_autonomy`
- recommended small-model choices for local or hosted planner inference

This document does not cover:

- Jetson-side robot execution details
- VLM grounding service implementation
- cloud deployment details beyond planner hosting fit

Those live in:

- `docs/STRAFER_AUTONOMY_ROS.md`
- `docs/STRAFER_AUTONOMY_INTERFACES.md`
- `docs/STRAFER_AUTONOMY_MVP_RUNTIME_DECISION.md`
- `docs/STRAFER_AUTONOMY_VLM_GROUNDING.md`

## Planner Role

The planner is a text-to-plan service.

It takes:

- one user command
- bounded robot context
- the available skill registry

It returns:

- one bounded `MissionPlan`

The planner does not execute anything itself.

The planner is responsible for:

- interpreting the user's natural-language command
- mapping that command onto the supported skill registry
- choosing a short, typed skill sequence
- filling planner-owned arguments such as `label`, `mode`, `standoff_m`, and `execution_backend`
- deciding whether the mission requires grounding

The planner is not responsible for:

- publishing ROS topics
- issuing raw navigation goals
- projecting image detections into map-frame targets
- running safety checks
- handling low-level retries inside robot control loops
- deciding motor commands

That split must remain strict.

## Runtime Placement

Chosen MVP placement:

- Jetson: `strafer_autonomy.executor`
- Windows workstation: `strafer_autonomy.planner` service

The planner is remote from the executor's point of view.

```text
Jetson executor
  -> POST /plan over LAN
  -> MissionPlan
  -> local robot execution
```

This is the right split because:

- planner requests are low-payload and synchronous
- planner latency is acceptable if it stays off the Jetson
- the robot remains safe if planner service becomes unavailable

## External Contract

The source of truth for the request and plan schema remains:

```text
source/strafer_autonomy/strafer_autonomy/schemas/mission.py
```

Current relevant types:

- `PlannerRequest`
- `MissionIntent`
- `MissionPlan`
- `SkillCall`

## `POST /plan`

### Request

The first planner API should accept the current `PlannerRequest` shape:

```json
{
  "request_id": "plan_001",
  "raw_command": "wait by the door for me",
  "robot_state": null,
  "active_mission_summary": null,
  "available_skills": [
    "capture_scene_observation",
    "locate_semantic_target",
    "project_detection_to_goal_pose",
    "navigate_to_pose",
    "wait",
    "cancel_mission",
    "report_status"
  ]
}
```

### Response

The first planner API should return a `MissionPlan`:

```json
{
  "mission_id": "mission_001",
  "mission_type": "wait_by_target",
  "raw_command": "wait by the door for me",
  "created_at": 1710000000.0,
  "steps": [
    {
      "step_id": "step_01",
      "skill": "capture_scene_observation",
      "args": {},
      "timeout_s": 1.0,
      "retry_limit": 0
    },
    {
      "step_id": "step_02",
      "skill": "locate_semantic_target",
      "args": {"label": "door"},
      "timeout_s": 8.0,
      "retry_limit": 1
    },
    {
      "step_id": "step_03",
      "skill": "project_detection_to_goal_pose",
      "args": {"standoff_m": 0.7},
      "timeout_s": 2.0,
      "retry_limit": 0
    },
    {
      "step_id": "step_04",
      "skill": "navigate_to_pose",
      "args": {
        "goal_source": "projected_target",
        "execution_backend": "nav2"
      },
      "timeout_s": 90.0,
      "retry_limit": 0
    },
    {
      "step_id": "step_05",
      "skill": "orient_relative_to_target",
      "args": {"mode": "face_away"},
      "timeout_s": 15.0,
      "retry_limit": 0
    },  // post-MVP — omit until orient skill is implemented
    {
      "step_id": "step_06",
      "skill": "wait",
      "args": {"mode": "until_next_command"},
      "timeout_s": null,
      "retry_limit": 0
    }
  ]
}
```

### Additional endpoint

- `GET /health`

## Recommended Internal Architecture

The external contract should stay simple:

- request in
- validated plan out

Internally, the planner should use a two-stage flow:

```text
PlannerRequest
  -> prompt builder
  -> LLM output
  -> parse MissionIntent-like structure
  -> deterministic plan compiler
  -> MissionPlan validator
  -> MissionPlan response
```

This is more robust than letting the model invent the final step list directly.

### Why use an internal `MissionIntent` stage

The model is good at:

- intent recognition
- target-label extraction
- orientation and wait-mode selection
- deciding whether grounding is needed

The model is not the right place to hard-code:

- exact step ordering templates
- default timeout values
- retry policies
- execution-backend defaults

Those should come from deterministic planner code.

Recommended split:

- LLM output: bounded `MissionIntent` plus optional planner notes
- deterministic compiler: `MissionIntent -> MissionPlan`

For the MVP, the external `POST /plan` response should still be only `MissionPlan`.

## MVP Supported Intents

The first planner should support only a small bounded intent set.

Recommended MVP intents:

- `go_to_target`
- `wait_by_target`
- `cancel`
- `status`

Possible examples:

| User command | Mission type | Requires grounding |
|---|---|---|
| `go to the door` | `go_to_target` | yes |
| `wait by the couch` | `wait_by_target` | yes |
| `stop` | `cancel` | no |
| `what are you doing` | `status` | no |

Do not support in the first planner:

- arbitrary multi-room navigation
- open-ended search loops
- follow-me
- patrol
- return-to-origin memory
- freeform multi-step decomposition beyond the bounded templates

Those can come later.

## Skill Templates

The deterministic compiler should use a small set of templates.

### `go_to_target`

```text
capture_scene_observation
locate_semantic_target(label=...)
project_detection_to_goal_pose(standoff_m=0.7)
navigate_to_pose(goal_source="projected_target", execution_backend="nav2")
```

### `wait_by_target`

```text
capture_scene_observation
locate_semantic_target(label=...)
project_detection_to_goal_pose(standoff_m=0.7)
navigate_to_pose(goal_source="projected_target", execution_backend="nav2")
orient_relative_to_target(mode=...)  # post-MVP — omit until orient skill is implemented
wait(mode="until_next_command")
```

### `cancel`

```text
cancel_mission
```

### `status`

```text
report_status
```

## Execution Mode Selection

The planner should understand the three robot-local execution modes already defined elsewhere:

- `nav2`
- `strafer_direct`
- `hybrid_nav2_strafer`

Planner rule for MVP:

- default to `nav2`

Later extensions:

- allow mission-level override from operator config
- allow planner hints for `strafer_direct` or `hybrid_nav2_strafer`
- keep final policy under deterministic validation, not raw model freedom

The planner should not be free to invent arbitrary execution-mode strings.

## Validation Rules

Every planner response must be validated before it reaches the executor.

Required validation:

- all skills must exist in `available_skills`
- no unknown top-level fields
- no duplicate `step_id`
- allowed args only for each skill
- enum fields must be from approved values
- all timeout and retry values must be in safe bounds
- unsupported mission types must fail closed

Recommended planner failure behavior:

- if the model output is malformed, return planner failure
- if the command is unsupported, return planner failure with a short reason
- if the command is ambiguous but recoverable, optionally return a clarification-needed failure later

For the MVP, it is acceptable to fail unsupported or ambiguous commands instead of asking follow-up questions.

## Prompting Strategy

The planner prompt should be explicit and narrow.

The system prompt should include:

- planner role
- allowed mission types
- allowed skill names
- allowed arg keys and enums
- examples of valid and invalid outputs
- instruction to return JSON only
- instruction not to emit prose

Recommended model behavior defaults:

- default planner mode: efficient non-thinking mode
- optional fallback mode: thinking mode only for hard commands, replans, or debug runs

Reason:

- routine mission planning is a short structured classification and mapping task
- reasoning mode is not needed for every command
- planner latency matters more than open-ended reasoning depth

## Recommended Project Structure

Planner code should live under:

```text
source/strafer_autonomy/strafer_autonomy/planner/
```

Recommended structure:

```text
source/strafer_autonomy/
  strafer_autonomy/
    planner/
      __init__.py
      app.py
      service.py
      model_config.py
      llm_runtime.py
      prompt_builder.py
      prompt_templates.py
      intent_parser.py
      plan_compiler.py
      validators.py
      examples.py
```

Suggested responsibilities:

- `app.py`
  - HTTP entry point
- `service.py`
  - request handling and orchestration
- `model_config.py`
  - chosen model id, sampling config, thinking-mode policy
- `llm_runtime.py`
  - model load and generation wrapper
- `prompt_builder.py`
  - structured prompt assembly from `PlannerRequest`
- `intent_parser.py`
  - parse model output into a bounded internal structure
- `plan_compiler.py`
  - deterministic `MissionIntent -> MissionPlan`
- `validators.py`
  - schema and safety validation

## LLM Selection Requirements

The planner model should be:

- text-only
- small enough to run comfortably on a workstation-class GPU
- workable under an 8 GB VRAM target
- strong at instruction following and structured output
- usable behind a simple local HTTP service
- permissively licensed if possible

For this planner, raw coding ability matters less than:

- reliable JSON output
- bounded tool or skill selection
- low latency
- stable instruction following

## Candidate Models

The following candidates are the strongest small-model fits I found from official model cards and vendor documentation.

### 1. `Qwen/Qwen3-4B`

Why it fits:

- 4.0B parameters
- Apache 2.0
- 32,768 native context, with documented YaRN extension to 131,072
- explicit thinking and non-thinking modes in one model
- explicit agentic and external-tool support
- documented deployment path through vLLM or SGLang with OpenAI-compatible serving

Best use here:

- primary workstation-hosted planner when 4-bit quantization is acceptable

Tradeoffs:

- at 4B dense parameters, bf16/fp16 loading is likely too tight for an 8 GB VRAM budget once runtime overhead is included
- use quantization if the 8 GB budget is strict

### 2. `Qwen/Qwen3-1.7B`

Why it fits:

- same planner-friendly Qwen3 feature set
- 1.7B parameters
- Apache 2.0
- 32,768 native context
- same thinking/non-thinking and agentic positioning as Qwen3-4B

Best use here:

- lowest-risk strict-VRAM planner choice
- easiest path if you want to stay in the Qwen ecosystem without relying on 4-bit quantization

Tradeoffs:

- lower ceiling than 4B-class models
- more likely to need strict prompt discipline and deterministic compilation

### 3. `microsoft/Phi-4-mini-instruct`

Why it fits:

- 3.8B parameters
- MIT license
- 128K context
- explicitly positioned for memory/compute-constrained and latency-bound use cases
- explicit tool-enabled function-calling format
- Microsoft reports strong small-model benchmark performance against several same-size and larger baselines

Best use here:

- strong alternative if you prioritize reasoning density and an MIT license

Tradeoffs:

- tool-calling format is more custom than the Qwen stack
- if you already plan to use Qwen for VLM and Qwen-compatible serving elsewhere, Phi adds another model family to operate

### 4. `meta-llama/Llama-3.2-3B-Instruct`

Why it fits:

- 3B parameters
- explicitly intended for agentic applications
- Meta documents quantized variants for limited-compute use cases
- official model card includes memory figures for bf16 and quantized variants

Best use here:

- conservative fallback if you want a very widely supported small instruct model

Tradeoffs:

- custom Meta license instead of Apache 2.0 or MIT
- less planner-specific tooling guidance than Qwen3

### 5. `google/gemma-3-4b-it`

Why it fits:

- 4B-class lightweight model
- 128K context
- Google positions Gemma 3 as a lightweight, state-of-the-art open model suitable for limited-resource deployment

Why it is not the first recommendation here:

- multimodal capability is not needed for the planner
- gated access under Gemma terms adds friction
- the planner does not benefit from carrying image capability here

## Recommendation

### Primary recommendation

Use:

- `Qwen/Qwen3-4B`

Planner serving mode:

- workstation-hosted
- non-thinking mode by default
- 4-bit quantized if the VRAM ceiling is truly under 8 GB

Why:

- best feature fit for this planner job
- same vendor family as the current Qwen VLM work
- explicit support for agentic use
- explicit support for switching between fast and reasoning-heavy behavior
- Apache 2.0
- easy path to an OpenAI-compatible planner endpoint

### Secondary recommendation

Use:

- `Qwen/Qwen3-1.7B`

when:

- the VRAM budget must stay safely below 8 GB without relying on aggressive quantization
- lower latency matters more than planner quality ceiling

### Best non-Qwen alternative

Use:

- `microsoft/Phi-4-mini-instruct`

when:

- you want a text-only model with a strong reasoning profile
- MIT license matters
- you are comfortable with its custom function-calling prompt format

## VRAM Guidance

This part is an engineering inference, not a vendor-published apples-to-apples comparison.

Rules of thumb:

- 1.7B dense model in bf16 is usually comfortable under 8 GB
- 3B to 4B dense models in bf16/fp16 are near or above the practical edge of an 8 GB card once runtime overhead and KV cache are included
- 3B to 4B models become much safer under an 8 GB target when loaded in 4-bit form

Relevant official datapoint:

- Meta's Llama 3.2 3B card reports about 7,419 MB memory size for the bf16 baseline and about 3,726 to 4,060 MB for quantized variants

Operational conclusion:

- if the 8 GB requirement is strict and simplicity matters, pick `Qwen3-1.7B`
- if quality matters more and 4-bit serving is acceptable, pick `Qwen3-4B`

## Implementation Order

1. Create `strafer_autonomy/planner/`
2. Implement `prompt_builder.py`
3. Implement `llm_runtime.py`
4. Implement `intent_parser.py`
5. Implement `plan_compiler.py`
6. Implement `validators.py`
7. Implement `app.py` with:
   - `POST /plan`
   - `GET /health`
8. Wire `strafer_autonomy.clients.planner_client` to the new service

## Sources

- Qwen3-4B model card: https://huggingface.co/Qwen/Qwen3-4B
- Qwen3-1.7B model card: https://huggingface.co/Qwen/Qwen3-1.7B
- Qwen2.5 blog: https://qwenlm.github.io/blog/qwen2.5/
- Phi-4-mini-instruct model card: https://huggingface.co/microsoft/Phi-4-mini-instruct
- Gemma 3 4B model card: https://huggingface.co/google/gemma-3-4b-it
- Llama 3.2 3B Instruct model card: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
