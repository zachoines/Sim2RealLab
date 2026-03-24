# Strafer Autonomy Roadmap

This document reframes the autonomy stack around typed robot skills instead of a single Phase 5 VLM integration step.

The main design change is simple:
- Qwen is not the autonomy layer.
- Qwen is one skill backend inside a larger task execution system.
- `strafer_ros` remains the safe execution runtime on the robot.
- `strafer_vlm` remains the perception package for grounding.
- A new `strafer_autonomy` package should own planning, mission execution, and skill orchestration.

This document does four things:
1. Defines the initial Strafer skill registry.
2. Maps each skill onto `strafer_ros`, `strafer_vlm`, and `strafer_autonomy`.
3. Defines an MVP built around a small LLM planner plus a deterministic executor.
4. Defines a staged roadmap that can eventually replace the current single `PHASE_5_VLM_INTEGRATION.md` framing.

## Design Principles

1. Keep real-time control local.
   - Navigation control, safety checks, and motor commands stay in `strafer_ros`.
   - Cloud or workstation models may advise, but they do not own the control loop.

2. Make every capability a typed skill.
   - A skill has a name, an input schema, an output schema, and clear failure modes.
   - This lets the system grow from a small LLM-planned MVP into a larger autonomy stack without changing the robot execution layer.

3. Keep the VLM narrow.
   - The VLM should answer perception questions such as "where is the door?"
   - It should not be responsible for mission planning, waiting logic, sequencing, or human dialogue.

4. Separate planning from execution.
   - The planner may be an LLM.
   - The executor should still be deterministic, stateful, and bounded.
   - Long-running actions like locate, navigate, wait, and cancel still require mission state and explicit execution control.

5. Put orchestration in its own package.
   - Planning logic does not belong in `strafer_ros`.
   - Mission logic does not belong in `strafer_vlm`.
   - `strafer_autonomy` should become the clean boundary for skill registry, mission plans, planner prompts, execution state, and tool clients.

## System Layers

| Layer | Responsibility | Main Runtime |
|------|----------------|--------------|
| User interface | Accept command text and show status | Web, mobile, console |
| Autonomy layer | Convert user intent into typed plans, execute missions, track step state | `strafer_autonomy` |
| Perception layer | Ground semantic targets in images and return detections | `strafer_vlm` |
| Robot execution layer | Execute typed skills safely and report progress | `strafer_ros` |
| Control layer | Local navigation, control, sensing, and hardware safety | Nav2, RL policy, ROS drivers |

## Initial Strafer Skill Registry

The initial registry is intentionally small. It is enough to support a useful MVP and a clean path to a larger system.

### Core robot and perception skills

| Skill | Purpose | Inputs | Outputs | Primary owner | Uses VLM | MVP |
|------|---------|--------|---------|---------------|----------|-----|
| `get_robot_state` | Return current robot state for planning and status | none | pose, velocity, battery, nav state, timestamp | `strafer_ros` | No | Yes |
| `capture_scene_observation` | Capture synchronized RGB-D and camera metadata | camera source, frame request id | image refs, depth refs, intrinsics, timestamp | `strafer_ros` | No | Yes |
| `locate_semantic_target` | Find a named object or place in the current scene | image, prompt text | found, bbox, label, confidence | `strafer_vlm` | Yes | Yes |
| `project_detection_to_goal_pose` | Convert a 2D detection into a reachable goal pose | bbox, depth, intrinsics, TF, robot pose | goal pose, approach pose, quality flags | `strafer_ros` | No | Yes |
| `navigate_to_pose` | Navigate the robot to a target pose | pose, tolerances, timeout | success, progress, final pose, failure code | `strafer_ros` | No | Yes |
| `orient_relative_to_target` | Rotate or face relative to a known target or heading | target pose or yaw rule | success, final yaw | `strafer_ros` | No | Yes |
| `wait` | Hold position until timeout or interruption | duration or wait mode | success, interrupted flag | `strafer_ros` | No | Yes |
| `cancel_mission` | Stop the current mission safely | mission id or active mission | canceled flag, final state | `strafer_ros` | No | Yes |
| `report_status` | Produce operator-facing mission status | mission id | mission state, current step, last error | `strafer_ros` | No | Yes |

### Autonomy-layer skills

| Skill | Purpose | Inputs | Outputs | Primary owner | Uses VLM | MVP |
|------|---------|--------|---------|---------------|----------|-----|
| `interpret_user_command` | Turn freeform text into a typed mission request | natural language command, robot context | structured mission intent | `strafer_autonomy` planner | No | Yes |
| `plan_skill_sequence` | Build a bounded skill list from mission intent | mission intent, skill catalog, world state | ordered `SkillCall` list | `strafer_autonomy` planner | Indirectly | Yes |
| `validate_plan` | Reject unsafe or unsupported plans before execution | proposed plan, skill catalog | accepted plan or validation errors | `strafer_autonomy` | No | Yes |
| `select_next_skill` | Choose the next executable step from mission state | mission state, world state | next skill call or mission completion | `strafer_autonomy` executor | No | Yes |
| `repair_plan` | Recover after a failed step | mission state, failure code, world state | revised plan or terminal failure | `strafer_autonomy` planner | Optional | Later |
| `ask_for_clarification` | Request user clarification when intent is ambiguous | command text, ambiguity reason | clarification question | `strafer_autonomy` planner | No | Later |

## Ownership Map

### `strafer_ros`

`strafer_ros` owns deterministic, stateful, safety-critical execution.

It should own:
- robot state access
- synchronized sensor capture
- depth-based pose projection
- navigation and orientation execution
- wait, cancel, and status reporting
- local action execution state
- safety constraints and action validation at the robot boundary

It should not own:
- open-ended natural language planning
- planner prompts and LLM backends
- freeform dialogue
- semantic grounding model inference

### `strafer_vlm`

`strafer_vlm` owns semantic visual grounding.

It should own:
- prompt formatting for grounding requests
- image preprocessing
- Qwen inference
- parsing model output into typed detections
- confidence and bbox quality checks
- training and evaluation workflows for the grounding model

It should not own:
- mission sequencing
- navigation control
- long-lived task memory
- planner or executor logic

### `strafer_autonomy`

`strafer_autonomy` should own mission-level planning and orchestration.

It should own:
- skill registry
- mission plan schemas
- planner prompts and model adapters
- plan validation
- mission execution state
- retries, timeouts, and cancel flow
- status summarization for operators
- client adapters to `strafer_ros` and `strafer_vlm`

It should not own:
- motor control
- TF, depth projection, or direct sensing
- Qwen training or evaluation
- hard real-time control loops

## Why The MVP Should Still Have An Executor

The real choice is not:
- hand-built parser
- or small LLM

The real split is:
- small LLM for intent to plan
- deterministic executor for plan to execution

Even with a small LLM, the system still needs mission state because these operations are asynchronous and long-running:
- locate target
- navigate to pose
- orient to target
- wait for next command
- cancel a running mission

That means the MVP should still include an explicit mission runner, even if the planner is an LLM from day one.

## MVP: Small LLM Planner Plus Deterministic Executor

The first useful MVP should use a small LLM, but it should not give that LLM direct control over ROS actions.

Instead, the MVP should have three explicit components:
- `strafer_autonomy` planner
- `strafer_autonomy` executor
- robot and perception skills implemented by `strafer_ros` and `strafer_vlm`

### MVP goal

Accept a constrained set of natural-language requests and convert them into bounded typed plans.

Examples:
- "go to the door"
- "go to the kitchen table"
- "wait by the door"
- "face the couch"
- "stop"

### MVP architecture

```text
User text
  -> strafer_autonomy planner
  -> typed mission plan
  -> strafer_autonomy executor
  -> strafer_ros / strafer_vlm skill calls
  -> mission status updates
```

### MVP execution flow

```text
User text
  -> interpret_user_command
  -> plan_skill_sequence
  -> validate_plan
  -> select_next_skill
  -> capture_scene_observation
  -> locate_semantic_target
  -> project_detection_to_goal_pose
  -> navigate_to_pose
  -> optional orient_relative_to_target
  -> optional wait
  -> report_status
```

### MVP supported command families

| Intent | Example | Expected plan shape |
|------|---------|---------------------|
| `go_to_target` | "go to the door" | locate -> project -> navigate |
| `go_and_wait` | "wait by the door" | locate -> project -> navigate -> wait |
| `go_and_face` | "face the couch" | locate -> project -> navigate -> orient |
| `cancel` | "stop" | cancel current mission |
| `status` | "what are you doing" | summarize active mission |

### MVP planner requirements

The planner should be small and tightly bounded.

It may:
- parse user intent into a small mission schema
- choose from a fixed skill registry
- produce short ordered plans with typed arguments
- ask for clarification only if the command is unusable

It should not:
- invent unsupported skills
- emit raw ROS messages or arbitrary code
- directly control motors or navigation loops
- manage safety-critical execution without the executor

### MVP plan contract

The planner should emit structured plans, not prose.

Example:

```json
{
  "mission_type": "wait_by_target",
  "steps": [
    {"skill": "capture_scene_observation", "args": {}},
    {"skill": "locate_semantic_target", "args": {"label": "door"}},
    {"skill": "project_detection_to_goal_pose", "args": {"standoff_m": 0.7}},
    {"skill": "navigate_to_pose", "args": {"goal_source": "projected_target"}},
    {"skill": "wait", "args": {"mode": "until_next_command"}}
  ]
}
```

### MVP mission state

The executor still needs explicit mission state:

```text
IDLE -> PLANNING -> VALIDATING -> EXECUTING_STEP -> WAITING_ON_SKILL -> COMPLETE
                                          \-> FAILED
                                          \-> CANCELED
```

That state may be implemented as a mission runner or state machine inside `strafer_autonomy`, but it should remain deterministic and inspectable.

## Suggested `strafer_autonomy` Package

A new package is the cleanest home for planning and orchestration.

Suggested structure:

```text
source/strafer_autonomy/
  pyproject.toml
  README.md
  strafer_autonomy/
    __init__.py
    schemas/
      mission.py
      skills.py
    planner/
      prompt.md
      llm_client.py
      plan_parser.py
    executor/
      mission_runner.py
      state_machine.py
    clients/
      ros_client.py
      vlm_client.py
    registry/
      skills.py
```

This keeps the planning boundary clean:
- `strafer_ros` is the safe executor on the robot.
- `strafer_vlm` is the grounding tool.
- `strafer_autonomy` is the planner and mission layer.

## Post-MVP: Full Skill-Oriented Autonomy

After the MVP, the next jump is not "make the VLM smarter".

The next jump is to make the autonomy layer more capable while keeping the same skill boundaries.

Example command:
- "wait by the door for me"

Example decomposed plan:
1. `capture_scene_observation()`
2. `locate_semantic_target(label="door")`
3. `project_detection_to_goal_pose(standoff_m=0.7)`
4. `navigate_to_pose(goal_source="projected_target")`
5. `orient_relative_to_target(mode="face_away")`
6. `wait(mode="until_next_command")`

Only the grounding step is VLM-backed. The rest remain deterministic skills.

## Staged Roadmap

The current single Phase 5 document is too narrow to represent the full endstate product. The roadmap should be split into stages that align with system boundaries.

### Stage 0: Current sim-to-real base

Goal:
- Hardcoded or manually supplied goal poses execute safely on the robot.

Focus:
- RL policy or Nav2 execution
- sensing, TF, depth, and control integration

### Stage 1: Grounding workstation pipeline

Goal:
- Qwen grounding is trained, evaluated, and validated offline.

Focus:
- dataset format
- prompt contract
- bbox quality
- train and eval routines
- workstation-first iteration

This is what the current `source/strafer_vlm` package mostly supports today.

### Stage 2: ROS grounding skill integration

Goal:
- Turn the offline grounding model into a robot-usable `locate_semantic_target` skill.

Focus:
- sensor capture from `strafer_ros`
- image/depth handoff to `strafer_vlm`
- 2D bbox to 3D goal conversion
- publish and debug goal candidates

### Stage 3: Autonomy-layer MVP

Goal:
- Support a constrained set of natural-language commands using a small LLM planner and a deterministic executor.

Focus:
- create `strafer_autonomy`
- define mission and skill schemas
- bounded planner prompt and structured outputs
- deterministic mission runner
- command, cancel, and status flows

### Stage 4: Deployment architecture

Goal:
- Make autonomy and VLM skills available both locally and remotely.

Focus:
- localhost development path
- remote demo deployment path
- service API contracts
- auth, logs, and cost control
- whether `strafer_autonomy` and `strafer_vlm` are co-located or split

This is where EC2, SageMaker, and Databricks comparisons belong.

### Stage 5: Robust orchestration

Goal:
- Expand the autonomy layer beyond the MVP without changing the core skill boundaries.

Focus:
- clarification loops
- plan repair
- richer world state
- memory across missions
- better operator explanations
- more advanced compound task handling

### Stage 6: Productized operator interfaces

Goal:
- Provide operator-facing interfaces and multi-mission visibility.

Focus:
- web or mobile command UI
- mission timelines
- audit logs
- remote monitoring
- fleet-ready patterns if needed later

## Document Implications

`PHASE_5_VLM_INTEGRATION.md` should eventually be replaced by a set of narrower stage documents.

Recommended future split:
- one document for grounding model development and evaluation
- one document for ROS grounding skill integration
- one document for `strafer_autonomy` MVP design
- one document for deployment architecture
- one document for advanced orchestration design

For now, keep `PHASE_5_VLM_INTEGRATION.md` as the grounding-specific implementation deep dive, but treat this roadmap as the higher-level system plan.

## Immediate Next Work

1. Create `source/strafer_autonomy` with mission and skill schemas.
2. Define the first callable skill contracts between `strafer_autonomy`, `strafer_ros`, and `strafer_vlm`.
3. Decide whether the first `strafer_autonomy` to `strafer_vlm` path is in-process, sidecar, or remote.
4. Draft the bounded planner prompt and JSON output schema for the MVP.
5. Split the old Phase 5 document once the Stage 2 and Stage 3 interfaces are settled.
