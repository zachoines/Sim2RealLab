# VLM & Planner Enhancement Tasks

This document tracks two planned features that address missing capabilities in
the autonomy stack.  Both are motivated by the same operational reality: **most
user commands will reference targets that are not in the robot's current field
of view.**

---

## Context

The current `go_to_target` and `wait_by_target` plan templates assume:

```
capture_scene_observation  →  locate_semantic_target  →  project  →  navigate
```

If `locate_semantic_target` fails (target not in frame), the mission aborts
after one retry.  The robot has no way to search for a target and no way to
describe what it *can* see — which means it cannot give useful feedback or
attempt recovery.

These two features close that gap:

| Feature | Package | Purpose |
|---|---|---|
| **A. `scan_for_target` skill** | `strafer_autonomy` + `strafer_ros` | Rotate-and-ground loop until target is found |
| **B. VLM scene description endpoint** | `strafer_vlm` + `strafer_autonomy` | Return a text summary of what the camera sees |

They are independent and can be built in either order, but **Feature A is the
higher-priority unblock** for real-world usability.

---

## Feature A: `scan_for_target` Skill

### Problem

The robot captures a single frame and asks the VLM "find the door."  If the
door is behind the robot, the mission fails immediately.

### Solution

A Jetson-side composite skill that rotates the robot in discrete heading
increments, calling `capture_scene_observation` + `locate_semantic_target` at
each step.  Returns success when the target is found or failure after a full
rotation.

### Behavior

```
scan_for_target(label="door", max_scan_steps=6, scan_arc_deg=360)

  heading  0°  → capture → POST /ground "door" → not found
  heading 60°  → capture → POST /ground "door" → not found
  heading 120° → capture → POST /ground "door" → FOUND ✓
      └── store observation + grounding result at this heading
      └── return success
```

- Default `max_scan_steps=6` with `scan_arc_deg=360` → 60° increments
- Mecanum strafe is not needed; pure yaw rotation in-place is sufficient
- Each step calls the existing `HttpGroundingClient.locate_semantic_target()`
- On success, sets `runtime.latest_observation` and `runtime.latest_grounding`
  so that `project_detection_to_goal_pose` works on the correct frame
- On full-rotation failure, returns `SkillResult(status="failed",
  error_code="target_not_found_after_scan")`
- Respects `runtime.cancel_event` between rotation steps

### Tasks

#### A1. ROS client: add `rotate_in_place` method
- **Package:** `strafer_ros` (Jetson)
- **File:** `strafer_autonomy/clients/ros_client.py` — add to `RosClient` protocol
- Add `rotate_in_place(*, step_id, yaw_delta_rad, tolerance_rad, timeout_s) -> SkillResult`
- Jetson implementation publishes a `geometry_msgs/Twist` with angular z or
  sends a Nav2 spin action
- The existing MPPI controller already has `max_rotational_vel: 1.0 rad/s` and
  `rotational_acc_lim: 3.2 rad/s²` configured in `nav2_params.yaml`
- Simplest path: use the Nav2 `Spin` behavior plugin (already in the Nav2 stack)

#### A2. Schema: add scan-related args
- **Package:** `strafer_autonomy`
- No new dataclass needed — `scan_for_target` is a composite skill that uses
  existing `SceneObservation` + `GroundingResult` types
- Skill args passed through `SkillCall.args`:
  - `label: str` — target to search for
  - `max_scan_steps: int` (default 6)
  - `scan_arc_deg: float` (default 360)
  - `prompt: str | None` — optional override for the VLM prompt
  - `max_image_side: int` (default 1024)

#### A3. Executor: implement `_scan_for_target` handler
- **Package:** `strafer_autonomy`
- **File:** `executor/mission_runner.py`
- Add `"scan_for_target"` to `DEFAULT_AVAILABLE_SKILLS`
- Add `_scan_for_target(runtime, step) -> SkillResult` method:
  1. Extract `label`, `max_scan_steps`, `scan_arc_deg` from `step.args`
  2. Compute `step_angle_rad = scan_arc_deg * π / 180 / max_scan_steps`
  3. Loop `max_scan_steps` times:
     a. `capture_scene_observation()` → store in runtime
     b. `locate_semantic_target()` → if found, store and return success
     c. Check `cancel_event`
     d. `rotate_in_place(yaw_delta_rad=step_angle_rad)`
  4. If loop exhausts, return failed with `target_not_found_after_scan`
- Add dispatch case in `_execute_step()`

#### A4. Planner: update plan templates
- **Package:** `strafer_autonomy`
- **File:** `planner/plan_compiler.py`
- Replace `capture_scene_observation` + `locate_semantic_target` with
  `scan_for_target` in `go_to_target` and `wait_by_target` templates:

```python
# go_to_target (updated)
scan_for_target(label=..., max_scan_steps=6, scan_arc_deg=360)   # was 2 steps
project_detection_to_goal_pose(standoff_m=0.7)
navigate_to_pose(goal_source="projected_target", execution_backend="nav2")

# wait_by_target (updated)
scan_for_target(label=..., max_scan_steps=6, scan_arc_deg=360)
project_detection_to_goal_pose(standoff_m=0.7)
navigate_to_pose(goal_source="projected_target", execution_backend="nav2")
wait(mode="until_next_command")
```

- Update `SYSTEM_PROMPT` available skills if the prompt lists them explicitly

#### A5. Planner: update prompt_builder
- **Package:** `strafer_autonomy`
- **File:** `planner/prompt_builder.py`
- No SYSTEM_PROMPT change needed — the prompt already says `Available skills:`
  is appended from `request.available_skills`, which comes from
  `DEFAULT_AVAILABLE_SKILLS` at runtime

#### A6. Tests: scan_for_target
- **Package:** `strafer_autonomy`
- Update `test_plan_compiler.py`:
  - `go_to_target` now has 3 steps (was 4), first skill is `scan_for_target`
  - `wait_by_target` now has 4 steps (was 5)
  - Validation tests still pass against `DEFAULT_AVAILABLE_SKILLS`
- New `test_scan_for_target.py` (executor-level):
  - Mock `ros_client.rotate_in_place()` and `grounding_client.locate_semantic_target()`
  - Test: target found on first heading (no rotation needed)
  - Test: target found on third heading
  - Test: target not found after full rotation → failure
  - Test: cancel during scan → canceled result
  - Test: grounding service unavailable → propagated error

### Definition of Done
- [ ] A1: `rotate_in_place` in `RosClient` protocol and Jetson implementation
- [ ] A2: Skill args documented
- [ ] A3: `_scan_for_target` handler passes executor tests with mocked ROS+VLM
- [ ] A4: Updated plan templates compile and pass validation
- [ ] A6: All new and updated tests pass
- [ ] End-to-end: robot successfully finds a target behind it via scan

---

## Feature B: VLM Scene Description Endpoint

### Problem

When a mission fails (target not found, even after scanning), the robot
returns a generic error.  It cannot tell the user *what it sees* — e.g.
"I see a hallway and a bookshelf, but no kitchen."  This limits operator
situational awareness and blocks future re-planning loops.

### Solution

Add a `POST /describe` endpoint to the existing `strafer_vlm` service that
accepts an image and returns a short text description of the scene.  Add a
corresponding `describe_scene` skill and client method on the autonomy side.

### VLM Service Changes

#### B1. Payloads: `DescribeRequest` / `DescribeResponse`
- **Package:** `strafer_vlm`
- **File:** `strafer_vlm/service/payloads.py`
- New Pydantic models:

```python
class DescribeRequest(BaseModel):
    request_id: str
    image_jpeg_b64: str
    prompt: str = "Describe the objects and layout visible in this image in one or two sentences."
    max_image_side: int = 1024

class DescribeResponse(BaseModel):
    request_id: str
    description: str
    latency_s: float = 0.0
```

#### B2. Endpoint: `POST /describe`
- **Package:** `strafer_vlm`
- **File:** `strafer_vlm/service/app.py`
- New endpoint using the same `Qwen2.5-VL` model already loaded in `_state`:
  - Decode JPEG, apply size guard
  - Run `run_grounding_generation()` with a description prompt instead of a
    grounding prompt
  - Return raw text output (no bbox parsing)
- The VLM model is already multimodal and capable of scene description — this
  is just a different prompt, not a different model
- Inference timeout applies the same way

#### B3. Inference: description prompt template
- **Package:** `strafer_vlm`
- **File:** `strafer_vlm/inference/parsing.py` or new `description.py`
- System prompt for description mode:
  ```
  You are a robot vision system. Describe the scene in the image concisely.
  List the main objects, surfaces, and spatial layout visible.
  Keep your response to 1-3 sentences. Do not speculate about objects not visible.
  ```
- This can reuse `run_grounding_generation()` with a different system prompt,
  or use a thin wrapper

### Autonomy Side Changes

#### B4. Schema: `SceneDescription`
- **Package:** `strafer_autonomy`
- **File:** `strafer_autonomy/schemas/observation.py`
- New frozen dataclass:

```python
@dataclass(frozen=True)
class SceneDescription:
    request_id: str
    description: str
    stamp_sec: float = 0.0
    latency_s: float = 0.0
```

#### B5. Grounding client: add `describe_scene` method
- **Package:** `strafer_autonomy`
- **File:** `strafer_autonomy/clients/vlm_client.py`
- Add to `GroundingClient` protocol:
  ```python
  def describe_scene(self, *, request_id: str, image_rgb_u8: Any,
                     prompt: str | None = None,
                     max_image_side: int = 1024) -> SceneDescription: ...
  ```
- Implement in `HttpGroundingClient`:
  - JPEG-encode image
  - POST to `/describe`
  - Parse response into `SceneDescription`
  - Same retry logic as `locate_semantic_target`

#### B6. Executor: `describe_scene` skill handler
- **Package:** `strafer_autonomy`
- **File:** `executor/mission_runner.py`
- Add `"describe_scene"` to `DEFAULT_AVAILABLE_SKILLS`
- Add `_describe_scene(runtime, step) -> SkillResult` method:
  1. Capture observation (or use `runtime.latest_observation`)
  2. Call `grounding_client.describe_scene()`
  3. Return success with `outputs={"description": result.description}`
- Add dispatch case in `_execute_step()`

#### B7. Planner: add `describe` intent type (optional, future)
- **Not required for initial implementation**
- The `describe_scene` skill can be used within `scan_for_target` failure
  recovery or as a standalone skill invoked when reporting status
- A future `describe` intent (user says "what do you see?") would compile to:
  ```
  capture_scene_observation → describe_scene → report_status
  ```
- Defer adding this to `intent_parser.py` and `plan_compiler.py` until needed

#### B8. Integration: attach description to scan failure message
- **Package:** `strafer_autonomy`
- **Depends on:** A3 + B6
- When `scan_for_target` exhausts all headings without finding the target:
  1. Call `describe_scene` on the last captured observation
  2. Include the description in the failure message:
     ```
     "Target 'kitchen' not found after full 360° scan.
      Last observation: I see a hallway with a closed door and a bookshelf."
     ```
- This gives the operator actionable information without requiring a re-plan

### Tasks

#### B9. Tests
- `test_describe_endpoint.py` (VLM service):
  - Mock VLM inference, verify `/describe` returns text
  - Verify image size guard
  - Verify 503 when model not loaded
- `test_describe_scene.py` (autonomy executor):
  - Mock grounding client, verify skill handler returns description in outputs
- Update `test_planner_client.py` if client gains `describe_scene`

### Definition of Done
- [ ] B1–B2: `/describe` endpoint returns scene descriptions using existing VLM model
- [ ] B3: Description prompt produces concise, useful scene summaries
- [ ] B4–B5: `SceneDescription` schema + client method with retry logic
- [ ] B6: `describe_scene` executor skill handler
- [ ] B8: Scan failure messages include scene description
- [ ] B9: All tests pass with mocked VLM

---

## Sequencing

```
Phase 1 (unblocks real-world use):
  A1 → A2 → A3 → A4 → A6           scan_for_target skill

Phase 2 (operator awareness):
  B1 → B2 → B3 → B9 (VLM side)     /describe endpoint
  B4 → B5 → B6 → B9 (autonomy)     describe_scene skill

Phase 3 (integration):
  B8                                 attach description to scan failures
  B7 (optional)                      "what do you see?" intent
```

Phase 1 and Phase 2 (VLM side) can be worked in parallel since they touch
different packages.

---

## Not In Scope

- Spatial memory / semantic map — requires persistent map annotations
- Multi-room search loops — requires topological navigation
- Re-planning based on scene description — requires Planner to accept
  perception context (future architecture change)
- `orient_relative_to_target` — remains deferred from MVP
