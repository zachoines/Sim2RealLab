# Strafer Autonomy — Next Phase

This document captures the next round of improvements to the autonomy system.
It is written from the working baseline established in `phase_13`:

- Jetson executor calling DGX Spark planner (Qwen3-4B, port 8200) and VLM
  (Qwen2.5-VL-3B, port 8100) over LAN HTTP
- End-to-end mission execution tested on real hardware: user command through
  CLI, LLM planning, VLM grounding, goal projection, Nav2 navigation
- 10 implemented skills, deterministic plan compiler, mission runner with
  cancel/timeout/retry

The document is organized into four areas:
0. Safety
1. Optimizations
2. New features
3. Deployment (Databricks integration path)

---

## 0. Safety

### 0.1 Navigation progress validation

**Problem.** The robot currently trusts the projected goal pose and hands it to
Nav2 with a 90-second timeout. If the VLM returns a spurious detection (e.g.,
misidentifies a wall texture as the target), the robot drives to an incorrect
location, realizes nothing is wrong, and reports success.

**Proposal: arrival verification.**  After `navigate_to_pose` completes, add an
optional `verify_arrival` step that:

1. Captures a new RGB frame at the arrival pose.
2. Calls the VLM with the same target label.
3. If the target is found within a configurable bbox-area threshold (i.e., the
   object should appear large and centered because the robot is now close),
   the step succeeds.
4. If the target is not found or the detection is small/off-center, the step
   fails with `arrival_verification_failed`, which the executor can use to
   trigger a retry or re-scan.

**Implementation sketch.**

```python
# New skill: verify_arrival
# Inserted after navigate_to_pose in the plan compiler output
SkillCall(
    step_id="step_04",
    skill="verify_arrival",
    args={
        "target_label": intent.target_label,
        "min_bbox_area_fraction": 0.05,   # target should occupy >5% of frame
        "max_center_offset_norm": 0.3,    # bbox center within 30% of image center
    },
    timeout_s=10.0,
    retry_limit=1,
)
```

**Where this runs.** The VLM call is remote (DGX or Databricks). The
verification logic (bbox area check, center offset check) is local in the
mission runner — keeping the safety decision on the robot.

### 0.2 Costmap collision pre-check

**Problem.** Nav2 sometimes rejects goals immediately if the projected pose
lands in an occupied or unknown costmap cell. The executor sees `goal_rejected`
but has no recovery path.

**Proposal.** Before sending a goal to Nav2, query the costmap at the
projected pose coordinates. If the cell is occupied or unknown:
- Offset the goal along the standoff vector (move it closer to the robot)
- Or re-project with a larger standoff distance
- Log a warning so the operator knows the original projection was adjusted

This can be implemented as a pre-check inside `navigate_to_pose` in the
`JetsonRosClient`, or as a separate `validate_goal_pose` skill step.

### 0.3 VLM confidence thresholding

**Problem.** The VLM returns a confidence score with each grounding result, but
the executor currently ignores it. A low-confidence detection can still drive
navigation.

**Proposal.** Add a configurable `min_grounding_confidence` threshold (default
0.5) to `MissionRunnerConfig`. If the VLM returns a detection below this
threshold, treat it as `not_found` and continue the scan rotation. Log the
rejected detection for debugging.

### 0.4 Odometry drift watchdog

**Problem.** If RTAB-Map loses tracking (e.g., featureless corridor), the
`map → odom` TF becomes stale and Nav2 navigates using drifting wheel odometry
alone.

**Proposal.** Monitor the `map → odom` TF age during navigation. If it exceeds
a threshold (e.g., 5 seconds without update), pause navigation and attempt
recovery:
- Rotate slowly to give RTAB-Map new visual features
- If TF age recovers, resume navigation
- If not, fail the step with `slam_tracking_lost`

### 0.5 Emergency stop on excessive velocity

The driver watchdog already stops motors if no `cmd_vel` arrives for 500ms.
Add a complementary check: if the measured wheel velocity exceeds
`MAX_LINEAR_VEL * 1.2` (a 20% margin), immediately zero the motors and log
a `velocity_limit_exceeded` diagnostic. This guards against runaway
controller output from Nav2 or future RL backends.

---

## 1. Optimizations

### 1.1 Direct planner-to-VLM orchestration (agentic planning)

**Current flow.**

```text
Jetson executor
  → POST DGX:8200/plan    (planner returns plan)
  → execute scan_for_target locally
    → capture frame locally
    → POST DGX:8100/ground (VLM returns bbox)
    → rotate if not found, repeat
  → project goal locally
  → navigate locally
```

The Jetson mediates every call. For each `scan_for_target` rotation step, a
full round-trip image upload goes from Jetson → DGX VLM → Jetson.

**Proposed: agentic orchestration on the compute node.**

When planner and VLM are co-located (same DGX, same Databricks workspace),
the planner can call the VLM directly during plan generation to pre-validate
targets:

```text
Jetson executor
  → POST DGX:8200/plan_with_grounding
    body: { raw_command, image_jpeg_b64 }
    DGX internally:
      planner → intent
      planner calls VLM with the image → grounding result
      planner returns plan + pre-grounding result
  → Jetson skips scan_for_target (target already grounded)
  → project goal locally
  → navigate locally
```

**Benefits:**
- Eliminates one LAN image upload round-trip (~2-3s saved)
- Planner has grounding context when generating the plan (can adjust if target
  not visible)
- Natural fit for Databricks model serving where both models are endpoints in
  the same workspace

**Implementation.**

New planner endpoint: `POST /plan_with_grounding`

```python
class PlanWithGroundingRequest(BaseModel):
    request_id: str
    raw_command: str
    image_jpeg_b64: str | None = None   # optional: current camera frame
    robot_pose: dict | None = None       # optional: current robot state

class PlanWithGroundingResponse(BaseModel):
    plan: MissionPlan
    pre_grounding: GroundingResult | None = None  # populated if image provided
```

The planner service internally calls the VLM (localhost or same-host endpoint)
if an image is provided and the intent requires grounding. If the VLM finds
the target, the returned plan can skip `scan_for_target` and go straight to
`project_detection_to_goal_pose` with the pre-grounded bbox.

**Fallback.** If no image is provided, or the VLM doesn't find the target in
the provided image, the planner returns a standard plan with `scan_for_target`
and the Jetson handles the rotation loop as before.

### 1.2 Image compression optimization

Currently every VLM call sends a full JPEG base64 image over LAN HTTP. For
640x360 frames this is ~100-200KB per request, manageable on LAN but
expensive at scale or over WAN.

**Proposal.** Add configurable image quality and max-side parameters to the
grounding client:

```python
@dataclass(frozen=True)
class HttpGroundingClientConfig:
    ...
    jpeg_quality: int = 85          # lower for faster transfer
    max_image_side: int = 640       # resize before encoding
```

For Databricks deployment with cloud endpoints, consider S3 pre-signed URL
upload instead of inline base64 for images larger than 1MB.

### 1.3 Parallel health checks at startup

The executor currently probes VLM health sequentially. When both services are
on the same host, probe planner and VLM health in parallel to halve startup
time.

---

## 2. New Features

### 2.1 Rotate skills

**Current state.** `rotate_in_place` exists in `JetsonRosClient` but only as
an internal helper for `scan_for_target`. `orient_relative_to_target` raises
`NotImplementedError`.

**Proposal: two new user-facing skills.**

| Skill | Args | Description |
|-------|------|-------------|
| `rotate_by_degrees` | `degrees: float`, `speed_scale: float` | Rotate the robot by a relative angle. Positive = CCW. |
| `orient_to_direction` | `direction: str` ("north", "south", "east", "west", "toward_target", "away_from_target") | Rotate to an absolute or target-relative heading. |

These would be exposed as new intent types in the planner and compiler:

```python
# New intent type
"rotate" → [SkillCall(skill="rotate_by_degrees", args={"degrees": 180})]

# New intent type
"face_direction" → [SkillCall(skill="orient_to_direction", args={"direction": "toward_target"})]
```

**Implementation.** `rotate_by_degrees` wraps the existing `rotate_in_place`
method. `orient_to_direction` requires TF lookup for absolute headings and
the target pose for relative headings.

### 2.2 Multi-target mission chaining

**Current state.** The planner supports one `target_label` per intent. Composite
commands like "go to the cup, then go to the door" are not supported.

**Proposal: `go_to_targets` intent type.**

The LLM outputs a list of targets:

```json
{
  "intent_type": "go_to_targets",
  "targets": [
    {"label": "cup", "standoff_m": 0.7},
    {"label": "door", "standoff_m": 0.7}
  ]
}
```

The compiler chains scan → project → navigate blocks for each target:

```python
def _compile_go_to_targets(intent: MissionIntent) -> list[SkillCall]:
    steps = []
    for i, target in enumerate(intent.targets):
        base = i * 3 + 1
        steps.extend([
            SkillCall(step_id=f"step_{base:02d}", skill="scan_for_target",
                      args={"label": target["label"]}, ...),
            SkillCall(step_id=f"step_{base+1:02d}", skill="project_detection_to_goal_pose",
                      args={"standoff_m": target.get("standoff_m", 0.7)}, ...),
            SkillCall(step_id=f"step_{base+2:02d}", skill="navigate_to_pose",
                      args={"goal_source": "projected_target"}, ...),
        ])
    return steps
```

**Schema change.** `MissionIntent` gains an optional `targets: list[dict]`
field. The `intent_parser` learns to extract it from the LLM output.
`target_label` remains for single-target intents for backward compatibility.

### 2.3 Scene description as operator feedback

**Current state.** `describe_scene` is implemented as a skill and the VLM
`POST /describe` endpoint exists. It is currently only used as a fallback when
`scan_for_target` exhausts all rotations without finding the target.

**Proposal.** Expose `describe_scene` as a user-facing intent:

```
"describe" → [SkillCall(skill="describe_scene", args={"prompt": "What do you see?"})]
```

The description text is returned in the mission result message. This gives
operators situational awareness without requiring a video feed.

### 2.4 Patrol / waypoint sequence

A natural extension of multi-target chaining: define a patrol route as an
ordered list of named locations or relative offsets. The robot visits each
in sequence and optionally loops.

```json
{
  "intent_type": "patrol",
  "targets": ["door", "window", "desk"],
  "loop": true
}
```

This builds on `go_to_targets` but adds loop control in the mission runner.

### 2.5 Plan repair on failure

**Current state.** If a skill fails, the mission fails. No automatic recovery.

**Proposal.** Add a `repair_plan` capability:
- On `scan_for_target` failure: ask the VLM to describe the scene, return
  the description to the planner, and request a revised plan
- On `navigate_to_pose` failure: re-project with a larger standoff, or
  try a different approach angle
- On `arrival_verification_failed`: re-scan from the current position

This requires the planner to accept a `PlanRepairRequest` with the current
mission state, the failure code, and optionally a scene description.

---

## 3. Deployment — Databricks Integration

### 3.1 Current architecture vs. Databricks target

**Current (DGX Spark on LAN):**

```text
Jetson (robot)                    DGX Spark (192.168.50.196)
  strafer_autonomy.executor         strafer_vlm service (:8100)
  strafer_ros                       planner service (:8200)
       ←——— LAN HTTP ———→
```

**Target (Databricks):**

```text
Jetson (robot)                    Databricks Workspace
  strafer_autonomy.executor         VLM serving endpoint
  strafer_ros                       Planner serving endpoint
       ←——— HTTPS ———→            (optional: agentic endpoint
                                    combining planner + VLM)
```

### 3.2 Databricks model serving fit

Both services are good candidates for Databricks custom model serving:

| Service | Payload | Latency tolerance | GPU | Fit |
|---------|---------|-------------------|-----|-----|
| Planner | ~1 KB JSON | 2-5s | Optional (Qwen3-4B runs on CPU with quantization) | Good — lightweight text-in/JSON-out |
| VLM | ~200 KB (image + prompt) | 3-5s | Required (Qwen2.5-VL-3B) | Good — within 16 MB payload limit |

Databricks constraints to design around:
- **Payload limit:** 16 MB per request (fine for 640x360 JPEG, would need S3
  for high-res images)
- **Execution timeout:** 297 seconds (fine for both services)
- **Scale-to-zero:** endpoints can scale to zero but cold start adds 30-60s
  latency. Not acceptable for real-time robot operation. Keep minimum
  instances = 1 for active deployments.
- **GPU instances:** Databricks serves GPU models via serving endpoints with
  GPU-enabled instance types

### 3.3 Recommended Databricks architecture

**Two serving endpoints:**

1. `strafer-planner` — custom Python model serving endpoint
   - Input: `PlannerRequest` (JSON)
   - Output: `MissionPlan` (JSON)
   - Model: Qwen3-4B loaded via transformers
   - Instance: CPU or small GPU, minimum 1 instance

2. `strafer-vlm` — custom Python model serving endpoint
   - Input: `GroundingRequest` (JSON with base64 image)
   - Output: `GroundingResult` (JSON)
   - Model: Qwen2.5-VL-3B-Instruct loaded via transformers
   - Instance: GPU (T4 or A10G minimum), minimum 1 instance

**Optional: agentic endpoint (planner + VLM combined):**

3. `strafer-agent` — single endpoint that wraps both models
   - Accepts `PlanWithGroundingRequest` (command + optional image)
   - Internally calls planner model, then VLM model if grounding is needed
   - Returns `PlanWithGroundingResponse` (plan + pre-grounding)
   - Reduces Jetson→cloud round-trips from 2+ to 1
   - Natural fit for Databricks since both models are in the same serving
     infrastructure

### 3.4 Design changes for Databricks readiness

**3.4.1 Abstract the transport layer.**

Currently `HttpPlannerClient` and `HttpGroundingClient` use raw `requests`
sessions. For Databricks, the transport changes to the Databricks serving
SDK or HTTPS with workspace auth tokens.

**Proposal: transport adapter pattern.**

```python
class PlannerTransport(Protocol):
    def plan(self, request: PlannerRequest) -> MissionPlan: ...

class HttpPlannerTransport:
    """LAN HTTP transport (current DGX Spark)."""
    ...

class DatabricksServing PlannerTransport:
    """Databricks model serving endpoint transport."""
    def __init__(self, endpoint_name: str, workspace_url: str, token: str): ...
    def plan(self, request: PlannerRequest) -> MissionPlan:
        # Uses databricks-sdk or HTTPS POST to serving endpoint
        ...
```

The executor's `build_command_server()` factory selects the transport based
on configuration (env vars or a config file):

```bash
# Current LAN mode
PLANNER_URL=http://192.168.50.196:8200

# Databricks mode
PLANNER_BACKEND=databricks
DATABRICKS_HOST=https://<workspace>.databricks.net
DATABRICKS_TOKEN=dapi...
PLANNER_ENDPOINT=strafer-planner
VLM_ENDPOINT=strafer-vlm
```

**3.4.2 Package models for Databricks serving.**

Each model needs a `MLflow` Python model wrapper:

```python
class StraferPlannerModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["model"])
        self.model = AutoModelForCausalLM.from_pretrained(context.artifacts["model"])

    def predict(self, context, model_input):
        # model_input is a pandas DataFrame with columns matching PlannerRequest
        ...
        return plan_json
```

Register with MLflow, then create a serving endpoint:

```python
import mlflow
mlflow.pyfunc.log_model(
    artifact_path="planner",
    python_model=StraferPlannerModel(),
    artifacts={"model": "Qwen/Qwen3-4B"},
)
```

**3.4.3 Unified health and readiness.**

Databricks serving endpoints don't expose custom `/health` routes. Instead,
the serving endpoint has a built-in readiness state. Update the executor's
health check logic to support both:

- LAN mode: `GET /health` (current behavior)
- Databricks mode: query endpoint status via Databricks SDK
  (`client.serving_endpoints.get("strafer-vlm").state.ready`)

### 3.5 Migration path

| Phase | Planner | VLM | Transport |
|-------|---------|-----|-----------|
| Current | DGX Spark LAN | DGX Spark LAN | HTTP (requests) |
| Phase A | Databricks endpoint | DGX Spark LAN | Mixed (test planner on Databricks) |
| Phase B | Databricks endpoint | Databricks endpoint | HTTPS (Databricks SDK) |
| Phase C | Databricks agentic endpoint | (combined) | HTTPS (single endpoint) |

Each phase is independently testable. The transport adapter pattern means the
Jetson executor code doesn't change between phases — only the configuration.

### 3.6 Cost and latency considerations

- **Keep minimum instances = 1** for both endpoints during active development
  and demos. Scale-to-zero cold starts (30-60s) are unacceptable for a robot
  waiting to execute a command.
- **Image size matters.** 640x360 JPEG at quality 85 is ~100 KB. Well within
  the 16 MB Databricks limit. Don't send full-resolution frames.
- **LAN vs. cloud latency.** Current LAN round-trip for VLM grounding is ~2-3s.
  Cloud adds ~100-200ms network overhead. Acceptable for the current skill
  execution model (robot is stationary during grounding).
- **Batch optimization.** For patrol/multi-target missions, consider batching
  multiple grounding requests in a single endpoint call to amortize network
  overhead.

---

## Summary of Proposed Changes

| Area | Item | Effort | Priority |
|------|------|--------|----------|
| Safety | Arrival verification skill | Medium | High |
| Safety | VLM confidence thresholding | Small | High |
| Safety | Costmap collision pre-check | Small | Medium |
| Safety | Odometry drift watchdog | Medium | Medium |
| Safety | Velocity limit watchdog | Small | Low |
| Optimization | Agentic planner+VLM endpoint | Medium | Medium |
| Optimization | Image compression tuning | Small | Low |
| Feature | `rotate_by_degrees` skill | Small | High |
| Feature | `orient_to_direction` skill | Medium | High |
| Feature | Multi-target chaining | Medium | High |
| Feature | Scene description as user intent | Small | Medium |
| Feature | Patrol / waypoint sequence | Medium | Low |
| Feature | Plan repair on failure | Large | Low |
| Deployment | Transport adapter pattern | Medium | High |
| Deployment | MLflow model packaging | Medium | Medium |
| Deployment | Databricks serving endpoints | Medium | Medium |
| Deployment | Agentic combined endpoint | Medium | Low |
