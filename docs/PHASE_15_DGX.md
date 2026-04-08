# Phase 15 — DGX Spark Workstream

**Branch:** `phase_15`
**Platform:** DGX Spark (Ubuntu ARM64, CUDA 13.0, Python 3.12 venv at `.venv_vlm`)
**Hardware:** Grace CPU (ARM64) + Blackwell GB10 GPU (sm_121)
**Services:** VLM on port 8100, Planner on port 8200
**Design doc:** `docs/STRAFER_AUTONOMY_NEXT.md` (read-only reference)
**Integration context:** `docs/INTEGRATION_DGX_SPARK.md` (full API surface)

This file defines the tasks assigned to the DGX agent. All work is scoped
to files that ONLY this agent touches — no overlap with the Jetson or
Windows workstreams.

---

## Platform context

The DGX Spark hosts two stateless FastAPI services. For full platform details,
NVRTC fix, env vars, endpoint schemas, and setup instructions, see
`docs/INTEGRATION_DGX_SPARK.md`.

### Key existing APIs you will extend

**VLM service** (`strafer_vlm/service/app.py`, port 8100):
- `GET /health` → `HealthResponse(status, model_loaded, model_name)`
- `POST /ground` → `GroundResponse(request_id, found, bbox_2d, label, confidence, raw_output, latency_s)`
- `POST /describe` → `DescribeResponse(request_id, description, latency_s)`
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Inference: `run_grounding_generation()` in `strafer_vlm/inference/qwen_runtime.py`
- Parsing: `strafer_vlm/inference/parsing.py` — JSON extraction, bbox coercion, `<ref>`/`<box>` tag parsing
- All payloads in `strafer_vlm/service/payloads.py` (Pydantic BaseModel)
- Bbox coordinates: Qwen outputs [0, 1000] normalized, NOT pixel coordinates

**Planner service** (`strafer_autonomy/planner/app.py`, port 8200):
- `GET /health` → `PlannerHealthResponse(status, model_loaded, model_name)`
- `POST /plan` → `PlanResponse(mission_id, mission_type, raw_command, steps, created_at)`
- Model: `Qwen/Qwen3-4B`
- Pipeline: `build_messages()` → `LLMRuntime.generate()` → `parse_intent()` → `compile_plan()`
- Intent parser: `intent_parser.py` — `_VALID_INTENTS = frozenset({"go_to_target", "wait_by_target", "cancel", "status"})`
- Plan compiler: `plan_compiler.py` — `_COMPILERS` dict maps intent type → compiler function
- Current compilers: `go_to_target` → 3 steps, `wait_by_target` → 4 steps, `cancel` → 1 step, `status` → 1 step

**Client protocols** (these define what the Jetson executor calls):
- `PlannerClient` Protocol: `plan_mission(PlannerRequest) -> MissionPlan`
- `GroundingClient` Protocol: `locate_semantic_target(GroundingRequest) -> GroundingResult`, `describe_scene(...) -> SceneDescription`
- `HttpPlannerClient` implements `PlannerClient` (LAN HTTP, retries, backoff)
- `HttpGroundingClient` implements `GroundingClient` (LAN HTTP, JPEG encode, retries)
- Wire helpers: `planner_request_to_payload()`, `mission_plan_from_payload()`, `grounding_request_to_payload()`, `grounding_result_from_payload()`
- Image encoding: `_encode_image_to_jpeg_b64(image_rgb_u8, *, quality=90)`

**Schemas** (`strafer_autonomy/schemas/mission.py`):
- `MissionIntent(intent_type, raw_command, target_label, orientation_mode, wait_mode, requires_grounding)`
- `SkillCall(skill, step_id, args, timeout_s, retry_limit)`
- `MissionPlan(mission_id, mission_type, raw_command, steps: tuple[SkillCall], created_at)`
- `PlannerRequest(request_id, raw_command, robot_state, active_mission_summary, available_skills)`

### How the DGX calls itself (for agentic endpoint)

The planner service on port 8200 can call the VLM service on port 8100
via `http://localhost:8100/ground`. Both run on the same machine. For the
`/plan_with_grounding` endpoint (Task 3), use `httpx.AsyncClient` to make
this internal call.

---

## Owned directories (do NOT edit files outside these)

```
source/strafer_vlm/                                          # VLM service, inference, training
source/strafer_autonomy/strafer_autonomy/planner/            # planner app, intent parser, plan compiler, LLM runtime
source/strafer_autonomy/strafer_autonomy/clients/planner_client.py   # HttpPlannerClient
source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py       # HttpGroundingClient
source/strafer_autonomy/strafer_autonomy/clients/databricks_planner_client.py  # NEW
source/strafer_autonomy/strafer_autonomy/clients/databricks_vlm_client.py      # NEW
source/strafer_autonomy/tests/test_planner_*.py
source/strafer_autonomy/tests/test_intent_parser.py
source/strafer_autonomy/tests/test_plan_compiler.py
```

**Shared file protocol:** The Jetson agent owns `mission_runner.py` and the
`_execute_step` dispatch. If you add a new intent type or skill, you provide
only the compiler function and parser change — the Jetson agent wires the
executor side.

---

## Tasks (ordered by priority, then dependency)

### Task 1: `POST /detect_objects` VLM endpoint (Section 1.12)

**Priority:** High | **Effort:** Medium

**Files to create/modify:**

```
source/strafer_vlm/strafer_vlm/service/payloads.py   # add DetectObjectsRequest, DetectedObject, DetectObjectsResponse
source/strafer_vlm/strafer_vlm/service/app.py         # add /detect_objects route
source/strafer_vlm/tests/test_detect_objects.py        # NEW
```

**What to implement:**

1. Add Pydantic models to `payloads.py`:
   ```python
   class DetectObjectsRequest(BaseModel):
       request_id: str
       image_jpeg_b64: str
       max_image_side: int = 1024
       max_objects: int = 20
       min_confidence: float = 0.3

   class DetectedObject(BaseModel):
       label: str
       bbox_2d: list[int]        # [x1, y1, x2, y2] pixels
       confidence: float

   class DetectObjectsResponse(BaseModel):
       request_id: str
       objects: list[DetectedObject]
       latency_s: float = 0.0
   ```

2. Add route handler in `app.py`:
   - Prompt Qwen2.5-VL with: `"List all visible objects with their bounding boxes."`
   - Parse `<ref>label</ref><box>(x1,y1),(x2,y2)</box>` tags from VLM output
   - Convert 0-1000 scaled coordinates to pixel coordinates
   - Filter by `min_confidence` (if the VLM output includes confidence, otherwise default to 1.0)
   - Return `DetectObjectsResponse`

3. Add `detect_objects` method to `GroundingClient` protocol in `vlm_client.py`:
   ```python
   def detect_objects(self, *, request_id: str, image_rgb_u8: Any,
                      max_image_side: int = 1024, max_objects: int = 20,
                      min_confidence: float = 0.3) -> DetectObjectsResponse: ...
   ```
   Make it optional (not in the Protocol, but available on `HttpGroundingClient`)
   so existing code doesn't break.

4. Add wire helpers in `vlm_client.py`:
   - `detect_objects_request_to_payload(...)` — same pattern as existing grounding helpers
   - `detect_objects_response_from_payload(...)`

5. Write tests that mock the VLM output and verify `<ref>`/`<box>` parsing.

---

### Task 2: New intent types — rotate, go_to_targets, describe, query, patrol (Section 3.1–3.5)

**Priority:** High | **Effort:** Medium

**Coordination with Jetson agent:** This task creates the PLANNER side
(intent parser + plan compiler). The EXECUTOR side (skill handlers in
`mission_runner.py`) is owned by the Jetson agent. The handshake is:
- You add `"rotate"`, `"go_to_targets"`, `"describe"`, `"query"`, `"patrol"` to `_VALID_INTENTS` and `_COMPILERS`
- The compiled plans emit existing skills (e.g., `scan_for_target`, `navigate_to_pose`) plus new ones: `rotate_by_degrees`, `orient_to_direction`, `query_environment`
- The Jetson agent adds handlers for `rotate_by_degrees`, `orient_to_direction`, and `query_environment` in `_execute_step`
- Skills like `describe_scene` already have handlers — the compiler just needs to emit them

**Files to modify:**

```
source/strafer_autonomy/strafer_autonomy/planner/intent_parser.py
source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py
source/strafer_autonomy/strafer_autonomy/planner/payloads.py       # if needed
source/strafer_autonomy/strafer_autonomy/schemas/mission.py        # add targets field
source/strafer_autonomy/tests/test_intent_parser.py
source/strafer_autonomy/tests/test_plan_compiler.py
```

**What to implement:**

1. `schemas/mission.py` — add `targets: list[dict[str, Any]] | None = None`
   to `MissionIntent`

2. `intent_parser.py` — add to `_VALID_INTENTS`:
   ```python
   frozenset({"go_to_target", "wait_by_target", "cancel", "status",
              "rotate", "go_to_targets", "describe", "query", "patrol"})
   ```
   Add validation for `go_to_targets` and `patrol` (require non-empty `targets` list).

3. `plan_compiler.py` — add compiler functions:
   - `_compile_rotate(intent)` — numeric degrees → `rotate_by_degrees`, cardinal → `orient_to_direction`
   - `_compile_go_to_targets(intent)` — chain scan→project→navigate per target
   - `_compile_describe(intent)` — single `describe_scene` step
   - `_compile_query(intent)` — single `query_environment` step
   - `_compile_patrol(intent)` — delegate to `_compile_go_to_target` per waypoint
   - Register all in `_COMPILERS` dict

4. Update LLM system prompt in `build_messages` (or wherever prompts are
   assembled) to teach the model about the new intent types.

5. Update tests: add parametrized cases for each new intent type in
   `test_intent_parser.py` and `test_plan_compiler.py`.

---

### Task 3: Agentic planner+VLM endpoint (Section 2.1)

**Priority:** Medium | **Effort:** Medium

**Files to modify:**

```
source/strafer_autonomy/strafer_autonomy/planner/payloads.py  # add PlanWithGroundingRequest/Response
source/strafer_autonomy/strafer_autonomy/planner/app.py       # add /plan_with_grounding route
source/strafer_autonomy/tests/test_planner_endpoints.py
```

**What to implement:**

1. Add `GroundResultPayload`, `PlanWithGroundingRequest(PlanRequest)`,
   `PlanWithGroundingResponse(PlanResponse)` to `payloads.py`

2. Add `/plan_with_grounding` route to `app.py`:
   - Reuse existing pipeline: `build_messages` → `LLMRuntime.generate` → `parse_intent` → `compile_plan`
   - If `image_jpeg_b64` is provided and intent requires grounding, call
     `POST localhost:8100/ground` via `httpx.AsyncClient`
   - Return plan + optional pre-grounding result

3. Add `pip install httpx` to planner dependencies if not already present.

---

### Task 4: Image compression — expose jpeg_quality config (Section 2.2)

**Priority:** Low | **Effort:** Small

**Files to modify:**

```
source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py
```

**What to implement:**

1. Add `jpeg_quality: int = 90` to `HttpGroundingClientConfig`
2. Pass `quality=self._config.jpeg_quality` to `_encode_image_to_jpeg_b64()`
   in `locate_semantic_target()` and `describe_scene()`

---

### Task 5: Databricks client implementations (Section 4.4)

**Priority:** High | **Effort:** Medium

**New files to create:**

```
source/strafer_autonomy/strafer_autonomy/clients/databricks_planner_client.py
source/strafer_autonomy/strafer_autonomy/clients/databricks_vlm_client.py
```

**What to implement:**

1. `DatabricksServingPlannerClientConfig` (frozen dataclass) +
   `DatabricksServingPlannerClient` implementing `PlannerClient` protocol:
   - `plan_mission(request) -> MissionPlan` — POST to Databricks serving endpoint
   - `health() -> dict` — query endpoint state via Databricks SDK or REST

2. `DatabricksServingGroundingClientConfig` +
   `DatabricksServingGroundingClient` implementing `GroundingClient` protocol:
   - `locate_semantic_target(request) -> GroundingResult`
   - `describe_scene(...) -> SceneDescription`
   - `health() -> dict`

3. Use `databricks-sdk` or raw `requests` with bearer token auth.

**Note:** Do NOT modify `main.py` (executor entry point) — the Jetson agent
owns that file. Instead, provide the classes and document the env-var
interface. The Jetson agent will wire the backend selection.

---

### Task 6: MLflow model packaging (Section 4.4.2)

**Priority:** Medium | **Effort:** Medium

**New files to create:**

```
source/strafer_autonomy/databricks/
    __init__.py
    planner_model.py      # StraferPlannerModel(mlflow.pyfunc.PythonModel)
    vlm_model.py           # StraferVLMModel(mlflow.pyfunc.PythonModel)
    register.py            # script to log models to MLflow
```

**What to implement:**

1. `StraferPlannerModel` — wraps `LLMRuntime` + `build_messages` +
   `parse_intent` + `compile_plan` pipeline
2. `StraferVLMModel` — wraps `VLMRuntime` for ground and describe modes
3. `register.py` — log both models with `mlflow.pyfunc.log_model()`

---

---

## Deferred tasks (not assigned to this phase)

These items from `STRAFER_AUTONOMY_NEXT.md` are explicitly deferred:

- **Parallel health checks** (Section 2.3) — Small effort but touches
  `executor/command_server.py` which is owned by the Jetson agent. Reassigned
  to Jetson workstream to avoid file ownership conflicts.
- **Agentic combined endpoint** (Section 4.3 item 3) — Medium effort, Low
  priority. Natural follow-up after Task 3 (agentic planner) is validated.
- **Databricks serving endpoints** (Section 4.5) — Infrastructure setup,
  depends on Tasks 5 and 6 being complete and tested.

---

## Build and test

```bash
# Activate venv
source .venv_vlm/bin/activate

# Run tests (skips ROS-dependent)
python -m pytest source/strafer_autonomy/tests/ source/strafer_vlm/tests/ \
  -m "not requires_ros" -v

# Start services for manual testing
make serve-vlm
make serve-planner
```

---

## What NOT to touch

- `source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py` — owned by Jetson agent
- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py` — owned by Jetson agent
- `source/strafer_ros/` — owned by Jetson agent
- `source/strafer_shared/` — owned by Jetson agent
- `source/strafer_lab/` — owned by Windows agent
- `Makefile` — shared, do not modify without coordination
- `docs/` — do not modify design docs during implementation
