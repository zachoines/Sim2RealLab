# strafer_autonomy

Autonomy planning, mission execution, shared schemas, and service clients for the Strafer robot.

`strafer_autonomy` is the autonomy-layer Python package that owns mission
planning (LLM planner service), mission execution (Jetson executor +
mission runner), the shared typed schemas every other package agrees on
at autonomy-layer boundaries, and the client library that connects the
executor to remote planner and VLM services. It is installed on both
the DGX Spark (planner service + registration tooling) and the Jetson
Orin Nano (executor + CLI + ROS client).

## Role in the system

| Surface | Host | Consumes | Produces |
|---|---|---|---|
| Planner service | DGX Spark:8200 | `POST /plan` requests carrying user commands | `MissionPlan` JSON |
| Executor + command server | Jetson Orin Nano | `execute_mission` ROS action goals | Mission progress feedback, `SkillResult` sequences |
| Operator CLI | any host with ROS on the Jetson domain | Terminal commands | ROS action goals, JSON status output |
| Semantic map manager | Jetson Orin Nano | VLM detections + robot pose | NetworkX graph + ChromaDB embeddings + CLIP image encodings |

Sibling packages it interacts with:

- **`strafer_vlm`** — grounding / description / detect-objects service, called over LAN HTTP from the executor and directly from the planner for pre-grounding.
- **`strafer_ros`** — local sensor streams, Nav2 action client, goal projection service, TF, `/cmd_vel`. Consumed through `JetsonRosClient`.
- **`strafer_shared`** — physical constants (robot dimensions, velocity limits, depth clip range) imported by the goal projection service on the ROS side. Not imported directly by the executor today.

## What ships today

- **LLM planner service** (`planner/app.py`) — FastAPI app wrapping `Qwen/Qwen3-4B` with a two-stage pipeline: LLM classifies commands into a `MissionIntent`, then a deterministic `plan_compiler` expands into a validated `MissionPlan`.
- **Nine mission intent types** — `go_to_target`, `wait_by_target`, `go_to_targets`, `patrol`, `rotate`, `describe`, `query`, `cancel`, `status`. Enumerated in `planner/intent_parser.py`; compiled by `planner/plan_compiler.py`.
- **Agentic `POST /plan_with_grounding`** — planner pre-grounds the target via a co-located VLM call when the intent requires grounding, saving one LAN image round-trip per mission.
- **Jetson executor** (`executor/mission_runner.py`, `executor/command_server.py`) — 13 implemented skills, mission state machine, cancel / timeout / retry, parallel VLM + planner health checks at startup.
- **Operator CLI** (`cli.py`) — `submit` / `status` / `cancel` commands over the ROS action / service interfaces.
- **Service clients** (`clients/`) — `HttpPlannerClient`, `HttpGroundingClient` for LAN HTTP; `DatabricksServingPlannerClient`, `DatabricksServingGroundingClient` for Databricks Model Serving; `JetsonRosClient` for local ROS interactions.
- **Semantic spatial map** (`semantic_map/`) — NetworkX graph + ChromaDB vector store + OpenCLIP encoder backing the `verify_arrival` and `query_environment` skills.
- **Databricks deployment tooling** (`databricks/`) — MLflow pyfunc wrappers (`StraferPlannerModel`, `StraferVLMModel`) and a `register.py` CLI for logging both models to a Databricks workspace.

## Contracts

The package exposes four contract surfaces. Each is the source of truth for the interface it describes.

### Python schemas (`schemas/`)

Frozen dataclasses shared across planner, executor, and client code.

| Schema | Purpose | Key fields |
|---|---|---|
| `PlannerRequest` | Executor → planner request | `request_id`, `raw_command`, `robot_state`, `active_mission_summary`, `available_skills` |
| `MissionIntent` | Planner LLM output, pre-compilation | `intent_type`, `target_label`, `orientation_mode`, `wait_mode`, `requires_grounding`, `targets` |
| `MissionPlan` | Planner → executor compiled plan | `mission_id`, `mission_type`, `raw_command`, `steps: tuple[SkillCall, ...]`, `created_at` |
| `SkillCall` | One executable step | `skill`, `step_id`, `args`, `timeout_s`, `retry_limit` |
| `SkillResult` | Normalized executor-facing result | `step_id`, `skill`, `status` (`succeeded`/`failed`/`canceled`/`timeout`), `outputs`, `error_code`, `message`, timestamps |
| `GroundingRequest` | Executor → VLM grounding request | `request_id`, `prompt`, `image_rgb_u8` (numpy uint8 HxWx3), `image_stamp_sec`, `max_image_side`, `return_debug_overlay` |
| `GroundingResult` | VLM → executor grounding result | `request_id`, `found`, `bbox_2d` (`[0, 1000]` normalized coords), `label`, `confidence`, `raw_output`, `latency_s`, `debug_overlay_jpeg_b64` |
| `GoalPoseCandidate` | Robot-side projection output | `request_id`, `found`, `goal_frame`, `goal_pose` (`Pose3D`), `target_pose`, `standoff_m`, `depth_valid`, `quality_flags` |
| `Pose3D` | 3D pose matching `geometry_msgs/Pose` | `x, y, z, qx, qy, qz, qw` |
| `SceneObservation` | Jetson-local synchronized observation | `observation_id`, `stamp_sec`, `color_image_bgr`, `aligned_depth_m`, `camera_frame`, `camera_info`, `robot_pose_map` |
| `SceneDescription` | VLM → executor free-text description | `request_id`, `description`, `latency_s` |

### Planner service (HTTP)

FastAPI app served by `uvicorn strafer_autonomy.planner.app:create_app --factory`.

| Endpoint | Purpose | Request | Response |
|---|---|---|---|
| `GET /health` | Readiness check | — | `{"status", "model_loaded", "model_name"}` |
| `POST /plan` | Command → validated mission plan | `PlannerRequest` | `PlanResponse` with compiled `steps` |
| `POST /plan_with_grounding` | Same, plus pre-grounding via co-located VLM | `PlannerRequest + image_jpeg_b64 + image_stamp_sec + max_image_side` | `PlanResponse + pre_grounding` |

`POST /plan_with_grounding` only pre-grounds when `intent.requires_grounding` is true AND `intent.target_label` is set AND an image was supplied. On any VLM error, the response falls through with `pre_grounding=null` and the executor's existing `scan_for_target` step is the safety net.

**Environment variables** (planner service):

| Variable | Default | Description |
|---|---|---|
| `PLANNER_MODEL` | `Qwen/Qwen3-4B` | HuggingFace model name or local path |
| `PLANNER_DEVICE_MAP` | `auto` | PyTorch `device_map` |
| `PLANNER_TORCH_DTYPE` | `auto` | Torch dtype (`auto`, `float16`, `bfloat16`) |
| `PLANNER_LOAD_4BIT` | `0` | Set `1` for 4-bit quantisation |
| `PLANNER_MAX_TOKENS` | `256` | Max new tokens per inference |
| `VLM_GROUND_URL` | `http://localhost:8100/ground` | Co-located VLM endpoint used by `/plan_with_grounding` |
| `VLM_GROUND_TIMEOUT_S` | `10.0` | Timeout on the pre-grounding call |

Host and port are controlled by uvicorn CLI arguments, not env vars.

### Executor (ROS interfaces)

Defined in [`strafer_ros/strafer_msgs`](../strafer_ros/strafer_msgs/), implemented by `AutonomyCommandServer` in `executor/command_server.py`.

| Interface | Type | Default name | Purpose |
|---|---|---|---|
| `strafer_msgs/action/ExecuteMission` | Action | `execute_mission` | Submit a mission; stream feedback; cancel mid-flight |
| `strafer_msgs/srv/GetMissionStatus` | Service | `get_mission_status` | Query current-or-last mission snapshot |

`ExecuteMission.action`:

```text
# Goal
string request_id
string raw_command
string source
bool   replace_active_mission
---
# Result
bool   accepted
string mission_id
string final_state
string error_code
string message
---
# Feedback
string mission_id
string state
string current_step_id
string current_skill
string message
float32 elapsed_s
```

`GetMissionStatus.srv` response includes `active`, `mission_id`, `state`, `raw_command`, `current_step_id`, `current_skill`, `message`, `elapsed_s`.

At startup, `build_command_server()` runs parallel health checks against the planner and VLM clients with a 10 s timeout; it fails fast if either is reachable but has no model loaded, and logs a warning if unreachable. Disable with `check_vlm_health=False` for tests.

### Skill registry (executor-side)

`DEFAULT_AVAILABLE_SKILLS` in `executor/mission_runner.py`:

| Skill | Purpose | Backend |
|---|---|---|
| `capture_scene_observation` | Pull newest synced RGB + depth + pose from ROS cache | `JetsonRosClient` |
| `locate_semantic_target` | Single VLM grounding call | `HttpGroundingClient.locate_semantic_target()` |
| `scan_for_target` | Rotate and ground at each heading until found | Composite: ROS rotate + grounding loop |
| `describe_scene` | VLM free-text description | `HttpGroundingClient.describe_scene()` |
| `project_detection_to_goal_pose` | 2D bbox → map-frame goal pose | Local `/strafer/project_detection_to_goal_pose` service |
| `navigate_to_pose` | Drive to goal via selectable local backend | `nav2` (shipped), `strafer_direct` / `hybrid_nav2_strafer` (deferred) |
| `verify_arrival` | CLIP top-k ranking against semantic map at arrival pose | `SemanticMapManager` + `CLIPEncoder` |
| `rotate_by_degrees` | Relative yaw rotation | `JetsonRosClient.rotate_in_place()` |
| `orient_to_direction` | Absolute cardinal heading | `JetsonRosClient.rotate_in_place()` with yaw delta |
| `query_environment` | Semantic-map lookup with no motion | `SemanticMapManager.query_*` |
| `wait` | Hold position until timeout or interruption | Local timer + cancel event |
| `cancel_mission` | Stop current mission safely | Local cancel + ROS action cancel |
| `report_status` | Produce operator-facing status | Mission runtime snapshot |

Every compiler that terminates in `navigate_to_pose` appends a `verify_arrival` step, so single-target missions are 4 steps (`scan → project → navigate → verify`), `wait_by_target` is 5, and `go_to_targets` / `patrol` emit 4 steps per target.

### Client protocols

```python
class PlannerClient(Protocol):
    def plan_mission(self, request: PlannerRequest) -> MissionPlan: ...

class GroundingClient(Protocol):
    def locate_semantic_target(self, request: GroundingRequest) -> GroundingResult: ...
    def describe_scene(self, *, request_id, image_rgb_u8, prompt=None, max_image_side=1024) -> SceneDescription: ...

class RosClient(Protocol):
    def capture_scene_observation(self) -> SceneObservation: ...
    def navigate_to_pose(self, *, step_id, goal_pose, execution_backend="nav2", timeout_s) -> SkillResult: ...
    def project_detection_to_goal_pose(self, *, request_id, image_stamp_sec, bbox_2d, standoff_m, target_label) -> GoalPoseCandidate: ...
    def rotate_in_place(self, *, step_id, yaw_delta_rad, tolerance_rad=0.1, timeout_s) -> SkillResult: ...
    def get_robot_state(self) -> dict: ...
    def cancel_active_navigation(self) -> bool: ...
```

`HttpGroundingClient` additionally exposes `detect_objects(...)`, which is **not** part of the `GroundingClient` protocol so existing implementations stay valid. Callers use `hasattr(client, "detect_objects")` to detect it.

## Install

### Base package (all hosts)

```bash
pip install -e source/strafer_autonomy
```

This installs the executor, CLI, schemas, clients, and semantic map modules. `rclpy` and ROS 2 message packages are sourced from the colcon workspace, not PyPI — the `[ros]` extra is intentionally empty.

### Planner service (DGX Spark only)

```bash
pip install -e "source/strafer_autonomy[planner]"
```

Adds `fastapi`, `uvicorn`, `transformers`, `torch`, `accelerate`. Expects a working CUDA toolchain; on DGX Spark the NVRTC fix for Blackwell `sm_121` must be in place before launching the service.

### Entry points registered

- `strafer-autonomy-cli` → `cli.main`
- `strafer-executor` → `executor.main.main`

## Run

### Planner service on DGX Spark

```bash
source .venv_vlm/bin/activate
uvicorn strafer_autonomy.planner.app:create_app --factory --host 0.0.0.0 --port 8200
```

Model downloads on first run (~8 GB for Qwen3-4B, cached to `~/.cache/huggingface/`).

### Executor on Jetson

```bash
source ~/strafer_ws/install/setup.bash
VLM_URL=http://192.168.50.196:8100 \
PLANNER_URL=http://192.168.50.196:8200 \
    strafer-executor
```

Both URLs are required — `executor/main.py` exits with an error message if either is unset. The executor probes both services' `/health` endpoints in parallel at startup and fails fast if either is reachable but reports `model_loaded=false`.

Alternatively, from the repo root Makefile:

```bash
VLM_URL=http://192.168.50.196:8100 PLANNER_URL=http://192.168.50.196:8200 \
    make launch-autonomy
```

This target launches the full autonomy stack: driver + perception + SLAM + Nav2 + goal projection + executor.

### Submit missions via the operator CLI

```bash
# Submit and follow feedback until the mission terminates:
strafer-autonomy-cli submit "go to the tennis ball"

# Submit and return immediately (feedback continues on the executor):
strafer-autonomy-cli submit "go to the tennis ball" --detach

# Query current-or-last mission status:
strafer-autonomy-cli status

# Cancel the active mission:
strafer-autonomy-cli cancel
```

CLI flags common to submit / status / cancel: `--action-name`, `--service-name`, `--node-name`, `--wait-timeout`. `submit` adds `--request-id`, `--source`, `--replace-active`, `--detach`.

### Databricks deployment (alternative to LAN HTTP)

Register both pyfunc models with MLflow / Databricks:

```bash
python -m strafer_autonomy.databricks.register \
    --planner-model-path /path/to/Qwen3-4B \
    --vlm-model-path /path/to/Qwen2.5-VL-3B-Instruct \
    --experiment /Shared/strafer \
    --register-name-planner strafer-planner \
    --register-name-vlm strafer-vlm
```

Point the executor at the serving endpoints via environment:

```bash
PLANNER_BACKEND=databricks
VLM_BACKEND=databricks
DATABRICKS_HOST=https://<workspace>.databricks.net
DATABRICKS_TOKEN=dapi...
PLANNER_ENDPOINT=strafer-planner
VLM_ENDPOINT=strafer-vlm
```

The Jetson never imports `mlflow` — only `strafer_autonomy.clients.databricks_*`, which uses plain `requests` for HTTPS to the serving endpoint. MLflow runs inside the serving container, not on the robot.

## Design

**Planner and executor stay separate.** The LLM is a text-to-plan service; it never emits raw skill sequences or ROS messages. The executor owns mission state, validation, retries, cancel, timeout, and every robot-side control decision. This split lets the planner move between workstation and cloud without touching the robot-side execution boundary.

**Deterministic plan compilation.** The LLM picks an intent and (for target-bearing intents) fills in labels; `plan_compiler.py` is the sole source of skill ordering, timeouts, retry limits, and argument defaults. Plans are bounded and testable.

**`verify_arrival` is always appended after navigation.** Every compiler that ends in `navigate_to_pose` emits a `verify_arrival` step that runs a CLIP top-k ranking against the semantic map at the arrival pose. Decision rule is ranking-based, not threshold-based: if ≥3 of the top-5 neighbors are within `goal_radius_m=3.0` of the goal, verified. Threshold-based checks were rejected because they drift with CLIP model variant and environment.

**Images never leave the executor as Python objects.** `HttpGroundingClient` JPEG-encodes and base64-encodes before sending over LAN. Databricks clients do the same. Protocol implementations are drop-in replaceable.

**`capture_scene_observation` returns a Jetson-local runtime object, not a network-portable schema.** Do not attempt to serialize `SceneObservation` for transport; it holds raw numpy arrays from ROS topic subscriptions.

**Bbox coordinate convention is `[0, 1000]` normalized throughout.** `GroundingResult.bbox_2d`, `POST /ground` responses, and the `ProjectDetectionToGoalPose.srv` request all use this convention. Pixel conversion happens inside the robot-side projection service using camera intrinsics.

**The executor is ROS-aware but ROS-import-lazy.** `strafer_autonomy` is a Python package, not an ament_python package — `rclpy` and `strafer_msgs` are imported lazily at CLI / executor entry, so the schemas, mission runner, and client classes remain testable from a plain Python environment.

## Testing

```bash
# All non-ROS tests (default for DGX / CI):
python -m pytest source/strafer_autonomy/tests/ -m "not requires_ros" -v

# Full test suite (requires ROS 2 + strafer_msgs built):
source ~/strafer_ws/install/setup.bash
python -m pytest source/strafer_autonomy/tests/ -v
```

Tests marked `requires_ros` exercise the command server, CLI, and `JetsonRosClient` against live ROS interfaces. Everything else — planner pipeline, schemas, intent parsing, plan compilation, HTTP clients, Databricks clients, semantic map — runs without a ROS install.

## Deferred / known limitations

Tracked in [`docs/DEFERRED_WORK.md`](../../docs/DEFERRED_WORK.md). Items currently open:

- `orient_relative_to_target` skill — handler is drafted but commented out of `DEFAULT_AVAILABLE_SKILLS`. Reinstate when the behavior is needed.
- RL execution backends for `navigate_to_pose` — `strafer_direct` (pure-RL) and `hybrid_nav2_strafer` (Nav2 waypoints + learned local controller) are defined in the interface but not implemented. Default remains `nav2`.
- `rotate_in_place` PID tuning on real hardware — open-loop `cmd_vel` with odom feedback; may need tolerance adjustment.

## References

- [`source/strafer_vlm/README.md`](../strafer_vlm/README.md) — grounding / description / multi-object detection service.
- [`source/strafer_ros/README.md`](../strafer_ros/README.md) — Jetson runtime, sensors, TF, Nav2, goal projection.
- [`docs/STRAFER_AUTONOMY_NEXT.md`](../../docs/STRAFER_AUTONOMY_NEXT.md) — design rationale for current features (verify_arrival, `/plan_with_grounding`, semantic map).
- [`docs/SYSTEM_FLOW_DIAGRAMS.md`](../../docs/SYSTEM_FLOW_DIAGRAMS.md) — end-to-end runtime flows spanning this package + siblings.
- [`docs/DEFERRED_WORK.md`](../../docs/DEFERRED_WORK.md) — open items that will roll into the next design round.
- [`docs/VALIDATE_ISAAC_SIM_AND_INFINIGEN.md`](../../docs/VALIDATE_ISAAC_SIM_AND_INFINIGEN.md) — DGX-side install and smoke-test runbook.
