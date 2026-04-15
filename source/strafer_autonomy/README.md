# strafer_autonomy

Autonomy planning, mission execution, shared schemas, and client interfaces for Strafer.

## Install

```bash
pip install -e source/strafer_autonomy
```

## Package Layout

```text
strafer_autonomy/
  clients/       # HTTP and Databricks Model Serving clients for planner / VLM,
                 #   ROS client for Jetson hardware
                 #     planner_client.py, vlm_client.py              (LAN HTTP)
                 #     databricks_planner_client.py                  (Databricks)
                 #     databricks_vlm_client.py                      (Databricks)
                 #     ros_client.py                                 (Jetson hw)
  executor/      # Jetson-local command server, mission runner, entry point
  planner/       # LLM planner service (FastAPI, Qwen3-4B, two-stage
                 #   intent→plan pipeline with 9 intent types)
  schemas/       # Typed data contracts for planner, executor, ROS, VLM
  databricks/    # MLflow pyfunc wrappers + register.py for deploying the
                 #   planner and VLM as Databricks Model Serving endpoints
  cli.py         # Operator CLI for submit/status/cancel
```

## Running the Executor on Jetson

The `strafer-executor` entry point starts the autonomy command server. It requires
`VLM_URL` and `PLANNER_URL` environment variables pointing at the DGX Spark services:

```bash
source ~/strafer_ws/install/setup.bash

VLM_URL=http://192.168.50.196:8100 \
PLANNER_URL=http://192.168.50.196:8200 \
    strafer-executor
```

Or use the full autonomy launch file (starts navigation stack + goal projection + executor):

```bash
VLM_URL=http://192.168.50.196:8100 PLANNER_URL=http://192.168.50.196:8200 \
    make launch-autonomy
```

The executor probes the VLM health endpoint at startup and fails fast if the service
is unreachable or the model is not loaded.

## Operator CLI

```bash
strafer-autonomy-cli submit "go to the tennis ball" --detach
strafer-autonomy-cli status
strafer-autonomy-cli cancel
```

## Planner Service

The LLM planner service runs on the DGX Spark (port 8200). Install with planner extras:

```bash
pip install -e "source/strafer_autonomy[planner]"
```

Launch:

```bash
uvicorn strafer_autonomy.planner.app:create_app --factory --host 0.0.0.0 --port 8200
```

| Variable | Default | Description |
|---|---|---|
| `PLANNER_MODEL` | `Qwen/Qwen3-4B` | HuggingFace model name or local path |
| `PLANNER_DEVICE_MAP` | `auto` | PyTorch device map |
| `PLANNER_TORCH_DTYPE` | `auto` | Torch dtype |
| `PLANNER_LOAD_4BIT` | `0` | Set `1` to enable 4-bit quantisation |
| `PLANNER_MAX_TOKENS` | `256` | Max new tokens per inference |
| `PLANNER_PORT` | `8200` | Bind port |

Architecture: the LLM classifies user commands into a `MissionIntent` (one of
**nine** intent types), then a deterministic `plan_compiler` expands it into a
validated `MissionPlan` with correct skill ordering, timeouts, and retry
limits. The LLM never generates raw skill sequences — it only picks an intent
and (for target-bearing intents) fills in labels. The compiler is the sole
source of skill structure, which makes the mission space bounded and testable.

### Mission intent types

| Intent | Purpose | Key fields | Plan shape |
|---|---|---|---|
| `go_to_target` | Navigate to one semantic target | `target_label` | scan → project → navigate → verify_arrival (4 steps) |
| `wait_by_target` | Navigate and wait indefinitely | `target_label`, `wait_mode` | scan → project → navigate → verify_arrival → wait (5 steps) |
| `go_to_targets` | Ordered multi-target chaining | `targets: list[{label, standoff_m?}]` | 4 steps × N targets |
| `patrol` | Waypoint cycle (loop control at executor level) | `targets: list[{label, standoff_m?}]` | 4 steps × N waypoints |
| `rotate` | Relative degrees or absolute cardinal heading | `orientation_mode` (numeric string or `"north"`/`"east"`/etc.) | `rotate_by_degrees` OR `orient_to_direction` (1 step) |
| `describe` | Operator-facing scene readback | — | `describe_scene` (1 step) |
| `query` | Semantic-map lookup with no motion | — | `query_environment` (1 step) |
| `cancel` | Cancel the active mission | — | `cancel_mission` (1 step) |
| `status` | Report current robot/mission state | — | `report_status` (1 step) |

Every compiler that terminates in `navigate_to_pose` appends a
`verify_arrival` step, which runs a CLIP top-k ranking against the semantic
map at the arrival pose to confirm the robot is where it thinks it is.
`go_to_target` went from 3 steps to 4, `wait_by_target` from 4 to 5,
`go_to_targets` / `patrol` emit 4 steps per target. See the compiler code in
`planner/plan_compiler.py` and the design rationale in
`docs/STRAFER_AUTONOMY_NEXT.md` §0.1.

### `/plan_with_grounding` — agentic endpoint

`POST /plan_with_grounding` extends the standard `PlanRequest` with an optional
`image_jpeg_b64` field. When the planner and VLM are co-located on the same
host (the DGX Spark default), the planner forwards the image to
`http://localhost:8100/ground` via `httpx.AsyncClient` during plan compilation
if the parsed intent has `requires_grounding=true` and a resolved
`target_label`. The response carries both the plan and an optional
`pre_grounding` field with the VLM's bbox / confidence. This saves one LAN
image round-trip per mission (~2-3s). On any VLM failure, the endpoint
gracefully falls through to a plan without `pre_grounding` — the executor's
existing `scan_for_target` step is the safety net.

Env vars:

| Variable | Default | Description |
|---|---|---|
| `VLM_GROUND_URL` | `http://localhost:8100/ground` | Where the planner forwards images for pre-grounding |
| `VLM_GROUND_TIMEOUT_S` | `10.0` | Timeout for the internal VLM call |

## VLM Integration

`HttpGroundingClient` connects to the `strafer_vlm` grounding service over LAN.

- Retry with exponential backoff on connection errors and 5xx responses
- Domain-specific `GroundingServiceUnavailable` exception on failure
- `build_command_server()` probes VLM health at startup before accepting missions
- Debug overlay support via `return_debug_overlay` / `debug_overlay_jpeg_b64`
- Additive `detect_objects()` method (not on the `GroundingClient` protocol so
  existing implementations remain valid) for the `/detect_objects` endpoint —
  used to populate the semantic map with all visible objects in one call
- Configurable `jpeg_quality` on `HttpGroundingClientConfig` so callers can
  lower it for WAN / Databricks deployments without forking the encoder

See `source/strafer_vlm/README.md` for service launch instructions and API details.

## Databricks Model Serving Backend

The executor can transparently swap from the LAN HTTP services to Databricks
Model Serving endpoints by setting `PLANNER_BACKEND=databricks` /
`VLM_BACKEND=databricks` at startup. Two pieces make this work:

**1. Client-side drop-in implementations** (in `clients/`):

- `DatabricksServingPlannerClient` implements the same `PlannerClient` protocol
  as `HttpPlannerClient`, using `{"inputs": [...]}` / `{"predictions": [...]}`
  wire format. Health checks map Databricks `state.ready` into the shape
  `build_command_server()` already expects.
- `DatabricksServingGroundingClient` does the same for `GroundingClient`. Each
  row carries `{request_id, image_b64, prompt, mode}` where `mode` is
  `"ground"` or `"describe"`.
- Both clients reuse the shared `planner_request_to_payload` /
  `grounding_result_from_payload` helpers from their LAN counterparts, so the
  two transports are guaranteed to parse responses identically.

**2. Server-side pyfunc wrappers** (in `databricks/`):

- `StraferPlannerModel` wraps the full `build_messages → LLMRuntime.generate
  → parse_intent → compile_plan` pipeline — a Databricks endpoint call
  returns the exact same `MissionPlan` JSON shape as `POST /plan`.
- `StraferVLMModel` wraps ground + describe using the same Qwen2.5-VL
  runtime the LAN service uses.
- Neither wrapper subclasses `mlflow.pyfunc.PythonModel` at import time.
  `mlflow` is an optional dependency — the Jetson never imports the
  `databricks/` subpackage, and deploy hosts that install `mlflow` use
  `register.py` to compose the wrappers with `PythonModel` at log time.
- `register.py` is a standalone CLI that logs both wrappers to an MLflow
  experiment with correct `pip_requirements` and optional
  `registered_model_name` for Unity Catalog.

```bash
python -m strafer_autonomy.databricks.register \
    --planner-model-path /path/to/Qwen3-4B \
    --vlm-model-path /path/to/Qwen2.5-VL-3B-Instruct \
    --experiment /Shared/strafer \
    --register-name-planner strafer-planner \
    --register-name-vlm strafer-vlm
```

Environment variables for the executor when using the Databricks backend:

```bash
PLANNER_BACKEND=databricks
VLM_BACKEND=databricks
DATABRICKS_HOST=https://<workspace>.databricks.net
DATABRICKS_TOKEN=dapi...
PLANNER_ENDPOINT=strafer-planner
VLM_ENDPOINT=strafer-vlm
```

The Jetson itself never imports `mlflow` — it only imports
`strafer_autonomy.clients.databricks_*`, which uses plain `requests` for
HTTPS to the serving endpoint. MLflow only lives inside the serving
container, not on the robot.
