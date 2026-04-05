# strafer_autonomy

Autonomy planning, mission execution, shared schemas, and client interfaces for Strafer.

## Install

```bash
pip install -e source/strafer_autonomy
```

## Package Layout

```text
strafer_autonomy/
  clients/   # HTTP clients for planner and VLM, ROS client for Jetson hardware
  executor/  # Jetson-local command server, mission runner, and entry point
  planner/   # LLM planner service (FastAPI, Qwen3-4B, two-stage intent->plan pipeline)
  schemas/   # Typed data contracts for planner, executor, ROS, and VLM boundaries
  cli.py     # Operator CLI for submit/status/cancel
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

Architecture: the LLM classifies user commands into a `MissionIntent` (one of 4 types),
then a deterministic `plan_compiler` expands it into a validated `MissionPlan` with
correct skill ordering, timeouts, and retry limits. The LLM never generates raw skill
sequences.

## VLM Integration

`HttpGroundingClient` connects to the `strafer_vlm` grounding service over LAN.

- Retry with exponential backoff on connection errors and 5xx responses
- Domain-specific `GroundingServiceUnavailable` exception on failure
- `build_command_server()` probes VLM health at startup before accepting missions
- Debug overlay support via `return_debug_overlay` / `debug_overlay_jpeg_b64`

See `source/strafer_vlm/README.md` for service launch instructions and API details.
