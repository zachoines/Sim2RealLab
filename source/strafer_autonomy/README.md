# strafer_autonomy

Autonomy planning, mission execution, shared schemas, and client interfaces for Strafer.

## Install

```powershell
python -m pip install -e source/strafer_autonomy
```

## Package Layout

```text
strafer_autonomy/
  clients/   # Planner, VLM, and ROS client protocols plus first transport stubs
  executor/  # Jetson-local command ingress and mission execution scaffolding
  planner/   # LLM planner service (FastAPI, Qwen3-4B, two-stage intent→plan pipeline)
  schemas/   # Typed data contracts for planner, executor, ROS, and VLM boundaries
  cli.py     # Operator CLI for submit/status/cancel
```

## Planner Service

The LLM planner service runs on the Windows workstation (port 8200) alongside the VLM
grounding service (port 8100). Install with planner extras:

```powershell
python -m pip install -e "source/strafer_autonomy[planner]"
```

Launch:

```powershell
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

`HttpPlannerClient` connects to the planner over LAN with retry and exponential backoff,
matching the `HttpGroundingClient` pattern.

## VLM Integration

`HttpGroundingClient` connects to the `strafer_vlm` grounding service over LAN.

- Retry with exponential backoff on connection errors and 5xx responses
- Domain-specific `GroundingServiceUnavailable` exception on failure
- `build_command_server()` probes VLM health at startup before accepting missions
- Debug overlay support via `return_debug_overlay` / `debug_overlay_jpeg_b64`

See `source/strafer_vlm/README.md` for service launch instructions and API details.
