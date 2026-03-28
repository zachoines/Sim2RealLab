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
  schemas/   # Typed data contracts for planner, executor, ROS, and VLM boundaries
  cli.py     # Operator CLI for submit/status/cancel
```

## VLM Integration

`HttpGroundingClient` connects to the `strafer_vlm` grounding service over LAN.

- Retry with exponential backoff on connection errors and 5xx responses
- Domain-specific `GroundingServiceUnavailable` exception on failure
- `build_command_server()` probes VLM health at startup before accepting missions
- Debug overlay support via `return_debug_overlay` / `debug_overlay_jpeg_b64`

See `source/strafer_vlm/README.md` for service launch instructions and API details.
