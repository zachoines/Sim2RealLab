# Integration Context: DGX Spark (Workstation)

## Role

GPU-powered model hosting. Runs two stateless FastAPI services: VLM grounding/description (port 8100) and LLM planner (port 8200). No ROS, no robot control. Accepts HTTP requests from the Jetson over LAN and returns structured JSON responses.

## System Architecture

```
Jetson Orin NX (Robot)                       DGX Spark (Workstation)
========================                     ========================
strafer_ros                                  strafer_vlm (port 8100)
  - RoboClaw motor driver                      - Qwen2.5-VL-3B-Instruct
  - RealSense D555 camera+IMU                  - POST /ground
  - RTAB-Map SLAM                              - POST /describe
  - Nav2 navigation (MPPI holonomic)           - GET /health
  - goal_projection_node (service)
                                             strafer_autonomy.planner (port 8200)
strafer_autonomy.executor                      - Qwen3-4B LLM
  - AutonomyCommandServer (ROS action)         - POST /plan
  - MissionRunner (skill dispatch)             - GET /health
  - JetsonRosClient (local ROS2)
  - HttpPlannerClient → DGX:8200
  - HttpGroundingClient → DGX:8100

     ←——— LAN HTTP (service calls) ———→
```

## Platform Details

- **Hardware**: NVIDIA DGX Spark — Grace CPU (ARM64) + Blackwell GB10 GPU
- **OS**: Ubuntu ARM64
- **CUDA**: 13.0 (system), PyTorch uses cu128 nightly index
- **Python**: 3.12
- **GPU compute capability**: sm_121 (Blackwell)

## Relevant Packages

### `strafer_vlm` — VLM Service (port 8100)

**Model**: `Qwen/Qwen2.5-VL-3B-Instruct` (~7GB download on first start)

**Endpoints**:

#### `GET /health`
```json
{"status": "ok", "model_loaded": true, "model_name": "Qwen/Qwen2.5-VL-3B-Instruct"}
```

#### `POST /ground` — Visual Grounding
Request:
```json
{
  "request_id": "req-001",
  "prompt": "red chair",
  "image_jpeg_b64": "<base64 JPEG bytes>",
  "image_stamp_sec": 0.0,
  "max_image_side": 1024,
  "return_debug_overlay": false
}
```
Response:
```json
{
  "request_id": "req-001",
  "found": true,
  "bbox_2d": [120, 340, 580, 890],
  "label": "chair",
  "confidence": 0.92,
  "raw_output": "{\"found\": true, ...}",
  "latency_s": 2.15,
  "debug_overlay_jpeg_b64": null
}
```
- `bbox_2d` is in Qwen normalized [0, 1000] coordinates (not pixels)
- Prompt is normalized to "Locate: ..." internally
- Image is decoded from base64, optionally resized, then inference runs in a single-thread pool
- System prompt instructs the model to return JSON with `found`, `bbox_2d`, `label`, `confidence`

#### `POST /describe` — Scene Description
Request:
```json
{
  "request_id": "desc-001",
  "image_jpeg_b64": "<base64 JPEG bytes>",
  "prompt": "Describe what you see in this image.",
  "max_image_side": 1024
}
```
Response:
```json
{
  "request_id": "desc-001",
  "description": "A kitchen with stainless steel appliances and a wooden table.",
  "latency_s": 1.87
}
```
- System prompt: "You are a robot vision system. Describe the scene in the image concisely. List the main objects, surfaces, and spatial layout visible. Keep your response to 1-3 sentences."
- Reuses the same `run_grounding_generation()` function with a different system prompt

**Environment Variables**:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROUNDING_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | HF model name or local path |
| `GROUNDING_DEVICE_MAP` | `auto` | Device map |
| `GROUNDING_TORCH_DTYPE` | `auto` | Torch dtype |
| `GROUNDING_LOAD_4BIT` | `0` | Enable 4-bit quantization |
| `GROUNDING_MAX_TOKENS` | `128` | Max new tokens per inference |
| `GROUNDING_MAX_IMAGE_MP` | `20` | Max decoded image megapixels (rejects larger) |
| `GROUNDING_INFERENCE_TIMEOUT` | `30` | Max seconds per inference call |
| `GROUNDING_HOST` | `0.0.0.0` | Bind host |
| `GROUNDING_PORT` | `8100` | Bind port |

**Key source files**:
- `strafer_vlm/service/app.py` — FastAPI factory with lifespan model loading, `/health`, `/ground`, `/describe` endpoints
- `strafer_vlm/service/payloads.py` — Pydantic request/response models
- `strafer_vlm/inference/qwen_runtime.py` — Model loading (`load_qwen_model_and_processor`) and generation (`run_grounding_generation`)
- `strafer_vlm/inference/parsing.py` — JSON extraction, bbox coercion, coordinate normalization, `GroundingTarget` dataclass

### `strafer_autonomy.planner` — LLM Planner Service (port 8200)

**Model**: `Qwen/Qwen3-4B` (~8GB download on first start)

**Endpoints**:

#### `GET /health`
```json
{"status": "ok", "model_loaded": true, "model_name": "Qwen/Qwen3-4B"}
```

#### `POST /plan` — Command-to-Mission Plan
Request:
```json
{
  "request_id": "plan-001",
  "raw_command": "go to the door",
  "robot_state": null,
  "active_mission_summary": null,
  "available_skills": [
    "capture_scene_observation", "locate_semantic_target", "scan_for_target",
    "describe_scene", "project_detection_to_goal_pose", "navigate_to_pose",
    "wait", "cancel_mission", "report_status"
  ]
}
```
Response:
```json
{
  "mission_id": "mission_abc123",
  "mission_type": "go_to_target",
  "raw_command": "go to the door",
  "steps": [
    {"step_id": "step_01", "skill": "scan_for_target", "args": {"label": "door", "max_scan_steps": 6, "scan_arc_deg": 360}, "timeout_s": 60.0, "retry_limit": 0},
    {"step_id": "step_02", "skill": "project_detection_to_goal_pose", "args": {"standoff_m": 0.7}, "timeout_s": 2.0, "retry_limit": 0},
    {"step_id": "step_03", "skill": "navigate_to_pose", "args": {"goal_source": "projected_target", "execution_backend": "nav2"}, "timeout_s": 90.0, "retry_limit": 0}
  ],
  "created_at": 1710000000.0
}
```

**Two-stage pipeline**:
1. **LLM inference** → raw JSON text from Qwen3-4B
2. **Intent parsing** → validates into `MissionIntent` (one of: `go_to_target`, `wait_by_target`, `cancel`, `status`)
3. **Plan compilation** → deterministic compiler expands intent into `MissionPlan` with concrete skill steps

**Plan compilation rules**:
- `go_to_target` → `[scan_for_target, project_detection_to_goal_pose, navigate_to_pose]`
- `wait_by_target` → same 3 steps + `wait`
- `cancel` → `[cancel_mission]`
- `status` → `[report_status]`

**Environment Variables**:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PLANNER_MODEL` | `Qwen/Qwen3-4B` | HF model name or local path |
| `PLANNER_DEVICE_MAP` | `auto` | Device map |
| `PLANNER_TORCH_DTYPE` | `auto` | Torch dtype |
| `PLANNER_LOAD_4BIT` | `0` | Enable 4-bit quantization |
| `PLANNER_MAX_TOKENS` | `256` | Max new tokens per inference |
| `PLANNER_PORT` | `8200` | Bind port |

**Key source files**:
- `strafer_autonomy/planner/app.py` — FastAPI factory, `/health`, `/plan` endpoints
- `strafer_autonomy/planner/llm_runtime.py` — Model loading and text generation
- `strafer_autonomy/planner/intent_parser.py` — Validates LLM JSON output → `MissionIntent`
- `strafer_autonomy/planner/plan_compiler.py` — `MissionIntent` → `MissionPlan` with skill steps
- `strafer_autonomy/planner/prompt_builder.py` — Constructs system/user prompts for the LLM
- `strafer_autonomy/planner/payloads.py` — Pydantic request/response models

### `strafer_shared` — Shared Constants

Robot dimensions, velocity limits, encoder parameters. Used by both `strafer_ros` (on Jetson) and `strafer_autonomy` (on both machines).

Key constants:
```python
CHASSIS_LENGTH = 0.43        # meters
CHASSIS_WIDTH = 0.43
WHEEL_RADIUS = 0.048
NAV_LINEAR_VEL = 0.5         # m/s
NAV_ANGULAR_VEL = 1.9        # rad/s
MAP_RESOLUTION = 0.05        # meters
```

## Setup & Launch Instructions

### Initial Setup
```bash
cd ~/Documents/repos/Sim2RealLab
git checkout main && git pull

# Create venv
python3.12 -m venv .venv_vlm
source .venv_vlm/bin/activate

# Install PyTorch (cu128 nightly for ARM64 Blackwell)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install packages
pip install -e source/strafer_shared -e source/strafer_vlm -e source/strafer_autonomy
```

### CRITICAL: Fix NVRTC for Blackwell GPU

PyTorch cu128 bundles NVRTC from CUDA 12.8, which doesn't support the Blackwell GB10's `sm_121` compute capability. JIT-compiled CUDA kernels fail with `nvrtc: error: invalid value for --gpu-architecture (-arch)`. Fix by replacing the bundled NVRTC with the system's CUDA 13.0 version:

```bash
NVRTC_DIR=".venv_vlm/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"

# Backup originals
mv "$NVRTC_DIR/libnvrtc.so.12" "$NVRTC_DIR/libnvrtc.so.12.bak"
mv "$NVRTC_DIR/libnvrtc-builtins.so.12.8" "$NVRTC_DIR/libnvrtc-builtins.so.12.8.bak"

# Symlink system CUDA 13.0 NVRTC
ln -s /usr/local/cuda-13.0/lib64/libnvrtc.so.13.0.88 "$NVRTC_DIR/libnvrtc.so.12"
ln -s /usr/local/cuda-13.0/lib64/libnvrtc-builtins.so.13.0.88 "$NVRTC_DIR/libnvrtc-builtins.so.12.8"
```

**This must be redone if `nvidia-cuda-nvrtc` is upgraded or the venv is recreated.**

### Start Services

```bash
# Terminal 1: VLM service (model downloads ~7GB on first run)
source .venv_vlm/bin/activate
uvicorn strafer_vlm.service.app:create_app --factory --host 0.0.0.0 --port 8100

# Terminal 2: Planner service (model downloads ~8GB on first run)
source .venv_vlm/bin/activate
uvicorn strafer_autonomy.planner.app:create_app --factory --host 0.0.0.0 --port 8200
```

### Verify

```bash
curl http://localhost:8100/health
# {"status":"ok","model_loaded":true,"model_name":"Qwen/Qwen2.5-VL-3B-Instruct"}

curl http://localhost:8200/health
# {"status":"ok","model_loaded":true,"model_name":"Qwen/Qwen3-4B"}
```

### Run Tests

```bash
# All 204 tests pass without any env vars
python -m pytest source/strafer_autonomy/tests/ source/strafer_vlm/tests/ -v
```

The planner endpoint tests use an `autouse` fixture that sets `PLANNER_MODEL=/nonexistent` to prevent model download during tests.

## What the Jetson Looks Like From Here

The Jetson is a ROS2 robot running:

- **Hardware**: GoBilda Strafer mecanum-drive chassis, dual RoboClaw motor controllers, Intel RealSense D555 (RGB-D + IMU)
- **ROS stack**: Driver (motor control + odometry) → Perception (camera + depth + IMU) → SLAM (RTAB-Map) → Navigation (Nav2 MPPI holonomic)
- **Autonomy executor**: `AutonomyCommandServer` is a ROS2 action server. It receives voice commands, calls the DGX planner for a mission plan, then dispatches skills. For grounding skills, it sends JPEG images from the D555 camera to the DGX VLM service. For navigation/projection, it uses local ROS2 services and actions.
- **The executor calls DGX via HTTP**: `HttpPlannerClient` → `:8200/plan`, `HttpGroundingClient` → `:8100/ground` and `:8100/describe`
- **Images arrive as**: 640×360 BGR8 from D555, converted to JPEG base64 by the grounding client before sending
- **Bbox coordinates flow**: DGX returns [0,1000] normalized → Jetson projection service converts to pixels → depth lookup → 3D map-frame pose

The DGX has no ROS dependencies and no knowledge of the robot's internal state. It is purely a stateless inference backend.

## Known Integration Gaps

| # | Gap | Description |
|---|-----|-------------|
| 1 | **Firewall** | Ports 8100 and 8200 must be accessible from the Jetson's LAN IP. Verify with `curl http://<DGX_IP>:8100/health` from the Jetson. |
| 2 | **NVRTC fix not persisted** | If the venv is recreated or `nvidia-cuda-nvrtc` package is upgraded, the symlinks break and JIT kernels fail silently. |
| 3 | **Postman collection incomplete** | Only `/health` and `/ground` are in the Postman collection. `/describe` and `/plan` endpoints are not included. |
| 4 | **Model download time** | First start: ~7GB (VLM) + ~8GB (planner). Models cache to `~/.cache/huggingface/`. Subsequent starts are fast. |
| 5 | **No Docker/compose** | Services run directly in a venv. No containerization or deployment automation exists. |
| 6 | **Single-threaded inference** | Both services use `ThreadPoolExecutor(max_workers=1)`. Concurrent requests queue. |
| 7 | **Latency** | VLM grounding: ~2-3s per image. Planner: ~1-2s per command. These are acceptable for MVP but add up during `scan_for_target` (6 rotations × 3s = ~18s). |
