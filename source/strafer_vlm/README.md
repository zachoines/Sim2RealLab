# strafer_vlm

VLM grounding, description, and multi-object detection service for the Strafer robot, plus the tooling to fine-tune the underlying model.

`strafer_vlm` wraps `Qwen/Qwen2.5-VL-3B-Instruct` behind a small HTTP
service that exposes three endpoints on one loaded model: single-object
grounding, free-text scene description, and multi-object detection. It
also ships the LoRA fine-tuning and offline evaluation scripts for the
grounding model, plus operator-facing CLI tools for smoke-testing and
live camera evaluation. The service runs on the DGX Spark (port 8100)
and is called over LAN HTTP by the Jetson executor in
[`strafer_autonomy`](../strafer_autonomy/README.md). An alternative
deployment target is Databricks Model Serving, wired through
[`strafer_autonomy.clients.databricks_vlm_client`](../strafer_autonomy/strafer_autonomy/clients/databricks_vlm_client.py).

## Role in the system

| Surface | Host | Consumes | Produces |
|---|---|---|---|
| Grounding service | DGX Spark:8100 | `POST /ground` / `/describe` / `/detect_objects` with JPEG base64 images | JSON responses: bboxes, descriptions, object lists |
| Fine-tuning tooling | DGX Spark or Windows GPU workstation | Grounding dataset (JSONL) + base model | LoRA adapter weights |
| CLI tools | any workstation with GPU | Local images / camera feed | Overlay images + console output |

Sibling packages it interacts with:

- **`strafer_autonomy`** — the executor's `HttpGroundingClient` is the primary caller of every endpoint. The planner's `/plan_with_grounding` also calls `POST /ground` internally for agentic pre-grounding. Both sides agree on the `GroundingRequest` / `GroundingResult` schemas defined in `strafer_autonomy.schemas`.
- **`strafer_lab`** — consumes fine-tune datasets produced by `strafer_lab.scripts.prepare_vlm_finetune_data` (primary SFT prep) and `strafer_lab.tools.dataset_export` (basic grounding subset).

`strafer_vlm` does **not** do depth projection, goal generation, reachability checks, or any robot-side reasoning. It stops at image-space bounding boxes and free text.

## What ships today

- **Grounding service** (`service/app.py`) — FastAPI app with three endpoints on one loaded Qwen2.5-VL model, served behind a single-worker `ThreadPoolExecutor` so concurrent requests queue deterministically.
- **`POST /ground`** — single-object grounding, one `<ref>/<box>` pair. Consumed by the `scan_for_target` executor skill.
- **`POST /describe`** — free-text scene description, 1-3 sentences. Consumed by the `describe_scene` executor skill and the scan-failure fallback.
- **`POST /detect_objects`** — multi-object detection in one inference pass. Consumed by the semantic map to populate observation nodes with every visible object without running grounding per label.
- **Inference runtime** (`inference/qwen_runtime.py`, `inference/parsing.py`) — shared model loading, generation, and output parsing (JSON extraction, `<ref>/<box>` parsing, bbox normalization to `[0, 1000]`).
- **Training tooling** (`training/`) — LoRA fine-tuning (`train_qwen25vl_lora.py`), offline evaluation with IoU metrics (`eval_qwen25vl_grounding.py`), and dataset loader (`dataset_io.py`) supporting both flat and chat JSONL formats.
- **Operator CLI tools** (`tools/`) — single-image grounding smoke test and live camera/video grounding with runtime prompt updates.

## Contracts

### HTTP endpoints

| Endpoint | Purpose | Key request fields | Key response fields |
|---|---|---|---|
| `GET /health` | Readiness check | — | `status` (`ok`/`loading`), `model_loaded`, `model_name` |
| `POST /ground` | Single-object grounding | `request_id`, `prompt`, `image_jpeg_b64`, `max_image_side`, `return_debug_overlay` | `found`, `bbox_2d` (normalized `[0, 1000]`), `label`, `confidence`, `raw_output`, `latency_s`, `debug_overlay_jpeg_b64` |
| `POST /describe` | Scene description | `request_id`, `image_jpeg_b64`, `prompt`, `max_image_side` | `description`, `latency_s` |
| `POST /detect_objects` | Multi-object detection | `request_id`, `image_jpeg_b64`, `max_image_side`, `max_objects`, `min_confidence` | `objects[]` with `label` + `bbox_2d` (**pixel** coords of the inference image) + `confidence`, `raw_output`, `latency_s` |

Coordinate conventions:
- `/ground` returns `bbox_2d` in Qwen normalized `[0, 1000]` space. The robot-side projection service converts to pixels using camera intrinsics.
- `/detect_objects` returns `bbox_2d` in **pixel coordinates of the (possibly resized) inference image**. The semantic map caller converts separately if it needs normalized space.

Image handling:
- All image inputs are JPEG-encoded, base64-encoded, sent as `image_jpeg_b64`.
- `max_image_side` triggers a PIL LANCZOS thumbnail if the longer side exceeds the limit. Zero or negative disables resizing.
- `GROUNDING_MAX_IMAGE_MP` (default 20 MP) is the hard guard; images above it return HTTP 400.
- `GROUNDING_INFERENCE_TIMEOUT` (default 30 s) is the per-request wall-clock cap; breaches return HTTP 504.

Interactive API docs at `http://localhost:8100/docs` (Swagger UI). Import [`source/SImToRealLab.postman_collection.json`](../SImToRealLab.postman_collection.json) for pre-built requests.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROUNDING_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | HuggingFace model name or local path |
| `GROUNDING_DEVICE_MAP` | `auto` | PyTorch `device_map` |
| `GROUNDING_TORCH_DTYPE` | `auto` | Torch dtype (`auto` / `float16` / `bfloat16`) |
| `GROUNDING_LOAD_4BIT` | `0` | Set `1` for 4-bit quantisation |
| `GROUNDING_MAX_TOKENS` | `128` | Max new tokens for `/ground` and `/describe` |
| `GROUNDING_DETECT_MAX_TOKENS` | `512` | Max new tokens for `/detect_objects` (multi-object output is longer) |
| `GROUNDING_MAX_IMAGE_MP` | `20` | Reject images exceeding this megapixel budget |
| `GROUNDING_INFERENCE_TIMEOUT` | `30` | Per-request inference timeout in seconds (0 = disable) |
| `GROUNDING_HOST` | `0.0.0.0` | Bind host (uvicorn CLI overrides) |
| `GROUNDING_PORT` | `8100` | Bind port (uvicorn CLI overrides) |

### Dataset format (training)

`dataset_io.py` supports flat and chat JSONL. All `bbox_2d` values are in normalized `[0, 1000]` (the coordinate space Qwen2.5-VL was trained with).

Flat format:
```jsonl
{"image": "images/frame_0001.jpg", "prompt": "the chair next to the couch", "target": {"found": true, "bbox_2d": [120, 340, 580, 890], "label": "chair"}}
{"image": "images/frame_0002.jpg", "prompt": "a dog", "target": {"found": false}}
```

Image paths are resolved relative to the JSONL file's directory. Prompts without a `Locate:` prefix are auto-normalized.

## Install

### Base package

```bash
pip install -e source/strafer_vlm
```

Base install only depends on `numpy` and `pillow` — no GPU or service deps. Useful for importing parsers, dataset loaders, and type stubs from plain Python environments.

### Optional extras

```bash
pip install -e 'source/strafer_vlm[qwen]'     # Qwen inference: torch, transformers, accelerate, peft, trl, datasets
pip install -e 'source/strafer_vlm[live]'     # Live camera tool: opencv-python
pip install -e 'source/strafer_vlm[service]'  # HTTP service: fastapi, uvicorn
pip install -e 'source/strafer_vlm[dev]'      # Tests: pytest, httpx
```

Typical DGX install:

```bash
pip install -e "source/strafer_vlm[qwen,service,dev]"
```

### Critical on DGX Spark: NVRTC fix for Blackwell `sm_121`

PyTorch cu128 bundles NVRTC from CUDA 12.8, which does not support the Blackwell GB10's `sm_121` compute capability. JIT-compiled CUDA kernels fail silently. Replace the bundled NVRTC with the system's CUDA 13.0 version after venv creation:

```bash
NVRTC_DIR=".venv_vlm/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
mv "$NVRTC_DIR/libnvrtc.so.12" "$NVRTC_DIR/libnvrtc.so.12.bak"
mv "$NVRTC_DIR/libnvrtc-builtins.so.12.8" "$NVRTC_DIR/libnvrtc-builtins.so.12.8.bak"
ln -s /usr/local/cuda-13.0/lib64/libnvrtc.so.13.0.88 "$NVRTC_DIR/libnvrtc.so.12"
ln -s /usr/local/cuda-13.0/lib64/libnvrtc-builtins.so.13.0.88 "$NVRTC_DIR/libnvrtc-builtins.so.12.8"
```

Must be redone if `nvidia-cuda-nvrtc` is upgraded or the venv is recreated. The repo's `make check-nvrtc` target verifies the symlinks; `make serve-vlm` runs it as a prerequisite.

## Run

### Grounding service

```bash
uvicorn strafer_vlm.service.app:create_app --factory --host 0.0.0.0 --port 8100
```

Or via the repo root Makefile (runs the NVRTC check first):

```bash
make serve-vlm
```

The model downloads on first run (~7 GB, cached to `~/.cache/huggingface/`). Subsequent starts are fast.

### Verify

```bash
curl http://localhost:8100/health
# {"status":"ok","model_loaded":true,"model_name":"Qwen/Qwen2.5-VL-3B-Instruct"}

IMAGE_B64=$(base64 -w0 photo.jpg)
curl -s http://localhost:8100/ground \
  -H "Content-Type: application/json" \
  -d "{\"request_id\":\"t\",\"prompt\":\"Locate: door\",\"image_jpeg_b64\":\"$IMAGE_B64\"}"
```

PowerShell equivalent:

```powershell
$bytes = [System.IO.File]::ReadAllBytes("photo.jpg")
$b64   = [Convert]::ToBase64String($bytes)
$body  = @{ request_id="t"; prompt="Locate: door"; image_jpeg_b64=$b64 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8100/ground -Method Post -Body $body -ContentType "application/json"
```

### Operator CLI tools

```bash
# Single-image smoke test (no service running):
python -m strafer_vlm.tools.test_qwen25vl_grounding \
    --image docs/artifacts/strafer_top.jpeg \
    --prompt "the robot chassis"

# Live camera/video grounding with runtime console prompts:
python -m strafer_vlm.tools.live_qwen25vl_grounding --source 0 --prompt "Locate: the robot chassis"
```

In live mode, typing any non-empty line swaps the active prompt; `status`, `clear`, `quit` are reserved.

### LoRA fine-tuning

```bash
python -m strafer_vlm.training.train_qwen25vl_lora \
    --train-dataset data/vlm/train.jsonl \
    --eval-dataset data/vlm/val.jsonl \
    --output-dir outputs/qwen25vl_lora_run1
```

Key flags:

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-VL-3B-Instruct` | Base checkpoint |
| `--train-dataset` | *(required)* | Path to training JSONL |
| `--eval-dataset` | `None` | Validation JSONL (auto-splits from train if omitted) |
| `--eval-split` | `0.1` | Auto-split fraction when no eval dataset supplied |
| `--output-dir` | *(required)* | Adapter output directory |
| `--load-4bit` | `false` | QLoRA 4-bit training |

Evaluate a checkpoint:

```bash
python -m strafer_vlm.training.eval_qwen25vl_grounding \
    --dataset data/vlm/val.jsonl \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --adapter outputs/qwen25vl_lora_run1 \
    --output-dir outputs/qwen25vl_eval_run1
```

Produces per-example predictions with IoU scores and an `evaluation_summary.json`.

The primary data prep for this training path lives in [`strafer_lab.scripts.prepare_vlm_finetune_data`](../strafer_lab/scripts/prepare_vlm_finetune_data.py) — it emits negatives, multi-object examples, and description-preservation examples that the minimal exporter in `strafer_lab.tools.dataset_export` does not.

## Design

**Narrow package boundary.** `strafer_vlm` does image-space grounding, description, and detection. It does not consume depth, publish `/strafer/goal`, emit map-frame poses, decide reachability, or choose navigation backends. All of that is robot-local and lives in [`strafer_ros`](../strafer_ros/README.md).

**One loaded model, three endpoints.** `/ground`, `/describe`, and `/detect_objects` all share the same Qwen2.5-VL model loaded once at service startup. Each endpoint runs the same `run_grounding_generation()` function with a different system prompt. This keeps GPU memory consumption flat regardless of endpoint mix.

**Single-threaded inference pool.** All three endpoints enqueue onto a shared `ThreadPoolExecutor(max_workers=1)`, so concurrent requests queue rather than fighting for the GPU. Latency accumulates under load; throughput is bounded by one inference at a time per service instance.

**Bbox convention is normalized `[0, 1000]` for grounding, pixel coords for detection.** `/ground` stays normalized so the downstream projection service (which holds camera intrinsics) can convert to pixels exactly once. `/detect_objects` returns pixel coordinates of the inference image because the semantic map caller does not round-trip through the projection service.

**The fine-tune target is the 3B model; the description generation pipeline in `strafer_lab` uses the 7B model.** Feeding the fine-tune target's own outputs back as training data causes collapse. See [`strafer_lab.scripts.generate_descriptions`](../strafer_lab/scripts/generate_descriptions.py) for the 7B description pipeline.

**`detect_objects()` is additive, not on the `GroundingClient` protocol.** Existing protocol implementations stay valid. Callers in `strafer_autonomy` use `hasattr(client, "detect_objects")` to detect the richer endpoint.

## Testing

```bash
python -m pytest source/strafer_vlm/tests/ -v

# Skip GPU-backed tests (default in CI):
python -m pytest source/strafer_vlm/tests/ -m "not gpu" -v
```

Tests marked `@pytest.mark.gpu` require a CUDA device and loaded model. Everything else — parsing, dataset I/O, service request/response shaping (mocked) — runs without a GPU. All 200+ tests pass without any service running; the service endpoint tests use an `autouse` fixture that sets `GROUNDING_MODEL=/nonexistent` to prevent model download during tests.

## Deferred / known limitations

Tracked in [`docs/DEFERRED_WORK.md`](../../docs/DEFERRED_WORK.md). Items currently open:

- No containerization / deployment automation — services run directly in a venv.
- Postman collection covers `/health` and `/ground` only; `/describe` and `/detect_objects` are missing.
- Grounding latency ~2-3 s per image, ~1.5-4 s for multi-object. `scan_for_target` runs 6 grounding calls per rotation (~18 s) before navigation; latency budget is acceptable for MVP but adds up.

## References

- [`source/strafer_autonomy/README.md`](../strafer_autonomy/README.md) — executor-side `HttpGroundingClient`, schemas, `/plan_with_grounding` agentic endpoint.
- [`source/strafer_lab/README.md`](../strafer_lab/README.md) — fine-tune dataset prep pipeline (description generation, SFT data prep).
- [`docs/STRAFER_AUTONOMY_NEXT.md`](../../docs/STRAFER_AUTONOMY_NEXT.md) — design rationale for the `/detect_objects` endpoint and semantic-map integration.
- [`docs/VALIDATE_ISAAC_SIM_AND_INFINIGEN.md`](../../docs/VALIDATE_ISAAC_SIM_AND_INFINIGEN.md) — DGX-side install and smoke-test runbook (covers NVRTC).
- [`docs/DEFERRED_WORK.md`](../../docs/DEFERRED_WORK.md) — open items.
