# strafer_vlm — Remaining Tasks

This document tracks remaining work for `source/strafer_vlm` against the project goals
defined in the autonomy docs. Items are ordered by priority within each section.

Status key: **REQUIRED** = blocks MVP integration, **RECOMMENDED** = quality/maintainability,
**DEFERRED** = post-MVP or when needed.

---

## 1. Repository Structure Refactor

The current flat layout (`qwen_vl_common.py` + CLI scripts in package root) should be
reorganised into the layered structure defined in `docs/STRAFER_AUTONOMY_VLM_GROUNDING.md`.

| # | Task | Priority | Notes |
|---|------|----------|-------|
| 1.1 | Create `inference/` subpackage | RECOMMENDED | Extract model loading, generation, and parsing from `qwen_vl_common.py` into `inference/qwen_runtime.py` and `inference/parsing.py`. Keep `qwen_vl_common.py` as a re-export shim for backwards compatibility. |
| 1.2 | Create `training/` subpackage | RECOMMENDED | Move `train_qwen25vl_lora.py` and `eval_qwen25vl_grounding.py`. Extract dataset I/O helpers into `training/dataset_io.py`. |
| 1.3 | Create `tools/` subpackage | RECOMMENDED | Move `test_qwen25vl_grounding.py` and `live_qwen25vl_grounding.py`. These are operator CLI tools, not library code. |
| 1.4 | Split `service/app.py` into `payloads.py` and `handlers.py` | RECOMMENDED | Move Pydantic request/response models to `service/payloads.py`. Move endpoint logic to `service/handlers.py`. Keep `app.py` as the factory + lifespan only. |
| 1.5 | Update `__init__.py` exports | RECOMMENDED | Export key public types (`GroundingTarget`, `GroundingExample`, `load_qwen_model_and_processor`, `run_grounding_generation`) and add `__version__`. |
| 1.6 | Update `pyproject.toml` entry points | RECOMMENDED | After moving scripts, update `[project.scripts]` if CLI entry points are added. |

Target layout:

```text
strafer_vlm/
  __init__.py
  inference/
    __init__.py
    qwen_runtime.py        # model load + generation
    parsing.py             # JSON extraction, bbox coercion, normalisation
  service/
    __init__.py
    app.py                 # factory + lifespan
    handlers.py            # endpoint implementations
    payloads.py            # Pydantic request/response models
  training/
    __init__.py
    train_qwen25vl_lora.py
    eval_qwen25vl_grounding.py
    dataset_io.py          # dataset loading/serialisation helpers
  tools/
    __init__.py
    test_qwen25vl_grounding.py   # single-image smoke test CLI
    live_qwen25vl_grounding.py   # live video grounding CLI
  qwen_vl_common.py       # re-export shim (backwards compat)
```

---

## 2. Tests

No pytest-based tests exist today. The current `test_qwen25vl_grounding.py` is a CLI tool,
not an automated test.

| # | Task | Priority | Notes |
|---|------|----------|-------|
| 2.1 | ~~Add `tests/test_parsing.py` — unit tests for JSON extraction, bbox coercion, normalisation, coordinate conversion~~ | ✅ DONE | 57 tests covering `extract_first_json_object`, `_coerce_*`, `parse_grounding_*`, bbox normalisation, coordinate conversion, IoU. |
| 2.2 | ~~Add `tests/test_prompt.py` — unit tests for `normalize_prompt`, `serialize_target`~~ | ✅ DONE | 14 tests covering prompt normalisation, target serialisation, `bbox_iou`. |
| 2.3 | ~~Add `tests/test_dataset_io.py` — unit tests for dataset loading (flat + chat format)~~ | ✅ DONE | 13 tests covering flat JSONL/JSON, chat format, alternate field names, and error cases. |
| 2.4 | ~~Add `tests/test_service_endpoints.py` — FastAPI `TestClient` tests for `/health` and `/ground`~~ | ✅ DONE | 8 tests with mocked model: health, prompt validation (400), invalid image (400), missing fields (422), found/not-found/unparseable inference. |
| 2.5 | ~~Add `tests/conftest.py` with shared fixtures~~ | ✅ DONE | `tiny_jpeg_b64`, `tiny_jpeg_path`, `mock_vlm_client` fixtures shared across all test modules. |
| 2.6 | ~~Add pytest config to `pyproject.toml`~~ | ✅ DONE | `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and `gpu` marker. `[dev]` extras group added. |
| 2.7 | ~~Verify tests pass in CI-like conditions (no GPU)~~ | ✅ DONE | All 115 unit tests run without torch/GPU. Integration tests use `@pytest.mark.gpu`. |

---

## 3. Service Hardening

The service works end-to-end but needs production-readiness improvements.

| # | Task | Priority | Notes |
|---|------|----------|-------|
| 3.1 | ~~Add prompt validation — reject empty or whitespace-only prompts with 400~~ | ✅ DONE | Returns HTTP 400 with `"prompt must be a non-empty string."` for empty/whitespace prompts. |
| 3.2 | ~~Add image size guard — reject payloads where decoded image exceeds a configurable max (e.g. 20 MP)~~ | ✅ DONE | `GROUNDING_MAX_IMAGE_MP` env var (default 20). Returns HTTP 400 with MP details. |
| 3.3 | ~~Add inference timeout — wrap `run_grounding_generation` with a timeout (configurable, default 30s)~~ | ✅ DONE | `GROUNDING_INFERENCE_TIMEOUT` env var (default 30s, 0=no limit). Returns HTTP 504 on timeout. Inference runs in a thread pool with `asyncio.wait_for`. |
| 3.4 | ~~Add graceful model-load failure — catch exceptions during lifespan, set `_state.ready = False`, return 503 on all requests instead of crashing~~ | ✅ DONE | Lifespan wraps model load in try/except, logs traceback, sets ready=False. |
| 3.5 | ~~Add request-id logging — log `request_id` with every `/ground` request for traceability~~ | ✅ DONE | Logs `[request_id] /ground prompt=...` on entry and `[request_id] found=... bbox=... latency=...` on exit. |
| 3.6 | ~~Add `debug_overlay_jpeg_b64` support — when `return_debug_overlay=true`, generate bbox overlay and return as base64 JPEG~~ | ✅ DONE | Returns overlay image inline as base64 JPEG in `debug_overlay_jpeg_b64`. Only populated when `found=true` and overlay requested. Replaces the old `debug_artifact_path` design. |

---

## 4. Documentation

| # | Task | Priority | Notes |
|---|------|----------|-------|
| 4.1 | ~~Update `README.md` — add service launch instructions, env var table, API examples (request/response JSON), and Postman/curl usage~~ | ✅ DONE | Full service section with launch command, env var table, request/response examples, curl + PowerShell snippets, Swagger UI note. |
| 4.2 | ~~Add docstrings to `service/app.py` Pydantic models~~ | ✅ DONE | All fields have `Field(description=...)` with coordinate convention, encoding, and semantic notes. |
| 4.3 | ~~Add OpenAPI metadata — `summary` and `description` on each endpoint for auto-generated docs at `/docs`~~ | ✅ DONE | App has `description=`, endpoints have `summary=`. Swagger UI at `/docs` is now informative. |

---

## 5. Client Alignment (in `strafer_autonomy`)

These live in `source/strafer_autonomy` but are tracked here because they complete the
VLM integration story.

| # | Task | Priority | Notes |
|---|------|----------|-------|
| 5.1 | ~~Add timeout + retry to `HttpGroundingClient.locate_semantic_target`~~ | ✅ DONE | `max_retries` (default 2) and `retry_backoff_s` (default 0.5) on config. Retries on ConnectionError/Timeout/5xx with exponential backoff. |
| 5.2 | ~~Add `HttpGroundingClient.health()` call to executor startup~~ | ✅ DONE | `build_command_server()` probes grounding client `.health()` before creating the runner. Raises `GroundingServiceUnavailable` if unreachable or model not loaded. Skippable via `check_vlm_health=False`. |
| 5.3 | ~~Add connection error handling — raise a domain-specific exception (e.g. `GroundingServiceUnavailable`) instead of raw `requests` exceptions~~ | ✅ DONE | `GroundingServiceUnavailable` raised on all retry exhaustion and client HTTP errors. Exported from `clients/__init__.py`. |

---

## 6. Deferred / Post-MVP

These are called out in the docs but are not needed for initial autonomy integration.

| # | Task | Priority | Notes |
|---|------|----------|-------|
| 6.1 | Hosted deployment (AWS / Databricks) | DEFERRED | Same service contract, different host. Stage 5 in VLM grounding doc. |
| 6.2 | Domain adaptation fine-tuning pipeline | DEFERRED | Stage 6 in VLM grounding doc. Training infra already exists. |
| 6.3 | LoRA adapter hot-swap at runtime | DEFERRED | Load different adapters without restarting the service. |
| 6.4 | Multi-model ensemble / fallback | DEFERRED | Run multiple VLM backends behind the same endpoint. |
| 6.5 | Structured logging + metrics (Prometheus, OpenTelemetry) | DEFERRED | Add when operating as a long-running production service. |
| 6.6 | Rate limiting / request queuing | DEFERRED | Add when multiple Jetson clients share one workstation. |
