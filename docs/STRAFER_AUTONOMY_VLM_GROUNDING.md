# Strafer Autonomy VLM Grounding

This document is the source of truth for the `strafer_vlm` package and its role in the autonomy stack.

## Scope

This document covers:

- the current `source/strafer_vlm` implementation on the Windows workstation
- the grounding contract between `strafer_autonomy` and `strafer_vlm`
- the target repository shape for turning the current Qwen tooling into a workstation or cloud grounding service
- the staged work required to move from offline evaluation to autonomy-compatible deployment

This document does not cover:

- Jetson-side ROS package ownership
- robot-local depth projection or reachability logic
- mission planning or natural-language orchestration
- cloud infrastructure in detail

Those live in:

- `docs/STRAFER_AUTONOMY_ROS.md`
- `docs/STRAFER_AUTONOMY_INTERFACES.md`
- `docs/STRAFER_AUTONOMY_DEPLOYMENT_MODES.md`

## Core Decision

`strafer_vlm` is a semantic grounding package, not a ROS package and not an autonomy package.

It owns:

- Qwen2.5-VL grounding inference
- grounding prompt formatting
- grounding dataset loading and serialization
- grounding training and evaluation routines
- workstation or cloud VLM service hosting

It does not own:

- mission plans
- user-command parsing
- robot-local depth usage
- TF transforms
- goal-pose generation
- navigation execution

The key split is:

- `strafer_autonomy` decides when grounding is needed
- `strafer_vlm` answers "where is the named target in this RGB image?"
- `strafer_ros` turns that grounding result into a robot-usable target and motion

## Current Implementation

The current package already works for workstation-first development on Windows with a GPU-connected camera.

Current package root:

```text
source/strafer_vlm/
  pyproject.toml
  README.md
  strafer_vlm/
    __init__.py
    qwen_vl_common.py
    test_qwen25vl_grounding.py
    live_qwen25vl_grounding.py
    train_qwen25vl_lora.py
    eval_qwen25vl_grounding.py
```

Current entry points:

| Entry point | Purpose | Current role |
|---|---|---|
| `python -m strafer_vlm.test_qwen25vl_grounding` | single-image grounding smoke test | working |
| `python -m strafer_vlm.live_qwen25vl_grounding` | live camera/video grounding with runtime prompt updates | working |
| `python -m strafer_vlm.train_qwen25vl_lora` | LoRA fine-tuning pipeline | working |
| `python -m strafer_vlm.eval_qwen25vl_grounding` | offline grounding evaluation and metrics | working |
| `strafer_vlm.qwen_vl_common` | shared model loading, prompt handling, parsing, bbox normalization, dataset helpers | working |

Current package strengths:

- base-model evaluation on still images
- live grounding evaluation on Windows video sources
- LoRA fine-tuning on workstation GPU
- offline grounding metrics and prediction artifacts
- corrected bbox normalization and overlay handling

Current package limitations:

- no LAN HTTP service yet
- no stable `strafer_autonomy.clients.vlm_client` transport implementation yet
- no robot-local depth projection
- no ROS node or ROS interface implementation

## Current Grounding Contract

The current grounding behavior is already narrow enough to map cleanly into the autonomy design.

### Input

- one RGB image
- one grounding prompt

Prompt normalization is already standardized in `qwen_vl_common.normalize_prompt()`:

- raw prompt example: `door`
- normalized prompt example: `Locate: door`

### Output

The current Qwen system prompt is already constrained to JSON grounding output:

```json
{
  "found": true,
  "bbox_2d": [234, 156, 456, 378],
  "label": "door",
  "confidence": 0.92
}
```

Semantics:

- `found`
  - `true` means the model claims the target is visible
  - `false` is a valid result, not a transport error
- `bbox_2d`
  - the package boundary normalizes boxes to `[0, 1000]`
  - internal parsing may accept either pixel or normalized model output, then normalize before returning
- `label`
  - optional model-provided object label
- `confidence`
  - optional model-provided score in `[0, 1]`

### Dataset format

The current package supports both:

- flat JSON or JSONL
- chat-style JSON or JSONL

The stable training/eval target format remains:

```json
{
  "found": true,
  "bbox_2d": [230, 112, 510, 620],
  "label": "door",
  "confidence": 0.92
}
```

This is compatible with the autonomy-layer `GroundingResult` semantics.

## Autonomy-Aligned Runtime Role

The chosen MVP runtime is:

- Jetson: `strafer_autonomy.executor`
- Jetson: `strafer_ros`
- Windows workstation: planner service
- Windows workstation: `strafer_vlm` grounding service

That means `strafer_vlm` is remote from the executor's point of view.

```text
Jetson executor
  -> ros_client.capture_scene_observation()
  -> vlm_client.locate_semantic_target() over LAN
  -> GroundingResult
  -> ros_client.project_detection_to_goal_pose()
  -> ros_client.navigate_to_pose()
```

This is the critical boundary:

- `strafer_vlm` returns image-space grounding only
- `strafer_ros` converts grounding into robot-space meaning

`strafer_vlm` should not:

- consume depth
- publish `/strafer/goal`
- emit map-frame poses
- decide reachability
- choose navigation backends

Those are robot-local concerns.

## Canonical Service Boundary

The source of truth for the logical request and response types remains:

```text
source/strafer_autonomy/strafer_autonomy/schemas/
```

`strafer_vlm` should implement that contract, not redefine it at the mission level.

The first service shape should remain:

- `POST /ground`
- `GET /health`

### `POST /ground`

Request body:

```json
{
  "request_id": "req_123",
  "prompt": "Locate: the door",
  "image_jpeg_b64": "...",
  "image_stamp_sec": 1710000000.25,
  "max_image_side": 1024,
  "return_debug_overlay": false
}
```

Response body:

```json
{
  "request_id": "req_123",
  "found": true,
  "bbox_2d": [230, 112, 510, 620],
  "label": "door",
  "confidence": 0.92,
  "raw_output": "{\"found\":true,\"bbox_2d\":[230,112,510,620],\"label\":\"door\"}",
  "latency_s": 2.84,
  "debug_artifact_path": null
}
```

Coordinate convention:

- `bbox_2d` values are in the Qwen normalized `[0, 1000]` coordinate space, not pixel coordinates
- the robot-side projection service (`ProjectDetectionToGoalPose`) is responsible for converting to pixel space using camera intrinsics before depth lookup
- this convention is shared across the `POST /ground` response, `GroundingResult` in `strafer_autonomy.schemas`, and the projection service request

Transport expectations:

- image crosses the network as compressed payload, not a raw NumPy object
- `found=false` is a valid grounding response
- parse failure is a failed service result
- network failure is separate from model failure

## Repository Structure Plan

The current flat package layout was the right first move for getting Qwen working quickly on Windows.

It is not the right final layout for the autonomy-compatible service boundary.

### Current structure

```text
source/strafer_vlm/strafer_vlm/
  qwen_vl_common.py
  test_qwen25vl_grounding.py
  live_qwen25vl_grounding.py
  train_qwen25vl_lora.py
  eval_qwen25vl_grounding.py
```

### Target structure

The next refactor should move toward:

```text
source/strafer_vlm/
  pyproject.toml
  README.md
  strafer_vlm/
    __init__.py
    inference/
      qwen_runtime.py
      qwen_vl_common.py
      parsing.py
    service/
      app.py
      handlers.py
      payloads.py
    training/
      train_qwen25vl_lora.py
      eval_qwen25vl_grounding.py
      dataset_io.py
    tools/
      test_qwen25vl_grounding.py
      live_qwen25vl_grounding.py
```

The intent of this structure is:

- `inference/`
  - reusable runtime code for Qwen loading and one-shot grounding
- `service/`
  - workstation or cloud HTTP service surface
- `training/`
  - dataset, fine-tuning, and offline evaluation code
- `tools/`
  - operator-facing scripts for workstation development

Important constraint:

- `strafer_autonomy` remains the source of truth for mission-layer schemas
- `strafer_vlm` should only mirror the HTTP payload fields it needs to serve the grounding contract

## Execution Model Support

`strafer_vlm` is intentionally independent of the robot-local motion backend.

It must work unchanged whether the robot later uses:

- `nav2`
- `strafer_direct`
- `hybrid_nav2_strafer`

That is why `strafer_vlm` stops at image-space grounding.

If `strafer_vlm` starts producing map-frame goals or backend-specific targets, it will collapse the clean boundary between:

- semantic perception
- robot-local projection
- motion execution

That would be the wrong design.

## Staged Work

### Stage 1: Workstation-first grounding tools

Status:

- already in progress
- mostly implemented today

Deliverables:

- single-image smoke test
- live camera/video grounding
- LoRA training
- offline evaluation

### Stage 2: Reusable grounding runtime

Goal:

- separate reusable Qwen grounding runtime code from CLI-only scripts

Deliverables:

- reusable runtime module for model load + inference
- clean parsing and payload helpers
- service-friendly function boundary

### Stage 3: Workstation grounding service

Goal:

- expose the current grounding runtime through the autonomy-aligned LAN API

Deliverables:

- `POST /ground`
- `GET /health`
- base64 or equivalent compressed image transport
- debug overlay artifact support

### Stage 4: Jetson executor integration

Goal:

- make `strafer_autonomy.clients.vlm_client` call the workstation service

Deliverables:

- LAN HTTP client
- failure handling
- timeout handling
- health checking and error codes

### Stage 5: Hosted grounding

Goal:

- move the same grounding service boundary to AWS or Databricks if needed

Deliverables:

- same `vlm_client` contract
- different service host
- no change to mission semantics

### Stage 6: Domain adaptation

Goal:

- improve grounding quality only if zero-shot performance is insufficient

Deliverables:

- curated grounding dataset
- LoRA checkpoints
- evaluation reports against the same grounding contract

## Explicit Non-Goals

This package should not become:

- a ROS node bundle
- a target-projection package
- a mission planner
- a web or mobile API gateway
- a robot execution backend

If any of those start accumulating inside `strafer_vlm`, the package boundary has slipped.

## Immediate Next Work

1. Refactor the current Qwen runtime into reusable service-friendly modules.
2. Implement the workstation-side `POST /ground` and `GET /health` service.
3. Implement the LAN HTTP `strafer_autonomy.clients.vlm_client`.
4. Keep training and evaluation code inside `source/strafer_vlm`, but separate it from service runtime code.
5. Leave depth projection and reachability logic on the Jetson in `strafer_ros`.
