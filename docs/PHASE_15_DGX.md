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
# Planner + VLM services
source/strafer_vlm/                                          # VLM service, inference, training
source/strafer_autonomy/strafer_autonomy/planner/            # planner app, intent parser, plan compiler, LLM runtime
source/strafer_autonomy/strafer_autonomy/clients/planner_client.py   # HttpPlannerClient
source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py       # HttpGroundingClient
source/strafer_autonomy/strafer_autonomy/clients/databricks_planner_client.py  # NEW
source/strafer_autonomy/strafer_autonomy/clients/databricks_vlm_client.py      # NEW
source/strafer_autonomy/tests/test_planner_*.py
source/strafer_autonomy/tests/test_intent_parser.py
source/strafer_autonomy/tests/test_plan_compiler.py

# Synthetic data — batch processing scripts (NOT Isaac Sim runtime)
source/strafer_lab/scripts/prep_room_usds.py                 # Infinigen → USD pipeline
source/strafer_lab/scripts/extract_scene_metadata.py         # NEW — metadata extraction
source/strafer_lab/scripts/generate_descriptions.py          # NEW — 4-stage description pipeline
source/strafer_lab/scripts/finetune_clip.py                  # NEW — CLIP fine-tuning
source/strafer_lab/scripts/prepare_vlm_finetune_data.py      # NEW — VLM SFT data prep
source/strafer_lab/strafer_lab/tools/scene_labels.py         # NEW — runtime label accessor
source/strafer_lab/strafer_lab/tools/spatial_description.py  # NEW — SpatialDescriptionBuilder
source/strafer_lab/strafer_lab/tools/dataset_export.py       # NEW — export to CLIP CSV + VLM JSONL
```

**Shared file protocol:** The Jetson agent owns `mission_runner.py` and the
`_execute_step` dispatch. If you add a new intent type or skill, you provide
only the compiler function and parser change — the Jetson agent wires the
executor side. `source/strafer_lab/strafer_lab/tools/__init__.py` is created
by the Isaac Sim host agent (Task 3) — your tools import into the existing file.

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

   **Coordination:** This file is owned by the Jetson agent. Add ONLY the
   `targets` field — do not modify other fields or classes. Coordinate with
   the Jetson agent to avoid merge conflicts on this file.

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

## Synthetic data tasks (DGX batch processing)

These tasks run on the DGX Spark and do NOT require Isaac Sim. They
process scene data, generate descriptions, and run model fine-tuning.
The Isaac Sim host (Windows or DGX) collects raw perception data (Tasks
in `PHASE_15_WINDOWS.md`); these tasks consume that data.

### Task 7: Infinigen high-quality scene generation (Section 5.3)

**Priority:** High | **Effort:** Medium

**Files to modify:**

```
source/strafer_lab/scripts/prep_room_usds.py
```

**What to implement:**

Generate Infinigen scenes at higher quality using the DGX Spark's 128GB
unified memory (vs. the Windows workstation's 16GB VRAM limit). Update
the generation config to produce scenes with:
- Higher polygon counts and more detailed meshes
- More diverse room types (Kitchen, Bedroom, LivingRoom, Hallway, Office)
- More objects per room (richer scenes for perception training)
- Multiple rooms per scene (multi-room layouts with connectivity)

The existing `prep_room_usds.py` handles Infinigen → USD conversion. If
Infinigen must run in Blender on DGX, verify Blender headless works on
ARM64 (Grace). If not, generate scenes on Windows and transfer USDs to
DGX for metadata extraction.

---

### Task 8: Scene metadata extraction (Section 5.3)

**Priority:** High | **Effort:** Medium

**New files to create:**

```
source/strafer_lab/scripts/extract_scene_metadata.py
source/strafer_lab/strafer_lab/tools/scene_labels.py
```

**What to implement:**

1. **Metadata extraction script.** Hook into Infinigen's Blender Python
   state during generation (or as a post-processing step on `.blend` files)
   to serialize the rich scene metadata that Infinigen tracks internally:

   - Per-room: `room_type` (from `Semantics` enum — Kitchen, Hallway, etc.),
     `footprint_xy` (from `shapely.Polygon`), `area_m2`, `story`
   - Per-object: `label`, `semantic_tags` (from Infinigen's hierarchical
     tag system — Furniture/Seating/Chair), `instance_id`, `position_3d`,
     `bbox_3d`, `room_idx`, `relations` (SupportedBy, StableAgainst,
     CoPlanar), `materials`
   - Room adjacency graph from `RoomGraph`
   - Spawn points (existing, from `scenes_metadata.json` `spawn_points_xy`)

   Output: `scene_metadata.json` per scene in `Assets/generated/scenes/`.

2. **USD prim labeling.** Write `semanticLabel` and `instanceId` as USD
   prim attributes so Replicator annotators produce labeled bboxes:
   ```python
   from pxr import Usd, Sdf
   stage = Usd.Stage.Open(scene_usd_path)
   for prim in stage.Traverse():
       label = metadata_lookup(prim.GetName())
       if label:
           prim.CreateAttribute("semanticLabel", Sdf.ValueTypeNames.String).Set(label)
           prim.CreateAttribute("instanceId", Sdf.ValueTypeNames.Int).Set(instance_id)
   stage.Save()
   ```

3. **Runtime label accessor** (`scene_labels.py`):
   ```python
   def get_scene_metadata(metadata_path: Path, scene_name: str) -> dict:
       """Read full scene metadata including rooms, objects, relations."""

   def get_scene_label_set(metadata_path: Path, scene_name: str) -> set[str]:
       """Return unique object labels in a scene."""

   def get_room_at_position(metadata: dict, xy: tuple[float, float]) -> dict | None:
       """Point-in-polygon lookup: which room contains this XY position?"""
   ```

**Key insight:** Infinigen's richest metadata (room types, spatial relations,
semantic tags) lives in the Blender Python `State` object during generation
and is NOT written to `saved_mesh.json` or USD by default. This task must
serialize that data.

---

### Task 9: Scene description pipeline — Stages 1-3 + spot-check (Section 5.6.2)

**Priority:** High | **Effort:** Medium

**New files to create:**

```
source/strafer_lab/strafer_lab/tools/spatial_description.py   # Stage 1
source/strafer_lab/scripts/generate_descriptions.py           # Stages 1-4 batch runner
```

**What to implement:**

A 4-stage batch pipeline that produces rich text descriptions for every
frame captured during teleop data collection. Runs after data collection
as a batch job on DGX Spark.

**Stage 1: Programmatic spatial analysis** (`spatial_description.py`)

```python
from shapely.geometry import Point, Polygon

class SpatialDescriptionBuilder:
    """Compute spatial relations from simulation ground truth."""

    def __init__(self, scene_metadata: dict):
        self.rooms = [
            {**r, "polygon": Polygon(r["footprint_xy"])}
            for r in scene_metadata["rooms"]
        ]
        self.objects = scene_metadata["objects"]

    def build(self, frame_data: dict) -> dict:
        robot_xy = frame_data["robot_pos"][:2]
        robot_yaw = quat_to_yaw(frame_data["robot_quat"])

        # Which room is the robot in? (point-in-polygon)
        current_room = None
        for room in self.rooms:
            if room["polygon"].contains(Point(robot_xy)):
                current_room = room
                break

        described_objects = []
        for obj in self._visible_objects(frame_data["bboxes"]):
            pos = np.array(obj["position_3d"][:2])
            dist = float(np.linalg.norm(pos - robot_xy))
            bearing = self._compute_bearing(pos, robot_xy, robot_yaw)
            region = self._classify_region(dist)

            described_objects.append({
                "label": obj["label"],
                "semantic_tags": obj.get("semantic_tags", []),
                "distance_m": round(dist, 1),
                "bearing": bearing,
                "region": region,
                "room_type": self.rooms[obj["room_idx"]]["room_type"]
                             if "room_idx" in obj else None,
                "relations": [
                    f'{r["type"]} {r["target"]}'
                    for r in obj.get("relations", []) if r["target"]
                ],
                "materials": obj.get("materials", []),
            })

        return {
            "robot_room_type": current_room["room_type"] if current_room else None,
            "visible_objects": described_objects,
        }
```

**Stage 2: VLM description generation** (Qwen2.5-VL-7B, standalone)

Load Qwen2.5-VL-7B as a standalone model via `transformers.AutoModelForVision2Seq`
in the `generate_descriptions.py` script. This is completely separate from the
`strafer_vlm` service package (port 8100, which runs the 3B model being
fine-tuned). Do NOT import from `strafer_vlm`. The 7B model fits easily on
128GB DGX alongside other workloads.

The VLM receives both the structured spatial facts JSON (from Stage 1) and
the actual RGB frame in a single prompt. This lets it produce descriptions
that are spatially accurate (from GT facts) AND visually grounded (lighting,
textures, occlusion, materials) — no separate LLM stage needed.

Prompt template:

```text
<image>
Given these spatial facts about a scene viewed from a ground robot's
camera (25cm height) and the image above, write 3 natural descriptions
at different levels of detail: brief (5-10 words), medium (15-25 words),
and detailed (30-50 words). Include spatial relationships from the facts
and visual details (lighting, textures, materials) from the image.
Each must only mention the objects listed in the facts.

Facts: {structured_json}
```

**Stage 3: Ground truth validation filter**

Cross-reference each description against scene_metadata.json.
Discard any description that mentions objects not in the scene's label set.
Log rejection rate for monitoring.

**CLI:**

```bash
python scripts/generate_descriptions.py \
  --perception-data data/perception/ \
  --scene-metadata Assets/generated/scenes/ \
  --output data/descriptions/ \
  --vlm-model Qwen/Qwen2.5-VL-7B-Instruct
```

Note on model loading:
- Load Qwen2.5-VL-7B directly via `transformers.AutoModelForVision2Seq`
  and `AutoProcessor` inside `generate_descriptions.py`. Do NOT import
  from `strafer_vlm` — the description pipeline must be completely
  independent of the 3B VLM service to avoid self-training contamination.

**Stage 4: Human spot-check** is manual and periodic — review 50 random
samples per batch of 1000 frames, score quality (1-5), flag systematic
errors, refine prompts.

---

### Task 10: Dataset export — CLIP CSV + VLM JSONL (Section 5.5.5, 5.6.4)

**Priority:** Medium | **Effort:** Small

**New file to create:**

```
source/strafer_lab/strafer_lab/tools/dataset_export.py
```

**What to implement:**

Convert perception data + descriptions into training-ready formats:

1. **CLIP image-text CSV** (for OpenCLIP fine-tuning):
   ```csv
   image_path,description
   episode_0001/frame_0010.jpg,"a hallway with a plant and red ball at the far end"
   episode_0001/frame_0010.jpg,"looking down a dimly lit hallway, table on the left"
   ```
   Multiple descriptions per image (3-5) at different detail levels.
   **Excludes ProcRoom frames** (filter on `"scene_type": "infinigen"`).

2. **VLM grounding JSONL** (for Qwen2.5-VL SFT):
   ```json
   {
     "image": "frame_0042.jpg",
     "conversations": [
       {"role": "user", "content": "<image>Locate the table in this image."},
       {"role": "assistant", "content": "<ref>table</ref><box>(321,450),(580,890)</box>"}
     ]
   }
   ```
   - Coordinates scaled to 0-1000 range
   - Negative examples (1:3 ratio) using `get_scene_label_set()` to query
     objects NOT in the frame
   - **Excludes ProcRoom frames**

**Depends on:** Tasks 8 (scene metadata) and 9 (descriptions)

---

### Task 11: CLIP image-text contrastive fine-tuning (Section 5.6.1)

**Priority:** High | **Effort:** Medium

**New file to create:**

```
source/strafer_lab/scripts/finetune_clip.py
```

**What to implement:**

- Load OpenCLIP ViT-B/32 with `open_clip.create_model_and_transforms`
- Standard CLIP contrastive loss (symmetric InfoNCE on image-text pairs)
- Both towers trained (NOT just vision tower — text alignment matters)
- Training: AdamW, lr=1e-5, weight_decay=0.01
- Export fine-tuned model to ONNX — **both towers separately**:
  - `torch.onnx.export(model.visual, ...)` → `clip_visual.onnx`
  - `torch.onnx.export(model.text, ...)` → `clip_text.onnx`
  - The Jetson's `clip_encoder.py` uses both `encode_image()` and
    `encode_text()` (for `query_by_text`), so both towers must be exported.
- Log experiments with MLflow
- CLI: `python scripts/finetune_clip.py --data data/clip_descriptions.csv --epochs 10 --output models/clip_finetuned/`

**Dataset:** 75k-250k (image, text) pairs from Task 10.

**Evaluation gate for arrival verification:** After Phase 1 (image-text)
training, evaluate the ranking-based verify_arrival logic on held-out sim
data: construct (arrival_frame, goal_area_observations, distractor_observations)
tuples and measure whether top-k retrieval correctly identifies the goal
region. If ranking accuracy (majority of top-k near goal) is below 85%,
proceed to Phase 2: add NT-Xent loss on (anchor, positive, negative) image
triples from same/different locations (optional, see design doc Section 5.6.1).

---

### Task 12: VLM grounding fine-tuning data prep (Section 5.7)

**Priority:** Medium | **Effort:** Medium

**New file to create:**

```
source/strafer_lab/scripts/prepare_vlm_finetune_data.py
```

**What to implement:**

Convert perception data + bboxes into Qwen2.5-VL SFT format:

1. **Single-object grounding examples** (primary):
   - Coordinates scaled to 0-1000 range
   - Negative examples (1:3 ratio)
   - Use `get_scene_label_set()` for negative query generation
   - **Exclude ProcRoom frames**

2. **Multi-object detection examples** (for `POST /detect_objects`):
   - Prompt: `"List all visible objects with their bounding boxes."`
   - Response: multiple `<ref>label</ref><box>(...)</box>` tags
   - Include frames with 2-10 visible objects
   - Aim for ~20% of training examples to be multi-object

3. **Scene description examples** (to preserve `POST /describe` quality):
   - Reuse Stage 2 descriptions from Task 9 as (image, description) SFT pairs
   - Include ~10% description examples in the training mix to prevent
     catastrophic forgetting of the describe capability during grounding SFT

**Fine-tuning method:** Use **LoRA** (rank 16-32) rather than full fine-tune
to preserve the pretrained model's general capabilities (scene description,
OCR, reasoning) while specializing grounding performance. This avoids
degrading `POST /describe` quality.

- Output: `.jsonl` ready for SFT on DGX

**Depends on:** Tasks 8 (scene labels) and Isaac Sim host tasks (perception data)

## Deferred tasks (not assigned to this phase)

These items from `STRAFER_AUTONOMY_NEXT.md` are explicitly deferred:

- **Parallel health checks** (Section 2.3) — Small effort but touches
  `executor/command_server.py` which is owned by the Jetson agent. Reassigned
  to Jetson workstream to avoid file ownership conflicts.
- **Agentic combined endpoint** (Section 4.3 item 3) — Medium effort, Low
  priority. Natural follow-up after Task 3 (agentic planner) is validated.
- **Databricks serving endpoints** (Section 4.5) — Infrastructure setup,
  depends on Tasks 5 and 6 being complete and tested.
- **Failure-to-sim feedback pipeline** (Section 5.8) — Large effort, Low
  priority. Depends on real-world failure logging (Jetson Task 10) and all
  synthetic data infrastructure being validated.

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
- `source/strafer_lab/scripts/collect_perception_data.py` — owned by Isaac Sim host agent
- `source/strafer_lab/scripts/sim_in_the_loop.py` — owned by Isaac Sim host agent
- `source/strafer_lab/strafer_lab/bridge/` — owned by Isaac Sim host agent
- `source/strafer_lab/strafer_lab/tools/bbox_extractor.py` — owned by Isaac Sim host agent
- `source/strafer_lab/strafer_lab/assets/` — owned by Isaac Sim host agent
- `source/strafer_lab/strafer_lab/envs/` — owned by Isaac Sim host agent
- `Makefile` — shared, do not modify without coordination
- `docs/` — do not modify design docs during implementation
