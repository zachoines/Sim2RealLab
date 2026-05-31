# Adopt the VLM grounding LoRA tooling onto the harness detections column

**Type:** task / new feature
**Owner:** DGX agent (VLM LoRA training + dataset loader; no Jetson code)
**Priority:** P2
**Estimate:** L (~multi-day: new LeRobot-v3 loader + a LoRA training run + eval pass)
**Branch:** task/vlm-grounding-finetune

## Story

As a **DGX operator who has orphaned VLM grounding LoRA tooling and a
new first-class detections column landing in the harness corpus**, I
want **the grounding fine-tune wired to consume that column, trained,
and evaluated**, so that **`scan_for_target` / planner target-grounding
quality improves and the clip-validation cascade gets a reliable supply
of sibling-detection labels and confidences**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/perception-backbone-architecture.md`](../../context/perception-backbone-architecture.md) — **light reference only.** This grounder is a VLM (Qwen2.5-VL-3B), *upstream* of the CLIP cascade; it does **not** consume the shared frozen trunk and is not a consumer in that module's table. Read it only to confirm the boundary, not to inherit a backbone choice.

Parent design doc:
[`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
§2.5 — the bad-grounding failure mode (wrong instance / off-by-one-room
grounding) that a better grounder directly attacks.

Schema this brief consumes:
[`active/harness/harness-architecture.md`](../../active/harness/harness-architecture.md)
— the LeRobot v3 dataset this fine-tune loads from. **Gate this brief on
the R1 detections column existing** (see [Pickup gate](#pickup-gate)).

## Context

### The orphaned tooling this brief adopts

A complete Qwen2.5-VL grounding LoRA pipeline already exists in-repo
but has **no brief and a retired data path**:

- [`source/strafer_vlm/strafer_vlm/training/train_qwen25vl_lora.py`](../../../../source/strafer_vlm/strafer_vlm/training/train_qwen25vl_lora.py)
  — LoRA fine-tune for `Qwen/Qwen2.5-VL-3B-Instruct` grounding.
- [`source/strafer_vlm/strafer_vlm/training/dataset_io.py`](../../../../source/strafer_vlm/strafer_vlm/training/dataset_io.py)
  — `load_grounding_dataset()` reads **JSON / JSONL** grounding records
  (flat `{image, prompt, target}` or chat-format) into
  `GroundingExample(image_path, prompt, target)` where `target` is
  `{found, bbox_2d (in 0..1000), label, confidence}`.
- [`source/strafer_vlm/strafer_vlm/training/eval_qwen25vl_grounding.py`](../../../../source/strafer_vlm/strafer_vlm/training/eval_qwen25vl_grounding.py)
  — offline eval (IoU, found-accuracy) against a held-out grounding set.
- [`source/strafer_lab/strafer_lab/tools/bbox_extractor.py`](../../../../source/strafer_lab/strafer_lab/tools/bbox_extractor.py)
  — Replicator `bounding_box_2d_tight` → typed `DetectedBbox`
  (`semantic_id`, `label`, `labels`, pixel `bbox_2d`, `occlusion_ratio`).
  This is **this brief's bbox producer** — see the caution below.

The tooling currently reads the **legacy JSON/JSONL grounding-record
path** that the harness consolidation retired. It has no consumer brief
because its prior data feeders
([`scripts/generate_descriptions.py`](../../../../source/strafer_lab/scripts/generate_descriptions.py),
[`scripts/prepare_vlm_finetune_data.py`](../../../../source/strafer_lab/scripts/prepare_vlm_finetune_data.py))
are slated for retirement in the harness implementation PRs (see
[`harness-architecture.md`'s "Retired downstream scripts"](../../active/harness/harness-architecture.md#retired-downstream-scripts)).
This brief re-homes the grounding fine-tune onto the harness corpus.

### The R1 detections column it consumes

The orchestrator is landing a **first-class detections column** in the
harness LeRobot v3 schema. The shape this brief targets is a **native
packed parquet column set** (NOT a depth-style sidecar tree):

| Column | Shape | Notes |
|---|---|---|
| `observation.detections.bbox` | `(max_n, 4)` | per-frame 2D boxes, zero-padded to `max_n` |
| `observation.detections.label_id` | `(max_n,)` | id into the label vocab |
| `observation.detections.occlusion` | `(max_n,)` | Replicator occlusion ratio (`0.0` visible → `1.0` occluded) |
| `observation.detections.valid` | `(max_n,)` | padding mask; only `valid` rows are real detections |

plus a `meta/detection_labels.json` id↔string vocab. The box / label /
occlusion fields map cleanly from `bbox_extractor.py`'s `DetectedBbox`
producer. This is a **packed native column**, so the loader reads it
directly from the per-shard parquet rows and resolves `label_id` through
`meta/detection_labels.json` — no per-episode sidecar walk.

**This brief references the column shape; it does not land it.** The
harness schema change is the **orchestrator's to land** — see
[Pickup gate](#pickup-gate).

### Why it serves both epics

- **multi-room** — a better-grounded `scan_for_target` and
  planner target-grounding directly reduce the wrong-instance /
  off-by-one-room grounding errors named in
  [`MISSION_VALIDATION_ARCHITECTURE.md` §2.5](../../../MISSION_VALIDATION_ARCHITECTURE.md).
- **clip-validation** — the grounder is the **primary source** of
  case-2 alternate phrases for the
  [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
  `TextAlignmentMonitor`: when same-label siblings are visible the
  grounder returns multiple bboxes, and the monitor consumes the
  **other** detection labels as the alternate-phrase contrast pool. A
  better grounder also raises the per-detection confidences the
  executor's `min_grounding_confidence` gate keys off
  ([`MISSION_VALIDATION_ARCHITECTURE.md` §2.5](../../../MISSION_VALIDATION_ARCHITECTURE.md)).

### CAUTION — do NOT mark `bbox_extractor.py` for deletion

`bbox_extractor.py` was orphaned when the harness retired its old
consumers (`generate_descriptions.py`, `prepare_vlm_finetune_data.py`),
but it is **this brief's bbox producer** — the detections column the
fine-tune loads from is populated by it. Any harness-retirement work
that sweeps for now-unused tools **must not** delete `bbox_extractor.py`;
its consumer is this brief, not the retired scripts that named it.

## Acceptance criteria

- [ ] A LeRobot-v3 → `GroundingExample` loader exists alongside the
      legacy JSON/JSONL path in
      [`dataset_io.py`](../../../../source/strafer_vlm/strafer_vlm/training/dataset_io.py):
      it reads `observation.images.perception` for the image, builds the
      `prompt` from the per-episode `target_label` (and/or `tasks`), and
      builds each `GroundingExample.target` from the
      `observation.detections.{bbox, label_id, occlusion, valid}` packed
      columns resolved through `meta/detection_labels.json`. Only `valid`
      rows produce targets; `bbox` is normalized to the 0..1000 grounding
      convention the existing parser expects.
- [ ] `train_qwen25vl_lora.py` runs end-to-end against a harness
      LeRobot-v3 dataset and produces a LoRA adapter for
      `Qwen/Qwen2.5-VL-3B-Instruct`.
- [ ] `eval_qwen25vl_grounding.py` runs against a held-out split (pin
      the split name per
      [`harness-architecture.md`'s splits section](../../active/harness/harness-architecture.md#train--val--held-out-splits))
      and reports IoU + found-accuracy; the adapter beats the base model
      on the held-out set.
- [ ] `bbox_extractor.py` is **not** deleted or marked for deletion by
      this PR, and the PR notes it as the detections-column producer so a
      later harness-retirement sweep doesn't remove it.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the legacy JSON/JSONL loader path or the
      `strafer_vlm` grounding inference service (smoke test: the existing
      `dataset_io` tests still pass and `/ground` still serves).

## Pickup gate

**Filed-on-trigger (parked).** Pick up only when the **R1 detections
column lands in the harness** — i.e. the orchestrator has shipped
`observation.detections.{bbox, label_id, occlusion, valid}` + a
`meta/detection_labels.json` vocab into the LeRobot v3 schema
([`harness-architecture.md`](../../active/harness/harness-architecture.md),
Tier 2 / Tier 3). Before that, the loader this brief writes has nothing
to read. Verify the column exists in a captured dataset's parquet (and
the vocab file in `meta/`) before un-parking.

## Investigation pointers

- Legacy loader to extend (don't replace):
  [`dataset_io.py:125`](../../../../source/strafer_vlm/strafer_vlm/training/dataset_io.py#L125)
  `load_grounding_dataset`.
- Grounding record shape the trainer/eval expect:
  `GroundingExample` / `GroundingTarget` in
  [`source/strafer_vlm/strafer_vlm/inference/parsing.py`](../../../../source/strafer_vlm/strafer_vlm/inference/parsing.py)
  (`target = {found, bbox_2d in 0..1000, label, confidence}`).
- Detections producer:
  [`bbox_extractor.py`](../../../../source/strafer_lab/strafer_lab/tools/bbox_extractor.py)
  (`DetectedBbox.bbox_2d` is `(x_min, y_min, x_max, y_max)` in pixels;
  `occlusion_ratio` is the Replicator field that maps to the column's
  `occlusion`).
- Harness schema + how packed columns differ from the depth sidecar:
  [`harness-architecture.md`'s features schema](../../active/harness/harness-architecture.md#per-frame-features-schema-declared-in-metainfojson)
  and [Depth representation](../../active/harness/harness-architecture.md#depth-representation--sidecar-png-sequence)
  (depth is a sidecar; detections are a native packed column — contrast,
  don't copy).
- Case-2 alternate-phrase consumer:
  [`validator-evaluation.md`'s `TextAlignmentMonitor` "Where the alternates come from" table](../../active/clip-validation/validator-evaluation.md).

## Out of scope

- **Landing the R1 detections column in the harness.** Orchestrator's;
  this brief is gated on it.
- **Any CLIP / cascade fine-tune.** This is a VLM grounder, upstream of
  the cascade. The CLIP-side work is
  [`cotrained-retrieval-augmented`](cotrained-retrieval-augmented.md).
- **Deleting or refactoring `bbox_extractor.py`.** It's the producer;
  leave it in place (see the caution above).
- **Jetson-side deployment of the new adapter.** Training + eval on the
  DGX only; wiring the merged adapter into the served `strafer_vlm`
  endpoint is a follow-up.
- **Retiring the legacy JSON/JSONL loader path.** Keep it working; this
  brief adds the LeRobot-v3 path beside it.
