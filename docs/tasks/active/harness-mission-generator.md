# Free-text mission generator with LLM-emitted waypoints (multi-room default)

**Type:** new feature
**Owner:** DGX agent
**Priority:** P2 (data-side foundation for VLA training scale-out
and oracle-driver consumption; canonical mission queue source for
multi-room missions, including path-shape language)
**Estimate:** L (~week+; LLM-as-planner pipeline + scene-summary
formatter + post-validation + caching + corpus-composition tooling)
**Branch:** task/harness-mission-generator

## Story

As an **operator who needs a canonical mission queue source for
multi-room missions, including path-shape language ("go to the
chair by hugging the south wall," "go to the kitchen via the
dining room") at scale beyond what a teleop operator can
hand-author**, I want **a free-text mission generator that emits
`(mission_text, paraphrases[], planned_path)` rows from scene
metadata + an LLM-as-planner pass, with multi-room support by
default**, so that **the oracle driver, the teleop driver's
auto-queue path, and v2 VLA training all consume a single
mission-queue contract that handles the full spectrum from
endpoint missions through path-shape constraint language**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../context/branching-and-prs.md)
- [`context/conventions.md`](../context/conventions.md)

Parent design context:
[`MISSION_VALIDATION_ARCHITECTURE.md` §3.6.a](../../MISSION_VALIDATION_ARCHITECTURE.md#36a-teleop-demos-primary-canonical) — teleop is the canonical primary corpus for VLA training; this brief produces mission queues at scale beyond what teleop alone can author.

Sibling briefs:
- [`multi-room-scene-connectivity-validation`](multi-room-scene-connectivity-validation.md) —
  produces the `connectivity[]` graph this brief consumes. Hard
  prerequisite.
- [`harness-teleop-driver`](harness-teleop-driver.md) —
  consumes the queue (operator picks missions to drive); the
  teleop brief's previously-described auto-queue is replaced by
  this brief's output.
- [`harness-oracle-driver`](harness-oracle-driver.md) —
  consumes `planned_path` (the LLM-emitted waypoints) as the
  oracle's path-tracking input.
- [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md) —
  defines the schema downstream of mission execution; this
  brief's `mission.json` plugs into that schema.

## Context

### Why free-text only (no structured constraint enumeration)

Earlier drafts of this brief proposed a structured
`path_constraints[]` schema (`wall_follow`, `via_room`, etc.)
emitted alongside `mission_text`. **That's been retired.** Three
reasons:

- **The VLA consumes free-text.** Every published wheeled VLA
  (RT-2, OpenVLA, NaVid, π0) trains on natural-language mission
  text and learns the constraint distribution implicitly.
  Structured constraints don't enter the model's input.
- **Internal consumers don't actually need the structured form.**
  Oracle path biasing is replaced by **LLM-emitted waypoints**;
  teleop just shows the operator the free-text mission;
  validators train on `(frames, mission_text, outcome)` triples
  with mission-text fusion. None requires a closed schema.
- **Bounded enumeration limits coverage.** Real users say things
  like "approach quietly," "stay in shadow," "come back the
  same way" — language no enumeration anticipates. Free-text
  with LLM-emitted waypoints is unbounded by construction.

A small `generator_metadata` block is kept for ablations and
debugging only — not consumed by the VLA.

### Output schema

Per mission, one row in `mission_queue.yaml`:

```yaml
- mission_id: 0123
  scene_name: scene_high_quality_dgx_000_seed0
  scene_seed: 0
  start_pose: {x: 0.5, y: 0.5, yaw: 0.0}
  target_label: chair
  target_position_3d: [4.2, 1.8, 0.0]
  target_room: living_room
  start_room: kitchen
  cross_room: true   # derived from start_room != target_room

  mission_text: "Go to the chair by hugging the south wall."
  paraphrases:
    - "Approach the chair while staying close to the south wall."
    - "Drive along the southern edge of the room to reach the chair."
    - "Stay close to the south wall as you go to the chair."

  planned_path:
    - {x: 0.5, y: 0.4}    # LLM-emitted waypoint near south wall
    - {x: 2.5, y: 0.3}
    - {x: 4.0, y: 0.5}
    - {x: 4.2, y: 1.8}    # final waypoint at target

  generator_metadata:
    llm_model: "Qwen3-4B"
    llm_seed: 42
    constraint_type_hint: "wall_follow_inferred"   # post-hoc tag for ablations
    llm_reasoning_trace: "Identified 'hug the south wall' as a wall-follow constraint; planned waypoints staying within 0.5m of y=0..."
    waypoint_validation: {bounds_ok: true, navigable_ok: true, connectivity_ok: true, retries: 0}
```

The VLA's training script consumes `mission_text` (or a
randomly-chosen paraphrase). The oracle consumes `planned_path`.
Teleop consumes `mission_text` for operator display.
`generator_metadata` is a downstream-analysis side-channel.

### Multi-room is the default

Per the project's MVP-multi-room decision: this brief generates
cross-room missions by default. The connectivity graph from
[`multi-room-scene-connectivity-validation`](multi-room-scene-connectivity-validation.md)
gates which `(start_room, target_room)` pairs are reachable; the
LLM is told the connectivity graph as part of its scene prompt so
it doesn't propose paths through closed doors / unreachable rooms.

For single-room scenes (the legacy set), the generator falls back
to same-room missions only — `cross_room: false` in every row.

### Waypoint generation pipeline

One LLM call per mission at generation time. Components:

1. **Compact scene summary.** Don't paste raw `scene_metadata.json`;
   distill into ~200–500 tokens of structured prose:
   ```
   Bounds: x ∈ [0, 8], y ∈ [0, 6]
   Rooms:
     - kitchen: polygon [(0,0),(4,0),(4,3),(0,3)]
     - living_room: polygon [(4,0),(8,0),(8,3),(4,3)]
   Connectivity:
     - kitchen ↔ living_room via doorway at (4, 1.5)
   Walls:
     - kitchen south: y=0, x ∈ [0,4]
     ...
   Objects:
     - chair_1 (chair): (4.2, 1.8) in living_room
     ...
   ```
2. **Few-shot prompt** with 3–5 worked examples covering
   endpoint missions, single-room path-shape, and cross-room
   transit. Output is JSON-schema-constrained:
   `{"waypoints": [{"x": ..., "y": ...}, ...], "rationale": "..."}`.
3. **Post-validation.** Each generated waypoint set runs
   through:
   - bounds check (within scene)
   - navigable-mask check (not inside walls/objects)
   - segment check (adjacent waypoints connectable through
     navigable space)
   - connectivity check (cross-room paths use known doorways)
   - target-proximity check (final waypoint within 0.5 m of target)
4. **Retry on failure.** Up to 3 retries per mission, re-prompted
   with the failure mode named. On persistent failure, fall back
   to mechanical A* shortest-path; tag in metadata as
   `path_shape_constraint_unsatisfied`. Mission still ships in
   the queue (label is the more useful supervision signal than
   structured constraint enforcement).
5. **Caching.** Key by
   `(scene_seed, start_pose, mission_text, llm_seed)`. First run
   pays the cost; re-runs are free. Cache lives at
   `data/mission_queue_cache/<scene>/<scene_seed>.json`.
6. **Model choice.** Benchmark Qwen3-4B (already cached on the
   DGX) vs. Qwen2.5-VL-7B vs. a larger LLM if available; pick
   based on hallucination rate on a held-out mission set. Text-only
   models work since the input is structured metadata, not
   images.

### Corpus composition

A single `mission_queue.yaml` is **per-scene + per-generation-pass**,
not a complete training corpus. The full corpus comes from the
cross-product:

| Axis | Typical count |
|---|---|
| Scenes (Infinigen seeds, multi-room by default) | ~10 |
| Targets per scene (`objects[]` entries) | ~30 |
| Paraphrase variants per target | ~5 |
| Constraint-bias variants per target (no-bias / wall-follow / via-room / etc.) | ~3 |
| Start-pose seeds per target | ~5 |
| **Total missions** | ~22.5k |

The brief ships the per-scene generator; corpus assembly is a
small wrapper script that runs the generator across scenes +
seeds and concatenates outputs. Each per-scene queue lives at
`data/mission_queues/<scene_name>/queue.yaml`; the corpus is
their union.

## Acceptance criteria

- [ ] **Generator entry point** at
      `source/strafer_lab/strafer_lab/tools/build_mission_queue.py`
      consumes `scene_metadata.json` (with `connectivity[]`
      block from
      [`multi-room-scene-connectivity-validation`](multi-room-scene-connectivity-validation.md))
      and emits `mission_queue.yaml` with the schema above. Per
      `--mode {endpoint, path-shape, mixed}`:
  - `endpoint`: one mission per object, no path-shape language;
    trivial paraphrase pass; no LLM waypoints (oracle uses
    plain A* shortest-path).
  - `path-shape`: missions with constraint-language variants
    (wall-follow, via-room, around-furniture, etc.); LLM-emitted
    waypoints.
  - `mixed`: blend of both; default.
- [ ] **LLM-as-planner pipeline.** One LLM call per `path-shape`
      / `mixed` mission emits waypoints; output is
      JSON-schema-constrained; post-validation pass + retry
      logic implemented; caching keyed by
      `(scene_seed, start_pose, mission_text, llm_seed)`.
- [ ] **Multi-room default.** For multi-room scenes, the
      generator emits cross-room missions by default; the
      connectivity graph filters unreachable pairs. For
      single-room scenes, all missions are same-room.
- [ ] **Paraphrase pass** reuses the 7B Qwen2.5-VL pipeline from
      [`generate_descriptions.py`](../../../source/strafer_lab/scripts/generate_descriptions.py)
      Stage 2.
- [ ] **Driver consumption.** Both
      [`harness-teleop-driver`](harness-teleop-driver.md)
      (operator-display) and
      [`harness-oracle-driver`](harness-oracle-driver.md)
      (waypoint-following) accept the generated
      `mission_queue.yaml` unchanged.
- [ ] **Hallucination benchmark.** Brief PR includes a small
      benchmark: 50 generated missions, manually inspect for
      LLM-hallucinated waypoints (off-mesh, through walls,
      wrong target). Report retry rate + post-validation
      rejection rate per scene.
- [ ] **Corpus assembly wrapper.**
      `Scripts/build_mission_corpus.py` runs the generator
      across scenes + seeds and emits a concatenated corpus
      manifest under `data/mission_queues/corpus.yaml`.
- [ ] **Doc surface.**
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
      Stage 5b/5c gain notes on consuming the generated queue.
      [`source/strafer_lab/README.md`](../../../source/strafer_lab/README.md)
      "Scripts and tools inventory" gains
      `build_mission_queue.py` + `build_mission_corpus.py`.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit.

## Investigation pointers

- LLM caching: Qwen3-4B is in `~/.cache/huggingface/hub/`;
  Qwen2.5-VL-3B and -7B too. Pick text-only first.
- JSON-schema-constrained generation: most modern LLM APIs
  support this directly (OpenAI-compatible
  `response_format: {"type": "json_schema", ...}`). For
  `transformers`-served models, use `outlines` or `guidance`
  for grammar-constrained generation.
- Few-shot examples should cover all three mission modes;
  hand-author them from the existing
  `scene_high_quality_dgx_000_seed0` scene.
- Navigable-mask post-validation: reuse the connectivity-validation
  brief's A* helper; same code path.
- Existing paraphrase pipeline:
  [`generate_descriptions.py`](../../../source/strafer_lab/scripts/generate_descriptions.py)
  Stage 2 prompt template — adapt for paraphrase generation
  (different prompt, same model loading).

## Out of scope

- **Runtime mission generation.** This brief is offline at
  scene-gen / data-prep time. Runtime free-text mission entry
  is the operator's job (CLI / API), not this brief's.
- **Free-form natural-language mission generation by an LLM
  with zero scaffolding.** Templates + LLM expansion + scene-grounded
  paraphrase are the bar. Pure LLM-without-grounding generation
  risks hallucinating constraints the scene can't satisfy.
- **Replay-with-perturbation.** Recording the gamepad event
  stream is in scope per
  [`harness-teleop-driver`](harness-teleop-driver.md); the
  *replay tool* is a future brief.
- **Real-robot mission generation.** Sim-only; depends on
  Infinigen scene metadata not available in real deployments.
- **Validating waypoint quality at runtime.** Post-validation
  here is at generation time. Runtime monitoring of whether
  the oracle's path actually follows the waypoints is the
  oracle brief's concern.
- **Schema for `path_constraints[]`.** Retired — see "Why
  free-text only" above. Internal `generator_metadata` tags
  preserve ablation hooks without committing to a closed
  schema.
