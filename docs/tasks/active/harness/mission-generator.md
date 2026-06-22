# Free-text mission generator with LLM-emitted waypoints (multi-room default)

**Type:** new feature
**Owner:** DGX agent
**Priority:** P2 (data-side foundation for VLA training scale-out
and scripted-driver consumption; canonical mission queue source
for multi-room missions, including path-shape language)
**Estimate:** L (~week+; LLM-as-planner pipeline + scene-summary
formatter + post-validation + caching + corpus-composition tooling)
**Branch:** task/harness-mission-generator

## Story

As an **operator who needs a canonical mission queue source for
multi-room missions, including groundable path-shape language ("go to
the chair, passing the dining table," "go to the kitchen via the
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
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/path-planning-architecture.md`](../../context/path-planning-architecture.md) —
  **required.** This brief's oracle path + waypoint-validation A* MUST
  build on the shared `path_planner` core, not a new planner. See
  "Path planning: build on the shared core" below.

Parent design context:
[`MISSION_VALIDATION_ARCHITECTURE.md` §3.6.a](../../../MISSION_VALIDATION_ARCHITECTURE.md#36a-teleop-demos-primary-canonical) — teleop is the canonical primary corpus for VLA training; this brief produces mission queues at scale beyond what teleop alone can author.

Sibling briefs:
- [`scene-connectivity-validation`](../../completed/scene-connectivity-validation.md) —
  produces the `connectivity[]` graph this brief consumes. Hard
  prerequisite.
- [`harness-architecture`](harness-architecture.md) —
  consumes this brief's `mission_queue.yaml` via the `queue`
  mission source. The
  [Driver: teleop](harness-architecture.md#driver-teleop) +
  [Driver: bridge](harness-architecture.md#driver-bridge) display
  `mission_text` to the operator / dispatch via the autonomy
  stack respectively; the
  [Driver: scripted](harness-architecture.md#driver-scripted)
  consumes `planned_path` as the waypoint-tracking input. Schema
  downstream of mission execution (LeRobot v3) is defined in the
  harness brief; this brief's per-row metadata maps onto the
  per-episode metadata columns under `meta/episodes/`.

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

### Path planning: build on the shared core

Every place this brief plans or validates a path — the `endpoint`
mode's plain shortest-path oracle, the navigable-mask /
segment-connectable post-validation checks, and the fallback A* when
LLM waypoints fail validation — **routes through the one shared planner**
([`path_planner.plan_path`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/path_planner/)),
per [`context/path-planning-architecture.md`](../../context/path-planning-architecture.md).
This is a settled architectural decision (2026-06-17): the project has
**one** grid A* core (shipped by `subgoal-env`), and Infinigen consumers
adapt their scene onto it rather than writing a second helper.

Concretely, this brief **loads the cached occupancy grid** the scene-gen
pipeline already produced and adapts it onto the planner — it does **not**
re-rasterize the scene. `scene-connectivity-validation` (shipped) generates
`<scene>/occupancy.npy` from the USD's physics colliders via Isaac Sim's
occupancy-map extension and the shared invert/inflate adapter lives at
[`scene_connectivity`](../../../../source/strafer_lab/strafer_lab/tools/scene_connectivity.py):

```python
from strafer_lab.tools.scene_connectivity import load_occupancy, occupancy_to_free_space
from strafer_lab.tasks.navigation.path_planner import plan_path
occ = load_occupancy(scene_dir)
free = occupancy_to_free_space(occ.grid, grid_res=occ.resolution_m)
path = plan_path(start_xy, goal_xy, free, grid_res=occ.resolution_m, grid_origin_xy=occ.origin_xy)
```

This is the **cached-occupancy seam** described in
[`context/path-planning-architecture.md`](../../context/path-planning-architecture.md);
it supersedes the earlier plan for a per-brief `scene_metadata.json →
free_space` rasterizer (that footprint+AABB rasterizer now lives only as
`validate_scene_connectivity.py`'s `--rasterize-fallback`). For the
reachability of cross-room pairs, prefer the already-verified
`connectivity[]` graph over re-planning. **Do not** reuse a separate
connectivity A* helper or hand-roll shortest-path: there is one planner.

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

  mission_text: "Go to the chair, passing the dining table."
  paraphrases:
    - "Head to the chair."
    - "Approach the chair near the dining table."
    - "Make your way over to the chair."

  planned_path:
    - {x: 0.5, y: 0.4}    # LLM-emitted waypoint
    - {x: 2.5, y: 0.5}
    - {x: 3.6, y: 1.0}    # routed close past the dining table
    - {x: 4.2, y: 1.8}    # final waypoint at target

  generator_metadata:
    llm_model: "Qwen3-4B"
    llm_seed: 42
    constraint_type_hint: "landmark_relative"   # post-hoc tag for ablations
    llm_reasoning_trace: "Routed close past the dining table on the way to the chair."
    waypoint_validation: {structure_ok: true, navigable_ok: true, segment_ok: true, target_ok: true, retries: 0}
```

**Spatial language must be groundable.** `mission_text` is a VLA
training label *and* the bridge dispatch payload, consumed by a
holonomic, camera + relative-odometry robot with no compass. So it
encodes **no absolute compass direction** ("the south wall", world
coordinates) and **no egocentric side** ("on your left/right") — the
strafer's heading is decoupled from its direction of travel and is
unknown when the text is authored, so neither is groundable. The only
spatial phrasing emitted is landmark proximity with no side ("passing
the {landmark}") and room-type / connectivity transit ("via the
{room}", "through the doorway into the {room}").

The VLA's training script consumes `mission_text` (or a
randomly-chosen paraphrase). The oracle consumes `planned_path`.
Teleop consumes `mission_text` for operator display.
`generator_metadata` is a downstream-analysis side-channel.

### Multi-room is the default

Per the project's MVP-multi-room decision: this brief generates
cross-room missions by default. The connectivity graph from
[`scene-connectivity-validation`](../../completed/scene-connectivity-validation.md)
gates which `(start_room, target_room)` pairs are reachable; the
LLM is told the connectivity graph as part of its scene prompt so
it doesn't propose paths through closed doors / unreachable rooms.

For single-room scenes (the legacy set), the generator falls back
to same-room missions only — `cross_room: false` in every row.

### Waypoint generation pipeline

One LLM call per mission at generation time. Components:

1. **Compact scene summary.** Don't paste the raw embedded metadata;
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
   - **start-frame VLM grounding** (see "Start-frame grounding"
     section below) — the perception camera at the proposed
     `start_pose` must be able to *see* the target (or the
     transit landmark for cross-room missions) before the mission
     is committed to the queue. This is the FoV-honest counterpart
     to the trajectory-first regime's structural grounding
     guarantee; without it, forward-generation queues silently
     accumulate "go to the chair" missions where the chair is
     occluded by another object at the start pose.
4. **Retry on failure.** Up to 3 retries per mission, re-prompted
   with the failure mode named. On persistent failure, fall back
   to mechanical A* shortest-path; tag in metadata as
   `path_shape_constraint_unsatisfied`. Mission still ships in
   the queue (label is the more useful supervision signal than
   structured constraint enforcement).
5. **Caching.** Key by
   `(scene_seed, start_pose, mission_text, llm_seed,
   prompt_template_hash, generator_version)`. The
   `prompt_template_hash` + `generator_version` keys are the
   guard against silent staleness when the few-shot template or
   the validator code change without an LLM-seed bump. First run
   pays the cost; re-runs are free. Cache lives at
   `data/mission_queue_cache/<scene>/<scene_seed>.json`.
6. **Model choice.** Benchmark Qwen3-4B (already cached on the
   DGX) vs. Qwen2.5-VL-7B vs. a larger LLM if available; pick
   based on hallucination rate on a held-out mission set. Text-only
   models work since the input is structured metadata, not
   images.

### Start-frame grounding

The forward-generation regime's structural blind spot is that an
LLM reading the scene metadata can name targets the camera
will *never see* from the proposed start pose (occluded by
nearer objects, hidden behind a doorway not yet traversed,
backside of furniture). The trajectory-first regime
([Scripted × captioner](harness-architecture.md#scripted--captioner-trajectory-first-path)
in `harness-architecture`)
sidesteps this because the captioner only sees frames the camera
actually captured. Forward generation has to enforce it
explicitly.

This brief adds one pass:

- After waypoint validation, render a single frame at
  `start_pose` from the perception camera's pose budget.
- Pass `(frame, mission_text)` to a 7B Qwen2.5-VL grounding
  prompt: "Is the target named in the instruction visible in
  this image? Answer yes / partial / no."
- **Yes / partial** → mission ships in the queue.
- **No** → either (a) the target is intended to be discovered
  via scan / transit, in which case the LLM rationale already
  records the transit step and the mission is kept with
  `start_frame_grounded: false` set; or (b) the target is
  same-room and unreachable from this start, in which case the
  mission is rejected with `rejected_reason:
  "target_not_visible_at_start"` and the start-pose seed is
  re-rolled.

This is a generation-time check; runtime behavior of the
operator / oracle is independent. The intent is to keep the
forward-generation corpus FoV-comparable to trajectory-first.

### Corpus composition

A single `mission_queue.yaml` is **per-scene + per-generation-pass**,
not a complete training corpus. The full corpus comes from the
cross-product:

| Axis | Typical count | Note |
|---|---|---|
| Scenes (Infinigen seeds, multi-room by default) | ~10 | See "Scene count vs. mission density" caveat below |
| Targets per scene (`objects[]` entries) | ~30 | |
| Paraphrase variants per target | ~5 | |
| Constraint-bias variants per target (no-bias / landmark / via-room / etc.) | ~3 | |
| Start-pose seeds per target | ~5 | |
| **Total missions** | ~22.5k | |

**Scene count vs. mission density.** HSSD (2023) found that
agents trained on 122 high-quality scenes outperformed agents
trained on 10,000 ProcTHOR scenes for ObjectNav generalization to
real environments — scene realism / quality dominated over scene
count once a threshold was crossed. Infinigen-generated scenes
take ~30 min apiece on the DGX, so the ~10-scene anchor in the
table above is reasonable for v1; the next scale-up step is
**adding missions per existing scene** (more paraphrases, more
start-pose seeds, more constraint variants) before adding more
scenes. Don't burn weeks generating 100 shallow Infinigen seeds
when 10× more missions per existing seed gets better
generalization. Revisit the scene-count axis only once
per-scene mission diversity has plateaued.

The brief ships the per-scene generator; corpus assembly is a
small wrapper script that runs the generator across scenes +
seeds and concatenates outputs. Each per-scene queue lives at
`data/mission_queues/<scene_name>/queue.yaml`; the corpus is
their union.

## Acceptance criteria

- [ ] **Generator entry point** at
      `source/strafer_lab/strafer_lab/tools/build_mission_queue.py`
      consumes the scene USD's `customData` (via
      `scene_metadata_reader.load`, with the `connectivity[]` block from
      [`scene-connectivity-validation`](../../completed/scene-connectivity-validation.md))
      and emits `mission_queue.yaml` with the schema above. Per
      `--mode {endpoint, path-shape, mixed}`:
  - `endpoint`: one mission per object, no path-shape language;
    trivial paraphrase pass; no LLM waypoints (oracle uses
    plain A* shortest-path).
  - `path-shape`: missions with groundable constraint-language
    variants (landmark proximity, via-room transit); LLM-emitted
    waypoints.
  - `mixed`: blend of both; default.
- [ ] **LLM-as-planner pipeline.** One LLM call per `path-shape`
      / `mixed` mission emits waypoints; output is
      JSON-schema-constrained; post-validation pass + retry
      logic implemented; caching keyed by
      `(scene_seed, start_pose, mission_text, llm_seed)`.
- [ ] **Shared planner.** The `endpoint`-mode oracle path, the
      post-validation navigable/segment checks, and the LLM-waypoint
      fallback all call `path_planner.plan_path` over the cached-occupancy
      seam — load `<scene>/occupancy.npy` via
      `scene_connectivity.load_occupancy` + `occupancy_to_free_space` (no new
      adapter, no re-rasterization, no second A*). See
      [`context/path-planning-architecture.md`](../../context/path-planning-architecture.md).
- [ ] **Multi-room default.** For multi-room scenes, the
      generator emits cross-room missions by default; the
      connectivity graph filters unreachable pairs. For
      single-room scenes, all missions are same-room.
- [ ] **Paraphrase pass** reuses the 7B Qwen2.5-VL pipeline from
      [`generate_descriptions.py`](../../../../source/strafer_lab/scripts/retired/generate_descriptions.py)
      Stage 2.
- [ ] **Start-frame grounding pass.** Each generated mission is
      run through the start-frame VLM check described above;
      same-room missions where the target is invisible at the
      start pose are either re-rolled (new start-pose seed) or
      rejected. Report `start_frame_grounded` rate per scene in
      the PR.
- [ ] **Cache key includes prompt_template_hash +
      generator_version.** Re-running the generator against a
      cache built under a changed template invalidates rather
      than silently reuses.
- [ ] **Driver consumption.** All three drivers in
      [`harness-architecture`](harness-architecture.md) accept
      the generated `mission_queue.yaml` unchanged when run with
      `--mission-source queue`:
      [Driver: teleop](harness-architecture.md#driver-teleop)
      (operator-display),
      [Driver: bridge](harness-architecture.md#driver-bridge)
      (autonomy-stack dispatch), and
      [Driver: scripted](harness-architecture.md#driver-scripted)
      (waypoint-following).
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
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md)
      Stage 5b/5c gain notes on consuming the generated queue.
      [`source/strafer_lab/README.md`](../../../../source/strafer_lab/README.md)
      "Scripts and tools inventory" gains
      `build_mission_queue.py` + `build_mission_corpus.py`.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit.

## Investigation pointers

- LLM caching: Qwen3-4B is in `~/.cache/huggingface/hub/`;
  Qwen2.5-VL-3B and -7B too. Pick text-only first for the
  waypoint-planning pass; the start-frame grounding pass needs a
  VL model (Qwen2.5-VL-7B).
- JSON-schema-constrained generation: most modern LLM APIs
  support this directly (OpenAI-compatible
  `response_format: {"type": "json_schema", ...}`). For
  `transformers`-served models, use `outlines` or `guidance`
  for grammar-constrained generation.
- Few-shot examples should cover all three mission modes;
  hand-author them from the existing
  `scene_high_quality_dgx_000_seed0` scene.
- Navigable-mask post-validation: plan with the shared
  [`path_planner`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/path_planner/)
  via this brief's Infinigen occupancy-grid adapter — see "Path
  planning: build on the shared core" and
  [`context/path-planning-architecture.md`](../../context/path-planning-architecture.md).
  One planner, not a connectivity-specific helper.
- Existing paraphrase pipeline:
  [`generate_descriptions.py`](../../../../source/strafer_lab/scripts/retired/generate_descriptions.py)
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
  [Driver: teleop](harness-architecture.md#driver-teleop)
  in `harness-architecture`; the *replay tool* is a future brief
  ([`cosmos-replay-perturbation`](../../parked/harness/cosmos-replay-perturbation.md)
  filed by the audit).
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
