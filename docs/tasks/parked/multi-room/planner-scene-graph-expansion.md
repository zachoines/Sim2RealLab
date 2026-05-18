# Expand `world_state` with object poses + room inventory for an LLM-emitting planner

**Type:** new feature / investigation (filed-on-trigger)
**Owner:** DGX agent (planner request schema + Jetson populate
helper + prompt + few-shot set; cross-lane like other
`world_state` work)
**Priority:** P3 — pick up only when the C → B migration's
shadow-mode step shows the LLM needs more scene context than
[`context/planner-request-schema.md`](../../context/planner-request-schema.md)
currently exposes.
**Estimate:** M–L (~3–5 days; schema extension + Jetson
populator + prompt + few-shot pass + token-budget measurement
+ A/B against the compiler's deterministic plan).
**Branch:** task/planner-scene-graph-expansion

**Pickup gate:** Blocked on the C → B migration step that adds
shadow-mode `staging_hops` logging (step 2 in
[`STRAFER_AUTONOMY_NEXT.md` §1.10.2](../../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c)).
That step must ship first AND produce ≥ a week of shadow data
showing the LLM emits `staging_hops` that **disagree with the
compiler's plan**. Triage of those disagreements is the
trigger condition — see "Trigger detail" below. Un-park by
`git mv parked/multi-room/<this>.md
active/multi-room/<this>.md` in the PR that picks it up, per
[`README.md`'s Directory layout](../../README.md#directory-layout).

## Story

As an **operator about to promote `staging_hops` from advisory
to authoritative under the C → B migration**, I want **the LLM
planner to receive object poses, per-room object inventories,
and (optionally) relational edges on every planner request**,
so that **the LLM can disambiguate targets ("the chair next to
the table"), choose intermediate landmarks relative to known
obstacles instead of room centroids, and reason about approach
angles — capabilities the current Option C `world_state` does
not expose, and which shadow-mode data identifies as the
bottleneck preventing `staging_hops` promotion**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/planner-request-schema.md`](../../context/planner-request-schema.md)
  — the canonical `world_state` shape this brief extends.
- [`planner-architecture-alignment`](../../active/multi-room/planner-architecture-alignment.md)
  — the Option C decision and the C → B migration plan this
  brief slots into.
- [`observation-derived-room-state`](../../active/multi-room/observation-derived-room-state.md)
  — the runtime room-state APIs whose `RoomEntry` shape this
  brief extends with object-level detail.

## Trigger detail — when to un-park

The C → B migration is gated on empirical data, not a calendar.
Specifically, this brief un-parks when **shadow-mode
`staging_hops` triage shows the LLM systematically misses
target-resolution decisions the compiler also cannot make**.
Concretely, after the shadow brief ships and ≥ a week of
production missions are logged:

1. Compute the `staging_hops` ↔ `compiler.plan` agreement rate
   over warm-start cross-room missions.
2. For the disagreements, classify each by failure mode:
   - **Room-level disagreement** ("LLM picked the wrong target
     room") → covered by improving `observed_objects` /
     `connectivity` precision; this brief NOT triggered.
   - **Intra-room landmark disagreement** ("LLM and compiler
     both agreed on the kitchen, but LLM proposed `door` as
     the intermediate, compiler proposed `room_anchor`") →
     this brief IS triggered if the LLM's choice is
     defensibly better-than-compiler ≥ 30% of the time.
   - **Target disambiguation failure** ("the chair next to
     the table" and the LLM picked the wrong chair because it
     can't see the table's pose) → this brief IS triggered;
     this is precisely the gap object poses close.
   - **Pure LLM hallucination** → don't trigger this brief;
     more scene context won't fix hallucination.

If the disagreement breakdown is dominated by row 1 (room
level), file a different brief that tightens room inference,
not this one. If row 3 or row 4 dominates, this brief is the
right move.

## Context

### What `world_state` exposes today

Per [`context/planner-request-schema.md`](../../context/planner-request-schema.md):

```python
class PlannerWorldState(BaseModel):
    robot_pose_map: Pose2D
    last_grounding: GroundingSummary | None
    target_known_poses: list[Pose2D]
    current_room: str | None
    known_rooms: list[RoomEntry]              # RoomEntry.observed_objects: list[str]
    connectivity: list[tuple[str, str]]
```

`RoomEntry.observed_objects` is **labels only**. The compiler's
target-room inference works from this — `"sink" in
known_rooms["kitchen"].observed_objects` ⇒ kitchen. But labels
discard the spatial relations the LLM needs for the queries
Option B is supposed to handle well.

### What B-shaped planners want (literature)

All three planner-LLM systems cited as
[`planner-architecture-alignment`](../../active/multi-room/planner-architecture-alignment.md)
precedents use hierarchical scene graphs with object poses:

- **CogNav** (ICCV 2025) — LLM emits state-aware transitions
  between exploration and identification; per-node object
  attributes drive the transitions.
- **FSR-VLN** (arXiv:2509.13733) — fast/slow reasoning with
  hierarchical scene graph (floor → room → view → object).
  Strafer is single-floor; the other three levels apply.
- **osmAG-LLM** (arXiv:2507.12753) — LLM reasons over a
  topometric map with textual semantic objects + room
  attributes. Object poses are first-class.

The consensus: object poses unlock the queries the LLM is
supposed to be good at. The disagreement is on representation
detail — pure list of triples vs. JSON tree vs. natural-language
description. That choice is empirical and is the point of this
brief's measurement phase.

### Proposed schema extension

Phase 1 — object poses per room (likely sufficient):

```python
class ObjectEntry(BaseModel):
    label: str
    pose_map: Pose3D
    confidence: float
    last_seen_s: float                # age — for staleness reasoning
    room_label: str | None

class RoomEntry(BaseModel):
    label: str
    member_node_ids: list[str]
    centroid_xy: tuple[float, float]
    confidence: float
    observed_objects: list[str]       # kept for backward compat
    objects: list[ObjectEntry]        # NEW
```

Phase 2 — pairwise spatial relations (only if Phase 1 is
insufficient):

```python
class RoomEntry(BaseModel):
    # ... existing + objects ...
    relations: list[tuple[str, str, str]]  # (obj_a_id, relation, obj_b_id)
    # relations include: "next_to", "on", "above", "below", "facing",
    # "between", with geometric definitions defined in this brief
```

Phase 3 — affordance flags (deferred unless task data demands):

- `is_open` for doors, `is_occupied` for chairs, `is_powered`
  for appliances, etc. Affordances depend on perception we
  don't ship today; file a separate brief if Phase 1 + 2 are
  not enough.

### Token budget — the real constraint

Every object in the prompt costs tokens. The literature shows
LLM-as-planner quality peaks somewhere between "no scene state"
and "full scene graph" — too much detail degrades reasoning and
inflates latency. Picking the cutoff requires data:

- Measure per-prompt token count with Phase 1 only, Phase 1+2,
  Phase 1+2+3.
- Measure planner LLM latency at each token count.
- Measure `staging_hops` ↔ compiler agreement rate at each
  level (the metric that triggered this brief).
- Choose the smallest representation that achieves the
  agreement-rate target.

The brief ships whichever phase the data justifies, not all
three by default.

## Approach

In order:

1. **Schema extension** in
   [`source/strafer_autonomy/strafer_autonomy/schemas/`](../../../../source/strafer_autonomy/strafer_autonomy/schemas/).
   Phase 1 first (object poses). Keep `observed_objects` for
   backward compatibility — the compiler still uses it.
2. **Jetson populator** in `mission_runner.py`. The
   `SemanticMapManager` already stores per-node
   `detected_objects: list[DetectedObjectEntry]` with poses
   and covariances (see
   [`source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)).
   The populator groups these by room label and emits the
   `ObjectEntry` list per `RoomEntry`. No new perception work.
3. **Prompt update** in
   [`planner/prompt_builder.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/prompt_builder.py).
   Decide the textual representation (JSON snippet vs.
   natural-language summary) by A/B on the shadow-mode
   agreement metric. Add 3–5 few-shot examples that exercise
   the new fields.
4. **A/B measurement.** Run the same shadow-mode pipeline used
   to trigger this brief, but with the extended `world_state`.
   Compute the agreement rate again; the brief is
   "successful" only if the disagreement classes identified in
   the trigger move (rows 3 + 4 from the trigger triage drop
   below row 1).
5. **Promote `staging_hops` to advisory.** This unblocks step
   3 of the C → B migration (see
   [§1.10.2](../../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c)).
   That promotion lives in a separate brief; this one ends
   when the schema, populator, prompt, and measurement are in
   place.

## Acceptance criteria

- [ ] **Phase 1 schema lands.** `ObjectEntry` + `RoomEntry
      .objects` in
      `source/strafer_autonomy/strafer_autonomy/schemas/`.
      Backward-compatible: `observed_objects` stays populated.
      Documented in
      [`context/planner-request-schema.md`](../../context/planner-request-schema.md).
- [ ] **Jetson populator wired.** `mission_runner.py` reads
      `SemanticMapManager`'s detected-object data and emits
      `ObjectEntry` per room. Reused from the existing
      semantic-map storage; no new perception code.
- [ ] **Prompt + few-shot update.** Planner prompt teaches the
      LLM to use object poses for disambiguation. Few-shot
      set covers at least:
  - Disambiguation by spatial relation ("the chair next to
    the table").
  - Approach-angle reasoning ("face the painting on the
    north wall").
  - Multi-room landmark choice ("the door from the living
    room that leads to the kitchen").
- [ ] **Token budget measured.** Per-prompt token count
      reported for Phase 1 alone (and Phase 1+2 if shipped).
      Per-mission latency overhead vs. baseline reported.
- [ ] **Agreement rate measured.** Shadow-mode
      `staging_hops` agreement-with-compiler rate measured
      with and without the schema extension on the same
      mission queue. The brief ships only if the
      previously-identified failure modes (per the trigger
      triage) measurably improve.
- [ ] **Phase 2 decision recorded.** Whether pairwise
      relations are needed is decided based on Phase 1's
      agreement-rate result. If Phase 2 is shipped in this
      brief, schema + populator + prompt updates land in the
      same PR. If Phase 2 is deferred, file a follow-up
      brief.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same
      commit. See
      [`conventions.md`'s user-facing documentation
      maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- Existing per-node object storage:
  [`semantic_map/manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
  — `SemanticNode.detected_objects: list[DetectedObjectEntry]`
  is already populated by `add_observation`. This brief
  consumes it; it does not add new perception.
- Reinforcement / Mahalanobis matching:
  [`STRAFER_AUTONOMY_NEXT.md` §1.12](../../../STRAFER_AUTONOMY_NEXT.md)
  — object-pose uncertainty is already tracked. Stale or
  low-confidence objects can be filtered out of the prompt
  to control token count.
- Reference architectures to cross-check against:
  - CogNav (ICCV 2025)
  - FSR-VLN (arXiv:2509.13733)
  - osmAG-LLM (arXiv:2507.12753)

## Out of scope

- **Constrained-output decoder for the planner LLM.** A
  separate prerequisite of the C → B migration; filed
  elsewhere when shadow-mode data motivates it.
- **VLA-style fine-tuning on scene graphs.** Filed at
  [`vla-v2-architecture`](../experimental/vla-v2-architecture.md).
  Strictly larger architectural move; this brief is prompt +
  schema, not training.
- **Persistent scene-graph storage to disk.** The
  `SemanticMapManager` already persists nodes + edges + CLIP
  embeddings via ChromaDB; this brief shapes only the
  request-time payload, not storage.
- **Multi-floor scene graphs.** Strafer is single-story.
- **Affordance perception.** Phase 3 above is gated on
  Phase 1+2 being measurably insufficient AND new perception
  for door-state / chair-occupancy / etc. existing. If
  shadow data demands affordances and they don't exist, file
  separate briefs for the perception work first.
