# Lift the multi-room runtime deferral: stored-map fallback + planner transit steps

**Type:** new feature
**Owner:** Either (planner-side changes are DGX-lane in
`strafer_autonomy.planner`; `scan_for_target` changes are
Jetson-lane in `strafer_autonomy.executor`. Cross-lane like the
clip-eval brief.)
**Priority:** P1 (unblocks multi-room missions across the entire
project — VLA training, harness data, validation briefs, and
real-deployment missions all assume multi-room going forward)
**Estimate:** M (~2–4 days; small executor change + planner
prompt + plan-compiler emission + scene-metadata wiring + acceptance run)
**Branch:** task/multi-room-autonomy-stack

## Story

As an **operator who wants the strafer MVP to handle the
canonical home-robot mission ("go to the kitchen table" while in
the living room)**, I want **`scan_for_target` to fall back to
navigating to a stored-map sighting + the planner to emit
explicit cross-room transit steps when the robot's *own
observation-derived* world state indicates the target lives in
a different room**, so that **§1.10.1's multi-room deferral is
lifted, the harness's multi-room scene set is exercised
end-to-end at runtime, and downstream briefs
(harness-mission-generator, clip-eval's multi-room re-test, v2
VLA training data) have a runtime stack they can lean on
**without** the live agent depending on training-time
ground-truth (`scene_metadata.json`) it would not have on a
real robot.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)

Parent design context:
[`STRAFER_AUTONOMY_NEXT.md` §1.10.1](../../../STRAFER_AUTONOMY_NEXT.md#1101-known-limitation-multi-room-navigation)
— the deferral this brief lifts. The mitigation path that
section already names (short-term stored-map fallback + medium-term
room-level transit) is what this brief ships.

Sibling briefs:
- [`scene-connectivity-validation`](scene-connectivity-validation.md)
  — produces the **training-time** connectivity graph in
  `scene_metadata.json`. Consumed by the harness / mission
  generator / grader, **not** by the live planner emitted from
  this brief. See "Sim-to-real boundary" below.
- [`observation-derived-room-state`](observation-derived-room-state.md)
  — produces the *runtime* room-membership + connectivity
  signal that this brief consumes (derived from the semantic
  map + RTAB-Map graph + Nav2 costmap). Hard prerequisite for
  the planner transit-step path. **Must ship first.**
- [`planner-architecture-alignment`](planner-architecture-alignment.md)
  — recorded Option C: planner stays an intent classifier, the
  compiler grows room-aware logic, an advisory `staging_hops`
  slot is reserved on the response schema but not consumed in
  this brief. The compiler-side acceptance criteria below
  match that decision. Decision lives in
  [`STRAFER_AUTONOMY_NEXT.md` §1.10.2](../../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c);
  shared schema lives in
  [`context/planner-request-schema.md`](../../context/planner-request-schema.md).
  **Must ship first.**

## Context

### Sim-to-real boundary (read this first)

The strafer autonomy stack runs in **both** sim-in-the-loop and
on the real D555 robot. On the real robot, `scene_metadata.json`
does not exist — there is no ground-truth room polygon, no
ground-truth object instance list, no Infinigen-emitted room
graph. The robot has its sensors, RTAB-Map's pose graph and
occupancy grid, Nav2's costmaps, the semantic-map
(NetworkX + ChromaDB + CLIP) populated from prior observations,
and the VLM grounding service.

Anything the autonomy stack does in sim must therefore be
**reproducible on the real robot using only those signals**.
`scene_metadata.json` is fair game for the harness (mission
selection, mission grading, oracle path generation,
training-data labels) but is **out of bounds for the live
runtime** — feeding it to the planner or the executor would
ship a system that "works in sim" by cheating on information
the real deployment cannot recover. This is the single most
important constraint the multi-room work must respect.

Concretely, in this brief:

- The **executor-side stored-map fallback** is fine — the
  semantic map is built from the robot's own observations.
- The **planner-side room-level world state** must come from
  [`observation-derived-room-state`](observation-derived-room-state.md)
  (semantic-map clustering + CLIP room labeling + RTAB-Map
  graph reachability), not from
  [`scene-connectivity-validation`](scene-connectivity-validation.md).
- The connectivity-validation brief still produces a graph in
  `scene_metadata.json`; that graph is read **by the harness**
  (mission generator filters cross-room missions to reachable
  pairs, grader scores arrival in the correct room). The
  planner never sees it.

### What §1.10.1 currently says

> `scan_for_target` only finds targets visible from the robot's
> current position. When a target is in a different room (e.g.,
> "go to the kitchen table" while in the living room), the scan
> rotation exhausts all headings without detecting the target
> and the mission fails. Multi-target commands (`go_to_targets`,
> `patrol`) that reference targets in different rooms fail at
> the first out-of-room target.

The section names three options and accepts the deferral. This
brief ships the first two of those three: **stored-map fallback**
and a **planner slice that emits transit steps from room-level
metadata**.

### Stored-map fallback (executor side)

Today's `scan_for_target` flow:

1. Query-before-scan: if the label was seen recently AND the
   current view's top-1 ANN match is within 2 m of the prior
   sighting, skip the scan.
2. Else: rotate up to `max_scan_steps` headings, ground at each.
3. If exhausted: fail with `target_not_found_after_scan`.

The new fallback fires between (2) and (3): **before failing,
check whether the semantic map has a stored sighting for the
label that the query-before-scan rejected for spatial reasons.**
If yes, navigate to that stored pose and re-scan from there.

```
On scan exhaustion in scan_for_target:
  if known := semantic_map.query_by_label(label, max_age_s=3600):
      navigate_to_pose(known.pose)
      re-scan_for_target(label, max_scan_steps=6)
  else:
      fail with target_not_found_after_scan
```

This is small — `_scan_for_target` in
[`mission_runner.py`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
gains a fallback branch before the failure path. The semantic
map query already exists. Recursion bound: one fallback attempt
per mission.

### Planner transit-step emission

The planner needs room-level knowledge to emit transit steps for
out-of-room targets. The runtime-legal source of that knowledge
is the **observation-derived room state** filed at
[`observation-derived-room-state`](observation-derived-room-state.md):

- `current_room: str | None` — best-guess label for the robot's
  current pose (CLIP zero-shot over the latest observation,
  cross-checked against semantic-map node clusters).
- `known_rooms: list[RoomEntry]` — clusters discovered in the
  semantic map so far. Empty at cold-start; populated as the
  robot explores. Each `RoomEntry.observed_objects: list[str]`
  is the deduplicated set of object labels seen in that cluster
  — the compiler's primary target-room inference signal
  (`intent.target_label in known_rooms[X].observed_objects` →
  target room is X).
- `connectivity: list[(room_a, room_b)]` — pairs the robot has
  traversed between (derived from the RTAB-Map / semantic-map
  graph), plus pairs Nav2's global costmap can plan a path
  between. Pessimistic by default — absence of an edge means
  "not yet proven reachable," not "unreachable."
- `target_known_poses: list[Pose2D]` — prior sightings of
  `intent.target_label` from the semantic map, populated per
  request via `SemanticMapManager.query_by_label`. Empty list
  means cold-start; the compiler routes those through
  `explore_until_visible`. See
  [`context/planner-request-schema.md`](../../context/planner-request-schema.md).

Per the Option C decision in
[`STRAFER_AUTONOMY_NEXT.md` §1.10.2](../../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c),
the multi-room logic lives in the compiler — the LLM prompt is
unchanged. The plan-compiler change in
[`planner/plan_compiler.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py)
grows room-aware logic: when the LLM emits a `go_to_target` intent
and the target's *inferred* room differs from the robot's current
room, the compiler prepends a navigation step toward the cross-room
threshold:

```
Standard plan:
  step_01: scan_for_target(label)
  step_02: project_detection_to_goal_pose
  step_03: align_to_goal_yaw
  step_04: navigate_to_pose
  step_05: verify_arrival

Multi-room plan (target room known + reachable):
  step_01: navigate_to_pose(<doorway-or-room-anchor>)
  step_02: scan_for_target(label)
  ...

Multi-room plan (target room unknown):
  step_01: explore_until_visible(label)   ← see frontier-exploration brief
  ...
```

If the LLM emits an advisory `staging_hops` field per
[`context/planner-request-schema.md`](../../context/planner-request-schema.md),
the compiler **ignores it** in this brief — that field is reserved
for the C → B migration filed in
[`STRAFER_AUTONOMY_NEXT.md` §1.10.2](../../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c)
and is not consumed yet.

The "room-anchor" for the transit step is **not** a Infinigen
room-centroid (that's privileged info). It is the most recent
semantic-map node tagged with the target room (if any), falling
back to a frontier pose toward unexplored space. The frontier
fallback ties to
[`frontier-exploration-primitive`](frontier-exploration-primitive.md).

For `go_to_targets` and `patrol` (multi-target intents), the
same transit-or-explore step is prepended per target whenever
the inferred current room differs from the target's inferred
room.

### What this brief is NOT shipping

- **Frontier exploration / room-hopping** for unmapped
  environments. §1.10.1's option 3. Filed separately at
  [`frontier-exploration-primitive`](frontier-exploration-primitive.md).
  This brief *invokes* the primitive when the target room is
  not yet known, but does not implement the primitive itself.
- **Observation-derived room labeling.** Filed separately at
  [`observation-derived-room-state`](observation-derived-room-state.md).
  Hard prerequisite — this brief consumes the manager API
  defined there.
- **Planner architecture choice.** Recorded as Option C in
  [`planner-architecture-alignment`](planner-architecture-alignment.md)
  /
  [`STRAFER_AUTONOMY_NEXT.md` §1.10.2](../../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c).
  This brief ships the compiler-side multi-room work that
  decision implies. The C → B migration (shadow-mode
  `staging_hops`, advisory read, authoritative promotion) is
  out of scope and will be filed as separate briefs at each
  step.
- **Door-state handling.** Doors are assumed open at scene-gen
  time per
  [`scene-connectivity-validation`](scene-connectivity-validation.md);
  runtime door-opening is a future brief.
- **Plan repair.** When a transit step itself fails, the mission
  fails — no automatic re-plan with new transit. Plan-repair
  work is filed separately in `STRAFER_AUTONOMY_NEXT.md` §3.6.

## Acceptance criteria

- [ ] **Stored-map fallback in `scan_for_target`.** When the
      heading-by-heading scan exhausts without grounding,
      `_scan_for_target` queries the semantic map by label;
      if a sighting exists, navigates to that pose, and re-scans
      from there. Bounded to one fallback attempt per mission.
      Unit-tested under
      [`source/strafer_autonomy/tests/`](../../../../source/strafer_autonomy/tests/).
- [ ] **No `scene_metadata.json` access in the live autonomy
      stack.** A grep of
      `source/strafer_autonomy/strafer_autonomy/` for
      `scene_metadata`, `scene_labels`, `get_room_at_position`,
      `room_adjacency`, and `infinigen` returns zero hits after
      this brief ships (test fixtures excepted). The Jetson
      executor and the DGX planner backend may both read the
      semantic map, RTAB-Map's outputs (`/map`, `/rtabmap/...`),
      and Nav2's costmaps; neither may read Infinigen
      training-time metadata.
- [ ] **Planner room-level world state — observation-derived
      only.** The planner request carries the `world_state` block
      per
      [`context/planner-request-schema.md`](../../context/planner-request-schema.md);
      `current_room`, `known_rooms`, and `connectivity` come from
      the runtime `SemanticMapManager` API defined in
      [`observation-derived-room-state`](observation-derived-room-state.md).
      No field on the block traces back to
      `scene_metadata.json`. The prompt itself stays unchanged
      under Option C — the compiler consumes the block, not the
      LLM.
- [ ] **Plan compiler emits transit-or-explore steps.**
      `_compile_go_to_target`, `_compile_go_to_targets`,
      `_compile_patrol`, and `_compile_wait_by_target` infer the
      target's room by scanning
      `world_state.known_rooms[*].observed_objects` for
      `intent.target_label`. Three cases:
      target room == current room → unchanged 5-step plan;
      target room ≠ current room AND target room is in
      `known_rooms` with a `connectivity` edge → prepend
      `navigate_to_pose(<semantic-map-anchor>)`; target room is
      unknown (label not found in any room's `observed_objects`)
      OR unreachable from current room in `connectivity` →
      prepend `explore_until_visible(label)` (see
      [`frontier-exploration-primitive`](frontier-exploration-primitive.md)).
      Anchor pose comes from `SemanticMapManager.room_anchor(
      target_room)` — **never** a scene-metadata centroid.
      Unit-tested under
      [`source/strafer_autonomy/tests/`](../../../../source/strafer_autonomy/tests/).
- [ ] **Smoke test — cold-start.** A mission in a multi-room
      Infinigen scene where the target lives in a different
      room than the robot's start pose AND the robot has no
      prior semantic-map sightings of the target room
      succeeds end-to-end via the bridge harness, with the
      executor logs showing the `explore_until_visible` path
      (not a privileged-centroid path).
- [ ] **Smoke test — warm-start.** Repeat the same mission
      after the robot has previously visited the target room
      in this map. The plan now contains the room-anchor
      transit step (not exploration) and completes faster than
      the cold-start mission. Both mission-summary excerpts in
      the PR description.
- [ ] **Doc surface updates.**
  - [`STRAFER_AUTONOMY_NEXT.md` §1.10.1](../../../STRAFER_AUTONOMY_NEXT.md#1101-known-limitation-multi-room-navigation):
    update the audit-era brief list to stamp this brief as
    shipped in `<this commit>`. Other briefs in the
    decomposition (frontier-exploration-primitive,
    observation-derived-room-state, etc.) update §1.10.1
    independently as they ship. Plan repair remains deferred
    (no brief filed) until that changes.
  - [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md):
    skill table's `scan_for_target` row gains a note about the
    fallback path; planner section mentions room-level world
    state.
  - [`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md):
    multi-room scope subsection updated to reference this brief
    as the runtime path.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in single-room missions. Smoke this in the
      PR description by running an existing single-room mission
      through the bridge harness and confirming the same plan
      structure (no spurious transit step).

## Investigation pointers

- The fallback insertion point:
  [`mission_runner.py:_scan_for_target`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
  around the `target_not_found_after_scan` failure branch.
- The plan compiler:
  [`planner/plan_compiler.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py).
- The planner prompt construction:
  [`planner/prompt_builder.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/prompt_builder.py).
- Connectivity graph schema:
  [`scene-connectivity-validation`](scene-connectivity-validation.md)
  defines it; this brief consumes it via the planner prompt and
  the plan-compiler's room-lookup helpers.
- The semantic map query is already exposed via
  `SemanticMapManager.query_by_label`; the executor's
  `_try_query_before_scan` already uses it for a different code
  path. Reuse the same call.

## Out of scope

- **Frontier exploration implementation.** §1.10.1 option 3.
  Filed at
  [`frontier-exploration-primitive`](frontier-exploration-primitive.md).
  This brief invokes the primitive; it does not implement the
  exploration policy, the frontier detector, or the gain
  function.
- **Observation-derived room labeling implementation.** Filed
  at
  [`observation-derived-room-state`](observation-derived-room-state.md).
  This brief consumes the manager API; it does not implement
  the CLIP zero-shot room classifier, the graph clustering,
  or the connectivity inference.
- **Planner architecture choice.** Filed at
  [`planner-architecture-alignment`](planner-architecture-alignment.md).
  This brief assumes a decision has been made and writes
  acceptance criteria for both outcomes (compiler-side OR
  LLM-side transit emission).
- **Live `scene_metadata.json` consumption.** Explicitly
  forbidden in the live planner / executor — see the
  Sim-to-real boundary subsection above. The harness, mission
  generator, oracle driver, and grader continue to read it.
- **Plan repair on transit failure.** §3.6 of
  `STRAFER_AUTONOMY_NEXT.md`. If the transit step itself fails,
  the mission fails — no auto-replan.
- **Closed-door handling at runtime.** Doors stay assumed-open
  at scene-gen time per the connectivity-validation brief.
  Articulated runtime doors are a separate brief if pursued.
- **Cross-room semantic map cold-start improvements.** The
  validator briefs (clip-eval, learned-validator) handle the
  measurement implications; the *map* mechanics stay as-is.
