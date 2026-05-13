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
explicit cross-room transit steps when room-level metadata
indicates the target lives in a different room**, so that
**§1.10.1's multi-room deferral is lifted, the harness's
multi-room scene set is exercised end-to-end at runtime, and
downstream briefs (harness-mission-generator, clip-eval's
multi-room re-test, v2 VLA training data) have a runtime stack
they can lean on**.

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

Sibling brief:
[`scene-connectivity-validation`](scene-connectivity-validation.md)
— produces the connectivity graph in `scene_metadata.json` that
this brief consumes.

## Context

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
out-of-room targets. The strafer pipeline has it in
`scene_metadata.json`'s `objects[]` (each object is in a room
polygon) plus the connectivity graph from
[`scene-connectivity-validation`](scene-connectivity-validation.md).

The plan-compiler change in [`planner/plan_compiler.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py):
when the LLM emits a `go_to_target` intent and the target's
known room differs from the robot's current room:

```
Standard plan:
  step_01: scan_for_target(label)
  step_02: project_detection_to_goal_pose
  step_03: navigate_to_pose
  step_04: verify_arrival

Multi-room plan:
  step_01: navigate_to_pose(target_room.centroid)
  step_02: scan_for_target(label)
  step_03: project_detection_to_goal_pose
  step_04: navigate_to_pose
  step_05: verify_arrival
```

The LLM's planner prompt is updated to surface room-level world
state (current room, target room, connectivity graph). The
plan-compiler's `_compile_go_to_target` consumes this and emits
the transit step when needed.

For `go_to_targets` and `patrol` (multi-target intents), the
compiler emits a transit step before each out-of-room target
in the sequence.

### What this brief is NOT shipping

- **Frontier exploration / room-hopping** for unmapped
  environments. §1.10.1's option 3. Out of scope; future brief.
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
- [ ] **Planner room-level world state.** The planner prompt
      receives `current_room` (derived from the robot's pose +
      scene metadata's `rooms[]`) and `objects_by_room[]`
      (target candidates grouped by room) and the connectivity
      graph from
      [`scene-connectivity-validation`](scene-connectivity-validation.md).
- [ ] **Plan compiler emits transit steps.** `_compile_go_to_target`,
      `_compile_go_to_targets`, `_compile_patrol`, and
      `_compile_wait_by_target` consult the LLM's emitted target
      room. When it differs from the current room, the compiler
      prepends a `navigate_to_pose(target_room.centroid)` step
      to that target's sub-plan. Unit-tested under
      [`source/strafer_autonomy/tests/`](../../../../source/strafer_autonomy/tests/).
- [ ] **Smoke test.** A mission in a multi-room Infinigen scene
      where the target lives in a different room than the
      robot's start pose succeeds end-to-end via the bridge
      harness. Captured as a one-line mission-summary excerpt
      in the PR description.
- [ ] **Doc surface updates.**
  - [`STRAFER_AUTONOMY_NEXT.md` §1.10.1](../../../STRAFER_AUTONOMY_NEXT.md#1101-known-limitation-multi-room-navigation):
    update from "deferred" to "stored-map fallback +
    planner transit steps shipped in `<this commit>`; frontier
    exploration + plan repair remain future work."
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

- **Frontier exploration / unmapped multi-room navigation.**
  §1.10.1 option 3. The stored-map fallback only works on repeat
  visits; first visits to an unmapped target room still fail.
  That's an acceptable v1 limitation; a future brief covers
  exploration.
- **Plan repair on transit failure.** §3.6 of
  `STRAFER_AUTONOMY_NEXT.md`. If the transit step itself fails,
  the mission fails — no auto-replan.
- **Closed-door handling at runtime.** Doors stay assumed-open
  at scene-gen time per the connectivity-validation brief.
  Articulated runtime doors are a separate brief if pursued.
- **Cross-room semantic map cold-start improvements.** The
  validator briefs (clip-eval, learned-validator) handle the
  measurement implications; the *map* mechanics stay as-is.
