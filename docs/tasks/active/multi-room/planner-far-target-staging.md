# Planner emits multi-hop plans for far / cross-room targets

**Type:** task / new feature
**Owner:** DGX agent
**Priority:** P2
**Estimate:** M–L (~2–3 days; world-state schema + planner prompt /
few-shot work + planner-output validation harness)
**Branch:** task/planner-far-target-staging

## Story

As a **mission operator watching the planner's emitted plan during
long-horizon missions**, I want **the LLM planner to produce explicit
multi-step plans (scan → drive → scan → drive → final) when the
operator's target is across a room or past the current SLAM
horizon**, so that **the trajectory is interpretable from the plan,
intermediate poses are chosen with the LLM's spatial reasoning
instead of geometric clamping, and the executor's reactive staging
loop is reserved for unexpected long-horizon cases instead of every
cross-room mission**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [completed/goal-projection-depth-range.md](../../completed/goal-projection-depth-range.md)
  — the predecessor that surfaced the long-horizon problem.
- [`nav2-far-goal-staging.md`](../../completed/nav2-far-goal-staging.md) — the
  Jetson-side reactive staging loop this brief layers on top of.
  **Must ship first.**
- [`planner-architecture-alignment`](planner-architecture-alignment.md)
  — decides whether the planner stays a thin intent classifier
  or is promoted to a multi-step planner. The acceptance
  criteria of this brief change depending on the outcome.
  **Must ship first.**
- [`autonomy-stack`](autonomy-stack.md) — also adds a
  `world_state`-shaped prompt input (room-level). Co-design the
  schema so the two briefs share one field, not two competing
  ones.

## Context

### Current planner state (verified 2026-05-13)

The DGX planner today is a thin **intent classifier**, not a
multi-step planner. It emits a single
`{intent_type, target_label, ...}` object per
[`planner/prompt_builder.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/prompt_builder.py);
the deterministic
[`plan_compiler.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py)
then expands every `go_to_target` intent into the same
5-step `scan → project → align → navigate → verify` sequence
via `_compile_single_target_steps`.

This brief proposes a real architecture shift — teaching the
LLM to emit multi-hop sequences. The
[`planner-architecture-alignment`](planner-architecture-alignment.md)
brief decides whether that shift is the right move at all, or
whether the staging logic should live in the compiler instead.
Until that brief lands, treat the "LLM emits multi-hop plans"
language below as one option, not a foregone conclusion.

### Why staging is needed regardless of where it lives

The Jetson-side reactive staging loop in
[`nav2-far-goal-staging.md`](../../completed/nav2-far-goal-staging.md) catches
off-costmap goals and drives the robot in stages along the approach
vector. It works for any mission, including ones the planner emits
as a single `navigate(target)` step — that's its safety-net role.

But geometric clamping is dumb: it always picks "halfway-ish along
the approach vector," which is rarely the *right* place to scan
from. Examples:

- The right intermediate landmark might be a doorway 3 m ahead, not
  the geometric midpoint along a robot→target line that runs
  through a wall.
- The right place to re-ground might be after a 90° turn into a
  hallway, not straight ahead.
- The clamped pose might land in a featureless region where the VLM
  can't re-ground reliably, forcing the staging loop to burn its
  budget on a useless re-projection.

The LLM planner has the context needed to pick better: the
operator's natural-language target description, the structure of
the room from prior grounding outcomes, and (with a small schema
addition) the current SLAM horizon. Teaching it to emit explicit
multi-hop plans for far targets produces interpretable plans and
keeps the Jetson reactive loop as a backstop for cases the planner
mispredicts.

## Approach

Three pieces, in dependency order:

**1. Co-design the `world_state` schema with autonomy-stack.**

Both this brief and
[`autonomy-stack`](autonomy-stack.md) propose adding a
`world_state` block to the planner request. They must share one
schema. Proposed shape:

```python
class PlannerWorldState(BaseModel):
    # Pose / costmap (this brief's primary need)
    robot_pose_map: Pose2D
    global_costmap_extent: Rectangle  # min_x, min_y, max_x, max_y
    last_grounding: GroundingSummary | None  # depth, stability, age

    # Room-level (autonomy-stack's primary need; observation-derived
    # only — see autonomy-stack's "Sim-to-real boundary")
    current_room: str | None
    known_rooms: list[RoomSummary]
    connectivity: list[tuple[str, str]]
```

The Jetson-side `mission_runner.py` populates pose + costmap +
last-grounding; the DGX-side `SemanticMapManager` populates the
room block (see
[`observation-derived-room-state`](observation-derived-room-state.md)).
Both sides must populate **all** fields they own — partial
`world_state` is a footgun for the LLM.

**2. Decide where staging lives (planner-architecture-alignment).**

If the planner stays an intent classifier: the compiler emits
staging hops from `world_state.global_costmap_extent` +
target-pose geometry. No LLM prompt-engineering needed for
staging; far-target staging becomes deterministic.

If the planner is promoted to a multi-step planner: the LLM
emits the staging sequence directly. More flexible (the
intermediate landmark can leverage the LLM's spatial reasoning
over the operator's description), more brittle (prompt
regressions can corrupt unrelated mission shapes).

This brief does not decide; it ships acceptance criteria for
both outcomes.

**3. Teach the chosen planner to emit multi-hop plans for far
targets.** Pattern: when the target is past a "near" threshold
(configurable, say 5 m given the 6 m SLAM horizon), emit
`scan_for_target → navigate(intermediate) → scan_for_target → navigate(final)`
instead of a single `navigate(target)`. Validate against a small
set of test missions (cross-room, down-hallway, around-corner).
The acceptance bar is that the Jetson reactive staging loop fires
≤1 time on these missions in the happy path — once the plan is
good, reactive staging only catches final-meter corrections.

## Acceptance criteria

- [ ] **One `world_state` schema, shared with autonomy-stack.**
      `source/strafer_autonomy/strafer_autonomy/schemas/`
      carries the combined room + pose + costmap +
      last-grounding schema co-designed with
      [`autonomy-stack`](autonomy-stack.md). Whichever of the
      two briefs lands second uses the schema the first one
      shipped — no parallel `PlannerWorldState` variants.
      Existing planner requests without `world_state` remain
      valid (additive change).
- [ ] **No `scene_metadata.json` access.** The DGX planner
      backend reads only `world_state` (observation-derived) +
      semantic-map state. A grep of
      `source/strafer_autonomy/strafer_autonomy/planner/` for
      `scene_metadata`, `scene_labels`, `room_adjacency`
      returns zero hits.
- [ ] `clients/planner_client.py` accepts and forwards `world_state`
      to the planner service. The Jetson-side populate is wired in
      `mission_runner.py` so the field is filled on every
      `planner.plan(...)` call (small, can ship in this commit;
      coordinate with the operator if Jetson lane review is
      preferred).
- [ ] **Multi-hop emission lives where
      `planner-architecture-alignment` says it lives.**
      - If "intent classifier (status quo)": the
        `plan_compiler` grows a `_compile_far_target_staging`
        helper that decomposes `go_to_target` into the
        multi-hop sequence when `world_state` flags the
        target as off-costmap. The LLM prompt is unchanged.
        New step compositions only — no new step types.
      - If "multi-step planner": the LLM prompt + few-shot
        set is updated to emit the multi-hop sequence
        directly. New plans pass the existing schema
        validator unchanged.
- [ ] On a representative sim mission to a >6 m cross-room target —
      the 2026-04-27 reproducer "Navigate to the open wood door on
      other side of the room" is canonical — the planner emits ≥3
      steps (e.g.,
      `scan → navigate(intermediate) → scan → navigate(final)`).
      The executor runs them linearly. The Jetson reactive staging
      loop in [`nav2-far-goal-staging.md`](../../completed/nav2-far-goal-staging.md)
      fires ≤1 time during the mission, allowing for final-meter
      corrections at the last stage.
- [ ] On a near-target mission (target visible and inside the
      costmap on first ground), the planner continues to emit a
      single `navigate` step. No regression in plan length on
      simple missions. Verify with a near-target reference mission.
- [ ] Test/playground harness for planner outputs. At minimum: a
      small set of canned `world_state` payloads paired with
      operator-prompt strings + expected plan shapes (length,
      step types). Runnable via `make test-dgx` or equivalent.
      Catches prompt regressions before they ship.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/planner/` — current
  prompt and few-shot examples. Read end-to-end before changing the
  multi-hop pattern; existing plans assume some structure that the
  new examples need to preserve.
- `source/strafer_autonomy/strafer_autonomy/schemas/` — request /
  response schemas. Look for any existing "world_state"-shaped slot
  before adding a new field; we may have one already from earlier
  work that just isn't populated.
- `source/strafer_autonomy/strafer_autonomy/clients/planner_client.py`
  — request payload construction. Backward-compat note: any change
  here must keep the request shape valid for older clients during
  the transition (server treats absent `world_state` as empty).
- `source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`
  — Jetson-lane call site for `planner_client.plan(...)`. Where the
  populated `world_state` goes in. The Jetson reactive staging loop
  shipping in [`nav2-far-goal-staging.md`](../../completed/nav2-far-goal-staging.md)
  already needs to query the global-costmap bounds; reuse that
  helper to populate `world_state.costmap_extent`.
- The failing run from 2026-04-27 reproduces the cross-room
  scenario. Once the Jetson staging task ships and the env override
  is re-enabled, the same scenario becomes the multi-hop test
  fixture.

## Out of scope

- **Executor-level reactive staging.** Tracked at
  [`nav2-far-goal-staging.md`](../../completed/nav2-far-goal-staging.md) (Jetson).
  This brief assumes that brief has shipped first and uses its
  reactive loop as a safety net.
- **Real-D555 depth range survey** — tracked at
  [`real-d555-depth-range-survey.md`](../investigations/real-d555-depth-range-survey.md).
  Orthogonal: it informs the projection cap and how often staging
  triggers, not how the planner stages.
- **Semantic-map integration.** Using the persistent semantic map
  to inform intermediate-landmark choice is a natural extension but
  not required here. Prompt-only reasoning over `world_state` is
  the bar for this brief. If plan quality plateaus from
  prompt-only reasoning, file a follow-up to feed semantic-map
  excerpts into `world_state`. Note that
  [`autonomy-stack`](autonomy-stack.md) already feeds
  observation-derived room state via `world_state` —
  intermediate-landmark choice from the semantic map is the
  natural next extension after both briefs land.
- **Plan validation against the static map.** Checking that
  intermediate landmarks are reachable before sending the plan.
  The Jetson reactive loop covers the unreachable-leg case as a
  fallback; pre-validation is a separate hardening task only worth
  filing if reactive fallbacks fire too often after this brief
  ships.
- **Replacing the LLM planner.** Out of scope; this brief tunes the
  existing planner. A different planner architecture (e.g.,
  hierarchical, tool-using) is a much larger task.
