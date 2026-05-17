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
  — recorded Option C: planner stays a thin intent classifier,
  the compiler is the staging authority, the LLM prompt is
  unchanged. The compiler-side acceptance criteria below match
  that decision. Decision lives in
  [`STRAFER_AUTONOMY_NEXT.md` §1.10.2](../../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c);
  shared `world_state` schema lives in
  [`context/planner-request-schema.md`](../../context/planner-request-schema.md).
  **Must ship first.**
- [`autonomy-stack`](autonomy-stack.md) — also consumes the
  shared `world_state` block (room-level fields). Whichever of
  the two briefs lands first defines the wire shape in
  `source/strafer_autonomy/strafer_autonomy/schemas/`; the
  second one consumes it.

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

Per the Option C decision in
[`planner-architecture-alignment`](planner-architecture-alignment.md)
/
[`STRAFER_AUTONOMY_NEXT.md` §1.10.2](../../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c),
this brief ships its far-target staging logic in the compiler:
the LLM still emits a single intent, and a new compiler helper
inflates it into the multi-hop sequence when the target is past
the costmap. The prompt itself is **unchanged**. The advisory
`staging_hops` slot on `MissionIntent` (per
[`context/planner-request-schema.md`](../../context/planner-request-schema.md))
is reserved for the C → B migration and **not consumed** here.

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

Two pieces, in dependency order:

**1. Use the canonical `world_state` schema.**

The shared schema lives at
[`context/planner-request-schema.md`](../../context/planner-request-schema.md)
and is co-owned with
[`autonomy-stack`](autonomy-stack.md). Whichever brief ships
first defines the wire types in
[`source/strafer_autonomy/strafer_autonomy/schemas/`](../../../../source/strafer_autonomy/strafer_autonomy/schemas/);
this brief consumes them.

The Jetson populates the entire `world_state` block in
`mission_runner.py`: pose + costmap + last-grounding from its ROS
clients, and the room block from the Jetson-resident
`SemanticMapManager` (see
[`observation-derived-room-state`](observation-derived-room-state.md)
for the runtime room-state methods, and
[`validator-evaluation`](../clip-validation/validator-evaluation.md)
for the wiring that puts the manager into the production
executor process). The DGX planner service is a pure consumer.
The Jetson must populate **all** fields it can fill — a partial
`world_state` is a footgun.

**2. Add `_compile_far_target_staging` to the compiler.** Pattern:
when the target pose is outside `world_state.global_costmap_extent`
(or its in-costmap projection is past a "near" threshold —
configurable, say 5 m given the 6 m SLAM horizon), the compiler
decomposes `go_to_target` into
`scan_for_target → navigate(intermediate) → scan_for_target → navigate(final)`
instead of a single 5-step plan. The intermediate pose is chosen
geometrically: cap the approach vector at the inside-costmap
boundary, or at the nearest semantic-map node tagged with the
target room if `world_state.current_room` and the target's
inferred room differ.

The LLM prompt is **unchanged** under Option C. No few-shot
multi-step examples; no constrained-output decoder; no
prompt-test harness in this brief. The compiler decision is
deterministic from `world_state`. Validate against a small set
of test missions (cross-room, down-hallway, around-corner). The
acceptance bar is that the Jetson reactive staging loop fires
≤1 time on these missions in the happy path — once the plan is
good, reactive staging only catches final-meter corrections.

## Acceptance criteria

- [ ] **One `world_state` schema, shared with autonomy-stack.**
      `source/strafer_autonomy/strafer_autonomy/schemas/`
      carries the combined room + pose + costmap +
      last-grounding schema documented in
      [`context/planner-request-schema.md`](../../context/planner-request-schema.md).
      Whichever of the two briefs lands second uses the schema
      the first one shipped — no parallel `PlannerWorldState`
      variants. Existing planner requests without `world_state`
      remain valid (additive change).
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
- [ ] **Multi-hop emission lives in the compiler (Option C).**
      Per
      [`STRAFER_AUTONOMY_NEXT.md` §1.10.2](../../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c):
      `plan_compiler` grows a `_compile_far_target_staging`
      helper that decomposes `go_to_target` into the multi-hop
      sequence when `world_state` flags the target as
      off-costmap (or past the near-threshold inside the
      costmap). The LLM prompt is **unchanged**. New step
      compositions only — no new step types. Any
      `staging_hops` field the LLM emits per
      [`context/planner-request-schema.md`](../../context/planner-request-schema.md)
      is logged but ignored — the C → B migration consumes it
      in a separate brief.
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
- [ ] Test harness for compiler outputs. At minimum: a small
      set of canned `(intent, world_state)` pairs with expected
      plan shapes (length, step types). Runnable via
      `make test-dgx` or equivalent. Catches compiler
      regressions before they ship. (Under Option C the LLM
      prompt is unchanged, so prompt-regression coverage is
      out of scope — that harness lands with the migration
      brief that promotes `staging_hops`.)
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
