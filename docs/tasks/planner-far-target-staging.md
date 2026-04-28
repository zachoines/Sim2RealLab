# Planner emits multi-hop plans for far / cross-room targets

**Type:** task / new feature
**Owner:** DGX agent
**Priority:** P2
**Estimate:** M–L (~2–3 days; world-state schema + planner prompt /
few-shot work + planner-output validation harness)

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
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [completed/goal-projection-depth-range.md](completed/goal-projection-depth-range.md)
  — the predecessor that surfaced the long-horizon problem.
- [`nav2-far-goal-staging.md`](nav2-far-goal-staging.md) — the
  Jetson-side reactive staging loop this brief layers on top of.
  **Must ship first.**

## Context

The Jetson-side reactive staging loop in
[`nav2-far-goal-staging.md`](nav2-far-goal-staging.md) catches
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

Two pieces:

**1. Give the planner enough world-state context to decide.**

The planner needs to know, at minimum:

- Current robot pose in the `map` frame.
- Current global-costmap extent (the SLAM horizon as a rectangle).
- Last VLM grounding result for this target if the executor has one
  cached (depth distance, bbox stability, frame age).

Add a `world_state` field to the planner request payload (or
augment whatever already exists in `schemas/`). The Jetson-side
`mission_runner.py` populates it from existing telemetry; the
DGX-side `planner_client.py` request schema accepts and forwards
it. The planner backend reads it during prompt assembly.

The Jetson populate-side is small enough that it can ship in this
brief's commit (the call site is `mission_runner.py`, but the
schema and payload shape are owned DGX-side). Coordinate via the
operator if Jetson-side review is preferred.

**2. Teach the LLM to emit multi-hop plans when warranted.**

Prompt updates + few-shot examples. The pattern: when the target is
past a "near" threshold (configurable, say 5 m given the 6 m
SLAM horizon), emit
`scan_for_target → navigate(intermediate) → scan_for_target → navigate(final)`
instead of a single `navigate(target)`. The intermediate pose comes
from the LLM's spatial reasoning over the operator's description and
`world_state`, not a geometric formula.

Validate against a small set of test missions (cross-room,
down-hallway, around-corner). The acceptance bar is that the Jetson
reactive staging loop fires ≤1 time on these missions in the happy
path — once the plan is good, reactive staging only catches
final-meter corrections.

## Acceptance criteria

- [ ] `world_state` schema in `source/strafer_autonomy/strafer_autonomy/schemas/`
      carries (at minimum) robot pose in `map`, current
      global-costmap extent rectangle, and a slot for the last
      grounding result. Documented as part of the planner request
      schema. Existing planner requests without `world_state`
      remain valid (additive change).
- [ ] `clients/planner_client.py` accepts and forwards `world_state`
      to the planner service. The Jetson-side populate is wired in
      `mission_runner.py` so the field is filled on every
      `planner.plan(...)` call (small, can ship in this commit;
      coordinate with the operator if Jetson lane review is
      preferred).
- [ ] The planner backend's prompt + few-shot set is updated to emit
      multi-hop plans for far targets. New plans pass the existing
      schema validator unchanged (no new step types — just
      compositions of `scan_for_target` and `navigate`).
- [ ] On a representative sim mission to a >6 m cross-room target —
      the 2026-04-27 reproducer "Navigate to the open wood door on
      other side of the room" is canonical — the planner emits ≥3
      steps (e.g.,
      `scan → navigate(intermediate) → scan → navigate(final)`).
      The executor runs them linearly. The Jetson reactive staging
      loop in [`nav2-far-goal-staging.md`](nav2-far-goal-staging.md)
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
  shipping in [`nav2-far-goal-staging.md`](nav2-far-goal-staging.md)
  already needs to query the global-costmap bounds; reuse that
  helper to populate `world_state.costmap_extent`.
- The failing run from 2026-04-27 reproduces the cross-room
  scenario. Once the Jetson staging task ships and the env override
  is re-enabled, the same scenario becomes the multi-hop test
  fixture.

## Out of scope

- **Executor-level reactive staging.** Tracked at
  [`nav2-far-goal-staging.md`](nav2-far-goal-staging.md) (Jetson).
  This brief assumes that brief has shipped first and uses its
  reactive loop as a safety net.
- **Real-D555 depth range survey** — tracked at
  [`real-d555-depth-range-survey.md`](real-d555-depth-range-survey.md).
  Orthogonal: it informs the projection cap and how often staging
  triggers, not how the planner stages.
- **Semantic-map integration.** Using the persistent semantic map
  to inform intermediate-landmark choice is a natural extension but
  not required here. Prompt-only reasoning over `world_state` is
  the bar for this brief. If plan quality plateaus from
  prompt-only reasoning, file a follow-up to feed semantic-map
  excerpts into `world_state`.
- **Plan validation against the static map.** Checking that
  intermediate landmarks are reachable before sending the plan.
  The Jetson reactive loop covers the unreachable-leg case as a
  fallback; pre-validation is a separate hardening task only worth
  filing if reactive fallbacks fire too often after this brief
  ships.
- **Replacing the LLM planner.** Out of scope; this brief tunes the
  existing planner. A different planner architecture (e.g.,
  hierarchical, tool-using) is a much larger task.
