# Move hybrid `/plan` replan ownership out of the autonomy client

**Type:** refactor
**Owner:** Jetson agent
**Priority:** P2
**Estimate:** M
**Branch:** task/hybrid-replan-ownership

## Story

As a **mission operator**, I want **one component to own plan freshness
for a hybrid mission**, so that **the client's replan cadence, the
generator's suppression budget, and the inference watchdog stop being
three separately-tuned mechanisms glued by topic freshness**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/conventions.md](../../context/conventions.md)

## Context

`_navigate_via_hybrid` (`strafer_autonomy/clients/ros_client.py`,
~line 1058) currently: sends `NavigateToPose` to the inference node,
then polls Nav2's `ComputePathToPose` action every
`hybrid_replan_period_s` (0.5 s wall) **purely so the planner server's
`/plan` topic side-effect stays fresh** for the subgoal generator. The
coupling is fragile enough that the client carries a runtime warning
when the period is not safely below the generator's ~1.0 s suppression
budget (~line 1090). Three components coordinate a single concern
(plan freshness) through two timeout budgets and a polling cadence —
a hand-rolled miniature of what Nav2's behavior tree does natively
(replan-at-rate + follow).

The elegant shape: the client sends **one** action goal — identical to
`strafer_direct` dispatch — and a mission-side component owns replanning
for the mission's duration.

## Design (decided: Plan A — generator owns replanning)

The subgoal generator learns the mission goal at mission start, calls
`ComputePathToPose` itself on its own cadence, and consumes the path
from the **action result**. The client's replan loop,
`hybrid_replan_period_s`, and the period-vs-suppression warning are
deleted; hybrid dispatch collapses to the `strafer_direct` shape (one
`NavigateToPose`).

**Rejected: B (inference execute loop owns replanning).** It puts an
action client + async futures inside the 50 ms blocking mission loop
in the `ReentrantCallbackGroup` — exactly the concurrency surface
PRs #130/#131 just spent two rounds taming — and couples the RL policy
node to Nav2's planner interface. Plan A keeps the policy node pure and
gives the plan lifecycle to the component that already consumes it.

Two load-bearing specifics:

1. **The goal hand-off is telemetry, not a command.** The inference
   node publishes its accepted goal on a **status** topic (e.g.
   `/strafer_inference/active_goal`), direction node → generator. This
   is not a re-introduction of a goal command channel:
   **`navigate_to_pose` remains the sole command channel into the
   policy** — do not "simplify" later by routing goals to the policy
   through this topic. Publish at goal accept **and** as a slow
   keep-alive (~1–2 Hz) while executing. The generator applies its
   existing freshness idiom (same shape as its `/plan` staleness
   guard): a stale/absent active-goal telemetry means "no mission →
   stop replanning," which handles mission-end for free, and
   publish-at-accept makes preemption re-fire trivially (a preempting
   goal is a fresh accept → fresh telemetry → generator retargets).

2. **Consume the path from the `ComputePathToPose` action result**, not
   the `/plan` topic side-effect. Keep the `/plan` subscription as a
   fallback input. This removes the planner-server topic-side-effect
   dependency that made the client's 0.5 s poll fragile in the first
   place (the poll existed only to keep `/plan` warm).

## Acceptance criteria

- [x] `_navigate_via_hybrid` sends one `NavigateToPose` and no
      `ComputePathToPose` polling; hybrid and direct dispatch differ
      only in backend selection.
- [x] The generator owns the `ComputePathToPose` cadence and consumes
      the path from the action **result**; `/plan` subscription stays
      only as a fallback input.
- [x] The active-goal hand-off is a node → generator **status** topic;
      `navigate_to_pose` remains the sole command channel into the
      policy (assert no goal-command path through the new topic).
- [x] Replan cadence lives in the generator with a single documented
      budget relative to (or replacing) its `/plan` suppression window.
- [x] Plan freshness failure still zero-twists within the existing
      ~2 s composed budget (subgoal watchdog source unchanged).
- [x] Unit coverage for the active-goal telemetry (accept + keep-alive
      + staleness → stop replanning + preemption retarget) and the
      replan trigger; `strafer_inference` + `strafer_autonomy` suites
      green on the Jetson.
- [ ] Hybrid sim mission on the DGX rig: rolling subgoal tracks a
      preempted/updated goal; killing the planner server zero-twists
      within budget.
- [x] If the work invalidates a fact in any referenced context module,
      package README, top-level `Readme.md`, or guide under `docs/`,
      update those in the same commit.

## Out of scope

- The RL policy, observation pipeline, and the subgoal selection math.
- Nav2 controller-plugin architecture (the "implement the policy as a
  Nav2 controller" asymptote — separate investigation if ever).
- `strafer_direct` dispatch.

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`
  `_navigate_via_hybrid`, `_GENERATOR_SUPPRESSION_BUDGET_S`,
  `hybrid_replan_period_s`.
- `source/strafer_ros/strafer_inference/strafer_inference/subgoal_generator_node.py`
  (plan subscription + suppression).
- Depends on / sequenced after
  [`inference-goal-preemption`](../../completed/inference-goal-preemption.md): goal
  updates arrive as preempting action goals, so the goal hand-off in
  Option A must re-fire on preemption.
