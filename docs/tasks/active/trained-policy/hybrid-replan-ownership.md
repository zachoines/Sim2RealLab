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

## Design options (decide before implementing)

**A. Generator owns replanning (recommended).** The subgoal generator
learns the mission goal at mission start, calls `ComputePathToPose`
itself on its own cadence, and consumes the path from the **action
result** (no `/plan` topic scraping; the topic subscription can remain
as a fallback input). Needs a goal hand-off: the inference node knows
the goal (it owns the mission) — e.g. it publishes the accepted goal
(+ a mission-active signal, or a keep-alive the generator treats with
the freshness idiom) to the generator. Keeps planning out of the RL
node; the generator is already the plan consumer.

**B. Inference execute loop owns replanning.** The blocking mission
loop (50 ms cadence, reentrant group) fires `ComputePathToPose` every
~0.5 s. No goal hand-off needed (it has the goal), but it puts an
action client + async plumbing inside the RL node's mission loop and
couples the policy node to Nav2's planner interface.

Either way the client's replan loop, `hybrid_replan_period_s`, and the
period-vs-suppression warning are deleted; hybrid dispatch collapses to
the `strafer_direct` shape.

## Acceptance criteria

- [ ] `_navigate_via_hybrid` sends one `NavigateToPose` and no
      `ComputePathToPose` polling; hybrid and direct dispatch differ
      only in backend selection.
- [ ] Replan cadence lives in the owning component with a single
      documented budget relative to the generator suppression window
      (or replaces the suppression window outright — decide and
      document).
- [ ] Plan freshness failure still zero-twists within the existing
      ~2 s composed budget (subgoal watchdog source unchanged).
- [ ] Unit coverage for the new goal hand-off and replan trigger;
      `strafer_inference` + `strafer_autonomy` suites green on the
      Jetson.
- [ ] Hybrid sim mission on the DGX rig: rolling subgoal tracks a
      preempted/updated goal; killing the planner server zero-twists
      within budget.
- [ ] If the work invalidates a fact in any referenced context module,
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
