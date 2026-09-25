# Size the executor's navigate budget on the policy backends to the policy, not to Nav2's speed

**Type:** bug (mission outcome, trained-policy backends)
**Owner:** Jetson (`strafer_autonomy` executor lane)
**Priority:** P1 — on the v3 sim-bridge gate's own timings, a language mission routed through
the executor would have been cancelled short of the goal on three of the seven v3 reaches, the
policy's action server having gone on to reach each of them. Goal-a's "via the autonomy CLI"
clause cannot close until this is fixed.
**Estimate:** S
**Branch:** `task/executor-policy-nav-budget`

## Story

As the **executor dispatching a navigate step to a trained-policy backend**, I want **the step's
time budget to follow the policy's own completion bound, or its measured closing rate**, so that
**a mission is not cancelled on a goal the policy is still converging on.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [measurements/goal-a-rig-gate-v3-2026-09-25](../../../measurements/goal-a-rig-gate-v3-2026-09-25/README.md)
  — the reach times this brief is sized against, and how the reaches were reached.

## Context

**The budget.**
- The planner's navigate step carries no `timeout_s`.
- So `MissionRunner._motion_timeout_s` (`executor/mission_runner.py`), with the default
  `nav_progress_aware=True`, gives it `compute_motion_budget_s`. That is
  `min(STRAFER_NAVIGATION_TIMEOUT_S = 90, max(5, 2·d / NAV_LINEAR_VEL + 5))` seconds, with
  `d` the straight-line distance at dispatch and `NAV_LINEAR_VEL` = 0.7841 m/s.
- The budget runs on the executor's node clock, which is sim time on the sim lanes.
- It is sized for Nav2 driving at its nominal speed.

**What the policy backends do with it.**
- `_navigate_via_hybrid` and `_navigate_via_strafer_direct` in `clients/ros_client.py` wait on
  the result with `tracker=None`, so they get the deadline without the progress (stall) watchdog
  the Nav2 path has.
- When the deadline passes they call `cancel_goal_async()` and return
  `navigation_timeout`. The policy is stopped where it is.

**Against the v3 gate.** That gate drove the node's action server directly, not through the
executor, so it was not affected. Its reach times against this budget:

| run | start distance | budget | reach time | |
|---|---:|---:|---:|---|
| G1 | 3.11 m | 12.9 s | 19.0 s | over |
| L1 | 2.07 m | 10.3 s | 11.1 s | over |
| L2 | 3.63 m | 14.3 s | 15.4 s | over |
| R1 | 2.16 m | 10.5 s | 6.6 s | under |
| F1 | 3.09 m | 12.9 s | 10.7 s | under |
| F2 | 3.09 m | 12.9 s | 6.8 s | under |
| F3 | 3.07 m | 12.8 s | 6.8 s | under |

The three over-budget reaches are the ones the record shows decided in the terminal
centimetres: 7–14 s sim held between 0.30 and 0.42 m. The node's own bound is `mission_timeout_s`
= 60 s sim.

## Acceptance criteria

- [ ] On the policy backends, the navigate step's budget follows the policy. Either it defers to
      the node's `mission_timeout_s` plus a margin, or it is derived from the policy's measured
      closing rate. The Nav2 backend keeps its progress-aware budget and stall watchdog unchanged.
- [ ] A test pins the policy-backend budget for a 3.1 m goal at or above the v3 gate's 19.0 s sim
      reach, and the Nav2 budget for the same goal at today's value.
- [ ] A CLI-submitted confirmation set, with the executor in the loop, runs on the sim-bridge lane
      and its results are recorded against the node-driven v3 gate.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports.

## Out of scope

- Changing `mission_timeout_s`, or the policy itself.
- The Nav2 backend's budget and stall watchdog.
- Adding a stall watchdog to the policy backends. The terminal-approach behaviour in the record
  (commanding with little chassis motion near the goal) would trip a progress watchdog sized for
  Nav2, so that needs its own measurement first.
