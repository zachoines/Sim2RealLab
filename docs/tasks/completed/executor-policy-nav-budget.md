# Size the executor's navigate budget on the policy backends to the policy, not to Nav2's speed

**Status:** Shipped 2026-10-03 in `9b80e99` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/235
**Follow-ups:** [`planner-translate-two-axis-sign`](../active/reliability/planner-translate-two-axis-sign.md) — the planner's two-axis "right" sign;
[`policy-time-to-goal-objective`](../active/trained-policy/policy-time-to-goal-objective.md) — a per-goal bound derived from training;
[`sim-bridge-cheatsheet-trim`](../active/tooling/sim-bridge-cheatsheet-trim.md) — the cheatsheet cut to commands and gotchas

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

- [context/repo-topology.md](../context/repo-topology.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [measurements/goal-a-rig-gate-v3-2026-09-25](../../measurements/goal-a-rig-gate-v3-2026-09-25/README.md)
  — the reach times this brief is sized against, and how the reaches were reached.

## Context

**The budget.**
- The planner's navigate step carries no `timeout_s`.
- So `MissionRunner._motion_timeout_s` (`executor/mission_runner.py`), with the default
  `nav_progress_aware=True`, gives it `compute_motion_budget_s`. That is
  `min(STRAFER_NAVIGATION_TIMEOUT_S = 90, max(5, 2·d / NAV_LINEAR_VEL + 5))` seconds, with
  `d` the straight-line distance at dispatch and `NAV_LINEAR_VEL` = 0.7841 m/s.
- The cap is `STRAFER_NAVIGATION_TIMEOUT_S`: 90 s by default, which the sim-bridge lane keeps;
  the sim-in-the-loop lane sets 180 s.
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

- [x] On the policy backends, the navigate step's budget follows the policy. Either it defers to
      the node's `mission_timeout_s` plus a margin, or it is derived from the policy's measured
      closing rate. The Nav2 backend keeps its progress-aware budget and stall watchdog unchanged.
      *Met 2026-10-03 in `9b80e99`, by deferring.*
      - *A goal that `strafer_direct` or `hybrid_nav2_strafer` executes gets
        `POLICY_MISSION_TIMEOUT_S` (60 s, new in `strafer_shared`) + `policy_budget_margin_s`
        (5 s) = 65 s on the executor's node clock. It is still capped by
        `STRAFER_NAVIGATION_TIMEOUT_S`, and an explicit step timeout still wins.*
      - *The constant is also the node's `mission_timeout_s` default. `inference.yaml` no
        longer pins it, and a config test keeps it out, so the node and the executor read
        one value.*
      - *The budget reaches only the policy branches of `navigate_to_pose`
        (`policy_timeout_s`). The per-mission fallback to Nav2 keeps `timeout_s` and the stall
        watchdog.*
      - *It applies to translate legs too: on a policy backend they go to the same action
        server, and the executor's documented CLI form for a pose goal is a translate.*
      - *Why not a closing rate: the slow part of a reach is the terminal approach (7–14 s sim
        on the gate's three over-budget reaches, 1.3–5.1 s on the others), and it does not
        scale with distance. A rate is specific to one artifact (v2 closed at
        0.014–0.083 m/s, v3 at 0.15–0.41). A rate-derived budget would still cancel short
        goals before the node decides, which is the failure being fixed. With the deferral the
        node decides every step, and `navigation_timeout` means only an executor backstop.*
- [x] A test pins the policy-backend budget for a 3.1 m goal at or above the v3 gate's 19.0 s sim
      reach, and the Nav2 budget for the same goal at today's value.
      *Met 2026-10-03:*
      - *`test_progress_aware_timeouts.py::TestPolicyBackendBudget` pins 65 s (above the gate's
        19.0 s reach) and above the node's own 60 s, on both policy backends.
        (2026-10-05: the test is now `test_3p1_m_goal_policy_budget_is_node_bound_plus_margin`;
        its 19.0 s assertion gave way to a 65 s value pin.) It pins the Nav2 budget at
        12.9072 s with its stall watchdog. It also covers the explicit-timeout, ceiling, legacy
        and translate paths, and checks distance independence.*
      - *`test_ros_client.py::TestNavigateToPoseDeadlineRouting` checks that only a policy
        backend receives the policy deadline, including through the Nav2 fallback.*
      - *The node and config suites pin the shared default.*
- [x] A CLI-submitted confirmation set, with the executor in the loop, runs on the sim-bridge lane
      and its results are recorded against the node-driven v3 gate.
      *Met 2026-10-03:*
      - *The record is
        [`goal-a-cli-confirmation-2026-10-03`](../../measurements/goal-a-cli-confirmation-2026-10-03/README.md),
        pre-registered.*
      - *Four of the five missions were submitted with `make submit-deploy`. Each ended on
        the node's own `SUCCEEDED`, the executor agreeing. There was no executor cancel, so the
        budget meets the outcome rule fixed before the first mission.*
      - *R1 was not submitted: the planner mis-signed its two-axis "right", and the dry-check
        caught it. Filed as `planner-translate-two-axis-sign`.*
      - *The four reaches (8.5–12.5 s sim) were all under the old budget too. The set confirms
        the CLI path under the new budget; it does not contain a mission the old one would
        have cancelled.*
- [x] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
      *Met 2026-10-03 in `9b80e99`:*
      - *`context/bridge-runtime-invariants.md` (the policy-backend budget);*
      - *`docs/sim_bridge_autonomy_cheatsheet.md` (mission completion);*
      - *the executor's env docstring;*
      - *the `strafer_autonomy` README's `RosClient` signature.*
- [x] No regression in the workflows the touched code supports.
      *Met 2026-10-03, at `9b80e99` against `main` (`ed0d5af`):*
      - *`make test-ros`: 822 passed.*
      - *Autonomy suite: 599 passed, 79 skipped (main 591).*
      - *`requires_ros` autonomy files: 91 passed (main 87).*
      - *`test-driver`: 60 passed.*
      - *`make test-autonomy` in the container lane fails collection on
        `test_sim_in_the_loop_runtime_env.py` (`isaaclab_tasks` not importable) identically on
        `main`. It is pre-existing and was ignored for the counts.*
      - *The container lane now mounts the tree's `strafer_shared`, as the ROS lane already did,
        so a shared-constant change is tested against the tree.*

## Open question: the 60 s bound itself

*2026-10-09.* This brief deferred the executor to the node's bound; it did not choose the bound.
- **What 60 s is.** The inference node's `mission_timeout_s` default since the DEPTH MVP
  (`e857d33`, 2026-05-25), counted on the node clock since `86e9590` (2026-07-03), and unchanged
  through v2, v3 and both gates. This brief moved it into `strafer_shared` without changing it.
- **What it is not derived from.**
  - *Training.* A training episode is one goal tracked along a planned path, capped at 20 s
    (`_DEFAULT_NAV_EPISODE_LENGTH_S`). v3's mean episode is 163.9 steps (about 5.5 s), with a
    time_out termination share of 0.0010. So 60 s is three times the horizon the policy was
    trained on, and a goal can run the recurrent policy for up to 1800 steps against training's
    600, which [`goal-a-rig-gate-2026-08-17`](../../measurements/goal-a-rig-gate-2026-08-17/README.md)
    lists as untested.
  - *A closing rate.* Nor is it derived from one.
- **Against the reaches measured so far.** Its headroom depends on the artifact.
  - **v3.** The gate's longest reach was 19.0 s sim (G1, 13.9 s of it between 0.42 m and the
    radius). The longest to date is 24.98 s sim for the same goal, through the executor with the
    livestream on (`goal-a-cli-video-2026-10-09`, #237), 18.7 s of it in that band. 60 s is
    about 3.2× and 2.4× those.
  - **v2.** Its one reach on the rig, the 2026-08-17 pilot (`PILOT_uncontrolled_heading`), took
    53.6 s sim for 3.03 m: 6.4 s inside the bound.
  - **Aborts at the bound.** Two v3 gate missions, R2 and R3, ended `ABORTED` there. R2 had held
    0.347–0.420 m for 52.7 s sim.
- **Distance.** No part of the bound scales with distance.
  - *A projected navigate goal* lies within the projection's depth range
    (`STRAFER_PROJECTION_DEPTH_MAX_M`: 6 m by default, 15 m on the sim-in-the-loop lane).
  - *Staging* drives such a goal in clamped legs, each with its own bound, only when it falls
    outside the global costmap. That is the mapped area, not a window.
  - *A translate* is dispatched as one goal without staging. Its displacement is what the step
    commands, which the executor does not cap; this time bounds only how long it may run.
- **Open.** A principled per-goal time bound for the policy backends has not been established.
  - Candidates: measured closing rates per artifact; a distance term with a floor, which has to
    hold a terminal approach that does not scale with distance; or a dwell rule at the arrival
    radius.
  - It belongs to the terminal-approach parity brief, to be filed before the real-lane gate is
    pre-registered.
  - *2026-10-10.* 60 s stays in the meantime. The bound is to come from training instead:
    [`policy-time-to-goal-objective`](../active/trained-policy/policy-time-to-goal-objective.md)
    makes time to goal an objective and derives the bound from the arrival times of the policy
    trained on it, or from v3's measured baseline if no objective is adopted. The deploy success
    rule, against training's dwell, stays with the terminal-approach parity brief.

## Out of scope

- Changing `mission_timeout_s`, or the policy itself.
- The Nav2 backend's budget and stall watchdog.
- Adding a stall watchdog to the policy backends. The terminal-approach behaviour in the record
  (commanding with little chassis motion near the goal) would trip a progress watchdog sized for
  Nav2, so that needs its own measurement first.
