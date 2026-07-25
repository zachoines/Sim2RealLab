# Deploy `navigate_to_pose` succeeds at a radius tighter than the policy was trained to park

**Status:** Shipped 2026-07-24 in `e52c5b1` (Either).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/165

**Type:** task / bug
**Owner:** Either — the deploy success gate is Jetson-side (`strafer_inference`), but the parity source is shared (`strafer_shared`) and referenced by the DGX training env.
**Priority:** P1 — blocks the goal-standoff VLM missions. A hybrid depth-policy mission never reports success and always times out, because the policy parks at its trained arrival distance just *outside* the deploy gate.
**Estimate:** S (a shared constant + reference swaps + a guard test).
**Branch:** task/goal-arrival-radius-parity

## Story

As **the autonomy executor running hybrid DEPTH_SUBGOAL missions**, I want **the deployed `navigate_to_pose` success radius to match the arrival distance the policy was actually trained to reach**, so that **a mission completes when the robot arrives, instead of the robot parking at the goal and the mission timing out.**

## Context

The DEPTH_SUBGOAL objective completes via a **dwell-park**: `path_complete` fires once the robot holds within `dwell_radius_m = 0.30 m` of the path's final point at ≤ `dwell_speed_max_m_s` for `dwell_steps` (`commands.py` `SubgoalCommand`, the `strafer_env_cfg.py` ProcRoom override; the +200 completion reward and the `path_complete` termination both read that one dwell flag). Nothing in the reward pulls the robot closer than the disc, and the low-speed dwell gate rewards *stopping* as soon as it enters — so the policy learns to decelerate and park near the ~0.30 m edge.

Deployment declared success on a tighter, pure-distance gate: `goal_reached_distance_m = 0.25 m` (`inference_node.py` execute loop, straight-line map→base_link TF). `0.25 < 0.30`, so a legitimate trained park in the **[0.25, 0.30] m** band is rejected: the policy holds `cmd_vel ≈ 0` (its objective met) and the mission runs out its budget. Observed directly on `strafer-nx`: a direct-injection mission drove to the goal, parked, and timed out at 240 s; a semantic VLM→policy mission drove to a grounded target, parked, and hit the mission-runner budget — in both, the robot had arrived.

**Diagnosis note.** This surfaced while investigating a suspected "7 s wall-clock nav timeout." That was a mis-diagnosis: the hybrid nav deadline is already sim-clock (it funnels through `_wait_for_nav_result`, converted by `nav-deadline-sim-time-audit`, PR #45). The "7 s" is a correct *sim-time* per-step budget (`compute_motion_budget_s`); it only fires because the policy never trips the too-tight success radius. See [`completed/nav-deadline-sim-time-audit.md`](nav-deadline-sim-time-audit.md).

## What shipped

Promote the arrival radius to one shared constant so the two lanes cannot drift on what counts as "reached":

- `strafer_shared/constants.py`: new **`GOAL_ARRIVAL_RADIUS_M = 0.30`**, beside `SUBGOAL_LOOKAHEAD_M`, same "pin both references here" rationale.
- **Training:** `SubgoalCommandCfg.dwell_radius_m` (default in `commands.py`) and the ProcRoom override (`strafer_env_cfg.py`) now read the constant — **value unchanged (0.30)**, pure plumbing.
- **Deploy:** `inference_node.py` `goal_reached_distance_m` default reads the constant; the redundant `inference.yaml` override is removed so the constant is the single source. **Effect: the deploy gate moves 0.25 → 0.30**, matching training.
- A guard test pins the deploy default to the constant so it cannot be re-hardcoded.

Zero change to commanded motion (the policy drives identically); only the deploy-side declaration of success moves. No retraining.

## Acceptance criteria

- [x] `GOAL_ARRIVAL_RADIUS_M` is the single source for both the training dwell radius and the deploy success radius; no drift-capable literal remains in production source/config (verified: exactly the two production sites route through it).
- [x] Training behavior unchanged — `dwell_radius_m` still 0.30; `test_subgoal_command.py`'s `== 0.3` assertions still hold (`0.30 == 0.3`).
- [x] Deploy completion logic intact — `strafer_inference` runtime suite green (75 passed / 3 skipped) against the change, including a new guard test asserting the deploy default equals `GOAL_ARRIVAL_RADIUS_M`.
- [ ] **Live confirm (rig, operator-run):** with the sim bridge up, a hybrid DEPTH_SUBGOAL mission to a reachable goal reports **SUCCEEDED** when the robot parks (~0.30 m), instead of timing out. Gated on the DGX bridge being up.

## Follow-ups (filed, not blocking)

- **Direct-goal arrival radius still hardcodes `0.3`** in the *separate* fixed-goal ProcRoom objective (`strafer_env_cfg.py` `goal_reached`/`goal_reached_reward`, the curriculum `goal_threshold`, `commands.py` `goal_reach_threshold`). It coincides with `GOAL_ARRIVAL_RADIUS_M` today but is not routed through it, so a retune would drift. The same deploy gate serves the `strafer_direct` backend, so folding these into the constant is a clean consolidation — deliberately left out of this fix's scope (a DGX-lane, broader change).
- **Stall watchdog is nav2-only.** `_navigate_via_hybrid` and `_navigate_via_strafer_direct` pass `tracker=None`, so the `_ProgressTracker` best-ever-distance stall detector runs only on the nav2 backend; a genuinely stuck hybrid mission has only the hard sim-clock deadline. Robustness gap surfaced during this investigation.
