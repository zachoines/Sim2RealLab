# Validate the `hybrid_nav2_strafer` backend end-to-end against the sim-in-the-loop rig

**Type:** task / validation
**Owner:** Either — Nav2 + bridge + sim env run on DGX; the inference
node + the hybrid runtime + the comparison scripts run on Jetson. The
brief can be picked up by whichever lane the operator is sitting in
when the un-park trigger fires.
**Priority:** P3 — gates the real-robot hybrid validation follow-up
but not the merge of the hybrid-mode runtime PR itself (the precedent
established by [`inference-package`](../../completed/inference-package.md)
→ [`strafer-direct-sim-validation`](../../active/trained-policy/strafer-direct-sim-validation.md)
is that the runtime PR can ship with unit-testable acceptance closed
and the operator-driven sim validation rides as a separate follow-up).
**Estimate:** M (1–2 days once every prerequisite has shipped and
the rig is up; longer if any prerequisite needs work first).
**Branch:** task/strafer-hybrid-sim-validation

## Story

As a **mission operator about to deploy `hybrid_nav2_strafer` on a
known map**, I want **sim-in-the-loop evidence that Nav2's global
path and the trained `NOCAM_SUBGOAL` policy's local control compose
correctly: subgoal selection matches what the policy was trained on,
the inference loop holds latency, and the cross-room reference
mission completes with sustained velocity**, so that **I can route
operator missions through hybrid mode without a real-robot debugging
session on each new map**.

## Un-park trigger

This brief is parked until **all** of the following have shipped:

1. [`subgoal-env`](../../completed/subgoal-env.md) (DGX,
   active P2) — supplies `PolicyVariant.NOCAM_SUBGOAL`, the
   `SubgoalCommand` term, the path planner, the trained checkpoint,
   and the registered task IDs.
2. [`hybrid-mode`](../../active/trained-policy/hybrid-mode.md) (Jetson, active P3 —
   ships across PRs A/B/C) — supplies the
   `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` dispatch, the Nav2
   `/plan` subscription in the inference node, the rolling-subgoal
   selection logic, and the hybrid-specific watchdog source for
   stale `/plan`.
3. Sim-in-the-loop rig usable (same prerequisite as
   [`strafer-direct-sim-validation`](../../active/trained-policy/strafer-direct-sim-validation.md);
   if it's already up by the time `hybrid-mode` ships, this clause
   is already satisfied).

Un-park by `git mv parked/trained-policy/strafer-hybrid-sim-validation.md
active/trained-policy/strafer-hybrid-sim-validation.md` in the PR that
picks it up.

## Why this brief exists separately

The same logic that extracted [`strafer-direct-sim-validation`](../../active/trained-policy/strafer-direct-sim-validation.md)
out of [`inference-package`](../../completed/inference-package.md):
operator-driven sim validation has no unit-test analog and would
gate the runtime PR's merge indefinitely behind a multi-day rig +
training availability. Pre-filing it here means
[`hybrid-mode`](../../active/trained-policy/hybrid-mode.md)'s implementer (whoever picks it up)
can extract its Phase 3 sim validation into this brief on first
pass, mirror the established precedent, and ship the hybrid runtime
PR with unit-testable acceptance closed.

## Context bundle

Read these before starting:

- [hybrid-mode.md](../../active/trained-policy/hybrid-mode.md) — the consumer-side runtime brief.
  This validation's acceptance criteria refine its Phase 3 ("End-to-end
  sim validation"); pick this brief up only once that one has shipped
  or is in flight on the same branch.
- [subgoal-env.md](../../completed/subgoal-env.md) — the
  DGX-side env + training brief. Its Phase 1 path-planner choice
  (Option A: Nav2 offline at training time, Option B: custom A* +
  per-tick noise) directly affects this brief's subgoal-parity
  acceptance — see "Acceptance criteria > Contract parity" below.
- [`strafer-direct-sim-validation.md`](../../active/trained-policy/strafer-direct-sim-validation.md)
  — the parallel brief on the `strafer_direct` side. Same shape;
  hybrid validation reuses the rig and most of the harness.
- [`inference-package`](../../completed/inference-package.md) —
  the architecture this brief validates an extension of. Phase 4's
  backend-dispatch fallback rules apply identically to the hybrid
  backend (mis-typed `STRAFER_NAV_BACKEND` falls back to nav2;
  inference server unavailable falls back per-mission).
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
  — point 4 (reset triggers) and point 5 (determinism) apply to
  `NOCAM_SUBGOAL` exactly as they apply to `DEPTH`; the assertions
  in this brief restate them in the hybrid-specific call sites.

## What does NOT carry over from `strafer-direct-sim-validation`

Three differences worth flagging before writing the comparison
scripts:

1. **No 4800-dim depth field.** `NOCAM_SUBGOAL` is 19 dims. The
   strafer-direct brief's depth-portion ≤ 1e-3 parity bound has no
   hybrid analog; only the 19-dim NOCAM-portion bound applies.

2. **The goal-related obs fields refer to a *rolling subgoal*, not
   a final goal pose.** Per
   [`subgoal-env`](../../completed/subgoal-env.md)
   Phase 2's `PolicyVariant.NOCAM_SUBGOAL` definition. The
   inference-side referent comes from Nav2's `/plan` topic via the
   rolling-lookahead pick; the training-side referent comes from
   `SubgoalCommand`'s internal pick. Parity therefore requires both
   the obs vector AND the subgoal-pose itself to agree — see the
   acceptance criteria.

3. **Mission start incurs a Nav2 plan latency.** The strafer-direct
   path goes goal-accept → first inference tick → first cmd_vel.
   The hybrid path goes goal-accept → Nav2 plan → `/plan` published
   → first inference tick → first cmd_vel. The plan-latency leg is
   new and must be measured separately from the steady-state
   per-tick latency.

## Acceptance criteria

### Contract parity (rig-only)

- [ ] **NOCAM-fields obs parity (19 dims)**: with a recorded
      sim-in-the-loop rosbag, the inference node's assembled obs
      under hybrid mode matches the
      `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-v0` env's obs
      at the same sim timestamp within ≤ 1e-5 max abs delta.
      Meaningful because both sides derive
      `body_velocity_xy` from the same encoder-FK signal chain (post-
      `observation-contract-cleanup`) and both consume the same
      `SubgoalCommand` referent for the goal-related fields.

- [ ] **Subgoal-pose parity**: at the same sim timestamp, the
      rolling subgoal the inference node picks off Nav2's `/plan` is
      within ≤ `MAP_RESOLUTION * 2` (10 cm) of the subgoal the
      training-time `SubgoalCommand` produces from the gym-env path.
      This is the load-bearing parity bound for the hybrid mode —
      if it doesn't hold, the policy at deployment is tracking a
      subgoal it never saw in training.

      **Tolerance depends on the path-planner choice in
      [`subgoal-env`](../../completed/subgoal-env.md)
      Phase 1.** Option A (Nav2 offline at training time) makes the
      training and deployment planners identical, so any disagreement
      is numerical noise — tighten to ≤ 5 cm if Option A is the
      shipped path. Option B (custom A* + per-tick noise during
      training) trains the policy to be robust within the noise
      envelope — keep the 10 cm bound and verify it falls inside the
      noise envelope the env used.

### Latency (needs the trained checkpoint + TRT runtime)

- [ ] **Per-tick latency p95 < 10 ms** (obs receive → cmd_vel
      publish), same bar as
      [`strafer-direct-sim-validation`](../../active/trained-policy/strafer-direct-sim-validation.md).
      NOCAM_SUBGOAL is 19 dims with no conv stack, so this should be
      easier than DEPTH; record it anyway to anchor the comparison.

- [ ] **Mission-start latency p95 < 500 ms** (goal accept → first
      `/plan` published → first non-zero `/cmd_vel`). Nav2's planner
      dominates this number; the policy contributes one tick. The
      bound is operator-facing — "did the robot start moving" within
      half a second of the goal being submitted. CPU-only fallback
      surfaced separately, not gating.

### End-to-end (gates on the trained checkpoint + the cross-room scene)

- [ ] **Cross-room reference mission**: "Navigate to the open wood
      door on the other side of the room" (from
      [`completed/nav2-far-goal-staging.md`](../../completed/nav2-far-goal-staging.md))
      completes under `STRAFER_NAV_BACKEND=hybrid_nav2_strafer`.
      Nav2's planner publishes the path; the `NOCAM_SUBGOAL` policy
      tracks rolling subgoals along it; the chassis reaches the
      door pose within `xy_goal_tolerance`.

- [ ] **Sustained median odom vx ≥ 1.0 m/s** on the straight
      segments of the cross-room mission. This is the architectural
      threshold the MPPI brief plateaued under at 0.632 m/s — same
      anchor as the strafer-direct validation, applied here as the
      ceiling of "yes, hybrid earns its complexity over Nav2 alone".

- [ ] **No regression on `strafer_direct`**: the translate-forward-
      3-m mission from
      [`strafer-direct-sim-validation`](../../active/trained-policy/strafer-direct-sim-validation.md)
      passes unchanged with `STRAFER_NAV_BACKEND=strafer_direct`.

- [ ] **No regression on `nav2`**: the same cross-room mission with
      unset `STRAFER_NAV_BACKEND` (Nav2-only baseline) reaches the
      door at the existing MPPI velocities. Confirms the hybrid
      dispatch isn't accidentally rerouting nav2 missions.

### Safety / robustness (hybrid-specific)

- [ ] **Plan-freshness watchdog**: unit-tested + sim-exercised. With
      Nav2 silenced mid-mission (kill the planner process or block
      the `/plan` topic), the inference node zero-twists once
      `path_timeout_s` is exceeded. The hybrid-mode brief introduces
      this as the 6th watchdog source — verify under load that it
      actually fires before the chassis stays committed to a stale
      plan.

- [ ] **Goal accept resets the policy**: per
      [`recurrent-policy-contract.md`](../../context/recurrent-policy-contract.md)
      point 4, a new `/navigate_to_pose` goal accept fires
      `policy.reset()`. The `NOCAM_SUBGOAL` checkpoint is recurrent
      under the same GRU-1×128 architecture as DEPTH (per
      [`subgoal-env`](../../completed/subgoal-env.md)
      Phase 5); the reset call sites must consume the canonical
      trigger set, not redefine it.

- [ ] **Per-mission fallback to nav2 when the hybrid runtime is
      unavailable**: the autonomy-side `JetsonRosClient` dispatch
      from
      [`inference-package`](../../completed/inference-package.md)
      Phase 4 falls back to nav2 when
      `wait_for_server` times out. Verify this holds for the
      `strafer_inference` action server under hybrid mode the same
      way it does under `strafer_direct`.

### Trust-boundary documentation (NOCAM_SUBGOAL-specific)

- [ ] **`PolicyVariant.NOCAM_SUBGOAL` docstring** (added by
      [`subgoal-env`](../../completed/subgoal-env.md)
      Phase 2) calls out the costmap-trust caveat: "this variant
      trusts Nav2's costmap absolutely; the deployment lane must
      include a costmap freshness watchdog and must not use it in
      dynamic-obstacle scenarios." Confirm the docstring shipped as
      written; if it didn't, file the fix back at the source brief
      before this validation ships.

- [ ] **Operator-facing runbook** (the
      [`docs/example_commands_cheatsheet.md`](../../../example_commands_cheatsheet.md)
      hybrid section, or a dedicated `docs/INTEGRATION_HYBRID_MODE.md`
      depending on where the hybrid-mode brief lands its docs) flags
      the dynamic-obstacle limitation explicitly. NOCAM_SUBGOAL
      cannot see a moved chair; the operator decides whether the
      scene satisfies the costmap-trust precondition before
      selecting the backend.

## Approach

Three independent runs — each closes one acceptance group and each
can land as its own PR (or fold all three into one). Recommend
running A before B before C; each one's outputs inform the next.

### A. Parity runs (rig-only; trained checkpoint optional)

Identical scaffolding to
[`strafer-direct-sim-validation`](../../active/trained-policy/strafer-direct-sim-validation.md)'s
parity runs, with two hybrid-specific deltas:

1. The gym-env dump instruments the
   `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-v0` env, not
   ProcRoom-Depth. Dump the assembled obs vector AND the
   `SubgoalCommand`'s subgoal pose at every step.
2. The inference-side dump records the obs AND the rolling-subgoal
   pose the runtime picks off Nav2's `/plan`. The comparison joins
   on sim timestamp and asserts both bounds.

Hand-built NOCAM_SUBGOAL-shaped dummy (any 19-dim → 3-dim mapping)
is fine for this run; the policy output is not the bar — the obs
assembly and the subgoal selection are.

### B. Latency benchmark (needs a real ONNX artifact)

1. Export a NOCAM_SUBGOAL-shaped ONNX stub via the export tooling
   (`--variant NOCAM_SUBGOAL`). Doesn't have to be the final
   trained checkpoint for latency — the inference graph shape is
   what matters.
2. Per-tick latency: extend the strafer-direct latency harness from
   [`strafer-direct-sim-validation`](../../active/trained-policy/strafer-direct-sim-validation.md)
   to run under `STRAFER_NAV_BACKEND=hybrid_nav2_strafer`.
3. Mission-start latency: instrument the `JetsonRosClient` dispatch
   path with `t_goal_accept`, `t_plan_first`, `t_cmd_vel_first`
   timestamps. Report p50 / p95 / p99 over ≥ 30 missions.

### C. End-to-end mission (gates on the trained checkpoint)

1. Operator rsyncs the converged `NOCAM_SUBGOAL` checkpoint from
   the DGX training run to the Jetson model path. Update
   `inference.yaml`'s `model_path` (or pass via launch overlay).
2. Bring up `bringup_sim_in_the_loop` against the cross-room
   reference scene.
3. Run the cross-room mission under each backend
   (`strafer_direct`, `hybrid_nav2_strafer`, unset → nav2);
   capture the `tune_capture.py` run-table for each.
4. Verify all three acceptance bullets in the End-to-end section
   above against the recorded data.

## Investigation pointers

- [`hybrid-mode.md`](../../active/trained-policy/hybrid-mode.md) — the consumer brief. Its
  Phase 3 "End-to-end sim validation" section is what this brief
  refines; pre-extracting it here means the hybrid-mode PR can
  ship its runtime with a pointer to this brief instead of
  carrying a half-validated Phase 3.
- [`subgoal-env.md`](../../completed/subgoal-env.md) —
  source of `PolicyVariant.NOCAM_SUBGOAL`, the training task IDs,
  the path-planner choice (Option A vs B), and the trained
  checkpoint. The "costmap-trust" caveat documented in subgoal-env
  Phase 2 is what the trust-boundary acceptance group validates
  shipped correctly.
- [`strafer-direct-sim-validation.md`](../../active/trained-policy/strafer-direct-sim-validation.md)
  — the parallel brief. Rig setup, rosbag harness, comparison-
  script structure all reuse.
- [`source/strafer_ros/strafer_navigation/scripts/tune_capture.py`](../../../../source/strafer_ros/strafer_navigation/scripts/tune_capture.py)
  — the run-table capture harness extended in
  strafer-direct-sim-validation's latency benchmark; reuse for the
  end-to-end velocity check.
- [`completed/nav2-far-goal-staging.md`](../../completed/nav2-far-goal-staging.md)
  — the "navigate to the open wood door" reference mission that
  hybrid-mode's Phase 3 already cites; same mission carries over
  here.
- [`completed/inference-package.md`](../../completed/inference-package.md)
  — Phase 4's backend-dispatch fallback rules apply to
  `hybrid_nav2_strafer` exactly as they apply to `strafer_direct`;
  re-confirm during the safety-acceptance runs.

## Out of scope

- **`DEPTH_SUBGOAL` validation.** The MVP target for hybrid is
  `NOCAM_SUBGOAL`. `DEPTH_SUBGOAL` is parked in
  [`subgoal-env`](../../completed/subgoal-env.md)'s
  Out of scope section as a follow-up; if it's picked up later it
  gets its own validation brief alongside.
- **Real-robot hybrid validation.** File as
  `strafer-inference-hybrid-real-robot-validation.md` once this
  brief's sim validation passes. Real-robot hybrid introduces TF
  freshness concerns (SLAM stalls), Nav2 replan latency under
  load, costmap-update lag, and moved-obstacle distribution shift
  that warrant their own scope.
- **Dynamic-obstacle stress tests in sim.** `NOCAM_SUBGOAL` is
  documented as unsafe in dynamic-obstacle scenarios by
  construction; sim-stressing that boundary is theatre, not
  validation. The real-robot follow-up brief is the right scope
  for that question.
- **Re-tuning the `NOCAM_SUBGOAL` checkpoint if validation surfaces
  a gap.** This brief is pass/fail against the bounds; tuning
  responses are filed as separate training-side briefs back in
  [`subgoal-env`](../../completed/subgoal-env.md)'s
  follow-up queue.
- **Comparing hybrid vs DEPTH-direct performance on the same
  mission.** Evaluation activity, useful for operator guidance but
  not a controller-correctness brief. Run it as a separate
  measurement task if the question becomes operationally relevant.
- **Sim-in-the-loop rig setup.** Already gates
  [`strafer-direct-sim-validation`](../../active/trained-policy/strafer-direct-sim-validation.md);
  if the rig is broken when this brief is picked up, file the
  breakage as its own follow-up rather than fixing it inline here.
