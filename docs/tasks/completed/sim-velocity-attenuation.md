# Investigate sim-in-the-loop `/cmd_vel` velocity attenuation

**Status:** Shipped 2026-04-29 (Jetson) — v1; deeper MPPI tuning
deferred to a follow-up brief (see below).
**PR:** _set after merge_

**Bisection conclusion:** Jetson lane. `/cmd_vel_nav ≡ /cmd_vel`
(velocity smoother passes MPPI's command through unchanged) and the
chassis can reach the cap end-to-end. Manual ground-truth check —
`ros2 topic pub /cmd_vel {linear.x: 1.5683}` at 20 Hz — produced
sustained `/strafer/odom.linear.x` of 1.37–1.45 m/s (sim instantaneous
samples), within ~7–13 % of commanded. So the bridge contract is
intact and the chassis ceiling is ≈ 1.4 m/s. MPPI, by contrast, was
plateauing at peak `/cmd_vel.vx ≈ 0.55 m/s` against the sim cap of
1.5683 m/s during a `translate forward 3 m` mission, with median
near 0.4 m/s — i.e. asking for ~33 % of the cap when the chassis
could deliver ~90 % of it.

**Fix shipped (v1):** `_resolved_nav_velocities` now returns an
`envelope_factor = resolved_scale / NAV_VEL_SCALE` (1.0 real, 2.0
sim), and `_patch_params` scales MPPI's `vx_std`, `vy_std`, `wz_std`,
and `prune_distance` linearly by that factor at launch. Identity at
`envelope_factor=1.0` keeps real-robot bringup byte-identical.
Empirically a super-linear (factor**1.5) std scaling pushed
exploration so wide that MPPI started preferring strafe/spin over
forward progress, so the shipped scaling is linear.

**Post-fix vs pre-fix (`translate forward 3 m`, sim):**

| | peak `/cmd_vel.vx` | peak `/strafer/odom.vx` |
|---|---|---|
| Pre-fix | 0.55 m/s | 0.45 m/s |
| Post-fix (v1, this PR) | 0.83 m/s (+50 %) | 0.53 m/s (+18 %) |
| Manual `ros2 topic pub` ground truth | 1.5683 m/s commanded | 1.37–1.45 m/s sustained |

Peak commanded velocity moves the right direction. Median commanded
velocity (~0.4 m/s) didn't shift meaningfully — that gap is critic
balance, not noise scale, and is the territory of the follow-up.

**Acceptance criteria — what landed:**
- [x] Bisection between `/cmd_vel` and `/strafer/odom` recorded with
      a clear "fix lives in the Jetson lane" conclusion.
- [x] `STRAFER_NAV_VEL_SCALE=1.0` confirmed propagating to Nav2 (via
      patched-yaml inspection — `vx_max=1.5683`, `wz_max=4.1866`).
- [x] No regression on real-robot bringup — `envelope_factor=1.0`
      branch keeps controller config byte-identical to YAML baseline,
      asserted in `test_nav_config.py`.
- [x] If your work invalidates a fact in any referenced context
      module, update that module in the same commit. *(No invalidations.)*

- [ ] **Median sustained `/strafer/odom.linear.x ≥ 0.8 × cap`** —
      not met by v1 (sustained ≈ 0.2–0.5 m/s vs target 1.255 m/s).
      Deferred to the MPPI critic-tuning follow-up below; the v1
      std-scaling lifts MPPI's *exploration* into the new envelope,
      but the critic landscape (especially `PathAlignCritic`,
      weight 14.0) still biases the chosen mean toward slow precise
      path-tracking.
- [ ] **`translate 3 m forward` completes inside its timeout** — not
      met, but for an orthogonal reason: every translate validation
      tripped the planner-side `plan_compiler.py:158`
      `timeout_s=60.0` hardcode, which silently overrides the
      executor's 180 s `STRAFER_NAVIGATION_TIMEOUT_S` sim-time
      budget. Same shape of bug the brief calls out for line 87
      (`navigate_to_pose=90.0`); filed as a separate follow-up.
- [ ] **Rotate-by-degrees regression check** + **far-goal staging
      regression check** — not run. Held pending the timeout
      follow-up; both will fire the same hardcoded-timeout path
      regardless of MPPI tuning, so re-running them today would not
      produce diagnostic data. Re-validate as part of the
      timeout-fix brief or the critic-tuning brief, whichever lands
      first.

**Follow-ups filed:**
- [`mppi-critic-tuning-for-sim-envelope.md`](../mppi-critic-tuning-for-sim-envelope.md)
  — the deeper tuning pass (PathAlign / PreferForward weights,
  PathFollow offset, iteration_count, temperature) needed to push
  median commanded vel toward the cap. Picks up where this v1 ends.
- [`plan-compiler-skill-timeouts.md`](../plan-compiler-skill-timeouts.md)
  — `plan_compiler.py:56-233` hardcodes per-skill `timeout_s`
  values that silently override the executor's
  `STRAFER_NAVIGATION_TIMEOUT_S=180` sim-time budget. Generalizes
  the navigate_to_pose=90 case the brief calls out to translate=60
  and rotate_by_degrees=30, both surfaced during this task's
  validation runs.

**Type:** task / bug
**Owner:** Either (bisection determines the lane — see Approach)
**Priority:** P1 — currently blocks end-to-end mission validation in sim
**Estimate:** S–M (~half day to bisect; M if the fix is an MPPI tuning
sweep with a re-validation lap)
**Branch:** task/sim-velocity-attenuation

## Story

As a **mission operator running sim-in-the-loop missions**, I want
**the robot to translate at the configured velocity envelope (~1.5 m/s
linear with `STRAFER_NAV_VEL_SCALE=1.0`)**, so that **a "navigate
forward 3 m" mission completes in the expected couple of seconds of
sim time instead of timing out at the navigation deadline**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)
- [nav2-far-goal-staging.md](nav2-far-goal-staging.md) — the
  predecessor task whose end-to-end validation surfaced this issue.

## Context

End-to-end validation of `nav2-far-goal-staging.md` on 2026-04-29
confirmed the staging loop works as designed: the global costmap
grows past the SLAM horizon, Nav2 plans across the room to a far
door, and the robot does move toward it. But the robot translates
so slowly that the per-leg navigation timeout trips before completing
the trip:

```
{
  "mission_id": "mission_1c2af7c9dc77",
  "final_state": "timeout",
  "error_code": "navigation_timeout",
  "message": "Navigation timed out after 90s."
}
```

`/strafer/odom` mid-mission, sampled with the robot in active forward
motion (no obstacle contact, no MPPI recovery state):

```
linear.x = 0.087 m/s         (expected near 1.57 m/s with NAV_VEL_SCALE=1.0)
angular.z = 0.063 rad/s      (expected near 4.19 rad/s envelope)
linear.y, linear.z ≈ 0 ± 0.001 m/s
angular.x, angular.y ≈ 0.06 rad/s (oscillation noise, not commanded)
```

The asymmetric per-axis attenuation — linear at 5.5% of cap, angular
at 1.5% — is inconsistent with a uniform downstream scaling factor. A
bridge-side normalization bug (e.g., double-divide by
`MAX_LINEAR_VEL`) would attenuate both axes by the same ratio. The
asymmetry instead points at MPPI commanding small values directly. But
that hypothesis hasn't been bisected against `/cmd_vel` yet — until
that data is in hand the actual cause isn't pinned down.

## Approach

1. **Bisect commanded vs. observed velocity.** With a `translate`
   mission running, capture in overlapping windows:
   - `ros2 topic echo /cmd_vel` on the Jetson — what Nav2 commands
   - `ros2 topic echo /strafer/odom` — what the robot actually does

   Outcomes:
   - `/cmd_vel` magnitude ≈ `/strafer/odom.twist` magnitude → MPPI /
     Nav2 cost-function configuration is the source. Fix lives in
     `source/strafer_ros/strafer_navigation/config/nav2_params.yaml`
     (Jetson lane).
   - `/cmd_vel` >> `/strafer/odom` → bridge or sim physics is
     attenuating between command and motion. Fix lives in
     `source/strafer_lab/strafer_lab/bridge/` or upstream env
     (DGX lane).

2. **Verify `STRAFER_NAV_VEL_SCALE=1.0` actually propagated.** Grep
   the `make launch-sim` console for the
   `STRAFER_NAV_VEL_SCALE=... overrides NAV_VEL_SCALE=...` info-log
   from `navigation.launch.py:_resolved_nav_velocities`. Missing →
   stale Nav2 process or an env-sourcing regression that needs
   straightening out before any further tuning.

3. **Apply the fix in the lane the bisection lands in.** Most likely
   candidates, ordered by current suspicion:
   - **MPPI critic weights / penalties** in `nav2_params.yaml`.
     Common culprits: `path_align_cost`, `prefer_forward_cost`,
     `goal_critic` weights, an undersized `time_steps × model_dt`
     rollout horizon, or conservative `max_accel` keeping the rollout
     from reaching cap on short horizons.
   - **Bridge `cmd_vel` normalization regression** — verify against
     the contract in `bridge-runtime-invariants.md` "cmd_vel
     normalization contract" (`/cmd_vel` arrives in m/s; bridge
     divides by `MAX_LINEAR_VEL`, clamps to [-1, 1], then
     `MecanumWheelAction.process_actions` re-multiplies by
     `_velocity_scale`).
   - **Sim wheel-stall / friction.** If `/cmd_vel` is full-scale but
     the chassis barely moves, the floor mesh / terrain plane
     friction or wheel-contact dynamics in the USD scene are throttling
     translation. Look at `lift_ground_plane_to_floor` and the wheel
     prim setup in the postprocessed scene USD.

4. **Validate.** A `translate 3 m forward` sim mission should complete
   well inside the configured navigation timeout, with median
   sustained `/strafer/odom.linear.x` close to the configured cap once
   past the acceleration ramp. Re-run the cross-room mission from
   `nav2-far-goal-staging.md` and confirm the staging loop still
   completes end-to-end at the higher velocity.

## Acceptance criteria

- [ ] Bisection between `/cmd_vel` and `/strafer/odom` is recorded in
      the PR description, with a clear "fix lives in the {Jetson,DGX}
      lane" conclusion supported by a brief data sample (sustained
      values from each topic over a 1–2 s window).
- [ ] `STRAFER_NAV_VEL_SCALE=1.0` is confirmed to be propagating to
      Nav2 in the `launch-sim` log. If it wasn't, the env-sourcing
      path is fixed in the same PR.
- [ ] A `translate 3 m forward` sim mission completes inside the
      configured navigation timeout with median sustained
      `/strafer/odom.linear.x` ≥ `0.8 × resolved velocity cap` once
      past the initial acceleration ramp.
- [ ] No regression on rotation: a `rotate_by_degrees 90` mission
      completes inside the rotation timeout with sustained
      `angular.z` close to the expected envelope.
- [ ] No regression on the `nav2-far-goal-staging.md` reference
      mission ("Navigate to the open wood door on other side of the
      room"): mission still completes end-to-end with ≥ 2 staging
      legs before the final goal, just faster.
- [ ] Real-robot bringup is unaffected (the env override sourced by
      `env_sim_in_the_loop.env` is sim-only; real bringup leaves
      `STRAFER_NAV_VEL_SCALE` unset and keeps the indoor 0.5 cap).
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- `source/strafer_ros/strafer_navigation/config/nav2_params.yaml:165-333`
  — local costmap + MPPI controller config. MPPI critic blocks are
  the obvious tuning surface; `controller_frequency`, `time_steps`,
  `model_dt`, and the per-critic `cost_weight` values together
  determine how aggressive the rollout is.
- `source/strafer_ros/strafer_navigation/launch/navigation.launch.py:62-98`
  — `_resolved_nav_velocities` resolves `STRAFER_NAV_VEL_SCALE`. Look
  for the override info-log line on launch; absence is diagnostic.
- `source/strafer_lab/scripts/run_sim_in_the_loop.py` (`cmd_vel`
  callback path) + `source/strafer_lab/strafer_lab/bridge/` —
  the bridge side of the cmd_vel normalization contract documented
  in `bridge-runtime-invariants.md`.
- `source/strafer_shared/strafer_shared/constants.py` —
  `MAX_LINEAR_VEL ≈ 1.57 m/s`, `MAX_ANGULAR_VEL ≈ 4.19 rad/s`,
  `NAV_VEL_SCALE = 0.5` (real-robot indoor default).
- Two earlier bridge fixes for related symptoms: commits `d642bff`
  and `70c4ba9` (cmd_vel unit/normalization). They are the contract
  this task should preserve, not regress.

## Related (out of scope here)

- **`plan_compiler.py:87` hardcodes `timeout_s=90.0`** on
  `navigate_to_pose` steps, which silently overrides the executor's
  `STRAFER_NAVIGATION_TIMEOUT_S=180.0` env knob. That's a separate
  bug — file a follow-up brief if, even after fixing the velocity
  attenuation, the 90 s vs. 180 s gap matters for cross-room
  missions in slow-RTF sim.
- **Lowering `STRAFER_PROJECTION_DEPTH_MAX_M`** to mask the slow
  motion by keeping all goals within the SLAM horizon. Don't — that
  re-introduces the off-costmap failure that
  `nav2-far-goal-staging.md` shipped to fix. Fix the cause, not the
  symptom.

## Out of scope

- **Re-tuning the trained RL policy.** Training runs with
  `decimation=4` and the policy is trained on its own velocity
  distribution. Nav2's MPPI is a separate controller; tuning here
  doesn't touch the policy.
- **Bridge / sim RTF improvements.** Sub-unity RTF on the DGX is a
  known constraint (see `bridge-runtime-invariants.md` and
  `PERF_INVESTIGATION_SIM_IN_THE_LOOP.md`). RTF affects wall-clock
  duration, not the m/s magnitude reported on `/strafer/odom`, so
  it cannot explain the value attenuation observed here.
- **Real-robot velocity envelope survey.** Orthogonal — the real
  chassis runs with `NAV_VEL_SCALE=0.5` for indoor safety and isn't
  affected by changes targeting sim.
