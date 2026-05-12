# Tune MPPI critics so sim missions actually use the lifted velocity envelope

**Status:** Shipped 2026-05-03 in `4b55093` (Jetson). Converged at
sustained median odom vx 0.632 m/s — ~63% of the literal acceptance
threshold; remaining gap split across two filed follow-up briefs
([`nav2-startup-unknown-donut-path-noise`](nav2-startup-unknown-donut-path-noise.md)
and [`strafer-inference-package`](../active/strafer-inference-package.md)).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/15

**Type:** task / tuning
**Owner:** Jetson agent
**Priority:** P2
**Estimate:** M (~1–2 days; tuning sweep + per-mission re-validation)
**Branch:** task/mppi-critic-tuning-for-sim-envelope

## Story

As a **mission operator running sim-in-the-loop missions**, I want
**MPPI to *command* velocities close to the configured envelope
(~1.5 m/s linear with `STRAFER_NAV_VEL_SCALE=1.0`) when the path is
clear**, so that **sim missions cover ground in sim time roughly
matching what the trained policy / joystick teleop achieve, instead
of plateauing at one third of the cap because the critic landscape
biases MPPI toward slow precise path-tracking**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../context/bridge-runtime-invariants.md)
- [completed/sim-velocity-attenuation.md](sim-velocity-attenuation.md)
  — the predecessor that proved this is the actual bottleneck and
  shipped the v1 std-scaling fix this brief deepens.

## Context

`completed/sim-velocity-attenuation.md` bisected the symptom and
established two ground truths on 2026-04-29:

1. **The bridge contract is intact and the chassis can hit the cap.**
   A direct manual `ros2 topic pub /cmd_vel … {linear.x: 1.5683}` at
   20 Hz produced sustained `/strafer/odom.linear.x` of 1.37–1.45 m/s
   (sim instantaneous samples) — within ~7–13 % of commanded, with
   the residual attributable to wheel slip / chassis bounce that's
   tracked separately and not the subject of this brief.
2. **MPPI is the bottleneck.** During a `translate forward 3 m`
   mission, `/cmd_vel_nav ≡ /cmd_vel` (smoother passthrough) and
   peak `/cmd_vel.vx` was 0.55 m/s pre-fix → 0.83 m/s after a 2× std
   scaling. Median commanded vx stayed near 0.4 m/s in both cases.
   So MPPI *can* be made to ask for higher velocities (peaks
   improved), but its central tendency stays low — the critic
   landscape rewards slow precise path-tracking over making forward
   progress at speed.

The shipped v1 fix scales `vx_std`, `vy_std`, `wz_std`, and
`prune_distance` by `envelope_factor` (1.0 real-robot, 2.0 sim) in
`navigation.launch.py:_patch_params`. That moved the *exploration
window* but didn't change which trajectories the critics prefer.
Empirically a super-linear (factor**1.5) std scaling pushed
exploration so wide that MPPI started preferring strafe/spin over
forward — i.e. the issue is now critic balance, not noise scale.

## Approach

Tune the MPPI critic weights and a few related knobs so the
preferred-trajectory cost is minimized at higher velocities along
straight paths, without losing precise path tracking around corners
or near goals. Likely surface, ranked by current suspicion:

1. **`PathAlignCritic.cost_weight: 14.0`** — by far the highest
   weight. Penalizes any trajectory whose sampled distance from the
   global path exceeds `max_path_occupancy_ratio`. At higher
   velocities, integration noise puts trajectories slightly off-path
   even on straight segments, so this critic actively suppresses
   speed. Try 8–10. Watch for over-cutting on cornering.
2. **`PreferForwardCritic.cost_weight: 3.0`** — rewards forward
   motion. Bumping to 6–8 should shift the center of mass of the
   sampled distribution toward higher vx (vs. the current preference
   for "any vx that follows path tightly").
3. **`PathFollowCritic.offset_from_furthest: 5`** — MPPI's
   forward-target along the path is 5 path indices past the furthest
   visited. With path resolution ≈ MAP_RESOLUTION (0.05 m), that's
   ~25 cm ahead. Bumping to 15–25 lets MPPI plan toward a target far
   enough that high-speed rollouts win on cost.
4. **`iteration_count: 1` → `2` (or `3`)** — one optimization step
   per control cycle leaves the rolling mean drifting slowly toward
   the optimum. More iterations converge faster but cost CPU.
   Profile on Jetson before committing.
5. **`temperature: 0.3`** — softmax temperature in MPPI's weighted
   trajectory averaging. Lower = greedier (locks onto best). Try
   0.15–0.2 if the noise is wide enough to find good trajectories
   reliably.
6. **`max_robot_pose_search_dist`** (Nav2 MPPI internal) — controls
   how far ahead MPPI searches for the closest path point. Default
   is large; mention here only because it could become relevant if
   `prune_distance` scaling exposes a different bottleneck.

The tuning is sim-only on the surface (we want sim missions to use
the lifted envelope) but the parameters live in
`source/strafer_ros/strafer_navigation/config/nav2_params.yaml` and
are baseline values. Three options for keeping real-robot bringup
unaffected, ordered by preference:

A. **Push critic-weight overrides into `_patch_params` gated on
   `envelope_factor > 1.0`.** Mirrors the existing pattern for std
   and prune scaling. Real-robot keeps YAML defaults verbatim; sim
   gets a tuned profile.
B. **Set new YAML defaults that work for both lanes.** Risky — a
   real-robot regression here is harder to bisect than a sim one,
   and the operator-facing impact (slightly different cornering)
   could be missed for weeks.
C. **A separate `nav2_params_sim.yaml` selected by the sim launch.**
   Cleanest separation but doubles the maintenance surface for any
   future Nav2 param change.

A is the recommended starting point.

## Acceptance criteria

- [ ] On a straight `translate forward 3 m` sim mission with
      `STRAFER_NAV_VEL_SCALE=1.0`, sustained `/strafer/odom.linear.x`
      reaches **≥ 1.0 m/s** (≥ 64 % of cap, ≥ 70 % of the 1.4 m/s
      manual-pub ceiling) over a sliding 1 s window once past the
      acceleration ramp. Recorded as a bisection-style snapshot in
      the PR description, comparable to the predecessor brief's
      run-table format.
- [ ] No regression on the
      [`completed/nav2-far-goal-staging.md`](nav2-far-goal-staging.md)
      reference mission ("Navigate to the open wood door on other
      side of the room"): mission still completes end-to-end with
      ≥ 2 staging legs and the executor still reports ≥ 2
      intermediate Nav2 goals. Faster is fine; slower is a
      regression.
- [ ] No regression on cornering: a mission with a 90° heading
      change inside the room (a `translate forward 1 m` followed by
      a `rotate_by_degrees 90` and another `translate forward 1 m`,
      or any LLM-emitted equivalent) lands at the goal with
      `xy_goal_tolerance: 0.15` and `yaw_goal_tolerance: 0.20`
      satisfied — no overshoot, no oscillation around the goal pose.
- [ ] Real-robot bringup is unaffected — verify via the
      `test_nav_config.py::TestConstantsInjection` test that
      envelope_factor=1.0 leaves the controller config byte-identical
      to the YAML baseline.
- [ ] Unit tests cover whichever option (A / B / C above) ships,
      with at least one assertion per critic / knob touched.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- `source/strafer_ros/strafer_navigation/config/nav2_params.yaml:131-196`
  — MPPI critic block.
- `source/strafer_ros/strafer_navigation/launch/navigation.launch.py:120-150`
  — `_patch_params` is where option (A)'s critic-weight overrides
  would land. The existing std/prune scaling on `envelope_factor` is
  the pattern to mirror.
- The predecessor's bisection scripts (`/tmp/sim-vel/capture.py` +
  `/tmp/sim-vel/analyze.py` on the Jetson at the time of writing,
  but trivially reproducible from the predecessor brief) — drop into
  whichever location makes them re-runnable for this task's
  validation, or rewrite as a small standalone harness committed
  alongside the change.
- Manual-pub ground truth for the chassis ceiling, from the
  predecessor:
  ```
  cmd_vel.linear.x = 1.5683 (commanded)
  /strafer/odom.linear.x sustained ≈ 1.37–1.45 m/s
  /strafer/odom.linear.y / angular.x,y,z all non-zero (~0.1–0.5)
  → ~7–13 % shortfall + chassis bounce; not in scope here.
  ```

## Out of scope

- **Sim physics / chassis-bounce / wheel-slip improvements.** The
  ~10 % shortfall between commanded and observed at the cap is
  physics, not control. Separate brief if it ever becomes a
  blocker.
- **`plan_compiler` per-skill timeout hardcodes.** That's
  [`plan-compiler-skill-timeouts.md`](../active/plan-compiler-skill-timeouts.md);
  unrelated path. Validation of the acceptance criteria here may
  need that brief landed first if mission-level deadlines matter
  during the validation runs (the controller-level acceptance does
  not depend on it).
- **Re-tuning the trained RL policy.** The policy is a separate
  controller from Nav2 MPPI; this brief doesn't touch it.
- **Real-robot critic tuning.** If sim-side tuning surfaces
  improvements that *would* benefit real-robot, file a third brief
  with its own real-robot validation lap. Do not piggy-back.
