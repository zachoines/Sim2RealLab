# Progress-aware nav timeouts in the executor

**Status:** Shipped 2026-05-09 in `7065c6b` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/21
**Follow-ups:** [`nav-stall-multilayer-watchdog.md`](../active/nav-stall-multilayer-watchdog.md) — multi-signal layered watchdog (chassis-wedge + Nav2 recovery-rate + the existing best-ever progress + absolute deadline). Filed-on-trigger; pick up only if the v1 best-ever-`distance_remaining` watchdog produces real-world false positives or false negatives.

**Type:** task / enhancement
**Owner:** Jetson agent
**Priority:** P2
**Estimate:** M (~1–2 days; executor + ros_client surgery, feedback
plumbing, multi-skill tests)
**Branch:** task/progress-aware-nav-timeouts

## Story

As a **mission operator running translate / rotate / navigate steps
across a wide range of distances**, I want **per-step timeouts to be
derived from how far / how much we're asking the chassis to move,
with stall detection on Nav2 feedback for free-form navigation**, so
that **short moves fail fast when stuck and long moves get the budget
they actually need, instead of every motion skill sharing one coarse
`STRAFER_NAVIGATION_TIMEOUT_S` backstop that's either too tight for
long traverses or too loose for short ones**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../context/bridge-runtime-invariants.md)
  — "Sim-time-aware navigation timeout (Jetson side)" section. The
  sim-clock convention this brief composes with.
- [`plan-compiler-skill-timeouts.md`](plan-compiler-skill-timeouts.md)
  — the predecessor. That brief stops the compiler from overriding
  the env-knob backstop; this brief replaces the backstop itself
  with a per-step budget derived from the skill's args.
- [`rotate-in-place-sim-clock-deadline.md`](rotate-in-place-sim-clock-deadline.md)
  — sibling that puts `rotate_in_place` on the sim-clock. This brief
  assumes that one has landed (or lands at the same time); the
  per-step budgets here only make sense if the deadline source
  itself is correct.
- [completed/sim-velocity-attenuation.md](../completed/sim-velocity-attenuation.md)
  — original surfacing context (translate-3m timeout at low RTF).

## Context

After the compiler stops emitting hardcoded `timeout_s` (predecessor
brief), every motion skill falls through to the executor's
`default_navigation_timeout_s`, sourced from
`STRAFER_NAVIGATION_TIMEOUT_S` (90 s real / 180 s sim). That is one
coarse value applied to:

| Skill | Realistic budget at `NAV_LINEAR_VEL=0.78 m/s` / `NAV_ANGULAR_VEL=2.1 rad/s` |
|-------|----------------------------------------------------------------------------|
| `translate dx=0.5 m`   | ~0.6 s of motion + setup + Nav2 plan ≈ 5–10 s |
| `translate dx=3 m`     | ~3.8 s of motion + Nav2 plan ≈ 15–30 s        |
| `translate dx=10 m`    | ~13 s + recoveries ≈ 60–120 s                 |
| `rotate_by_degrees 30` | ~0.25 s of motion + settle ≈ 3–5 s            |
| `rotate_by_degrees 180`| ~1.5 s of motion + settle ≈ 6–10 s            |
| `navigate_to_pose` (projected target, unknown distance) | depends on Nav2 path |

A single 180 s knob covers all of them today. Short moves that get
stuck waste 3 minutes before failing; long moves still risk being
clipped. Industry practice is **distance-based budgets for
known-displacement skills** plus a **stall watchdog on Nav2 feedback
for free-form navigation**.

The chassis already exposes the constants we need:
[`source/strafer_shared/strafer_shared/constants.py:171-175`](../../../source/strafer_shared/strafer_shared/constants.py)
defines `NAV_LINEAR_VEL ≈ 0.78 m/s`, `NAV_ANGULAR_VEL ≈ 2.1 rad/s`.

Nav2's `nav2_msgs/action/NavigateToPose.Feedback` already publishes
`distance_remaining`, `estimated_time_remaining`, `navigation_time`,
and `number_of_recoveries` on every action tick. The current client
at [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:420-537`](../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
calls `send_goal_async` without a `feedback_callback`, so this signal
is available for the price of one kwarg + a handler.

## Approach

Three layered changes:

1. **Distance-based budgets for `translate`** ([`mission_runner.py:_translate`](../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)).
   When `step.timeout_s` is None, compute:
   ```
   budget = hypot(dx_m, dy_m) / NAV_LINEAR_VEL * SAFETY_FACTOR + SETUP_OVERHEAD_S
   ```
   with `SAFETY_FACTOR ≈ 2.0` (sim cluttered scenes need slack for MPPI
   recoveries) and `SETUP_OVERHEAD_S ≈ 5.0` (Nav2 plan + accel/decel).
   Pass to `ros_client.navigate_to_pose` as the per-step budget.

2. **Distance-based budgets for `rotate_by_degrees`** /
   `orient_to_direction` ([`mission_runner.py:_rotate_by_degrees`](../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py),
   [`mission_runner.py:_orient_to_direction`](../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)).
   Same shape:
   ```
   budget = abs(yaw_delta_rad) / NAV_ANGULAR_VEL * SAFETY_FACTOR + SETUP_OVERHEAD_S
   ```

3. **Stall watchdog on Nav2 feedback** for `navigate_to_pose`
   ([`ros_client.py:420-537`](../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)).
   - Register a `feedback_callback` on `send_goal_async` that records
     `distance_remaining` + a sim-clock timestamp into a small ring.
   - Replace the single-shot `_wait_for_future(result_future, timeout)`
     with a polling loop that, in addition to the existing absolute
     deadline, checks: "if `distance_remaining` has not decreased by
     ≥ `STALL_PROGRESS_M` (default 0.10 m) in `STALL_WINDOW_S` (default
     20 s sim-time), cancel the goal and return
     `error_code=navigation_stalled`."
   - Initial budget for the absolute deadline can be derived from the
     first feedback's `distance_remaining`:
     `max(MIN_NAV_BUDGET_S, distance / NAV_LINEAR_VEL * SAFETY_FACTOR)`,
     bounded above by `STRAFER_NAVIGATION_TIMEOUT_S` as the operator's
     final word.
   - All deadlines remain on `node.get_clock().now()` (sim-clock
     convention from `bridge-runtime-invariants.md`).

`STRAFER_NAVIGATION_TIMEOUT_S` is **demoted to absolute backstop**,
not the primary deadline. Operators who want the old behavior can set
`STRAFER_NAV_PROGRESS_AWARE=0` (escape hatch — see acceptance criteria).

## Acceptance criteria

- [ ] `translate(dx_m, dy_m)` with `step.timeout_s=None` computes its
      budget from `hypot(dx, dy) / NAV_LINEAR_VEL * 2.0 + 5.0`. Unit
      test asserts a 0.5 m translate gets ~6 s, a 10 m translate gets
      ~30 s, instead of both getting `default_navigation_timeout_s`.
- [ ] `rotate_by_degrees(deg)` with `step.timeout_s=None` computes its
      budget from `abs(rad) / NAV_ANGULAR_VEL * 2.0 + 5.0`. Unit test
      mirrors the translate one with 30°, 180°, 720° cases.
- [ ] `orient_to_direction` budget derives from the *delta* yaw, not
      a flat 30 s. Unit test exercises the worst-case (current=east,
      target=west → π rad delta) path.
- [ ] `ros_client.navigate_to_pose` registers a `feedback_callback`
      and exposes a stall watchdog. Test (with a mocked Nav2 action
      server stub) where `distance_remaining` plateaus for 20 s
      sim-time aborts with `error_code=navigation_stalled`; a goal
      whose feedback decreases monotonically completes normally.
- [ ] When the per-step budget would exceed
      `STRAFER_NAVIGATION_TIMEOUT_S`, the env-knob value wins (it's
      the operator's max). Test that.
- [ ] Escape hatch: `STRAFER_NAV_PROGRESS_AWARE=0` falls back to
      pre-brief behavior (single env-knob deadline, no stall
      watchdog, no distance-based budgets) so an operator can
      bisect a regression. Documented in the executor's main module
      env-var preamble.
- [ ] Sim repro from the predecessor brief
      ([`plan-compiler-skill-timeouts.md`](plan-compiler-skill-timeouts.md))
      still passes: `translate forward 3 m` at
      `STRAFER_NAV_VEL_SCALE=1.0` with the env knob at default
      90 s — the per-step budget computes ~30 s and that's what the
      executor enforces. Record before/after in the PR.
- [ ] Real-robot bringup unaffected (no `use_sim_time` flip; sim-clock
      and wall-clock collapse on real hardware). Smoke test on
      `bringup_real.launch.py` if available, otherwise note the lack.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit. In particular,
      the `bridge-runtime-invariants.md` sim-clock note may want a
      cross-reference to the new progress-aware behavior.

## Investigation pointers

- `nav2_msgs/action/NavigateToPose.action` — feedback message
  definition. Fields: `current_pose`, `navigation_time`,
  `estimated_time_remaining`, `number_of_recoveries`,
  `distance_remaining`. We want `distance_remaining` (and possibly
  `number_of_recoveries` as a secondary stall signal — N recoveries
  means Nav2 is also struggling).
- `rclpy.action.ActionClient.send_goal_async(goal, feedback_callback=...)`
  — register the feedback handler at goal-send time.
- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:778-820`
  — existing `_wait_for_future` polling-with-sim-clock helper. The
  stall watchdog can be a sibling polling loop that also checks the
  feedback ring on each iteration.
- `source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py:1468-1525`
  — `_translate` is the simplest place to land the distance-based
  budget pattern; `_rotate_by_degrees` and `_orient_to_direction`
  follow the same shape.
- `source/strafer_shared/strafer_shared/constants.py:171-175` —
  `NAV_LINEAR_VEL`, `NAV_ANGULAR_VEL`. Re-use these (don't redefine).
  Note shared boundary in `ownership-boundaries.md`: append-only
  across the boundary; we're reading existing constants, not
  modifying them.
- The `SAFETY_FACTOR=2.0` and `SETUP_OVERHEAD_S=5.0` defaults are
  starting points — tune against the velocity-attenuation bisection
  data once the plumbing lands.

## Out of scope

- **Compiler-side hardcode removal.** Predecessor brief
  [`plan-compiler-skill-timeouts.md`](plan-compiler-skill-timeouts.md)
  owns that. This brief assumes the compiler emits `timeout_s=None`
  for motion skills and the executor synthesizes the budget.
- **`rotate_in_place` sim-clock deadline.** Sibling brief
  [`rotate-in-place-sim-clock-deadline.md`](rotate-in-place-sim-clock-deadline.md).
  Both can land in either order; this one logically composes on top.
- **Re-tuning `NAV_LINEAR_VEL` / `NAV_ANGULAR_VEL`.** Those are
  chassis specs, not nav-policy knobs. If the safety factor needs to
  go above 3.0 to hit acceptance, the right next step is investigating
  why the chassis isn't tracking nominal speed, not raising the
  factor indefinitely.
- **MPPI / Nav2 planner tuning.** Separate
  [`completed/mppi-critic-tuning-for-sim-envelope.md`](../completed/mppi-critic-tuning-for-sim-envelope.md)
  territory.
- **Replacing the absolute deadline entirely with a stall-only
  watchdog.** Stall-only is risky — a slow drift toward the goal
  could run forever. Keep the absolute deadline as a backstop; this
  brief just makes it scale with the work and adds the stall signal.
