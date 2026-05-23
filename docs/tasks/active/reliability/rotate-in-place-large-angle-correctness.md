# `rotate_in_place` no-ops on full turns and takes the short way past 180°

**Type:** task / bug
**Owner:** Jetson agent (`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py` — `rotate_in_place`)
**Priority:** P2
**Estimate:** S–M (~½–1 day; closed-loop target rework + tests + low-RTF re-run)
**Branch:** task/rotate-in-place-large-angle-correctness

## Story

As a **mission operator issuing rotations of any magnitude (including
full 360° sweeps and turns greater than 180°)**, I want **`rotate_in_place`
to traverse the full requested arc in the requested direction**, so that
**a `rotate_by_degrees 360` actually spins the robot once around instead
of returning instant success, and a `rotate_by_degrees 270` doesn't
silently rotate −90° the short way**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [completed/rotate-in-place-sim-clock-deadline.md](../../completed/rotate-in-place-sim-clock-deadline.md)
  — the deadline-source fix; explicitly left convergence/angle
  correctness out of scope ("Convergence-tolerance / physics-bouncing").
- [completed/progress-aware-nav-timeouts.md](../../completed/progress-aware-nav-timeouts.md)
  — where the per-rotation budget (`|yaw_delta_rad| / NAV_ANGULAR_VEL …`)
  is computed; that budget assumes the *full* arc is traversed.

## Context

Surfaced during the `nav-deadline-sim-time-audit` end-to-end
validation (PR #45) on the DGX bridge at RTF ≈ 0.085, 2026-05-22:

- **`rotate_by_degrees 360` returns `succeeded` in ~5 s without
  rotating.** The mission goes `planning → succeeded` with no motion;
  `/cmd_vel.angular.z` stays 0.
- **`rotate_by_degrees 180` works** (verified: 77–82 s wall of
  `0.5 rad/s` rotation, final yaw error 4.1° < tolerance, success).

The cause is in
[`ros_client.py` `rotate_in_place`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py):

```python
initial_yaw = self._yaw_from_quaternion(...)
target_yaw  = self._normalize_angle(initial_yaw + yaw_delta_rad)   # wraps to (-π, π]
...
error = self._normalize_angle(target_yaw - current_yaw)
if abs(error) < tolerance_rad:                                     # already "there"
    return succeeded
```

The controller closes the loop on a **single normalized target yaw**,
which discards how many turns (and which way) were requested:

- `yaw_delta = 2π` → `target = normalize(initial + 2π) = initial` →
  `error ≈ 0` on the first iteration → instant success, **0° rotated**.
- `yaw_delta = 3π/2` (270°) → `target = normalize(initial + 3π/2) =
  initial − π/2` → the robot rotates **−90° the short way**, not +270°.
- Any `|yaw_delta| > π` is collapsed to its `(-π, π]` representative,
  so the *direction* and *magnitude* past a half-turn are both lost.

The `direction = 1.0 if yaw_delta_rad >= 0 else -1.0` line picks the
sign from the *requested* delta, but the loop termination is governed
by the normalized `error`, so for `|delta| > π` the robot can drive in
the requested direction yet stop a half-turn early (or immediately, at
exact multiples of 2π).

This was masked before because the executor rarely issued > 180°
rotations, and the per-step budget / sim-clock work (PRs #21, #24,
#45) all used ≤ 180° test cases.

## Approach

Make the rotation track **accumulated** progress rather than a single
normalized setpoint. Options (pick during design):

1. **Integrate traversed yaw.** Accumulate the unwrapped per-iteration
   yaw delta (`Σ normalize(current - prev)`) and stop when the
   accumulated magnitude reaches `|yaw_delta_rad| - tolerance_rad`,
   keeping `direction` as the commanded sign. Robust to multi-turn.
2. **Decompose into ≤ π segments.** Split `yaw_delta` into a sequence
   of sub-π setpoints and run the existing closed-loop per segment.

Either way:
- Preserve the sim-clock deadline + `_ClockStallDetector` from PR #45
  unchanged (this brief is orthogonal to the deadline source).
- Keep `tolerance_rad` semantics for the final segment.

## Acceptance criteria

- [ ] `rotate_by_degrees 360` traverses a full turn (≈ 2π of
      accumulated yaw) and returns `succeeded`; `/cmd_vel.angular.z` is
      non-zero for the duration. Unit test with a stubbed odom stream
      that advances yaw, asserting accumulated travel ≈ 2π.
- [ ] `rotate_by_degrees 270` rotates +270° in the requested direction,
      not −90°. Unit test asserts direction + magnitude.
- [ ] `rotate_by_degrees 90 / 180` regression: still complete within
      tolerance (no behavior change for ≤ 180°).
- [ ] Low-RTF (≤ 0.1) bridge re-run: a `rotate_by_degrees 360` mission
      completes the full arc without spurious termination, matching the
      180° result already recorded in PR #45.
- [ ] Real-robot bringup unaffected (no clock-source change; pure
      setpoint-tracking logic).

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py` —
  `rotate_in_place` body; `_normalize_angle`, `_yaw_from_quaternion`.
- `source/strafer_autonomy/tests/test_ros_client.py` —
  `TestRotateInPlace*`; add the multi-turn / large-angle cases here.
- PR #45 comment with the e2e evidence (360° no-op vs 180° success at
  RTF ≈ 0.085).

## Out of scope

- **Rotation direction-sign at the planner.** Separate brief
  [`planner-rotate-direction-prompt.md`](planner-rotate-direction-prompt.md)
  (DGX prompt). This brief is purely the executor-side arc tracking.
- **Cancel-mid-rotation.** Separate brief
  [`executor-cancel-mid-motion-cmd-vel-zero.md`](../../completed/executor-cancel-mid-motion-cmd-vel-zero.md).
- **Sim-clock deadline / stall detector.** Shipped in PR #45
  (`nav-deadline-sim-time-audit`); do not re-touch the deadline path.
- **MPPI / Nav2 traverse-speed tuning.** Tracked under
  [`completed/mppi-critic-tuning-for-sim-envelope.md`](../../completed/mppi-critic-tuning-for-sim-envelope.md)
  and its follow-ups; unrelated to in-place rotation.
