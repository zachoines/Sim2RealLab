# Multi-layer Nav2 stall watchdog

**Type:** task / enhancement (filed-on-trigger)
**Owner:** Jetson agent
**Priority:** P3 — pickup only when the v1 watchdog produces real-world
false positives or false negatives.
**Estimate:** M–L (~2–4 days; multi-signal integration + tuning + tests)
**Branch:** task/nav-stall-multilayer-watchdog

## Story

As a **mission operator running long-horizon multi-room missions**, I
want **the executor's stall detector to combine chassis-motion,
plan-health, and goal-progress signals**, so that **single-layer
failure modes (re-plan to a much-longer path, planner thrash, motor
wedge masked by commanded motion) cannot spuriously abort or silently
hang missions that the v1 best-ever-`distance_remaining` watchdog
mishandles**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../context/bridge-runtime-invariants.md)
  — "Sim-time-aware navigation timeout" section.
- [completed/progress-aware-nav-timeouts.md](../completed/progress-aware-nav-timeouts.md)
  *(after this brief lands)* — the v1 watchdog this brief generalizes.

## Context

The v1 stall watchdog shipped in `progress-aware-nav-timeouts.md`
tracks the best-ever `distance_remaining` from Nav2 feedback and
fires when no improvement happens within `STRAFER_NAV_STALL_WINDOW_S`.
This is robust to the most common failure mode (re-plan jitter that
bounces `distance_remaining` up and down around a descending trend),
but it has a documented limit and two adjacent failure modes it
cannot detect:

1. **Re-plan to a much-longer path.** A single re-plan that lengthens
   the path by more than `stall_window_s × NAV_LINEAR_VEL` (default
   ~15 m at 20 s × 0.78 m/s) will trip the watchdog because the
   robot can't drive far enough to beat its previous best within the
   window. Single-room scenes don't hit this; large multi-room
   re-plans might.
2. **Planner thrash.** Nav2 keeps re-planning between conflicting
   routes and incrementing `number_of_recoveries`. The robot may
   appear to make occasional progress (resetting the watchdog) while
   never converging on the goal.
3. **Chassis wedge.** `/cmd_vel` commands non-zero motion but
   wheels aren't turning (motor stall, e-stop, jammed against a wall
   the costmap doesn't see). The robot's `distance_remaining`
   sometimes still ticks down due to costmap updates around the
   robot's reported pose, masking the physical stall.

These are real failure modes in production nav stacks. None is a
day-one blocker on the Strafer MVP — they're follow-ups for if the
v1 watchdog produces false positives in cluttered sim or false
negatives in a wedge incident.

## Approach

Layered watchdog in the executor's `_dispatch_nav_goal` (and
`_translate`'s direct `navigate_to_pose` call), with each layer an
independent abort signal:

| Layer | Signal | Default window | Fires when |
|------:|:-------|:--------------:|:-----------|
| 1     | `\|cmd_vel\|` non-zero AND `\|odom.twist\|` near zero | ~3 s    | Physical chassis stall (wedged, e-stopped, motor failure) |
| 2     | `number_of_recoveries` rate (deltas per second) | ~30 s   | Planner thrash — Nav2 itself signaling trouble |
| 3     | Best-ever `distance_remaining` not improved (v1 logic) | `STRAFER_NAV_STALL_WINDOW_S` (20 s) | Goal-relative progress stall |
| 4     | Absolute deadline (`STRAFER_NAVIGATION_TIMEOUT_S`)   | 90 s real / 180 s sim | Pathological / runaway mission |

Each layer aborts with a distinct `error_code` so the operator (and
log analysis) can tell which signal fired:
- Layer 1 → `chassis_wedged`
- Layer 2 → `nav2_recovery_thrash`
- Layer 3 → `navigation_stalled` (existing; v1 semantics preserved)
- Layer 4 → `navigation_timeout` (existing)

Layers compose by OR: any one firing aborts the goal. No layer
"votes" or waits for multi-signal consensus — that's a more
sophisticated design point for a future iteration if the OR-aggregator
turns out to be too aggressive.

Implementation lives in `JetsonRosClient`; `MissionRunner` only sees
new `error_code` values. The `_ProgressTracker` from v1 stays as the
Layer 3 implementation. Layers 1 and 2 add new helpers (e.g.
`_ChassisStallTracker`, `_RecoveryRateTracker`) with the same pure-
helper unit-testable shape.

## Acceptance criteria

- [ ] Each new layer is implemented as a pure helper class with
      `record(...)` + `is_stalled() -> bool`, mirroring the v1
      `_ProgressTracker` pattern.
- [ ] `ros_client.navigate_to_pose` accepts new optional kwargs for
      Layer 1 / Layer 2 thresholds, defaulting to disabled (None) so
      callers that don't opt in get pre-brief behavior. `MissionRunner`
      opts in only when `nav_progress_aware=True`.
- [ ] Each layer fires with a distinct `error_code`; no layer
      shadows another. Tests cover:
      - Layer 1 fires when `cmd_vel.linear` is published with
        non-zero magnitude and `/strafer/odom` reports near-zero
        velocity for the configured window.
      - Layer 2 fires when `number_of_recoveries` increments by ≥ K
        within the configured window (K and window both env-tunable).
      - Layer 3 still passes the v1 re-plan-jitter test verbatim.
      - Layer 4 still respects `STRAFER_NAVIGATION_TIMEOUT_S`.
- [ ] New env knobs follow the existing convention:
      `STRAFER_NAV_CHASSIS_STALL_WINDOW_S`,
      `STRAFER_NAV_RECOVERY_RATE_THRESHOLD`,
      `STRAFER_NAV_RECOVERY_RATE_WINDOW_S`. Each plumbed through
      `executor/main.py` with a doc-string entry, and tested in
      `test_executor_main.py`.
- [ ] The Layer 3 long-detour edge case test
      (`test_replan_to_much_longer_path_eventually_stalls`) is
      adjusted: with Layers 1+2 disabled, behavior is unchanged;
      with all layers enabled, the abort comes from whichever layer
      fires first.
- [ ] Sim repro: a deliberately wedged chassis in Isaac Sim (e.g.
      `--mode bridge` with the robot driven into a wall) trips
      Layer 1 in ≤ 5 s sim-time instead of waiting for the Layer 3
      / Layer 4 fallbacks.
- [ ] If your work invalidates a fact in any referenced context
      module, update it in the same commit. In particular, the
      `bridge-runtime-invariants.md` "Sim-time-aware navigation
      timeout" section's tunables list and behavior description need
      to reflect the new layers.

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py` —
  the v1 `_ProgressTracker` and `navigate_to_pose` watchdog plumb-in
  point. Layers 1 and 2 add peer helpers and additional checks in
  the `_wait_for_nav_result` loop.
- `nav2_msgs/action/NavigateToPose.Feedback` — `number_of_recoveries`
  and `distance_remaining` are both available; Layer 2 reads the
  former from the same `feedback_callback` Layer 3 already uses, so
  no new subscription is needed.
- `JetsonRosClient._latest_odom` — already subscribed at
  `/strafer/odom`; Layer 1 reads `twist.linear` / `twist.angular`
  from the cache.
- `geometry_msgs/Twist` — Layer 1 needs to compare against the
  `/cmd_vel` the executor / Nav2 is publishing. Easiest to subscribe
  to `/cmd_vel` in the client (read-only) rather than thread it
  through from the controller.
- Multi-room re-plan repro for Layer 3 long-detour case requires the
  `multi-room-autonomy-stack` brief to ship first (or a manual
  cross-room mission setup); single-room sim doesn't exercise it.

## Out of scope

- **Probabilistic / Bayesian stall estimation.** Multi-signal voting
  with confidence weights is a research-paper-sized project. This
  brief uses simple OR-aggregation of independent thresholds.
- **Learned failure classifiers.** Same — out of scope for the Nav2
  wrapper layer.
- **Replacing Layer 3 with robot-pose Euclidean distance to goal.**
  Considered during the v1 design; rejected because pose-Euclidean
  is wrong when there's an obstacle (legitimate detours increase it).
  Path-length is the correct measure when the planner is stable; the
  v1 best-ever fix handles the unstable-planner case adequately for
  most scenes.
- **Tuning the v1 thresholds.** The four
  `STRAFER_NAV_BUDGET_*` / `STRAFER_NAV_STALL_*` env knobs from the
  predecessor brief stay in operator's hands; this brief only adds
  new layers, doesn't re-tune existing ones.
