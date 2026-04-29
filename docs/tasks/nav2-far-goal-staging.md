# Stage navigation through the SLAM horizon for far goal-projection targets

**Type:** task / new feature
**Owner:** Jetson agent
**Priority:** P1
**Estimate:** M (~1–2 days; executor staging loop + costmap-bounds query
helper + tests, plus re-enable of the sim env override on green)
**Branch:** task/nav2-far-goal-staging

## Story

As a **mission operator running sim-in-the-loop missions to far
targets**, I want **the executor to drive the robot in stages when a
VLM-grounded goal lands outside the Nav2 global costmap**, so that
**a "navigate to the door across the room" mission completes
end-to-end instead of failing at the planner with `goal off the
global costmap`**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)
- [completed/goal-projection-depth-range.md](completed/goal-projection-depth-range.md)
  — the immediate predecessor; explains the projection-side env
  override that surfaced this failure mode.

## Context

End-to-end testing of the goal-projection env override (commit
`35018b1`) on 2026-04-27 surfaced a Nav2 failure that the prior 6 m
projection cap was implicitly masking:

```
[goal_projection-13] STRAFER_PROJECTION_DEPTH_MAX_M=15.0 overrides projection depth max (=15.0000 m)
[goal_projection-13] Projected 'open wood door' at depth 6.43m → goal (6.10, 5.18) yaw 40°
[planner_server-7]   worldToMap failed: mx,my: 133,133, size_x,size_y: 92,103
[planner_server-7]   The goal sent to the planner is off the global costmap. Planning will always fail to this goal.
```

Cause chain:

- Nav2's global costmap `static_layer` (`nav2_params.yaml:311-314`)
  subscribes to `/rtabmap/map`. Its physical extent is bounded by
  whatever RTAB-Map publishes.
- RTAB-Map runs with `Grid/RangeMax=6.0`, so its grid only fills cells
  within ~6 m of observed keyframe poses.
- Pre-shipped, the goal-projection cap of 6.0 m happened to match.
  Projection couldn't produce a goal past the SLAM horizon, so the
  planner never saw an off-costmap goal — an implicit coupling.
- With sim using `STRAFER_PROJECTION_DEPTH_MAX_M=15.0`, projection
  admits goals up to 15 m. Goals past the explored region land
  outside the costmap rectangle and fail at `worldToMap`.

The mitigation in commit `<this brief's commit>` disabled the sim
override by commenting it out in `env_sim_in_the_loop.env` so
end-to-end coherence is restored (sim reverts to projection-side
reject for far targets). This task ships the real fix and re-enables
the override as part of acceptance.

## Approach

When the projected goal would land outside the global costmap, the
executor drives in stages along the approach vector:

1. Query the current global costmap extent (origin, width, height,
   resolution).
2. If the projected goal is inside the rectangle (with a small
   safety margin for the robot footprint), navigate normally — done.
3. Otherwise, clamp the goal to a point inside the costmap along the
   robot→target vector, with a footprint-radius worth of cells of
   margin from the boundary.
4. Send the clamped pose as a Nav2 goal. Wait for arrival.
5. Once arrived, re-ground via the existing VLM client and
   re-project.
6. Repeat until the projection lands inside the costmap, or a step
   budget (e.g., 4 stages) is exhausted with a structured failure.

The staging loop lives in `mission_runner.py` — wrapping the
existing navigate-with-projection path. **The projection node stays
single-purpose**: this task does NOT add a "clamp to bounds"
parameter to `goal_projection_node.py`. Bounds-awareness is the
executor's responsibility; the projection node remains a pure
pixel→pose mapper.

## Acceptance criteria

- [ ] `clients/ros_client.py` gains a costmap-bounds query helper
      (subscribe to `/global_costmap/costmap` or use Nav2's costmap
      service). Returns a `(min_x, min_y, max_x, max_y)` rectangle in
      the `map` frame, or a structured `not_yet_available` if the
      costmap hasn't been received.
- [ ] The executor's navigate-with-projection path in
      `mission_runner.py` invokes the staging loop. When the
      projected goal is on-costmap, it behaves identically to today
      (no extra Nav2 goals issued, no extra VLM calls). Verify by
      counting Nav2 goals + VLM calls on a near-target reference
      mission before/after.
- [ ] Off-costmap goals trigger staging: at least one intermediate
      Nav2 goal is sent, the robot arrives, the VLM is re-called,
      and projection is re-attempted. The mission log records each
      stage with its clamped goal pose and the resulting projection
      so the operator can read the trajectory after the fact.
- [ ] A bounded step budget (default 4, env-overridable via
      `STRAFER_NAV_STAGING_BUDGET` mirroring the existing
      `STRAFER_NAV_*` pattern) prevents runaway loops. Exhausting the
      budget produces a structured failure with a distinct error
      code (`navigate_via_staging_exhausted` or similar) —
      distinguishable from `goal_projection_failed` and bare Nav2
      timeouts.
- [ ] Re-enable `STRAFER_PROJECTION_DEPTH_MAX_M=15.0` in
      `env_sim_in_the_loop.env` (un-comment the line disabled in
      this brief's predecessor commit). Restore the original
      explanatory comment block.
- [ ] Reproduce the failing mission from 2026-04-27 — "Navigate to
      the open wood door on other side of the room" — runs end-to-end
      on the InfinigenPerception sim scene with the executor
      producing ≥2 intermediate Nav2 goals before the final.
- [ ] Unit tests cover: clamp-along-approach-vector with various
      target positions (inside, just-outside, far-outside,
      diagonal), costmap-bounds query helper (including
      not-yet-available state), step-budget exhaustion path. Mock
      the costmap subscription with a fixed `OccupancyGrid` fixture.
- [ ] No regression on near-target missions (target inside the
      costmap on first projection): same number of Nav2 goals as
      before, no extra VLM calls, no extra rotation steps.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`
  — the navigate-with-projection skill handler. Search for the
  "Projected goal pose:" log line; the `send_nav_goal` call directly
  after is where the staging loop wraps. Existing skill handlers
  (`_scan_for_target` etc.) are a reasonable pattern to mirror for
  the new staging helper.
- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`
  — pattern for the new bounds-query helper. Look at existing
  subscriber+latest-cache patterns in this file (e.g., the way the
  client tracks the latest TF or odom snapshot).
- `source/strafer_ros/strafer_navigation/config/nav2_params.yaml:298-333`
  — global costmap config. Note `track_unknown_space: true` and
  `always_send_full_costmap: true`. The latter means subscribing to
  `/global_costmap/costmap` gets the full grid each update; no need
  to track local updates.
- The failing run from 2026-04-27 reproduces deterministically with
  the env override re-enabled. Use it as the integration smoke
  test before flipping the un-comment in `env_sim_in_the_loop.env`.
- Test scene: InfinigenPerception sim scene (default
  `make sim-bridge-gui` config); mission text "Navigate to the open
  wood door on other side of the room" or any far cross-room target.

## Out of scope

- **Smarter planner-side staging.** The LLM planner emitting
  explicit multi-hop plans is the parallel DGX-side task
  [`planner-far-target-staging.md`](planner-far-target-staging.md).
  It assumes this brief has shipped (and uses this loop as its
  safety net). Plan/code coordination between the two tasks goes
  through the operator, not directly.
- **Changing RTAB-Map's `Grid/RangeMax`.** That value affects
  multiple consumers (semantic mapping, costmap content quality);
  separate decision, separate task if/when it's needed. This brief
  lives entirely above the SLAM layer.
- **Changing the projection-side cap defaults.** The 6.0 / 0.3
  defaults shipped in `35018b1` stay. This brief only re-enables
  the sim-side env override that admits 15 m.
- **Real-D555 depth range survey** — orthogonal; tracked at
  [`real-d555-depth-range-survey.md`](real-d555-depth-range-survey.md).
  Affects how often staging triggers on the real robot, not whether
  staging is correct.
- **Plan validation against the static map.** Checking whether
  intermediate landmarks (chosen by the planner in Task 2) are
  reachable before sending each leg. This brief's reactive loop
  already covers the unreachable-leg case as a fallback. Could be a
  separate hardening task if reactive fallbacks fire too often once
  Task 2 ships.
