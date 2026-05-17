# Precheck RTAB-Map tracking health before each motion step

**Type:** task / bug
**Owner:** Jetson agent (`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`, `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`)
**Priority:** P2
**Estimate:** S–M (~1 day; reuse existing `check_slam_tracking` helper; add a wait-and-retry per skill + tests)
**Branch:** task/executor-slam-tracking-precheck-mid-mission

## Story

As a **mission operator running long-horizon missions where RTAB-Map
may drop tracking mid-run (loop-closure rejected, kidnapped-robot,
featureless corridor)**, I want **each motion step to verify RTAB-Map
is still localized before dispatch and to bound the wait when it
isn't**, so that **the executor stops driving on stale `map → odom`
TF and either recovers (waits for re-localization) or aborts cleanly
with `slam_tracking_lost` instead of silently navigating to a phantom
pose**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [completed/progress-aware-nav-timeouts.md](../../completed/progress-aware-nav-timeouts.md)
  — the watchdog pattern this brief borrows the env-knob style from.

## Context

The executor never queries SLAM tracking health. `JetsonRosClient`
already has the primitive at
[`ros_client.py:456-474`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
(`check_slam_tracking()`), but no call site in
[`mission_runner.py`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
invokes it before dispatching motion. Failure modes the gap admits:

1. **Loop-closure rejection.** RTAB-Map's
   `Rtabmap.cpp:3069::process() Rejected loop closure ... Not enough
   inliers 0/15` log line (observed on cold-start in
   [`rtabmap-cold-start-determinism.md`](rtabmap-cold-start-determinism.md))
   does not actually reset SLAM, but it means the current frame
   didn't match any known place. If many consecutive frames reject,
   the `map → odom` transform drifts with wheel odometry alone (no
   gyro fusion — see
   [`imu-yaw-drift-no-magnetometer.md`](../../parked/reliability/imu-yaw-drift-no-magnetometer.md)).
2. **Kidnapped-robot.** Operator picks the chassis up and moves it
   (debugging, real-robot lab moves). RTAB-Map keeps publishing
   `map → odom` until it finds a relocalization, but the TF lookup
   the executor uses to anchor goal poses is wrong during the
   intervening seconds-to-minutes.
3. **Featureless-corridor tracking loss.** A long blank wall, a dim
   room, a static white ceiling — RTAB-Map's odometry-only fallback
   pose drifts until it can re-anchor.
4. **Cold-start identity-reset.** Already
   [filed](rtabmap-cold-start-determinism.md) but it's the same family
   — the executor would benefit from refusing to dispatch motion while
   RTAB-Map's `map → odom` transform is in flux.

Today's failure mode is silent: the executor reads a stale TF, dispatches
a Nav2 goal in `map` frame, Nav2 plans to a phantom pose, the robot
drives somewhere wrong. There's no error code. The only signal is
that `_verify_arrival` (when it's wired) fails at the end —
expensive, late.

The reference primitive
([`ros_client.py:456-474`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py))
already exists; this brief uses it.

```python
def check_slam_tracking(self) -> dict[str, Any]:
    """Return SLAM tracking health snapshot from RTAB-Map status topic."""
    # ... reads /rtabmap/info or equivalent; returns {"tracking": True/False, "reason": str | None}
```

Peer reference: RTAB-Map's own
[FAQ](https://github.com/introlab/rtabmap/wiki/FAQ) recommends
checking the `MapData.loop_closure_id` and `info.loop_closure_id` /
`info.proximity_closure_id` topics for tracking health, and using
`/rtabmap/get_map_data` if a finer query is needed. The auto-reset
identity-pose increment-map-id branch
([issue #80](https://github.com/introlab/rtabmap_ros/issues/80)) is
the canonical "you've lost tracking" signal that this brief surfaces
to the executor.

## Approach

### A. Precheck + bounded wait before each motion step (recommended)

A new helper `_await_slam_tracking(timeout_s)` on `MissionRunner`:

```python
def _await_slam_tracking(self, timeout_s: float, started_at: float) -> SkillResult | None:
    """Return None if SLAM is tracking; SkillResult(failed, slam_tracking_lost)
    otherwise. Waits up to timeout_s sim-seconds for tracking to recover."""
    deadline = self._ros_client.clock_now() + timeout_s   # sim-time aware
    while True:
        status = self._ros_client.check_slam_tracking()
        if status.get("tracking", True):
            return None
        if self._ros_client.clock_now() >= deadline:
            return self._failed_result(
                step, f"SLAM tracking lost: {status.get('reason')}",
                "slam_tracking_lost", started_at,
            )
        time.sleep(0.1)
```

Call sites that invoke this before motion:
- `_navigate_to_pose` / `_dispatch_nav_goal` ([`mission_runner.py:959-1014`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py))
- `_translate` ([`mission_runner.py:1572-1660`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py))
- `_align_to_goal_yaw` ([`mission_runner.py:1667-1719`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py))
- `_navigate_via_staging` ([`mission_runner.py:1066-1365`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)) — once at entry, plus before each leg dispatch.

`_rotate_by_degrees` is OK without — rotation doesn't consume map-frame
data. Same for `_scan_for_target` (rotates between scan headings; the
grounding itself is camera-relative).

Env knobs (following the
[`progress-aware-nav-timeouts`](../../completed/progress-aware-nav-timeouts.md)
convention):
- `STRAFER_SLAM_PRECHECK_TIMEOUT_S` (default 10 s sim-time)
- `STRAFER_SLAM_PRECHECK_ENABLED` (default `1`; set `0` to disable
  entirely — escape hatch in case the precheck causes regressions)

### B. Subscribe to `/rtabmap/info` and abort in-flight

The same signal threaded through into the in-flight watchdog
loops, so a tracking-loss mid-`navigate_to_pose` aborts the
goal immediately instead of waiting for a deadline / stall. This is
nat­urally Layer 5 of
[`nav-stall-multilayer-watchdog.md`](../../parked/reliability/nav-stall-multilayer-watchdog.md)
— file there if the parked brief gets picked up. **Out of scope here.**

### C. Defer entirely; lean on `verify_arrival` to catch silent failures

Status quo. Rejected: too expensive. A mission already past a 30 s
nav leg before discovering it was driving on stale TF is 30 s of
unnecessary motion in a real-world environment, with no recoverable
state at the failure boundary.

**Recommended:** A. B follows naturally once the multilayer-watchdog
ships.

## Acceptance criteria

- [ ] `MissionRunner._await_slam_tracking` exists and is invoked at the
      four call sites above. Skipping is gated on
      `STRAFER_SLAM_PRECHECK_ENABLED=0`.
- [ ] Returns `slam_tracking_lost` error code when the bounded wait
      expires without tracking recovery. The `message` field includes
      the reason RTAB-Map reported.
- [ ] Sim-time-aware: the wait deadline uses
      `node.get_clock().now()`, not `time.monotonic()` (mirrors
      [`rotate-in-place-sim-clock-deadline`](../../completed/rotate-in-place-sim-clock-deadline.md)).
- [ ] `_scan_for_target` and `_rotate_by_degrees` are explicitly *not*
      pre-checked (they don't consume map-frame data; the precheck
      would just add per-scan-heading latency).
- [ ] Unit tests cover (a) tracking healthy → precheck no-op,
      (b) tracking lost → wait → recovers → motion proceeds,
      (c) tracking lost → wait → still lost at deadline → failure,
      (d) cancel during wait → `mission_canceled`.
- [ ] Integration sim repro: inject a tracking-loss by publishing an
      identity-pose `/strafer/odom` to `_on_odom`
      ([`ros_client.py:_on_odom`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)),
      observe a `go_to_target` mission step pause for the precheck
      window then either resume or fail with `slam_tracking_lost`.
- [ ] No regression on cold-start without a DB (`make clean-map &&
      make launch-sim`) — the precheck must complete inside its
      timeout once `donut_warmup` finishes.
- [ ] Real-robot bringup unaffected — at the operator's discretion,
      `STRAFER_SLAM_PRECHECK_ENABLED=0` is an escape hatch.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:456-474`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py) —
  `check_slam_tracking()` already returns a dict with `tracking` /
  `reason`. Verify what RTAB-Map topic / service it actually reads
  before this brief lands — if it's a stub, the first commit on this
  branch should wire it to `/rtabmap/info` properly.
- [`source/strafer_ros/strafer_slam/config/rtabmap_params.yaml`](../../../../source/strafer_ros/strafer_slam/config/rtabmap_params.yaml) —
  the `info` topic is published by default.
- [`docs/tasks/active/reliability/rtabmap-cold-start-determinism.md`](rtabmap-cold-start-determinism.md) —
  the cold-start case overlaps. If that brief lands first with the
  localization-mode-by-default disposition, the precheck mostly
  catches kidnapped-robot and featureless-corridor cases; if it lands
  after, the precheck *is* the cold-start workaround.
- RTAB-Map peer reference: [introlab/rtabmap_ros#80](https://github.com/introlab/rtabmap_ros/issues/80)
  on the identity-pose increment-map-id branch.

## Out of scope

- **In-flight tracking-loss abort.** That's Layer 5 of
  [`nav-stall-multilayer-watchdog.md`](../../parked/reliability/nav-stall-multilayer-watchdog.md);
  this brief only adds the precheck.
- **Fixing RTAB-Map's tracking loss itself.** RTAB-Map tuning
  (`Vis/MaxFeatures`, `Reg/Strategy`, etc.) lives in
  [`rtabmap-cold-start-determinism.md`](rtabmap-cold-start-determinism.md)
  and any follow-ups. This brief makes the executor robust to
  tracking loss — it does not prevent the loss.
- **Real-robot relocalization hardware-watchdog timing.** If real-robot
  RTAB-Map drops tracking for >10 s on the lab carpet, the precheck
  timeout knob is the operator's lever; tuning it for real-robot is a
  bringup task.
