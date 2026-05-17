# Cancel-mid-motion must zero `/cmd_vel` for direct-publish skills

**Type:** task / bug
**Owner:** Jetson agent (`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`, `source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`)
**Priority:** P1 — correctness bug; real-robot blocker. The current behavior is "user pressed cancel, robot keeps rotating until tolerance or wall-cap." That's not acceptable on hardware near humans / furniture.
**Estimate:** S (~half day; thread a cancel-aware abort signal through `rotate_in_place` + add a single test)
**Branch:** task/executor-cancel-mid-motion-cmd-vel-zero

## Story

As a **mission operator pressing cancel on an in-flight rotation
(scan-for-target rotate, align-to-goal-yaw rotate) or translation**, I
want **`/cmd_vel` to publish a zero `Twist` within ~100 ms of the
cancel signal**, so that **the chassis actually stops on hardware
instead of completing the current `rotate_in_place` loop's deadline or
tolerance condition while ignoring my cancel.**

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [completed/rotate-in-place-sim-clock-deadline.md](../../completed/rotate-in-place-sim-clock-deadline.md)
  — touched the deadline loop that this brief threads cancel through.

## Context

The executor has two cancel entry points
([`mission_runner.py:321`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
`cancel_active_mission` and
[`mission_runner.py:1900-1912`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
`_cancel_step`). Both ultimately call
`_cancel_robot_actions`:

```python
# mission_runner.py:2017
def _cancel_robot_actions(self) -> None:
    try:
        self._ros_client.cancel_active_navigation()
    except Exception:
        return
```

`cancel_active_navigation`
([`ros_client.py:825-845`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py))
**only cancels the Nav2 action goal**:

```python
def cancel_active_navigation(self) -> bool:
    with self._nav_lock:
        goal_handle = self._active_goal_handle
    if goal_handle is None:
        return False
    cancel_future = goal_handle.cancel_goal_async()
    ...
```

This is correct for `navigate_to_pose` — Nav2's cancel handler is
responsible for ramping `/cmd_vel` to zero via the controller server.
But two of our motion skills bypass Nav2 and publish `/cmd_vel`
directly:

1. **`rotate_in_place`** ([`ros_client.py:1109-1197`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)).
   The loop only checks `clock.now() >= deadline`, `time.monotonic() >=
   wall_cap`, and `abs(error) < tolerance_rad`. There is **no cancel
   signal** wired into the loop. A cancel arriving via
   `cancel_active_navigation` returns immediately (the Nav2 lock holds
   no goal) and the rotation continues to either tolerance or timeout.
2. **`donut_warmup`** ([`donut_warmup.py:252-296`](../../../../source/strafer_ros/strafer_bringup/strafer_bringup/donut_warmup.py))
   — same loop shape, but this node is one-shot at bringup and not
   part of any cancellable mission, so it's out of scope here.

Field consequence on the existing sim lane: cancel during
`_scan_for_target` (which calls `rotate_in_place` between headings,
[`mission_runner.py:736`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py))
or during `_align_to_goal_yaw` (final rotate at goal,
[`mission_runner.py:1714`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py))
ignores the cancel for up to `default_rotate_timeout_s` of sim time.
On the real robot this is a near-miss waiting to happen — the operator
hits "stop" because the robot is about to collide with something the
costmap doesn't see, and the rotation completes anyway.

Note that the scan path *does* check `runtime.cancel_event.is_set()`
*between* headings ([`mission_runner.py:721`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)),
which is too late — the cancel arrives during a 6-second rotate and
the check happens only after the rotate completes.

## Approach

### A. Thread a `cancel_event` into `rotate_in_place` (recommended)

1. `JetsonRosClient.rotate_in_place` accepts an optional
   `cancel_event: threading.Event | None = None` kwarg.
2. The deadline loop adds `if cancel_event is not None and
   cancel_event.is_set(): break` alongside the existing deadline /
   wall-cap checks. On break, publish a final zero `Twist` (same as
   the existing timeout-path zero at [`ros_client.py:1191`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py))
   and return a `SkillResult` with `status="canceled"`,
   `error_code="rotation_canceled"`.
3. `MissionRunner._dispatch_step` (the dispatcher around line 499)
   passes `runtime.cancel_event` whenever it calls
   `self._ros_client.rotate_in_place(...)` directly. There are three
   call sites: `_scan_for_target`'s inter-heading rotate, `_align_to_goal_yaw`,
   and `_rotate_by_degrees`.
4. `_cancel_robot_actions` is extended to also publish a single zero
   `Twist` to `/cmd_vel` as a belt-and-braces fail-safe. This handles
   the corner where a cancel races a `rotate_in_place` call that
   hasn't yet entered its loop (the cancel sets the event, the rotate
   reads `is_set()` immediately and returns). The duplicate zero is
   harmless — `rotate_in_place` always publishes a zero on exit.

### B. Subscribe to `/cancel_motion` topic in the ROS client (rejected)

A separate cancel topic would let the cancel propagate via DDS the
same way `/cmd_vel` does, but it adds a subscription, a callback
thread, and shared mutable state. The `cancel_event` already exists
on `_MissionRuntime`; threading it down is simpler and avoids a new
inter-thread protocol.

### C. Use Nav2's `nav2_velocity_smoother` `velocity_timeout` (partial)

[Nav2's velocity smoother](https://docs.nav2.org/configuration/packages/configuring-velocity-smoother.html)
has a `velocity_timeout` param that publishes zero if no `cmd_vel`
arrives within a wall-clock window. We currently have it set to 1.0 s
in [`nav2_params.yaml:373`](../../../../source/strafer_ros/strafer_navigation/config/nav2_params.yaml).
This is a defense-in-depth backstop, **not** a cancel mechanism: it
requires the publisher to stop publishing first. Our `rotate_in_place`
loop publishes at 50 Hz, so the smoother timeout will never fire
during an active rotate. Useful as a tertiary fail-safe; not a fix.

**Recommended:** A. C stays in place as a fail-safe for the "executor
itself crashed mid-rotate" case.

## Acceptance criteria

- [ ] `rotate_in_place` accepts a `cancel_event: threading.Event |
      None` kwarg (default None for backward compatibility with any
      direct callers).
- [ ] When `cancel_event.is_set()` is observed inside the rotation
      loop, the function (1) publishes a single zero `Twist` to
      `/cmd_vel`, (2) returns `SkillResult` with `status="canceled"`
      and `error_code="rotation_canceled"`, within one publish period
      (≤ 20 ms of the check).
- [ ] All three call sites in `mission_runner.py`
      (`_scan_for_target`, `_align_to_goal_yaw`, `_rotate_by_degrees`)
      pass `runtime.cancel_event` through.
- [ ] `_cancel_robot_actions` publishes one zero `Twist` to `/cmd_vel`
      after calling `cancel_active_navigation` as a belt-and-braces
      fail-safe.
- [ ] Unit test: a `rotate_in_place` call against a stub `_node` and
      `_latest_odom` (mirroring the existing `test_ros_client.py` test
      shape) with `cancel_event.set()` invoked from a separate thread
      ~50 ms in completes with `status="canceled"`, publishes a final
      `angular.z == 0.0` twist, and returns within `≤ 100 ms` wall.
- [ ] Integration smoke: `strafer-autonomy-cli submit "scan for door"`
      followed by `cancel` mid-rotation produces a `/cmd_vel` topic
      trace whose final message (`ros2 topic echo --once /cmd_vel`
      immediately after cancel) has all zeros, observed in Foxglove.
- [ ] No regression on `nav2-far-goal-staging` and
      `nav2-startup-unknown-donut-path-noise` reference missions —
      neither path issues a cancel mid-rotation, so behavior must be
      identical to pre-brief.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:1109-1197`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py) —
  `rotate_in_place` loop. Add the `cancel_event` check next to the
  existing deadline / wall-cap branches around line 1158-1167.
- [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:825-845`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py) —
  `cancel_active_navigation`. The Nav2-side cancel stays unchanged.
- [`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py:2017-2021`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py) —
  `_cancel_robot_actions`. Add the zero-Twist publish after the Nav2
  cancel.
- [`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py:721`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py),
  [`mission_runner.py:1714`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py) —
  the call sites that pass `runtime.cancel_event` through.
- Nav2 reference: [Velocity Smoother config](https://docs.nav2.org/configuration/packages/configuring-velocity-smoother.html)
  for the `velocity_timeout` fail-safe semantics.
- The `_publish_grounding_overlay` `getattr(...)` pattern is a sibling
  smell tracked in [`grounding-publisher-extraction.md`](grounding-publisher-extraction.md);
  this brief introduces no new such patterns.

## Out of scope

- **Cancel-mid-`navigate_to_pose`.** Nav2's action server already
  zeros `/cmd_vel` on cancel via the controller server. If Foxglove
  traces show otherwise, that's a separate Nav2 config bug worth a
  follow-up brief, not a code-path fix here.
- **Donut warmup cancel.** Out of scope; the one-shot warmup is not
  part of any cancellable mission flow.
- **A separate `chassis_wedged` watchdog layer** that fires when
  `/cmd_vel` is non-zero but `/strafer/odom` is near-zero — that's
  [`nav-stall-multilayer-watchdog.md`](../../parked/reliability/nav-stall-multilayer-watchdog.md)
  Layer 1, parked. This brief only fixes the cancel path; the watchdog
  catches a different failure (motor wedge, e-stop, jammed) that has
  no executor-side cancel to honor.
- **Adding a `cancel_event` to the `RosClient` Protocol abstract
  method.** The current Protocol declares `rotate_in_place` without a
  cancel kwarg; threading it as an optional kwarg keeps Protocol
  conformance broader (any client implementation may ignore it).
  Tightening the Protocol is a separate refactor if it ever matters.
