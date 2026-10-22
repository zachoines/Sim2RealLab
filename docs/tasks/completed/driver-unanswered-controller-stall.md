# Stop both controllers once the watchdog window has passed when one RoboClaw stops answering

**Status:** Shipped 2026-10-22 in `73649a0` and `4070d7b` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/253

**Type:** bug (one controller that stays on USB but does not answer keeps the driver out of its
error state for about 3 s, while the other controller holds the last command)
**Owner:** Jetson (`strafer_driver`)
**Priority:** P2.
- No such controller has been seen. On the robot a main-power cut drops a RoboClaw off USB, which
  fails fast and reaches the error state in 0.155 s (the record's USB pull).
- The case needs a controller that is enumerated and silent, such as a firmware hang.
- On a policy mission the inference node zeroes `/cmd_vel` 0.20–0.24 s after `joint_states`
  stops, which lands mid-tick. The overdue next tick runs the old command, and the take applies at
  the tick after. So the other controller is sent the zero about 0.65 s after the last good tick:
  one period plus two stalled ticks of about 0.31 s. The 3 s stands for Nav2 and bench
  publishers.

**Estimate:** S
**Branch:** `task/driver-stall-and-reconnect-hardening`, with
[`driver-stale-cmd-vel-rearm`](driver-stale-cmd-vel-rearm.md) and
[`driver-reconnect-reapplies-gains`](driver-reconnect-reapplies-gains.md)

## Story

As **whoever relies on the driver to stop the robot when a controller fails**, I want **the
driver to stop both controllers at its first failure after the watchdog window has passed since
the last good exchange**, so that
**a silent controller cannot leave the other one driving for seconds.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [`chassis-stop-modes-2026-10-20`](../../measurements/chassis-stop-modes-2026-10-20/README.md)
  §4, finding 2 and "What this does not show".

## Context

Read at `72b9ba7`.

- **One tick with a silent controller costs about 0.31 s.**
  - Each tick sends both drive commands first. `_send_with_ack` waits `read_timeout` (0.1 s) for
    the silent controller's ACK, then returns False without raising, and nothing retries it.
  - The reads come next. The silent controller's first read times out (0.1 s), `_retry` sleeps
    5 ms, retries (0.1 s) and raises.
  - The tick then ends in `_handle_serial_failure`.
- **The escalation counts failures only.** `MAX_CONSECUTIVE_FAILURES` is 10, so the stop to both
  and the error state come after about 3.1 s.
- **The other controller holds the last command meanwhile.** Every tick starts with its drive
  command, about 0.31 s apart, inside its own 0.5 s serial timeout.
- **Seen on the robot, the short form.** In the record's USB pull no tick completed for about
  0.2 s before the kernel saw the disconnect, while the loop waited out the rear's timeouts. The
  disconnect then made every write fail at once, and the error state came at +0.155 s.
- **The fix.**
  - In `_handle_serial_failure`, escalate when `time.monotonic() - self._last_success_time`
    exceeds `WATCHDOG_TIMEOUT_SEC`, as well as on the count.
  - A successful reconnect also counts as a success. Otherwise the first failure after a long
    error state would escalate at once.
  - With one silent controller the error state then comes at the second failing tick, about
    0.6 s after the last good one.
- **The residual.** The first failure after any gap over the window escalates at once, the safe
  direction: after an executor delay of about 0.3 s, a single failing tick is enough.

## Acceptance criteria

- [x] **A fake-bus test with one controller enumerated and silent fails on `main`.** On `main`
      the driver reaches its error state only at the tenth failure; with the fix, at the first
      failure more than `WATCHDOG_TIMEOUT_SEC` after the last success. Both controllers are sent a
      stop.
- [x] **A transient failure still does not escalate.** One failed tick between good ones leaves
      the driver out of its error state.
- [x] **On hardware this is stated as not reproducible.** A power cut drops a RoboClaw off USB,
      so the silent-but-enumerated case is covered by the fake bus only.
- [x] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance).
- [x] No regression: `make test-ros` passes.

## Investigation pointers

- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py`: `_timer_callback`,
  `_handle_serial_failure`, `_last_success_time`, `MAX_CONSECUTIVE_FAILURES`.
- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py`: `_send_with_ack`,
  `_retry`, `_read_bytes`.
- `source/strafer_ros/strafer_driver/test/test_serial_timeout.py` (`Unanswered`, a fake controller
  that sends no reply) and `test_port_fence.py` (`FakeBus`, `build_node`).

## Out of scope

- **`_send_with_ack` returning False without raising.** The driver ignores the result of its
  drive commands; this brief keys the stop on time instead.
- **The loop's structure,** such as a reader thread per controller.
- **The stale-message re-arm:** [`driver-stale-cmd-vel-rearm`](driver-stale-cmd-vel-rearm.md).
