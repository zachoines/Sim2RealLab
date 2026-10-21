# Stop the RoboClaw driver from running a stale `cmd_vel` as fresh after a stall

**Type:** bug (after a stall of the driver, a `cmd_vel` that waited through the stall drives the
wheels for another 0.5 s, also after the watchdog has stopped them)
**Owner:** Jetson (`strafer_driver`)
**Priority:** P1.
- The watchdog is the stop for a publisher that falls silent, and a stall can undo it with a
  command that is no longer current.
- On the robot a spin at least 1.0 s old re-ran for 0.5 s after the watchdog's stop. At the
  1.57 m/s top speed, 0.5 s is about 0.8 m.
- It takes a stall of over 0.5 s, and a publisher that sends into it and then falls silent
  without a final zero. On the robot only `docker pause` has produced the stall.

**Estimate:** S–M
**Branch:** `task/driver-stale-cmd-vel-rearm`

## Story

As **anything that stops the robot by no longer publishing `cmd_vel`** (a publisher that exits,
crashes or loses its link), I want **the driver to run no command after the watchdog's window
from its send**, so that **a stop does not come undone when the driver stalls.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`chassis-stop-modes-2026-10-20`](../../../measurements/chassis-stop-modes-2026-10-20/README.md)
  §2 and finding 1, and its deposit's `pause_2/` and `pause_3/`.

## Context

Read at `97c035f`, and seen on the robot on 2026-10-20.

- **The watchdog.** `roboclaw_node.py`'s `_cmd_vel_callback` stores the message and stamps
  `_last_cmd_vel_time` with the node's clock when it takes it. The 50 Hz timer zeroes the motors
  once that stamp is more than `WATCHDOG_TIMEOUT_SEC` (0.5 s) old. A `Twist` has no header, so
  the take is the only age the driver knows.
- **The subscription** is `QoSProfile(depth=1, reliability=RELIABLE)`. After a stall, a message
  sent into it is waiting to be taken.
- **The order.** Humble's rclpy executor yields ready timers before ready subscriptions
  (`executors.py`, `_wait_for_ready_callbacks`).
  - After a stall, the overdue tick runs first and zeroes the motors: `No cmd_vel for <stall>s`.
  - The callback then takes the waiting message and stamps it with the current time.
  - The next ticks drive it until the watchdog fires again 0.5 s later.
- **Seen on the robot (pause_3).** `docker pause` froze `base` for 3 s mid-spin. The publisher
  kept sending for 2 s into the pause, then stopped.
  - The driver logged `No cmd_vel for 3.09s` 0.039 s after the unpause.
  - By the second watchdog line, it then took a message at +0.048 to +0.058 s. It was one sent
    into the pause, at least 1.0 s old.
  - It drove the spin until `No cmd_vel for 0.51s` at +0.565 s. Its counts moved from +0.089
    to +0.667 s, at up to 5.27 rad/s, and the wheels turned 2.47–2.64 rad.
  - In pause_2 the driver took the last message before the freeze, so nothing was waiting, and
    nothing moved.
  - In pause_1 a message was waiting, but a main-power cut hid the outcome.
- **The general case.** The watchdog counts from the take, so a delay between send and take adds
  to a command's life.
  - A stall of over 0.5 s, during which the publisher sent and then fell silent, restarts the
    wheels after the watchdog's stop.
  - A shorter stall lengthens the run by up to the stall.
  - A link outage with reliable retransmission may deliver a stale message the same way. That is
    a reading of the QoS, not measured.
- **What Humble gives.** The subscription's `take_message` returns the message with its
  `rmw_message_info`, which carries the writer's `source_timestamp`. Humble's executor drops it:
  `_take_subscription` returns `msg_info[0]`. So the callback has no send time as it stands.

## Acceptance criteria

- [ ] **A test reproduces the re-arm and fails on `main`.**
  - It drives the node through a real executor, with a publisher in the same process and the fake
    serial layer.
  - It spins while a 0.8 rad/s spin is published, then stops spinning: that is the stall.
  - It publishes for another 0.5 s, stops publishing, and waits until the newest message is 1 s
    old. Then it spins again.
  - On `main` the fake bus receives a non-zero speed after the watchdog's zero; with the fix it
    receives none.
- [ ] **A live publisher still drives after a stall.** The same test, with the publisher still
      sending when the spin resumes: non-zero speeds return within one publisher period.
- [ ] **The fix and its reason are recorded,** against the other candidates.
  - Candidates: the writer's `source_timestamp`, through a take that keeps the message info; or
    a stall check in the timer that drops a message taken right after an overdue tick.
  - The choice weighs the clock offset between hosts for a remote publisher, and the link-outage
    case.
- [ ] **On the robot,** with the wheels lifted, the record's pause_3 run again shows no re-arm.
  - That is one watchdog line after the unpause, and `joint_states` at zero.
  - The run is deposited, and a record states it.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No regression: `make test-ros` passes.

## Investigation pointers

- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py`: `_cmd_vel_callback`,
  `_timer_callback`, `WATCHDOG_TIMEOUT_SEC`, and the subscription's QoS.
- Humble's `rclpy/executors.py`: `_wait_for_ready_callbacks`, where timers are yielded before
  subscriptions, and `_take_subscription`, which drops the message info.
- `source/strafer_ros/strafer_driver/test/test_port_fence.py` (`FakeRoboClaw`, `FakeBus`,
  `build_node`) and `test_serial_timeout.py`
  (`test_node_never_sends_timeout_or_flash_commands`, which drives `_timer_callback` directly).
- The record's deposit: `pause_3/spin.csv`, `pause_3/base_log.txt`, and `scripts/pause_run.sh`
  with `STOP_DELAY=2`, which keeps the publisher sending 2 s into the pause.

## Out of scope

- **`TwistStamped` across the stack.** Nav2's Humble controller and the inference node publish
  `Twist`.
- **The driver's blocking reads.** One controller that does not answer holds the whole loop in
  0.1 s timeouts (the record's finding 2). A controller that stops answering but stays on USB is
  not measured.
- **The controllers' serial timeout,** which already stops the wheels when the driver stops
  sending.
