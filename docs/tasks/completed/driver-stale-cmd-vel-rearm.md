# Stop the RoboClaw driver from running a stale `cmd_vel` as fresh after a stall

**Status:** Shipped 2026-10-22 in `73649a0` and `4070d7b` (Jetson); the on-robot box stays open until the bench record.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/253

**Type:** bug (after a whole-executor stall of the driver, a `cmd_vel` that waited through the
stall drives the wheels for another 0.5 s, also after the watchdog has stopped them)
**Owner:** Jetson (`strafer_driver`)
**Priority:** P2, by exposure.
- It needs a stall of the driver's executor over 0.5 s, and a publisher that sends into the stall
  and then stops without a zero, on one fault. No deployed publisher does that: the inference node
  ends every mission with a zero, and zeroes `/cmd_vel` about 0.2 s after `joint_states` stops
  (0.2 s plus up to one 1/30 s tick).
- On the robot only `docker pause` has produced the stall, and only a bench publisher the silence.
- The re-armed 0.5 s covers 0.39 m at the 0.784 m/s cap the policy lane and Nav2 share
  (`navigation.launch.py` replaces the smoother's YAML 0.5 m/s with `NAV_LINEAR_VEL`). Nav2 caps
  each axis on its own, so a Nav2 diagonal reaches 1.11 m/s, 0.55 m. It reaches 0.8 m only from
  an uncapped publisher.

**Estimate:** S
**Branch:** `task/driver-stall-and-reconnect-hardening`, with
[`driver-unanswered-controller-stall`](driver-unanswered-controller-stall.md) and
[`driver-reconnect-reapplies-gains`](driver-reconnect-reapplies-gains.md)

## Story

As **anything that stops the robot by no longer publishing `cmd_vel`**, I want **the driver to run
no message that waited through a stall of its executor**, so that **a stop does not come undone
when the driver resumes.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [`chassis-stop-modes-2026-10-20`](../../measurements/chassis-stop-modes-2026-10-20/README.md)
  §2 and finding 1, and its deposit's `pause_2/` and `pause_3/`.

## Context

Read at `72b9ba7`, and seen on the robot on 2026-10-20.

- **The watchdog.** `roboclaw_node.py`'s `_cmd_vel_callback` stores the message and stamps
  `_last_cmd_vel_time` with the node's clock when it takes it. The 50 Hz timer zeroes the motors
  once that stamp is more than `WATCHDOG_TIMEOUT_SEC` (0.5 s) old. A `Twist` has no header, so
  the take is the only age the driver knows.
- **The order.** Humble's rclpy executor yields ready timers before ready subscriptions
  (`executors.py`, `_wait_for_ready_callbacks`), and rcl schedules a timer's next call on its
  period grid. After a stall the overdue tick zeroes the motors, and a catch-up tick can follow
  within milliseconds, before the take.
  - In pause_3 two ticks, with clock reads at +0.037 and +0.045 s, sent zero; the callback then
    took the waiting message (+0.046 to +0.056 s) and stamped it as new; the next ticks drove it.
  - The margin: the take window's lower edge is 0.6 ms after the second tick's clock read, and that
    tick's own body, about 5 ms, puts the take at least that much later. The executor's first wait
    after the unpause must have missed the subscription.
  - pause_2 shows the same pair of ticks, at +0.039 and +0.047 s.
- **The subscription keeps one message** (`QoSProfile(depth=1, reliability=RELIABLE)`), so only
  the newest is taken. Where it waited during the stall is a reading, not measured.
- **Seen on the robot (pause_3).** `docker pause` froze `base` for 3 s mid-spin. The publisher
  kept sending for 2 s into the pause, then stopped.
  - The driver logged `No cmd_vel for 3.09s` 0.039 s after the unpause.
  - By its second watchdog line it took a message at +0.046 to +0.056 s; the publisher's last
    message was then 1.05–1.06 s old.
  - It drove the spin until `No cmd_vel for 0.51s` at +0.563 s. The counts moved from +0.081 to
    +0.661 s by the samples' stamps, and the wheels turned 2.47–2.64 rad.
  - In pause_2 the driver took the last message before the freeze, so nothing was waiting and
    nothing moved. In pause_1 a message was waiting, but a main-power cut hid the outcome.
- **The precondition is a stall of the whole executor, longer than the watchdog window.** The
  record's USB pull is the contrast.
  - Its loop waited on the rear's serial timeouts for about 0.2 s, under the window.
  - The executor served the subscription in the error state that followed: the reconnect's
    watchdog line dates a take at +0.217–0.227 s, at the publisher's last message, and nothing
    re-armed after the reconnect.
- **The fix.**
  - `_timer_callback` records the time from its previous start to this one, on
    `time.monotonic()`.
  - A gap over `WATCHDOG_TIMEOUT_SEC` marks a stall.
  - The short-gap ticks after it leave the mark in place. A mark that every tick reassigned would
    already be clear at pause_3's take, because the catch-up tick ran first.
  - `_cmd_vel_callback` drops the first message it takes while the mark is set, and clears it.
  - A tick more than `WATCHDOG_TIMEOUT_SEC` after the mark also clears it. A stall with nothing
    waiting, as in pause_2, then does not cost a later fresh message.
  - The threshold is the watchdog window, not two timer periods. A loop that runs slow every tick
    (one silent controller costs about 0.31 s per tick) must not drop every message.
- **Rejected: the writer's `source_timestamp`.**
  - The subscription's `take_message` returns it in the message info, but Humble's executor
    drops the info (`_take_subscription` returns `msg_info[0]`). Reaching it needs a custom take.
  - An age check against it also needs the publisher's clock to agree with the driver's, which a
    remote publisher's does not promise.
  - It would also cover a delay outside the driver, such as a link outage with reliable
    retransmission. The stall flag does not.
- **The residual.**
  - A stall under 0.5 s still lengthens a run by up to the stall.
  - A delay outside the driver is not covered.
  - If a stall left more than one stale message for the subscription after the dropped one, the
    next would run. With one message kept and the overdue ticks' serial I/O between the wake
    and the take, that is not expected.
  - A start-to-start gap over the window whose watchdog has not fired, because the previous tick
    was slow, drives the old twist one more tick and may drop a queued zero, bounded by that
    tick's duration.
  - A publisher that starts within 0.5 s after a stall loses its first message.
  - A freeze inside a tick body, in an executor pass that already held the subscription, would
    take before the late tick: rare, and unmeasured.

## Acceptance criteria

- [x] **A test through a real executor reproduces the re-arm and fails on `main`.**
  - It uses the fake serial bus and a publisher in the same process.
  - It spins while a 0.8 rad/s spin is published, then stops spinning, and publishes on into the
    stall.
  - It stops publishing, waits 1 s, then spins again.
  - It runs twice: with the take straight after the overdue tick, and in pause_3's order, with a
    catch-up tick between them.
  - On `main` the fake bus receives a non-zero speed after the watchdog's zero; with the fix,
    none.
- [x] **A live publisher still drives after a stall.** The same test, with the publisher still
      sending when the spin resumes: non-zero speeds return within one publisher period.
- [x] **The gap is measured start to start on `time.monotonic()`, against
      `WATCHDOG_TIMEOUT_SEC`. The mark holds through the short-gap ticks until the first take, or
      until a tick `WATCHDOG_TIMEOUT_SEC` after it.**
- [ ] **On the robot,** with the wheels lifted, the record's pause_3 run again shows one watchdog
      line after the unpause, `joint_states` at zero and a still camera. The run is deposited and
      a record states it. *Open until the bench record.*
- [x] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance).
- [x] No regression: `make test-ros` passes.

## Investigation pointers

- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py`: `_cmd_vel_callback`,
  `_timer_callback`, `WATCHDOG_TIMEOUT_SEC`, and the subscription's QoS.
- Humble's `rclpy/executors.py`: `_wait_for_ready_callbacks`, where timers are yielded before
  subscriptions, and `_take_subscription`, which drops the message info.
- `source/strafer_ros/strafer_driver/test/test_port_fence.py` (`FakeRoboClaw`, `FakeBus`,
  `build_node`).
- The record's deposit: `pause_3/spin.csv`, `pause_3/base_log.txt`, and `scripts/pause_run.sh`
  with `STOP_DELAY=2`, which keeps the publisher sending 2 s into the pause.

## Out of scope

- **`TwistStamped` across the stack.** Nav2's Humble controller and the inference node publish
  `Twist`.
- **A delay outside the driver,** such as a link outage with retransmission (the residual above).
- **The loop's blocking reads:** [`driver-unanswered-controller-stall`](driver-unanswered-controller-stall.md).
- **The controllers' serial timeout,** which already stops the wheels when the driver stops
  sending.
