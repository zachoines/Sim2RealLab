# Re-apply the RoboClaw gains on reconnect, and read them back before driving

**Status:** Shipped 2026-10-22 in `73649a0` and `4070d7b` (Jetson); the on-robot box stays open until the bench record.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/253

**Type:** bug (after a main-power cut, the driver's reconnect restores the link but not the
velocity gains, so its next drive runs the rear on zero gains)
**Owner:** Jetson (`strafer_driver`)
**Priority:** P1.
- A main-power cut while `base` runs is an ordinary bench and field event, and the reconnect
  clears the error state by itself.
- The rear controller's saved gains are all zero, so after the reconnect its velocity commands do
  not work until `base` restarts. `diagnose_roboclaw.py` warns of exactly this for all-zero gains.

**Estimate:** S
**Branch:** `task/driver-stall-and-reconnect-hardening`, with
[`driver-stale-cmd-vel-rearm`](driver-stale-cmd-vel-rearm.md) and
[`driver-unanswered-controller-stall`](driver-unanswered-controller-stall.md)

## Story

As **whoever powers the RoboClaws down and up while `base` runs**, I want **the driver to put its
gains back before it drives again**, so that **a reconnect leaves both controllers as the driver
configured them, or leaves the driver stopped.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [`chassis-stop-modes-2026-10-20`](../../measurements/chassis-stop-modes-2026-10-20/README.md)
  §2 (pause_1) and finding 5, and its deposit's `step0/step0.txt` and
  `pause_1/base_log_full_after.txt`.

## Context

Read at `72b9ba7`, and seen on the robot on 2026-10-20.

- **The driver writes its gains once, at start.** `_configure_pid` sets P 15000, I 750, D 0 and
  QPPS 2796 on both motors of each controller, into RAM.
- **A power-up loads the controllers' saved settings instead.** Step 0 read them before the
  driver's first start: the front's M1 P 14999, I 749 (M2 15000, 750), and the rear's P, I, D and
  QPPS all 0.
- **The reconnect does not write them back.**
  - In the error state the timer calls `reconnect()` on both controllers every 2 s.
  - `reconnect()` reopens the path detection chose at start (`self.port`) and probes it with a
    battery read at that controller's address. It does not detect again. When both answer, the
    driver clears its error state and drives on.
- **Seen on the robot (pause_1).**
  - Main power was cut, and both controllers dropped off USB. When it came back, both
    re-enumerated on their old names.
  - The driver logged `RoboClaw reconnected successfully.` at its first attempt after both had,
    then `No cmd_vel for 86.66s -- stopping motors.`: its first command after the reconnect was
    the watchdog's zero.
  - `base` was stopped 4 s later, so no `cmd_vel` was driven on the rear's zero gains.
- **A write reads back exactly.** At 23:06:19 UTC in the bring-up, after the driver had run with no
  power cycle since, both controllers read P 15000, I 750, D 0 and QPPS 2796, including the rear,
  whose saved gains are zero (`chassis-bringup-2026-10-19`'s deposit,
  `drive/step6_polarity_drive.txt`). So a read-back can be compared exactly.
- **The names held because the enumeration order held.** The bring-up saw the two swap across a
  power cycle. After a swap the path chosen at start leads to the other controller, which does
  not answer at this one's address. `reconnect()` then keeps failing, the driver stays in its
  error state with both stopped, and restarting `base` re-runs detection.
- **The fix.**
  - After `reconnect()` succeeds on both ports, re-run the gain writes on both, and read the gains
    back (`read_velocity_pid_m1`, `_m2`), before clearing the error state.
  - If any read-back disagrees, or a write or read fails, stay in the error state and log which.
  - **At startup too.** The startup gain writes are read back the same way. Until both
    controllers hold the gains the driver stays in its error state, whose reconnect re-applies
    and reads them back every 2 s, so nothing drives on a controller's saved gains.
  - The enumeration swap is documented with restart as the recovery, not re-detected. So is a
    controller replugged before the first reconnect attempt releases the descriptor the driver
    holds, which returns under a new `ttyACM` number.

## Acceptance criteria

- [x] **A fake-bus test fails on `main`.** Fake controllers lose their gains across a simulated
      power cycle, and the driver reconnects. On `main` the error state clears with the rear's
      gains at zero; with the fix, both read 15000/750/0/2796 when it clears.
- [x] **A disagreeing read-back keeps the error state.** A fake controller that reads back other
      gains after the write leaves the driver in its error state, and the log names it.
- [x] **At startup, unconfirmed gains hold the error state.** A fake controller that leaves the
      startup gain writes unanswered keeps the driver in its error state, with no drive command
      sent, until it answers; then the error state clears and driving resumes.
- [x] **The enumeration swap is documented** where the driver's recovery is described, with restart
      as the recovery.
- [ ] **On the robot,** with `base` idle and no publisher, cut main power with the USB in.
  - After `RoboClaw reconnected`, the driver's logged read-back shows P 15000, I 750, D 0 and
    QPPS 2796 on both, without a restart of `base`.
  - Then one 0.8 rad/s spin turns all four wheels.
  - Then, with `base` stopped, `diagnose_roboclaw.py --read-only` shows the same gains on both. A
    stop leaves the RAM gains in place; the read waits for the stop because two processes cannot
    share a controller's port.
  - The run is deposited and a record states it. *Open until the bench record.*
- [x] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance).
- [x] No regression: `make test-ros` passes.

## Investigation pointers

- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py`: `_configure_pid`, the
  error-state branch of `_timer_callback`.
- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py`: `reconnect`,
  `set_velocity_pid_m1`/`_m2`, `read_velocity_pid_m1`/`_m2`.
- `source/strafer_ros/strafer_driver/test/test_port_fence.py` (`FakeRoboClaw`, `FakeBus`,
  `build_node`).

## Out of scope

- **Saving the gains to the controllers' flash.** That is
  [`chassis-velocity-pid-tuning`](../active/reliability/chassis-velocity-pid-tuning.md)'s decision of where the gains
  live.
- **Re-detecting by serial number on reconnect.** Restart stays the recovery from a swap.
- **The other driver stalls:** [`driver-stale-cmd-vel-rearm`](driver-stale-cmd-vel-rearm.md),
  [`driver-unanswered-controller-stall`](driver-unanswered-controller-stall.md).
