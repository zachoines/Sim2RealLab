# Re-detect the RoboClaws by address when a reconnect fails, so a swapped enumeration recovers by itself

**Type:** bug (after a main-power cut that swaps the controllers' enumeration order, every
reconnect fails until `base` is restarted)
**Owner:** Jetson (`strafer_driver`)
**Priority:** P2. The failure is safe: the driver stays in its error state with both controllers
stopped. But the recovery needs a person to restart `base`, and on the robot one of two main-power
cuts with the USB in swapped the names. A robot whose controllers lose power mid-mission stays
stopped until someone restarts it.
**Estimate:** S
**Branch:** `task/driver-reconnect-redetects-by-address`

## Story

As **whoever runs the robot after its controllers lose and regain main power**, I want **the
driver to find each controller by its address when its old path no longer answers**, so that **a
swapped enumeration recovers by itself, as an unswapped one already does.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`driver-hardening-bench-2026-10-25`](../../../measurements/driver-hardening-bench-2026-10-25/README.md)
  §4 and finding 4, and its deposit's `step3/`, `host/kernel_usb_events.txt` and
  `host/base_log_all.txt`.
- [`driver-reconnect-reapplies-gains`](../../completed/driver-reconnect-reapplies-gains.md), whose
  reconnect this extends.

## Context

Read at `af7d798`, and seen on the robot on 2026-10-25.

- **The names follow the enumeration order.** `99-strafer.rules` names every RoboClaw
  (`03eb:2404`) `/dev/roboclaw%n` after its `ttyACM` number, and the D555 holds `ttyACM0`. So the
  controller the kernel enumerates first is `/dev/roboclaw1`.
- **The order changes across power-ups.** On 2026-10-25 the kernel enumerated the rear first at
  the day's first power-up and at one power cycle, and the front first after both main-power cuts
  with the USB in.
- **Detection runs only at start.** With `auto_detect` (the default), `RoboClawNode.__init__`
  calls `detect_roboclaws` over the fence's glob, `ROBOCLAW_UDEV_GLOB` (`/dev/roboclaw*`). It
  refuses to start if an address is missing, or if both answer on one device (`same_device`), and
  builds each `RoboClawInterface` on the path its address answered on.
- **The reconnect reopens those paths.** In the error state `_timer_callback` calls `reconnect()`
  on both interfaces every `RECONNECT_INTERVAL_SEC` (2 s). `reconnect()` closes and reopens
  `self.port` and probes it with a battery read at `self.address`. After a swap each path leads to
  the other controller, which does not answer at this one's address, so every attempt fails.
- **Seen on the robot.** `base` had detected front `/dev/roboclaw2` and rear `/dev/roboclaw1`.
  After the first USB-in cut the front came back as `ttyACM1` and the rear as `ttyACM2`. All 25
  reconnect attempts, one every 2 s from the error state to the restart, failed; the 20 after the
  controllers came back met the wrong controller, both probes timing out. A restart of
  `base` re-ran detection, and the driver read its gains back and drove again.
- **The fix.**
  - When `reconnect()` fails on either interface, close both ports and run `detect_roboclaws` over
    the fence's glob.
  - Point each interface at the path its address answers on, and log the mapping as the start's
    `Auto-detect:` line does.
  - Keep the start's two checks. If an address is missing, or both answer on one device, adopt
    nothing and stay in the error state until the next interval.
  - Then reopen both, and re-apply and read back the gains as the reconnect does now, before the
    error state clears.
- **What it costs.**
  - `probe_address` waits up to 0.15 s for an answer, with no retry, and a scan probes each path
    for each address not yet found. After a swap one probe goes unanswered, about 0.17 s, as at
    each start on 2026-10-25. With two paths and nothing answering, the scan takes about 0.6 s.
  - The scan follows the two failed `reconnect()` calls, which held the executor 0.42–0.43 s on the
    robot. So the tick that re-detects after a swap runs about 0.6 s, in the error state, where
    nothing drives.
  - A tick over 0.5 s sets the stall mark at the next tick, and the first `cmd_vel` taken within
    0.5 s after that is dropped: with a live publisher, its next message.
  - A message that the long tick's own executor pass already held is taken before the mark is set.
    If the re-detection clears the error state, that message then drives: the residual that
    [`driver-stale-cmd-vel-rearm`](../../completed/driver-stale-cmd-vel-rearm.md) names.
- **The restart stays the recovery until this ships.** `source/strafer_ros/README.md`'s driver
  paragraph says so, for a swap and for a controller replugged onto a new `ttyACM` number. A scan
  of the glob would also find a replugged controller's new name.

## Acceptance criteria

- [ ] **A fake-bus test with swapped names fails on `main`.**
  - Two fake controllers start on `/dev/roboclaw1` and `/dev/roboclaw2`, and the driver detects
    them.
  - A simulated main-power cut exchanges the two paths.
  - On `main` every reconnect fails and the error state holds. With the fix the next reconnect
    re-detects, maps front and rear to their new paths, re-applies and reads back the gains, and
    clears the error state.
- [ ] **The re-detection keeps the start's checks.** A missing address, or both addresses on one
      device, leaves the error state set, with nothing written beyond the probes.
- [ ] **A reconnect on the old names is unchanged.** The existing reconnect tests in
      `test_stall_and_reconnect.py` pass as they are.
- [ ] **`source/strafer_ros/README.md`'s driver paragraph** says that a swap recovers by
      re-detection.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No regression: `make test-ros` passes.

## Investigation pointers

- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py`: the auto-detect block in
  `__init__`, and the error-state branch of `_timer_callback`.
- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py`: `reconnect`,
  `detect_roboclaws`, `probe_address`, `same_device`, `ROBOCLAW_UDEV_GLOB`.
- `source/strafer_ros/strafer_driver/test/test_port_fence.py` (`FakeBus`, `robot()`), and
  `test_stall_and_reconnect.py` (`GainsClaw`, `power_cycle_and_reconnect`).
- The record's deposit: `step3/kernel_usb.txt`, and `host/base_log_all.txt` from 16:24:41 to
  16:25:33 UTC.

## Out of scope

- **Naming the controllers by hub port** in `99-strafer.rules`. That would remove the swap at its
  source, but it ties the rules to one wiring. By serial number they cannot be told apart: the
  rules' header says the two share identical vendor, product and serial attributes.
- **Configured ports** (`auto_detect: false`). They keep today's reconnect.
- **Exiting on a failed reconnect so that compose restarts `base`.** The startup refusal would then
  loop for as long as a controller is off.
- **A swap on the robot.** A main-power cut's enumeration order is the kernel's. A replug of both
  USB cables, after the driver's first reconnect attempt has released the old descriptors and in
  the opposite order to `base`'s detection, would stage one; that is untried. The next bench that
  cuts main power with the USB in, or stages the swap, records whether the driver re-detected.
