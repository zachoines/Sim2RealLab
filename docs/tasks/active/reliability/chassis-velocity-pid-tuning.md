# Tune the chassis' velocity PID, so every wheel follows a low commanded speed

**Type:** task (on the real chassis, a 0.1 m/s command leaves the front-right wheel still)
**Owner:** maintainer + Jetson (hands on the chassis; `tune_pid.py`, `strafer_shared/constants.py`)
**Priority:** P1 for the real lane. At 0.1 m/s the front-right wheel did not turn and the
rear-right started late. A command at that speed, from the trained policy or Nav2, turns the
chassis instead of driving it straight.
**Estimate:** M
**Branch:** `task/chassis-velocity-pid-tuning`

## Story

As **whoever drives the real chassis**, I want **each wheel to follow the commanded speed from
low speeds up**, so that **a straight command drives straight and a slow approach still moves.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`chassis-bringup-2026-10-19`](../../../measurements/chassis-bringup-2026-10-19/README.md) §4
  and §5, and its deposit's `drive/` runs.
- [`docs/WIRING_GUIDE.md`](../../../WIRING_GUIDE.md) §5 (Motion Studio).

## Context

Measured on 2026-10-19, with the wheels lifted, at `371e848` (record §5).

| commanded | FL | FR | RL | RR |
|---|---|---|---|---|
| 0.1 m/s, +2.08 rad/s each | +1.11 / +1.71 | 0.00 / 0.00 | +1.17 / +1.54 | 0.00 / +0.22 |
| 0.3 m/s, +6.25 rad/s each | +6.02 | +7.18 | +6.14 | +6.81 |

Mean wheel speed in rad/s over two runs at 0.1 m/s and one at 0.3 m/s.

- **The gains.** At every start the driver writes `ROBOCLAW_PID_P` 15000, `ROBOCLAW_PID_I` 750,
  `ROBOCLAW_PID_D` 0 and `ROBOCLAW_QPPS` 2796 to both motors of both controllers.
  `roboclaw_node.py`'s `_configure_pid` writes them to RAM, not flash.
- **The saved values differ by controller.** The front's flash holds P≈15000 and I≈750. The
  rear's holds zeros, so the rear has never had gains saved.
- **The low-speed command is small against QPPS.** 0.1 m/s is 178 ticks/s, 6 % of QPPS. The pack
  read 10.2–11.1 V at the controllers.
- **The motors and encoders work.** The open-loop duty test moved every wheel the right way at
  duty 32/127 (record §4). At 0.3 m/s all four follow, and the right side overshoots by 9–15 %.
- **Saving gains and saving the timeout go together.** Command 94 saves every active setting.
  `diagnose_roboclaw.py --write-nvm` reloads the saved settings (95) before its save, so it never
  saves gains the node wrote to RAM. A gains save made here would be a deliberate choice.

## Acceptance criteria

- [ ] **Each motor's breakaway speed is measured:** the lowest commanded speed at which it turns,
      on blocks and on the floor, with the pack voltage recorded.
- [ ] **Each motor's QPPS is measured** at that voltage.
- [ ] **Gains are chosen** (shared or per controller), with the reason recorded.
- [ ] **Where the gains live is decided:** the node's constants, written at every start, or each
      controller's flash. The other is made consistent with it.
- [ ] **On the floor at 0.1 m/s,** all four wheels turn and the chassis' heading drift over a
      measured distance is recorded.
- [ ] **At 0.3 m/s,** each wheel is within ±5 % of its target.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No regression: `make test-ros` passes.

## Investigation pointers

- `source/strafer_ros/tune_pid.py` (`--read`, `--tune`, `--set`).
- `source/strafer_shared/strafer_shared/constants.py` (`ROBOCLAW_PID_*`, `ROBOCLAW_QPPS`).
- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py` (`_configure_pid`).

## Out of scope

- **The joint-position wrap:**
  [`roboclaw-encoder-count-wrap`](roboclaw-encoder-count-wrap.md).
- **The serial timeout**, set and saved on 2026-10-19.
- **The battery readings.** The two controllers read 10.2–10.3 V and 11.1 V from one pack. Check
  them with a meter before tuning, since gains tuned on a low pack differ.
