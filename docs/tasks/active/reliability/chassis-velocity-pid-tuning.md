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

Measured on 2026-10-19, with the wheels lifted, at `371e848` (record §4 and §5).

| commanded | FL | FR | RL | RR |
|---|---|---|---|---|
| 0.1 m/s, +2.08 rad/s each | +1.11 / +1.71 | 0.00 / 0.00 | +1.17 / +1.54 | 0.00 / +0.22 |
| 0.3 m/s, +6.25 rad/s each | +6.02 | +7.18 | +6.14 | +6.81 |

Mean wheel speed in rad/s over each hold (0.6 s after the first message to the last), two runs at
0.1 m/s and one at 0.3 m/s.

**The gains are the same on both controllers.** At every start the driver writes
`ROBOCLAW_PID_P` 15000, `ROBOCLAW_PID_I` 750, `ROBOCLAW_PID_D` 0 and `ROBOCLAW_QPPS` 2796 to both
motors of both controllers. `roboclaw_node.py`'s `_configure_pid` writes them to RAM. So what each
controller has saved does not decide the running gains. The wheel that never moved, FR, is on the
front controller.

What the deposit shows:
- **A right-side (M2) deficit, open loop, under equal duty.** In the duty test (record §4) each
  motor ran at duty 32/127 until the controllers' 0.5 s timeout cut it. FR turned 229 ticks and RR
  195, against 253 for FL and RL.
- **Gains untuned for the low speed.** 0.1 m/s is 178 ticks/s, 6 % of QPPS. At 0.3 m/s all four
  follow, and the right side overshoots by 9–15 %.
- **The two controllers read different voltages from one pack at no load:** 10.3 V at one, 11.1 V
  at the other. A charged 3S pack reads 12.6 V.

**A gains save goes with the timeout save.** Command 94 saves every active setting.
`diagnose_roboclaw.py --write-nvm` reloads the saved settings (95) before its save, so it never
saves gains the node wrote to RAM. A gains save would be a deliberate choice.

## Acceptance criteria

- [ ] **The pack is metered and charged.** Meter the pack, and each controller's `+`/`−`
      terminals, at no load. Record the three readings beside each controller's own reading, then
      charge the pack and record its charged voltage.
- [ ] **The M2 deficit is re-measured with the charged pack.** Run the open-loop duty test under
      equal duty, with the wheels lifted, and compare FR and RR against FL and RL. Record whether
      the deficit holds, and by how much.
- [ ] **Tuning, only after both boxes above:**
      - each motor's breakaway speed, the lowest commanded speed at which it turns, measured on
        blocks and on the floor at a recorded voltage;
      - each motor's QPPS, measured at that voltage;
      - gains chosen, shared or per controller, with the reason;
      - where the gains live decided: the node's constants or each controller's flash, with the
        other made consistent.
- [ ] **On the floor at 0.1 m/s,** all four wheels turn, and the chassis' heading drift over a
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
