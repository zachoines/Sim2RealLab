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
- **One 12 V supply:** a 4S LiPo through a 12 V buck and a Y harness feeds both controllers. The
  controllers' readings here, 10.3 V and 11.1 V, are within 0.1 V of their 2026-10-20 readings,
  10.2 V and 11.2 V, taken against a metered 12 V at their terminals: about 1.8 V and 0.8 V low.
  The evidence favours their own sensing; which instrument is right stays open (corrected
  2026-10-21 from
  [`chassis-stop-modes-2026-10-20`](../../../measurements/chassis-stop-modes-2026-10-20/README.md)
  §1). ~~A charged 3S pack reads 12.6 V.~~

**A gains save goes with the timeout save.** Command 94 saves every active setting.
`diagnose_roboclaw.py --write-nvm` reloads the saved settings (95) before its save, so it never
saves gains the node wrote to RAM. A gains save would be a deliberate choice.
*2026-11-01: it was made. `diagnose_roboclaw.py --set-gains --write-nvm` writes the driver's
gains, saves them, and checks what each controller loads at the reset; both controllers' flash
now holds P 15000, I 750, D 0, QPPS 2657 ([`chassis-velocity-loop-2026-10-31`](../../../measurements/chassis-velocity-loop-2026-10-31/README.md)).*

## Acceptance criteria

- [x] **The pack is metered and charged.** Meter the pack, and each controller's `+`/`−`
      terminals, at no load. Record the three readings beside each controller's own reading, then
      charge the pack and record its charged voltage.
      *2026-10-20 ([`chassis-stop-modes-2026-10-20`](../../../measurements/chassis-stop-modes-2026-10-20/README.md) §1):
      the chain is a 4S LiPo, a 300 W 12 V buck converter and a Y harness to both controllers.
      Charged, the pack read 16.8 V; the buck's output and both controllers' terminals read 12 V,
      by the maintainer's meter at idle, read to the volt. The controllers read 10.2 V (front) and
      11.2 V (rear). The pack was not metered before the charge: the maintainer found it at a 4S
      storage charge and charged it first, so every reading here followed the charge.*
- [x] **The M2 deficit is re-measured with the charged pack,** at a metered 12 V supply. Run the
      open-loop duty test under equal duty, with the wheels lifted, and compare FR and RR against FL and RL. Record whether
      the deficit holds, and by how much.
      *2026-10-31 ([`chassis-velocity-loop-2026-10-31`](../../../measurements/chassis-velocity-loop-2026-10-31/README.md)): on FR it does not hold. At duty 32/127 over 4 s holds, FR ran
      2.8 % over the FL/RL mean forward and RR 0.8 % over; RR's 4.1 % backward shortfall was
      within three repeat spreads (it held, at 3.1 %, in the session after the maintainer's work). The 2026-10-19
      figures came from half-revolution runs from rest. FR had a slow spot locked to its wheel
      angle instead, which stalled its low-speed starts; it went after the maintainer re-seated
      FR's miter gears and oiled every wheel.*
- [ ] **Tuning, only after both boxes above:**
      - each motor's breakaway speed, the lowest commanded speed at which it turns, measured on
        blocks and on the floor at a recorded voltage;
      - each motor's QPPS, measured at that voltage;
      - gains chosen, shared or per controller, with the reason;
      - where the gains live decided: the node's constants or each controller's flash, with the
        other made consistent.
      *2026-11-01 ([`chassis-velocity-loop-2026-10-31`](../../../measurements/chassis-velocity-loop-2026-10-31/README.md)): all but the floor. On blocks, at the metered 12.07 V, the
      breakaway speeds at the old gains; QPPS 2630.5, 2658.0, 2672.0 and 2666.0 ticks/s, 1.6 % apart, so
      one shared set: P 15000, I 750, D 0, QPPS 2657, the lowest-gain candidate that passed. They
      live in `strafer_shared.constants`, and both controllers' flash holds the same. The
      breakaway on the floor is open.*
- [ ] **On the floor at 0.1 m/s,** all four wheels turn, and the chassis' heading drift over a
      measured distance is recorded.
- [x] **At 0.3 m/s,** each wheel is within ±5 % of its target.
      *2026-11-01, on blocks, through the driver ([`chassis-velocity-loop-2026-10-31`](../../../measurements/chassis-velocity-loop-2026-10-31/README.md)): within 2.1–4.0 % in the first
      run and 0.2–0.4 % in the second; at 0.1 m/s every wheel turned within 0.099–0.123 s.*
- [x] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).
      *2026-11-01: `source/strafer_ros/README.md` (the gains, and `--set-gains`).
      `SIM_TO_REAL_TUNING_GUIDE.md` is interim and left as it is.*
- [x] No regression: `make test-ros` passes.
      *2026-11-01: 989 passed.*

## Investigation pointers

- `source/strafer_ros/tune_pid.py` (`--read`, `--tune`, `--set`).
- `source/strafer_shared/strafer_shared/constants.py` (`ROBOCLAW_PID_*`, `ROBOCLAW_QPPS`).
- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py` (`_configure_pid`).

## Out of scope

- **The joint-position wrap:**
  [`roboclaw-encoder-count-wrap`](roboclaw-encoder-count-wrap.md).
- **The serial timeout**, set and saved on 2026-10-19.
