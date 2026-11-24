#!/usr/bin/env python3
"""PID tuning script for RoboClaw velocity controllers.

Programmatically sets PID/QPPS values and runs step-response tests that
measure each motor: it drives with command 37, as the driver does, reads the
encoder counts (commands 16 and 17), passes a motor whose mean speed tracks
the target within ±5 % on the default step, and reports the dead time and
time constant fitted to the count. The robot is tuned for tracking and then
measured; nothing here aims at a time constant.

Usage:
    # Read current PID on both controllers
    python3 tune_pid.py --read

    # Set PID values and run step-response test
    python3 tune_pid.py --tune

    # Set specific PID values without running a test
    python3 tune_pid.py --set --p 15000 --i 1000 --d 0

    # Run step-response only (after PID is already set)
    python3 tune_pid.py --step-response

    All commands accept --controller front|rear|both (default: both)
    and --motor m1|m2|both (default: both). The step's options, --speed,
    --hold and --settle, default to the values below, each with its reason;
    the box and the safety limits are fixed.
"""

from __future__ import annotations
from strafer_shared.constants import (
    RADIANS_TO_ENCODER_TICKS,
    ROBOCLAW_BAUD_RATE,
    ROBOCLAW_FRONT_ADDRESS,
    ROBOCLAW_PID_D,
    ROBOCLAW_PID_I,
    ROBOCLAW_PID_P,
    ROBOCLAW_QPPS,
    ROBOCLAW_REAR_ADDRESS,
    WHEEL_RADIUS,
)
from strafer_driver.roboclaw_interface import (
    RoboClawInterface,
    detect_roboclaws,
)

import argparse
import atexit
import bisect
import math
import signal
import sys
import termios
import time
from typing import NamedTuple


# Global list of open controllers for cleanup on exit
_open_controllers: list[RoboClawInterface] = []

# ---------------------------------------------------------------------------
# Step-response parameters
# ---------------------------------------------------------------------------

# The gains the driver writes at every start; --tune writes them unless given others.
DEFAULT_P = ROBOCLAW_PID_P
DEFAULT_I = ROBOCLAW_PID_I
DEFAULT_D = ROBOCLAW_PID_D

# The step and its box: the step the driver's gains were chosen on, and the box
# they passed. Stepped to 0.3 m/s at the wheel and held 6 s, each wheel's mean
# speed from 0.6 s after the first command to the last was within ±5 % of its
# target. The integral term closes the last few percent over seconds, so the
# box judges only that step: --speed and --hold vary the step, and off their
# defaults a run reports its tracking with no verdict. The box's start and
# width are fixed.
STEP_SPEED_MPS = 0.3
STEP_VELOCITY_TICKS = round(STEP_SPEED_MPS / WHEEL_RADIUS * RADIANS_TO_ENCODER_TICKS)  # 535
STEP_HOLD_S = 6.0
TRACK_START_S = 0.6  # the rise is over by then
TRACK_TOLERANCE = 0.05
# --settle: the wait after a stop, before the rest check and between
# controllers. A lifted wheel stopped at duty 0 from about 600 ticks/s is at
# rest within it.
SETTLE_TIME_S = 0.2
# --hold is at least MIN_HOLD_S: the tracking window then spans 20 cycles or
# more, so one tick is under 1 % of the travel at the default step, and the
# stall abort, from STALL_AFTER_S, watches half the hold or more.
MIN_TRACK_WINDOW_S = 0.4
MIN_HOLD_S = TRACK_START_S + MIN_TRACK_WINDOW_S
MAX_HOLD_S = 30.0  # a cap on a mistyped --hold or --settle: five default holds
# Each cycle re-sends command 37 and reads the counts, at the driver's rate
# (roboclaw_node's publish_rate, 50 Hz).
SAMPLE_PERIOD_S = 0.02

# Rest: a count that moves more than REST_TICKS over PRE_S (ten cycles) before
# the step is not at rest, and one that moves more than that over STILL_CHECK_S,
# from STILL_CHECK_S after a stop, is turning. Either check catches every wheel
# turning at 20 ticks/s or more, under 4 % of the default step. A wheel at rest
# moves no tick: two let an encoder resting on an edge flicker a count, with one
# to spare. A lifted wheel stopped at duty 0 from about 600 ticks/s is at rest
# within STILL_CHECK_S.
PRE_S = 0.2
REST_TICKS = 2
STILL_CHECK_S = 0.2

# A step drives only while the controller's packet-serial timeout (command 15)
# reads in this range, so the controller stops the wheels if this tool dies
# mid-step: command 14 sets it in 0.1 s steps (0 disables it), and 1.0 s bounds
# how long a dead tool's wheels run.
SERIAL_TIMEOUT_RANGE_S = (0.1, 1.0)
# Each motor's duty-0 stop is sent up to STOP_TRIES times, until acknowledged:
# three ride out two lost packets. A final stop still unacknowledged is left to
# the controller's timeout: no packet for SILENT_S, two of command 14's 0.1 s
# steps past the longest timeout a step accepts.
STOP_TRIES = 3
SILENT_S = SERIAL_TIMEOUT_RANGE_S[1] + 0.2

# Aborts: a motor turning against its target (reversed motor or encoder
# leads), running away from it, or stalling (a dead encoder, or a jammed wheel
# or one not broken away). Both motors are watched, the one held at 0 too. A
# speed is the count's change over one cycle, so it moves in steps of
# 1 tick / SAMPLE_PERIOD_S = 50 ticks/s, and a count jitters by a tick.
#
# Against: over AGAINST_TPS, 3 ticks a cycle against the target, three times
# the jitter, for AGAINST_SAMPLES cycles (60 ms) in a row, so one glitched count
# does not stop the run.
AGAINST_TPS = 150
AGAINST_SAMPLES = 3
# Runaway: over RUNAWAY_FACTOR times |target|, far past a working loop's
# overshoot, plus RUNAWAY_MARGIN_TPS (6 ticks a cycle), so a low step's jitter
# does not trip it and a motor held at 0 has a limit too, for RUNAWAY_SAMPLES
# cycles (0.2 s) in a row, so only a sustained runaway trips it.
RUNAWAY_FACTOR = 2
RUNAWAY_MARGIN_TPS = 300
RUNAWAY_SAMPLES = 10
# Stall: from STALL_AFTER_S, when its window starts 0.2 s in, past a working
# wheel's dead time, a stepped motor whose count over the last STALL_WINDOW_S
# (15 cycles) moved under STALL_FRACTION of its target's travel. A working wheel
# moves most of that travel, and at the default step a tenth of it is 16 ticks,
# well over the count's one-tick steps; a dead encoder or a jammed wheel moves
# none. STALL_FRACTION also flags a stalled mean in the report.
STALL_AFTER_S = 0.5
STALL_WINDOW_S = 0.3
STALL_FRACTION = 0.1
# Count reads failed, or command 37 unacknowledged, MAX_MISSED cycles in a row.
# A missed cycle is a command 37 or its acknowledgement lost (the interface does
# not retry it), or a count read failed twice, with the interface's one retry;
# one is let pass, two in a row are a failing link.
MAX_MISSED = 2

# A count read is timed when it returns, and its count was latched somewhere
# inside it. A read longer than SAMPLE_PERIOD_S (a retry, or a pause after the
# controller answered) cannot be placed within its cycle, so the tracking and
# the fit leave its sample out; the abort rules take it. A read longer than the
# interface's tries wait unpaused (_longest_read_s) held a pause, which took
# that cycle's timing, not only its sample's (the next cycle runs at once): it
# ends the step, as the pause checks do. A pause under a cycle is left: it can
# make either end of a window up to a cycle late, which at the default step
# moves the mean by up to 0.4 % either way, and a sample in the rise up to a
# cycle late, which can move the printed 63 % point by about 3 ms and the dead
# time by up to about 18 ms, unflagged.

# The largest step keeps the runaway limit RUNAWAY_HEADROOM of ROBOCLAW_QPPS,
# the motors' mean full speed, so a motor slower than the mean (the slowest is
# 1 % under it) and the full-speed count's scatter still trip it.
RUNAWAY_HEADROOM = 0.9
MAX_STEP_TICKS = (int(RUNAWAY_HEADROOM * ROBOCLAW_QPPS) - RUNAWAY_MARGIN_TPS) // RUNAWAY_FACTOR
# The smallest step is a tick a cycle over AGAINST_TPS, so the against abort sees
# a reversed motor in every cycle whose reads are 15 ms or more apart: under it a
# reversed wheel turns 3 ticks in some cycles, which reads exactly AGAINST_TPS and
# is not counted, so the abort can miss it for a whole hold.
MIN_STEP_TICKS = round(AGAINST_TPS + 1 / SAMPLE_PERIOD_S)

# The fit covers the rise, before the box opens: after it, the integral term's
# slow approach to the target is no first-order lag, and over a whole hold it
# pins the dead time to 0 and stretches tau.
FIT_END_S = TRACK_START_S
# The fit's search: dead time on [0, 0.3] s and tau on [5 ms, 3 s]. The dead
# time's 0 is physical, but a fit held there has not resolved the split between
# the dead time and tau, and its sum may be short. Its 0.3 s, half the window,
# is far past a working wheel's (under 20 ms) and leaves the rest of the window
# to the rise. Tau's floor is a quarter of a cycle: a shorter lag rises within
# one cycle, so at most one read falls in its rise and the rest see only its
# sum with the dead time. Past its 3 s, five windows, a rise is a ramp inside
# the window, which fixes only its slope. A fit at any of these four bounds is
# reported as no measurement.
DEAD_TIME_MAX_S = 0.3
TAU_MIN_S = 0.005
TAU_MAX_S = 3.0
MIN_FIT_SAMPLES = 10  # 0.2 s from the first command, for three parameters
# The search starts on a coarse grid, 60 taus each 11 % above the last and the
# dead time in half-cycle steps, then runs a pattern search until its dead-time
# step is under FIT_STEP_MIN_S, far below the fit's scatter.
TAU_GRID_POINTS = 60
DEAD_TIME_GRID_S = SAMPLE_PERIOD_S / 2
FIT_STEP_MIN_S = 1e-6

LOW_SPEED_FRACTION = 0.5  # a mean under this fraction of the target: wiring or load, not the gains
EXIT_NO_VERDICT = 3  # a step off the default one: its tracking reported, with no verdict
EXIT_INTERRUPTED = 130  # 128 + SIGINT, a shell's status for a Ctrl-C


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


class StepFit(NamedTuple):
    """A dead time and a first-order lag fitted to a step's encoder count."""

    dead_time_s: float
    tau_s: float
    speed: float  # the level the rise settles toward, ticks/s
    rms_ticks: float  # the count's rms residual about the fit


class StepTrace:
    """One motor's step: its count from rest, timed from the first command."""

    def __init__(self, motor: str, target: int) -> None:
        self.motor = motor
        self.target = target
        self.times: list[float] = []  # s, when each count read returned
        self.counts: list[int] = []  # ticks from the count at rest
        self.slow: set[int] = set()  # the samples whose read took longer than a cycle
        self.zero_unknown = False  # the first command 37 went unacknowledged
        self.abort: str | None = None  # why the step did not run its hold
        self.fault: str | None = None  # what the check after the stop found


def count_delta(a: int, b: int) -> int:
    """Signed change from count *a* to count *b*, folding the 32-bit wrap."""
    return ((b - a + 2**31) % 2**32) - 2**31


def _step_sse(
    times: list[float], counts: list[float], sum_cc: float, dead: float, tau: float
) -> tuple[float, float]:
    """(sum of squared residuals, best final speed) for one dead time and tau."""
    sum_gg = sum_gc = 0.0
    for t, c in zip(times, counts):
        x = t - dead
        if x <= 0.0:
            continue
        g = x - tau * (1.0 - math.exp(-x / tau))
        sum_gg += g * g
        sum_gc += g * c
    if sum_gg == 0.0:
        return sum_cc, 0.0
    return sum_cc - sum_gc * sum_gc / sum_gg, sum_gc / sum_gg


def fit_dead_time_and_tau(
    times: list[float], counts: list[float], end_s: float = FIT_END_S
) -> StepFit | None:
    """Fit a dead time and a time constant to a step's encoder count.

    The speed is a first-order lag behind a dead time; its integral is the count:

        count(t) = V * (x - tau * (1 - exp(-x / tau))),  x = max(0, t - dead_time)

    with *times* in seconds from the first command and *counts* in ticks from
    the count at rest, over the samples before *end_s*. For each (dead time,
    tau) the best V is linear least squares. The pair comes from a coarse grid,
    then a pattern search from its best point: a 3x3 stencil that moves while
    it improves and halves its steps when it does not, so it follows the
    valley where more dead time trades for less tau. The dead time is bounded
    below by 0; a fit held there has not resolved the split. Fitting the count,
    not the reported speed (commands 18 and 19), avoids the speed reading's lag
    behind the wheel.

    The dead time plus tau (the time to 63 % of the level) is better resolved
    than the split between them. From whole-tick counts at 50 Hz alone, for a
    rise that levels off near 510 ticks/s with tau near the sample period, the
    sum scatters by under 1 ms (one standard deviation) and the split by about
    2 ms; both grow as the level falls, to about 2 and 5 ms at a level near
    140 ticks/s.

    Returns None with fewer than MIN_FIT_SAMPLES samples, or a count that
    never leaves rest.
    """
    pts = [(t, float(c)) for t, c in zip(times, counts) if 0.0 <= t < end_s]
    if len(pts) < MIN_FIT_SAMPLES or max(abs(c) for _, c in pts) <= REST_TICKS:
        return None
    ts = [t for t, _ in pts]
    cs = [c for _, c in pts]
    sum_cc = sum(c * c for c in cs)

    def trial(dead, tau, best):
        dead = min(max(dead, 0.0), DEAD_TIME_MAX_S)
        tau = min(max(tau, TAU_MIN_S), TAU_MAX_S)
        sse, v = _step_sse(ts, cs, sum_cc, dead, tau)
        return (sse, dead, tau, v) if sse < best[0] else best

    ratio = (TAU_MAX_S / TAU_MIN_S) ** (1.0 / (TAU_GRID_POINTS - 1))
    best = (math.inf, 0.0, TAU_MIN_S, 0.0)
    for i in range(round(DEAD_TIME_MAX_S / DEAD_TIME_GRID_S) + 1):
        for k in range(TAU_GRID_POINTS):
            best = trial(i * DEAD_TIME_GRID_S, TAU_MIN_S * ratio**k, best)

    dead_step, tau_step = DEAD_TIME_GRID_S, ratio  # tau steps are factors
    while dead_step > FIT_STEP_MIN_S:
        center = best
        for dd in (-1, 0, 1):
            for dt in (-1, 0, 1):
                best = trial(center[1] + dd * dead_step, center[2] * tau_step**dt, best)
        if best is center:
            dead_step /= 2.0
            tau_step = math.sqrt(tau_step)

    sse, dead, tau, v = best
    return StepFit(
        dead_time_s=dead,
        tau_s=tau,
        speed=v,
        rms_ticks=math.sqrt(max(sse, 0.0) / len(ts)),
    )


def tracking_speed(
    times: list[float], counts: list[float], start: float, end: float
) -> float | None:
    """Mean speed over [start, end] from the count, in ticks/s.

    The count's change between the first and last samples in the window, over
    the time between them. None with fewer than two samples there.
    """
    win = [(t, c) for t, c in zip(times, counts) if start <= t <= end]
    if len(win) < 2 or win[-1][0] <= win[0][0]:
        return None
    (t0, c0), (t1, c1) = win[0], win[-1]
    return (c1 - c0) / (t1 - t0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hold_signals(held: list[str], handlers: dict) -> None:
    """Hold a Ctrl-C, Ctrl-Z, SIGTERM or SIGHUP in *held* while it is empty, and raise
    KeyboardInterrupt on one after, storing in *handlers* the handlers this replaces; off the
    main thread none is installed."""

    def hold(signum: int, _frame) -> None:
        if held:
            raise KeyboardInterrupt(f"signal {signum}")
        held.append(f"signal {signum}")

    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGTSTP):
            handlers[sig] = signal.signal(sig, hold)
    except ValueError:
        pass  # not the main thread: nothing is held


def _restore_signals(handlers: dict) -> None:
    """Put back the handlers _hold_signals replaced."""
    for sig, handler in handlers.items():
        signal.signal(sig, signal.SIG_DFL if handler is None else handler)


def hard_stop(rc: RoboClawInterface) -> bool:
    """Force-stop both motors using duty-cycle commands (bypasses PID).

    This sends forward_m1(0) and forward_m2(0) which switch the RoboClaw
    out of velocity PID mode into duty-cycle mode with 0% duty = stopped.
    This is critical because drive_m*_speed(0) only sets the PID target
    to 0, and with accumulated I-term or reversed motors, the PID may
    continue driving.

    Each motor's stop is sent on its own and retried until acknowledged, up
    to STOP_TRIES times. A Ctrl-C, Ctrl-Z, SIGTERM or SIGHUP meanwhile is held
    and raised once both have been sent; a second one in the same call raises
    at once, abandoning this call's stops (a write that hangs, say) to the
    controller's serial timeout. Returns True when both were acknowledged.
    """
    held: list[str] = []
    handlers = {}
    try:
        _hold_signals(held, handlers)  # off the main thread the stops go out unheld
        acked = True
        for stop in (rc.forward_m1, rc.forward_m2):
            for _ in range(STOP_TRIES):
                try:
                    if stop(0):
                        break
                except Exception:
                    pass  # Best-effort during cleanup
            else:
                acked = False
    finally:
        _restore_signals(handlers)
    if held:
        raise KeyboardInterrupt(held[0])
    return acked


def _atexit_stop_all() -> None:
    """Emergency stop all motors on script exit."""
    for rc in _open_controllers:
        hard_stop(rc)
        try:
            rc.close()
        except Exception:
            pass
    _open_controllers.clear()


atexit.register(_atexit_stop_all)


def open_controller(port: str, address: int) -> RoboClawInterface:
    """Open and return a RoboClawInterface for the given port.

    It is opened as the driver opens it, with the interface's read timeout and
    retries, and listed for the stop on exit before its first stop, so an
    interrupted one is stopped on the way out.
    """
    rc = RoboClawInterface(port=port, address=address, baud_rate=ROBOCLAW_BAUD_RATE)
    rc.open()
    _open_controllers.append(rc)
    hard_stop(rc)  # Ensure motors are stopped on connect
    return rc


def close_controller(rc: RoboClawInterface) -> None:
    """Hard-stop and close a controller, removing it from the cleanup list."""
    hard_stop(rc)
    try:
        rc.close()
    except Exception:
        pass
    if rc in _open_controllers:
        _open_controllers.remove(rc)


# Cached auto-detected port mapping
_detected_ports: dict[int, str] | None = None


def _get_detected_ports() -> dict[int, str]:
    """Auto-detect RoboClaw ports (cached)."""
    global _detected_ports
    if _detected_ports is None:
        _detected_ports = detect_roboclaws(
            ROBOCLAW_FRONT_ADDRESS, ROBOCLAW_REAR_ADDRESS, ROBOCLAW_BAUD_RATE
        )
        if _detected_ports:
            print(
                f"Auto-detected: {', '.join(f'0x{a:02X}->{p}' for a, p in _detected_ports.items())}"
            )
        else:
            print(
                "WARNING: No RoboClaws found on /dev/roboclaw* (the only names scanned; "
                "`make udev` installs 99-strafer.rules)."
            )
    return _detected_ports


def get_controllers(which: str) -> list[tuple[str, str, int]]:
    """Return list of (label, port, address) based on --controller arg."""
    detected = _get_detected_ports()
    controllers = []
    if which in ("front", "both"):
        port = detected.get(ROBOCLAW_FRONT_ADDRESS)
        if port:
            controllers.append(("Front", port, ROBOCLAW_FRONT_ADDRESS))
        else:
            print(f"WARNING: Front RoboClaw (0x{ROBOCLAW_FRONT_ADDRESS:02X}) not detected.")
    if which in ("rear", "both"):
        port = detected.get(ROBOCLAW_REAR_ADDRESS)
        if port:
            controllers.append(("Rear", port, ROBOCLAW_REAR_ADDRESS))
        else:
            print(f"WARNING: Rear RoboClaw (0x{ROBOCLAW_REAR_ADDRESS:02X}) not detected.")
    return controllers


def get_motor_list(which: str) -> list[str]:
    """Return list of motor names based on --motor arg."""
    if which == "m1":
        return ["M1"]
    if which == "m2":
        return ["M2"]
    return ["M1", "M2"]


def _read_count(rc: RoboClawInterface, motor: str) -> tuple[float, int, float]:
    """(monotonic time the read returned, raw count, monotonic time it started) for one motor."""
    started = time.monotonic()
    count, _ = rc.read_encoder_m1() if motor == "M1" else rc.read_encoder_m2()
    return time.monotonic(), count, started


def _longest_read_s(rc: RoboClawInterface) -> float:
    """The longest the tries of a read through *rc* wait unpaused: each waits out the read
    timeout, the last answered at the end of it, with the retry pause between them."""
    return (1 + rc.max_retries) * rc.read_timeout + rc.max_retries * rc.retry_pause


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_read(args: argparse.Namespace) -> None:
    """Read and display current PID values."""
    for label, port, addr in get_controllers(args.controller):
        print(f"\n{'='*50}")
        print(f"  {label} RoboClaw (port={port}, addr=0x{addr:02X})")
        print(f"{'='*50}")
        try:
            rc = open_controller(port, addr)
            pid_m1 = rc.read_velocity_pid_m1()
            pid_m2 = rc.read_velocity_pid_m2()
            rc.close()
            print(
                f"  M1 PID: P={pid_m1['P']:>6}  I={pid_m1['I']:>6}  D={pid_m1['D']:>6}  QPPS={pid_m1['QPPS']}"
            )
            print(
                f"  M2 PID: P={pid_m2['P']:>6}  I={pid_m2['I']:>6}  D={pid_m2['D']:>6}  QPPS={pid_m2['QPPS']}"
            )
        except Exception as e:
            print(f"  ERROR: {e}")


def cmd_set(args: argparse.Namespace) -> None:
    """Set PID values on selected controllers/motors."""
    p = args.p
    i = args.i
    d = args.d
    qpps = args.qpps if args.qpps is not None else ROBOCLAW_QPPS
    motors = get_motor_list(args.motor)

    for label, port, addr in get_controllers(args.controller):
        print(f"\n{label} RoboClaw (0x{addr:02X}):")
        try:
            rc = open_controller(port, addr)

            for motor in motors:
                if motor == "M1":
                    ok = rc.set_velocity_pid_m1(p, i, d, qpps)
                else:
                    ok = rc.set_velocity_pid_m2(p, i, d, qpps)
                status = "OK" if ok else "FAILED"
                print(f"  {motor} SET P={p} I={i} D={d} QPPS={qpps} -> {status}")

            # Verify by reading back
            pid_m1 = rc.read_velocity_pid_m1()
            pid_m2 = rc.read_velocity_pid_m2()
            print(
                f"  Readback M1: P={pid_m1['P']} I={pid_m1['I']} D={pid_m1['D']} QPPS={pid_m1['QPPS']}"
            )
            print(
                f"  Readback M2: P={pid_m2['P']} I={pid_m2['I']} D={pid_m2['D']} QPPS={pid_m2['QPPS']}"
            )

            rc.close()
        except Exception as e:
            print(f"  ERROR: {e}")


def run_step_response(
    rc: RoboClawInterface,
    motors: list[str],
    speed_ticks: int,
    hold_s: float,
    settle_s: float = SETTLE_TIME_S,
) -> list[StepTrace]:
    """Step the selected motors of one controller to *speed_ticks* and record their counts.

    Command 37 sets both motors at once, as the driver does, and is re-sent
    every SAMPLE_PERIOD_S for *hold_s*, timed from the first one's return: a
    pause before the controller hears it is no dead time, one after it
    shortens the dead time and the 63 % point by about its length, unflagged
    until the dead time reads 0, and an unacknowledged first one flags the fit.
    A motor not selected is held at 0. Each cycle reads
    both motors' counts (commands 16 and 17); a sample whose read took longer
    than a cycle is marked slow, and the report leaves it out. The step does
    not start unless the controller's serial timeout is armed, the stop is
    acknowledged and, *settle_s* later, both counts are at rest. It stops, for
    both motors, when a motor turns against its target, runs away from it, or
    (if stepped) stalls, when reads fail or command 37 goes unacknowledged
    MAX_MISSED cycles in a row, when a cycle starts, or its command 37 ends,
    later than the serial timeout after the last cycle started (the rest
    check's last read, for the first; the controller may have stopped the
    wheels), or when a count read takes longer than the interface's tries can
    (the process was paused): the step is not resumed, nor taken for a finished
    hold. An unacknowledged final stop is followed by SILENT_S without a
    packet, and a motor turning STILL_CHECK_S after the stop fails.
    """
    traces = {m: StepTrace(motor=m, target=speed_ticks) for m in motors}
    target = {m: speed_ticks if m in motors else 0 for m in ("M1", "M2")}

    def abort(reason: str, by: str | None = None) -> list[StepTrace]:
        for m, tr in traces.items():
            tr.abort = reason if by in (None, m) else f"stopped with {by}: {reason}"
        return list(traces.values())

    try:
        timeout = rc.read_serial_timeout()
    except Exception as e:
        return abort(f"serial timeout unread ({e}); not stepped")
    lo, hi = SERIAL_TIMEOUT_RANGE_S
    if not lo <= timeout <= hi:
        return abort(
            f"serial timeout {timeout:g} s (0 disables it); a step needs {lo:g} to {hi:g} s, "
            "so the controller stops the wheels if this tool dies. Not stepped; set it with "
            "diagnose_roboclaw.py --set-serial-timeout."
        )

    # Hard-stop both motors before test (duty-cycle, bypasses PID)
    if not hard_stop(rc):
        return abort("the duty-0 stop went unacknowledged; not stepped")
    time.sleep(settle_s)

    # The counts at rest: the last before the first command is each motor's zero.
    rest: dict[str, list[int]] = {m: [] for m in target}
    end_pre = time.monotonic() + PRE_S
    while time.monotonic() < end_pre:
        for m in target:
            _, count, heard = _read_count(rc, m)
            rest[m].append(count)
        time.sleep(SAMPLE_PERIOD_S)
    for m in target:
        moved = max(abs(count_delta(rest[m][0], c)) for c in rest[m])
        if moved > REST_TICKS:
            hard_stop(rc)
            return abort(
                f"{m} not at rest before the step (count moved {moved} ticks); not stepped", m
            )

    zero = {m: rest[m][-1] for m in target}
    last = {m: (0.0, 0) for m in target}
    against = {m: 0 for m in target}
    runaway = {m: 0 for m in target}
    missed_reads = missed_acks = 0
    longest = _longest_read_s(rc)
    try:
        # The rest check's last read stands for the cycle before the first: the controller
        # last heard a packet in it, so the first cycle's checks count from its start.
        t0, sent = None, heard
        while True:
            # The gap is checked before the hold's end, so a pause that carries the
            # clock past the end is not taken for a finished hold.
            now = time.monotonic()
            if now - sent > timeout:
                abort(
                    f"{now - sent:.2f} s since the last cycle (or the rest check), over the "
                    f"{timeout:g} s serial timeout that stops the wheels (the process was "
                    "paused, or packets went unanswered); the step is not resumed"
                )
                break
            if t0 is not None and now - t0 >= hold_s:
                break
            prev, sent = sent, now
            missed_acks = 0 if rc.drive_m1m2_speed(target["M1"], target["M2"]) else missed_acks + 1
            written = time.monotonic()
            # The controller has heard nothing since the last cycle's reads: a pause
            # after the clock read, which the gap above misses, shows here, and the
            # stop follows the late command 37 at once.
            late = written - prev
            if late > timeout:
                abort(
                    f"{late:.2f} s from the last cycle (or the rest check) to the end of this "
                    f"one's command 37, over the {timeout:g} s serial timeout that stops the "
                    "wheels (the process was paused, or packets went unanswered); the step is "
                    "not resumed"
                )
                break
            if t0 is None:
                # The step is timed from its first command 37's return, as each sample is from
                # its read's: a pause before the controller hears that command is no dead time.
                # One after it shortens the dead time and the 63 % point by about its length,
                # unflagged until the dead time reads 0. Unacknowledged, the return says nothing
                # of when the controller heard it (if at all), so the fit is flagged.
                t0 = next_cycle = written
                for tr in traces.values():
                    tr.zero_unknown = missed_acks > 0
            if missed_acks >= MAX_MISSED:
                abort(f"command 37 unacknowledged {MAX_MISSED} cycles in a row")
                break
            try:
                reads = {m: _read_count(rc, m) for m in target}
            except Exception:
                missed_reads += 1
                if missed_reads >= MAX_MISSED:
                    abort(f"{MAX_MISSED} count reads failed in a row")
                    break
                reads = None
            stop = None
            if reads is not None:
                missed_reads = 0
                took = {m: t_read - t_start for m, (t_read, _, t_start) in reads.items()}
                slowest = max(took, key=took.get)
                if took[slowest] > longest:
                    abort(
                        f"{slowest}'s count read took {took[slowest]:.2f} s, over the "
                        f"{longest:g} s its tries can take (the process was paused); the step "
                        "is not resumed"
                    )
                    break
                for m, (t_read, raw, _) in reads.items():
                    t, c = t_read - t0, count_delta(zero[m], raw)
                    if m in traces:
                        if took[m] > SAMPLE_PERIOD_S:
                            traces[m].slow.add(len(traces[m].times))
                        traces[m].times.append(t)
                        traces[m].counts.append(c)
                    t_prev, c_prev = last[m]
                    last[m] = (t, c)
                    if t <= t_prev:
                        continue
                    v = (c - c_prev) / (t - t_prev)
                    goal = target[m]
                    backward = goal != 0 and v * math.copysign(1, goal) < -AGAINST_TPS
                    against[m] = against[m] + 1 if backward else 0
                    fast = abs(v) > RUNAWAY_FACTOR * abs(goal) + RUNAWAY_MARGIN_TPS
                    runaway[m] = runaway[m] + 1 if fast else 0
                    if against[m] >= AGAINST_SAMPLES:
                        stop = (
                            f"{m} turns against its target ({v:+.0f} ticks/s for "
                            f"{goal:+d}): REVERSED motor/encoder direction? "
                            f"Swap {m}A / {m}B."
                        )
                    elif runaway[m] >= RUNAWAY_SAMPLES:
                        stop = f"{m} ran away: {v:+.0f} ticks/s for a target of {goal:+d}"
                    elif goal != 0 and t >= STALL_AFTER_S:
                        stop = _stalled(traces[m], goal)
                    if stop:
                        hard_stop(rc)
                        abort(stop, m)
                        break
            if stop:
                break
            # The cycles keep the 50 Hz grid. A cycle that ends past the next grid
            # point moves the grid to one period after its own start, and the missed
            # cycles are not run. After a pause between this cycle's clock read and
            # its command 37 reaching the controller, that is past too: the next cycle
            # runs at once, its command 37 one cycle's I/O after this one's, and no
            # rule trips on it.
            next_cycle += SAMPLE_PERIOD_S
            if next_cycle <= time.monotonic():
                next_cycle = now + SAMPLE_PERIOD_S
            time.sleep(max(0.0, next_cycle - time.monotonic()))
    finally:
        # Hard-stop motor (duty-cycle, not PID)
        stopped = hard_stop(rc)
        if not stopped:
            print(
                f"    STOP NOT ACKNOWLEDGED: silent {SILENT_S:g} s so the controller's serial "
                "timeout stops the wheels."
            )
            time.sleep(SILENT_S)

    # Verify the motors stopped: a wheel turning STILL_CHECK_S after the stop is
    # stopped again, and fails.
    faults = [] if stopped else [f"the duty-0 stop went unacknowledged (silent {SILENT_S:g} s)"]
    moving = _moving_after_stop([rc], list(target))[0]
    if isinstance(moving, Exception):
        faults.append(f"the stop could not be checked ({moving})")
    elif moving:
        hard_stop(rc)
        faults.append(f"{', '.join(moving)} still moving after the stop")
    for tr in traces.values():
        tr.fault = "; ".join(faults) or None

    return list(traces.values())


def _moving_after_stop(
    controllers: list[RoboClawInterface], motors: list[str]
) -> list[list[str] | Exception]:
    """Each controller's motors turning after its stop, judged on the count, or the error that
    ended its check.

    From STILL_CHECK_S after the last stop, two reads of every controller's counts,
    STILL_CHECK_S apart: a motor whose count moves more than REST_TICKS between them is turning.
    So each controller's window starts at least STILL_CHECK_S after its own stop.
    """

    def counts(rc: RoboClawInterface) -> dict[str, int] | Exception:
        try:
            return {m: _read_count(rc, m)[1] for m in motors}
        except Exception as e:
            return e

    time.sleep(STILL_CHECK_S)
    before = [counts(rc) for rc in controllers]
    time.sleep(STILL_CHECK_S)
    after = [counts(rc) for rc in controllers]
    return [
        b if isinstance(b, Exception) else a if isinstance(a, Exception)
        else [m for m in motors if abs(count_delta(b[m], a[m])) > REST_TICKS]
        for b, a in zip(before, after)
    ]


def _stalled(trace: StepTrace, goal: int) -> str | None:
    """Why a stepped motor has stalled, or None.

    It has, when its count over the last STALL_WINDOW_S moved under
    STALL_FRACTION of the travel its target asks for.
    """
    t, c = trace.times[-1], trace.counts[-1]
    k = bisect.bisect_right(trace.times, t - STALL_WINDOW_S) - 1
    if k < 0:
        return None
    moved, span = abs(c - trace.counts[k]), t - trace.times[k]
    if moved >= STALL_FRACTION * abs(goal) * span:
        return None
    return (
        f"{trace.motor}'s count moved {moved} ticks in {span:.2f} s at {goal:+d} ticks/s: "
        "a dead encoder, a jammed wheel, or one not broken away at this speed"
    )


def _report_step_result(
    trace: StepTrace, hold_s: float, verbose: bool, judged: bool = True
) -> bool:
    """Print one motor's tracking and its fit, then its verdict. False on a fail.

    It passes when its mean speed over [TRACK_START_S, *hold_s*] is within
    TRACK_TOLERANCE of its target. Off the default step (*judged* False) its
    tracking is printed with no verdict, and only a failed measurement fails.
    """
    speed = trace.target
    fault = f"; {trace.fault}" if trace.fault else ""
    if trace.abort:
        print(f"    ABORT: {trace.abort}{fault}")
        _print_counts(trace, verbose)
        return False

    if not trace.times:
        print(f"    No data collected -- check motor/encoder connection{fault}")
        return False

    # The tracking and the fit take only the samples whose read took at most a cycle.
    timed = [k for k in range(len(trace.times)) if k not in trace.slow]
    times, counts = [trace.times[k] for k in timed], [trace.counts[k] for k in timed]
    tracked = tracking_speed(times, counts, TRACK_START_S, hold_s)
    within = (
        f" ({len(timed)} read within a cycle, {len(trace.slow)} left out)" if trace.slow else ""
    )
    print(f"    Samples:      {len(trace.times)}{within}")
    if tracked is None:
        left_out = trace.slow and tracking_speed(
            trace.times, trace.counts, TRACK_START_S, hold_s
        ) is not None
        why = ": those whose read took longer than a cycle are left out" if left_out else ""
        print(
            f"    FAIL -- too few samples in [{TRACK_START_S:g}, {hold_s:g}] s to measure "
            f"tracking{why}{fault}"
        )
        return False
    # A sample left out at a window end moves the window the tracking used: print that one.
    full = [t for t in trace.times if TRACK_START_S <= t <= hold_s]
    used = [t for t in times if TRACK_START_S <= t <= hold_s]
    window = (
        f"[{used[0]:.2f}, {used[-1]:.2f}]" if (full[0], full[-1]) != (used[0], used[-1])
        else f"[{TRACK_START_S:g}, {hold_s:g}]"
    )
    error = tracked / speed - 1.0
    print(
        f"    Tracking:     {tracked:.0f} ticks/s over {window} s "
        f"(target={speed}, error={error*100:+.1f}%)"
    )

    fit = fit_dead_time_and_tau(times, counts)
    if fit is None:
        print(
            f"    Fit:          none (too few samples before {FIT_END_S} s, "
            "or the count did not leave rest)"
        )
    else:
        dead_at = (
            " (the fit's floor)" if fit.dead_time_s <= 0.0
            else " (the fit's bound)" if fit.dead_time_s >= DEAD_TIME_MAX_S else ""
        )
        tau_at = " (the fit's bound)" if not TAU_MIN_S < fit.tau_s < TAU_MAX_S else ""
        print(
            f"    Dead time:    {fit.dead_time_s*1000:.1f} ms{dead_at}   "
            f"tau: {fit.tau_s*1000:.1f} ms{tau_at}   "
            f"63% at {(fit.dead_time_s + fit.tau_s)*1000:.1f} ms"
        )
        print(
            f"                  (fit to the count before {FIT_END_S} s: "
            f"level {fit.speed:.0f} ticks/s, rms {fit.rms_ticks:.1f} ticks)"
        )
        if fit.dead_time_s <= 0.0:
            print(
                "    WARNING: The dead time sits on its 0 floor: its split from tau is not "
                "measured, and the 63% time may be short. Do not record these."
            )
        if tau_at or fit.dead_time_s >= DEAD_TIME_MAX_S:
            print(
                f"    WARNING: The fit sits at a bound: the rise before {FIT_END_S} s is not "
                "measured. Do not record these."
            )
        if trace.zero_unknown:
            print(
                "    WARNING: The first command 37 went unacknowledged, so when the step started "
                "is not known. Do not record these."
            )

    if abs(tracked) < abs(speed) * STALL_FRACTION:
        print(f"    WARNING: Mean speed is <{STALL_FRACTION*100:g}% of target.")
        print(f"             Motor may be stalled, disconnected, or wheels loaded.")
        print(f"             Lift the robot so wheels spin freely for tuning.")
    elif abs(tracked) < abs(speed) * LOW_SPEED_FRACTION:
        print(f"    WARNING: Mean speed is <{LOW_SPEED_FRACTION*100:g}% of target.")
        print(f"             Check motor wiring and ensure wheels are unloaded.")

    if trace.fault:
        print(f"    FAIL -- {trace.fault}")
        passed = False
    elif not judged:
        print(
            "    NO VERDICT -- the box judges only the default step, "
            f"{STEP_VELOCITY_TICKS} ticks/s for {STEP_HOLD_S:g} s"
        )
        passed = True
    elif abs(error) <= TRACK_TOLERANCE:
        print(f"    PASS -- tracks within {TRACK_TOLERANCE*100:g}% of target")
        passed = True
    else:
        side = "under" if error < 0 else "over"
        print(
            f"    FAIL -- tracks {abs(error)*100:.1f}% {side} target "
            f"(box: ±{TRACK_TOLERANCE*100:g}%)"
        )
        passed = False
    _print_counts(trace, verbose)
    return passed


def _print_counts(trace: StepTrace, verbose: bool) -> None:
    """With -v, the step's raw counts."""
    if verbose:
        print(f"\n    {'Time(ms)':>10}  {'Count':>10}")
        for t, c in zip(trace.times, trace.counts):
            print(f"    {t*1000:10.1f}  {c:10d}")


def _step_controller(
    rc: RoboClawInterface, motors: list[str], speed: int, args: argparse.Namespace
) -> tuple[bool, bool]:
    """Step one controller's selected motors and report each, as *args* asks.

    Returns (no motor failed, the step was aborted or failed its check after the stop).
    """
    print(f"  Testing {', '.join(motors)} (command 37, {args.hold:g} s hold)...")
    traces = run_step_response(rc, motors, speed, args.hold, args.settle)
    passed = True
    for trace in traces:
        print(f"  {trace.motor}:")
        passed &= _report_step_result(trace, args.hold, args.verbose, _judged(args))
    time.sleep(args.settle)
    return passed, any(trace.abort or trace.fault for trace in traces)


def _judged(args: argparse.Namespace) -> bool:
    """Whether the step is the default one, the only step the tracking box judges."""
    return args.speed == STEP_VELOCITY_TICKS and args.hold == STEP_HOLD_S


def _summary(passed: bool, args: argparse.Namespace) -> str:
    """The run's tracking verdict, and the box that gives it, as the summary prints them.

    Off the default step the box gives none, so a run that fails there failed above: an
    abort, a failed measurement or check, or an error.
    """
    box = (
        f"±{TRACK_TOLERANCE * 100:g}% of the target, from {TRACK_START_S:g} s after the first "
        "command to the last"
    )
    if _judged(args):
        return f"{'PASS' if passed else 'FAIL'} -- the box is {box}"
    default = f"the default step, {STEP_VELOCITY_TICKS} ticks/s for {STEP_HOLD_S:g} s"
    if not passed:
        return f"FAIL -- the run failed above (the box judges only {default})"
    return (
        f"NO VERDICT -- the box is {box}, and it judges only {default}: this step, "
        f"{args.speed} ticks/s for {args.hold:g} s, has no verdict"
    )


def _selected_controllers(which: str) -> list[tuple[str, str, int]]:
    """The controllers --controller names, or none unless every one was detected."""
    controllers = get_controllers(which)
    if len(controllers) < (2 if which == "both" else 1):
        print("ERROR: not every selected controller was detected; nothing is stepped.")
        return []
    return controllers


STOP_AFTER_ABORT = "\n  Not stepping the remaining controllers after the abort."


def cmd_step_response(args: argparse.Namespace) -> bool:
    """Run the step-response test; report tracking, dead time and tau. True if all passed."""
    motors = get_motor_list(args.motor)
    speed = args.speed
    controllers = _selected_controllers(args.controller)
    passed = bool(controllers)

    for label, port, addr in controllers:
        print(f"\n{'='*50}")
        print(f"  {label} RoboClaw (0x{addr:02X}) -- Step Response")
        print(f"{'='*50}")

        try:
            rc = open_controller(port, addr)

            # Show current PID
            pid_m1 = rc.read_velocity_pid_m1()
            pid_m2 = rc.read_velocity_pid_m2()
            print(
                f"  Current M1 PID: P={pid_m1['P']} I={pid_m1['I']} D={pid_m1['D']} QPPS={pid_m1['QPPS']}"
            )
            print(
                f"  Current M2 PID: P={pid_m2['P']} I={pid_m2['I']} D={pid_m2['D']} QPPS={pid_m2['QPPS']}"
            )
            read = {"M1": pid_m1["QPPS"], "M2": pid_m2["QPPS"]}
            qpps = sorted({read[m] for m in motors})
            share = ", ".join(f"{speed / q * 100:.0f}% of QPPS {q}" for q in qpps if q)
            print(f"  Step target: {speed} ticks/s ({share or 'QPPS 0'})")
            print()

            ok, aborted = _step_controller(rc, motors, speed, args)
            passed &= ok
            close_controller(rc)
            if aborted:
                print(STOP_AFTER_ABORT)
                break
        except Exception as e:
            print(f"  ERROR: {e}")
            passed = False
            print(STOP_AFTER_ABORT)
            break
    print()
    print(f"  Tracking: {_summary(passed, args)}.")
    print("  The dead time and tau above are measurements, not targets: nothing here")
    print("  passes or fails on them.")
    return passed


def cmd_tune(args: argparse.Namespace) -> bool:
    """Set PID values and immediately run the step-response test. True if all passed."""
    p = args.p if args.p is not None else DEFAULT_P
    i_val = args.i if args.i is not None else DEFAULT_I
    d = args.d if args.d is not None else DEFAULT_D
    qpps = args.qpps if args.qpps is not None else ROBOCLAW_QPPS
    motors = get_motor_list(args.motor)
    speed = args.speed
    controllers = _selected_controllers(args.controller)
    passed = bool(controllers)

    for label, port, addr in controllers:
        print(f"\n{'='*60}")
        print(f"  {label} RoboClaw (0x{addr:02X}) -- Tune + Step Response")
        print(f"{'='*60}")

        try:
            rc = open_controller(port, addr)

            # Set PID
            written = True
            for motor in motors:
                if motor == "M1":
                    ok = rc.set_velocity_pid_m1(p, i_val, d, qpps)
                else:
                    ok = rc.set_velocity_pid_m2(p, i_val, d, qpps)
                written &= bool(ok)
                status = "OK" if ok else "FAILED"
                print(f"  {motor} SET P={p} I={i_val} D={d} QPPS={qpps} -> {status}")

            # Verify
            for motor in motors:
                if motor == "M1":
                    pid = rc.read_velocity_pid_m1()
                else:
                    pid = rc.read_velocity_pid_m2()
                written &= (pid["P"], pid["I"], pid["D"], pid["QPPS"]) == (p, i_val, d, qpps)
                print(
                    f"  {motor} readback: P={pid['P']} I={pid['I']} D={pid['D']} QPPS={pid['QPPS']}"
                )
            if not written:
                print("  ERROR: the gains were not confirmed; not stepping on other gains.")
                close_controller(rc)
                passed = False
                print(STOP_AFTER_ABORT)
                break

            print(
                f"\n  Step target: {speed} ticks/s ({speed / qpps * 100:.0f}% of QPPS)"
            )
            print()

            ok, aborted = _step_controller(rc, motors, speed, args)
            passed &= ok
            close_controller(rc)
            if aborted:
                print(STOP_AFTER_ABORT)
                break
        except Exception as e:
            print(f"  ERROR: {e}")
            passed = False
            print(STOP_AFTER_ABORT)
            break

    print(f"\n{'='*60}")
    print("  Tuning Summary")
    print(f"{'='*60}")
    print(f"  Gains:       P={p} I={i_val} D={d} QPPS={qpps}")
    print(f"  Tracking:    {_summary(passed, args)}")
    print()
    print("  The dead time and tau above are measurements: record them with the gains.")
    print("  They are not targets, and nothing here passes or fails on them.")
    print()
    print("  Iterate by running:  python3 tune_pid.py --tune --p <new_P> --i <new_I>")
    print("  PID values stored in RAM only — set automatically on startup")
    print("  by the driver node and test scripts.")
    return passed


def cmd_stop(args: argparse.Namespace) -> bool:
    """Emergency stop all motors. Every detected controller is opened and stopped before any is
    checked, so no check holds back another's stop, and an error on one keeps no other from its
    stop; a controller that answers nothing holds back the next one's stop by its retries alone.
    A Ctrl-C, Ctrl-Z, SIGTERM or SIGHUP from before detection is held by hard_stop's rule: the
    first is raised once every detected controller has been sent its stop. A second, in
    detection, in a port's open or in the same stop, raises at once: in detection it ends the
    run, and otherwise it abandons that controller's open or stop alone (a port that hangs, say)
    while the others' stops go on; in a later stop it is held to that stop's end. The check
    after a step then runs on all of them at once. True if every selected controller was
    detected and opened, each acknowledged its stop, no motor's count moves more than REST_TICKS
    over STILL_CHECK_S, from STILL_CHECK_S after the last stop, and its reported speeds were
    read: a failed read is an error, and the speeds read judge nothing."""
    opened, status, held, handlers = [], {}, [], {}
    try:
        _hold_signals(held, handlers)
        controllers = get_controllers(args.controller)
        stopped = len(controllers) == (2 if args.controller == "both" else 1)
        for label, port, addr in controllers:
            try:
                rc = open_controller(port, addr)
                # hard_stop is called by open_controller, but call again explicitly
                opened.append((label, rc, hard_stop(rc)))
            except KeyboardInterrupt as e:
                held.append(str(e))  # held by hard_stop to its stops' end, or a second one
            except Exception as e:
                status[label], stopped = f"ERROR: {e}", False
    finally:
        _restore_signals(handlers)
    if held:
        raise KeyboardInterrupt(held[0])
    checks = _moving_after_stop([rc for _, rc, _ in opened], ["M1", "M2"])
    for (label, rc, acked), moving in zip(opened, checks):
        try:
            if isinstance(moving, Exception):
                raise moving
            # Commands 18 and 19 read a filtered speed that lags the wheel by about
            # 0.1 s, so a stopped wheel reads as turning for longer than it turns.
            speeds = {"M1": rc.read_speed_m1()[0], "M2": rc.read_speed_m2()[0]}
            why = [] if acked else ["unacknowledged"]
            if moving:
                why.append(f"{', '.join(moving)} turning {STILL_CHECK_S:g} s after the stop")
            verdict = f"FAILED, {'; '.join(why)}" if why else "OK"
            status[label] = (
                f"{verdict} (reported speeds M1={speeds['M1']}, M2={speeds['M2']} ticks/s)"
            )
            stopped &= not why
        except Exception as e:
            status[label], stopped = f"ERROR: {e}", False
        finally:
            close_controller(rc)
    for label, _, addr in controllers:
        print(f"  Stopping {label} (0x{addr:02X})... {status[label]}")
    return stopped


def _travel_m(speed_ticks: int, hold_s: float) -> float:
    """A wheel's travel over a step, in metres."""
    return abs(speed_ticks) * hold_s / RADIANS_TO_ENCODER_TICKS * WHEEL_RADIUS


def _confirm_lifted(args: argparse.Namespace) -> bool:
    """The operator's typed confirmation, before the first step, that every wheel is off the
    ground; refused without a terminal unless --no-confirm."""
    text = (
        f"The step turns each selected wheel at {args.speed} ticks/s for {args.hold:g} s, about "
        f"{_travel_m(args.speed, args.hold):.2f} m of travel: every wheel must be off the ground."
    )
    if args.no_confirm:
        print(f"(--no-confirm) {text}")
        return True
    if not sys.stdin.isatty():
        print(f"ERROR: {text} That needs a typed confirmation at a terminal (or --no-confirm).")
        return False
    print(text)
    termios.tcflush(sys.stdin, termios.TCIFLUSH)  # only an answer typed after the prompt counts
    try:
        answer = input("Type yes to step: ").strip()
    except EOFError:
        answer = ""
    if answer != "yes":
        print("Not confirmed: nothing was stepped.")
        return False
    return True


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    lo, hi = SERIAL_TIMEOUT_RANGE_S
    tolerance = TRACK_TOLERANCE * 100
    still = STILL_CHECK_S
    travel = _travel_m(STEP_VELOCITY_TICKS, STEP_HOLD_S)
    longest = _travel_m(MAX_STEP_TICKS, MAX_HOLD_S)
    parser = argparse.ArgumentParser(
        description="RoboClaw velocity PID tuning for Strafer chassis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python3 tune_pid.py --read                          # Show current PID
  python3 tune_pid.py --tune                          # Set defaults + test
  python3 tune_pid.py --tune --p 20000 --i 500        # Custom PID + test
  python3 tune_pid.py --set --p 15000 --i 750 --d 0   # Set PID only (RAM)
  python3 tune_pid.py --step-response                  # Test only
  python3 tune_pid.py --step-response -v               # Test with raw data
  python3 tune_pid.py --tune --controller front        # Front only
  python3 tune_pid.py --tune --motor m1                # M1 only (all controllers)

The step drives the wheels: lift the chassis so every wheel is off the ground.
At the defaults each stepped wheel turns about {travel:.1f} m, and up to {longest:.1f} m
at the largest --speed and --hold. --speed is {MIN_STEP_TICKS} to {MAX_STEP_TICKS} ticks/s either
way, the steps the aborts guard. A typed confirmation comes first (--no-confirm
skips it), and each controller's serial timeout must be armed ({lo:g} to {hi:g} s).
By default it is the step the driver's gains were chosen on, {STEP_VELOCITY_TICKS} ticks/s
({STEP_SPEED_MPS:g} m/s) for {STEP_HOLD_S:g} s, and the box they passed: each wheel's mean
speed from {TRACK_START_S:g} s after the first command to the last within ±{tolerance:g} %
of its target. The box judges only that step: at another --speed or --hold a run
prints its tracking with no verdict.
Exit status of --step-response and --tune: 0 if every motor passed; 1 if a motor
fails tracking, a step is aborted or errs, a selected controller is missing, the
step is not confirmed or (--tune) the gains are not confirmed; {EXIT_NO_VERDICT} if the step
was not the default one and nothing failed. Of --stop: 1 if a selected controller
is missing, a stop is not acknowledged or errs, a read fails, or a wheel's count
moves more than {REST_TICKS} ticks over {still:g} s from {still:g} s after the stop; the reported
speeds lag, and judge nothing. 2 on a usage error; {EXIT_INTERRUPTED} if interrupted (Ctrl-C,
Ctrl-Z, SIGTERM, SIGHUP); else 0.
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--read", action="store_true", help="Read current PID values")
    mode.add_argument("--tune", action="store_true", help="Set PID + run step-response")
    mode.add_argument("--set", action="store_true", help="Set PID values only")
    mode.add_argument(
        "--step-response", action="store_true", help="Run step-response only"
    )
    mode.add_argument("--stop", action="store_true", help="Emergency stop all motors")

    parser.add_argument(
        "--controller",
        choices=["front", "rear", "both"],
        default="both",
        help="Which controller(s) to operate on (default: both)",
    )
    parser.add_argument(
        "--motor",
        choices=["m1", "m2", "both"],
        default="both",
        help=(
            "Which motor(s) to operate on (default: both); a step holds the other at 0 "
            "with its speed loop"
        ),
    )

    parser.add_argument(
        "--p", type=int, default=None,
        help=f"P gain (default for --tune: {DEFAULT_P}, the driver's, from strafer_shared)",
    )
    parser.add_argument(
        "--i", type=int, default=None,
        help=f"I gain (default for --tune: {DEFAULT_I}, the driver's, from strafer_shared)",
    )
    parser.add_argument(
        "--d", type=int, default=None,
        help=f"D gain (default for --tune: {DEFAULT_D}, the driver's, from strafer_shared)",
    )
    parser.add_argument(
        "--qpps", type=int, default=None,
        help=f"QPPS (default: {ROBOCLAW_QPPS}, the motors' mean full speed, from strafer_shared)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=STEP_VELOCITY_TICKS,
        help=(
            f"Step target in ticks/s (default: {STEP_VELOCITY_TICKS}, {STEP_SPEED_MPS:g} m/s at "
            "the wheel through strafer_shared's WHEEL_RADIUS and RADIANS_TO_ENCODER_TICKS, the "
            f"speed the driver's gains were chosen at); from {MIN_STEP_TICKS}, so the against "
            f"abort sees a reversed motor in every full-length cycle, to {MAX_STEP_TICKS} either "
            "way, so the "
            f"runaway abort trips {(1 - RUNAWAY_HEADROOM) * 100:g} %% under full speed. Another "
            "speed gets no verdict"
        ),
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=STEP_HOLD_S,
        help=(
            f"Step hold in seconds (default: {STEP_HOLD_S:g}, the hold the driver's gains were "
            f"chosen on); from {MIN_HOLD_S:g}, so the tracking window spans "
            f"{MIN_TRACK_WINDOW_S:g} s and the stall abort runs, to {MAX_HOLD_S:g}. Another hold "
            "gets no verdict"
        ),
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=SETTLE_TIME_S,
        help=(
            "Seconds to wait after a stop, before the counts' rest check and between "
            f"controllers (default: {SETTLE_TIME_S:g}: a lifted wheel stopped at duty 0 from "
            f"about 600 ticks/s is at rest within it); from 0 to {MAX_HOLD_S:g}. The check that "
            f"the wheels stopped after a step waits a fixed {STILL_CHECK_S:g} s"
        ),
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help=(
            "Step without the typed confirmation that every wheel is off the ground, for a "
            "run with no terminal"
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print raw step-response counts"
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Validate --set requires at least one PID value
    if (
        args.set
        and args.p is None
        and args.i is None
        and args.d is None
        and args.qpps is None
    ):
        parser.error("--set requires at least one of --p, --i, --d, --qpps")
    if (args.tune or args.set) and args.qpps is not None and args.qpps <= 0:
        parser.error("--qpps must be positive: it is the motors' full speed in ticks/s")
    if args.tune or args.step_response:
        # Each bound is a chained comparison, so NaN fails it too.
        if not MIN_HOLD_S <= args.hold <= MAX_HOLD_S:
            parser.error(
                f"--hold must be from {MIN_HOLD_S:g} s, so the tracking window spans "
                f"{MIN_TRACK_WINDOW_S:g} s and the stall abort runs, to {MAX_HOLD_S:g} s"
            )
        if not 0 <= args.settle <= MAX_HOLD_S:
            parser.error(f"--settle must be from 0 to {MAX_HOLD_S:g} s")
        if abs(args.speed) < MIN_STEP_TICKS:
            parser.error(
                f"--speed must be at least {MIN_STEP_TICKS} ticks/s either way: below it the "
                "against abort cannot see a reversed motor every cycle"
            )
        if abs(args.speed) > MAX_STEP_TICKS:
            parser.error(
                f"--speed must be at most {MAX_STEP_TICKS} ticks/s either way, so the runaway "
                f"abort trips {(1 - RUNAWAY_HEADROOM) * 100:g} % under full speed"
            )

    # A terminal hang-up, a kill or a Ctrl-Z takes the same path as Ctrl-C, so
    # the motors are stopped on the way out.
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGTSTP):
        signal.signal(sig, _interrupt)

    try:
        if args.stop:
            return 0 if cmd_stop(args) else 1
        elif args.read:
            cmd_read(args)
        elif args.set:
            # For --set, default unspecified values to 0 rather than the tuning defaults
            if args.p is None:
                parser.error("--set requires --p")
            if args.i is None:
                args.i = 0
            if args.d is None:
                args.d = 0
            cmd_set(args)
        elif args.step_response or args.tune:
            if not _confirm_lifted(args):
                return 1
            if not (cmd_tune(args) if args.tune else cmd_step_response(args)):
                return 1
            return 0 if _judged(args) else EXIT_NO_VERDICT
    except KeyboardInterrupt:
        print("\nInterrupted: stopping the motors.")
        _atexit_stop_all()
        return EXIT_INTERRUPTED
    return 0


def _interrupt(signum: int, _frame) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


if __name__ == "__main__":
    raise SystemExit(main())
