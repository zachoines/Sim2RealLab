"""Tests for tune_pid.py: the step it drives, the fit to the count, and the tracking verdict.

The tuner drives each controller with command 37, as the driver does, and reads the encoder
counts (commands 16 and 17). On the default step a motor passes when its mean speed from 0.6 s
after the first command to the end of the hold is within ±5 % of the target; off it, the tuner
reports the tracking with no verdict. It reports a dead time and a time constant fitted to the
count before 0.6 s, and aims at neither. The fit runs on
synthetic steps and on two bench holds of the real wheels (data/step_holds/SOURCE.md). The
step runs on test_port_fence's fake bus with a simulated motor behind each channel.
"""

import csv
import functools
import importlib.util
import inspect
import math
import os
import re
import signal
import statistics
import threading
import struct
import sys
from unittest.mock import MagicMock, patch

import pytest

from strafer_driver.roboclaw_interface import _crc16
from strafer_shared.constants import (
    ROBOCLAW_PID_D,
    ROBOCLAW_PID_I,
    ROBOCLAW_PID_P,
    ROBOCLAW_QPPS,
    ROBOCLAW_SERIAL_TIMEOUT_S,
)

# test/ has no __init__.py, so pytest puts it on sys.path; the imported autouse fixture
# applies here.
from test_port_fence import (  # noqa: F401
    CAMERA_PORT,
    FRONT,
    REAR,
    FakeBus,
    FakeCamera,
    FakeRoboClaw,
    patched,
)

TUNER = os.path.join(os.path.dirname(__file__), "..", "..", "tune_pid.py")
TRACES = os.path.join(os.path.dirname(__file__), "data", "step_holds")
CHANNELS = ("F1", "F2", "R1", "R2")
HOLD_S = 6.0  # the bench's hold for these runs

# The tracking box, the one the driver's gains passed, and the fit's end.
BOX_TOLERANCE = 0.05
BOX_START_S = 0.6

# The manual's numbers, written out as in test_port_fence.
DRIVE_M1M2_SPEED = 37
READ_BATTERY = 24
DRIVE_ONE_SPEED = {35, 36}
READ_COUNTS = {16, 17}
READ_SPEEDS = {18, 19}
DUTY_FORWARD = {0, 4}
SET_PID = {28, 29}
MOTOR_COMMANDS = {0, 1, 4, 5, 35, 36, DRIVE_M1M2_SPEED}


def load_tuner():
    spec = importlib.util.spec_from_file_location("tune_pid", TUNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tuner = load_tuner()


def test_the_box_and_the_default_step():
    """±5 % from 0.6 s, the fit before it, and 0.3 m/s at the wheel: the bench's 535 ticks/s."""
    assert (tuner.TRACK_TOLERANCE, tuner.TRACK_START_S) == (BOX_TOLERANCE, BOX_START_S)
    assert tuner.FIT_END_S == BOX_START_S
    assert tuner.STEP_VELOCITY_TICKS == 535


def test_the_largest_step_leaves_the_runaway_rule_room():
    """At the largest step the runaway limit, RUNAWAY_FACTOR times the step plus the margin, stays
    10 % under the motors' mean full speed; at a step a tick per second larger it would not."""
    limit = tuner.RUNAWAY_FACTOR * tuner.MAX_STEP_TICKS + tuner.RUNAWAY_MARGIN_TPS
    assert limit <= 0.9 * ROBOCLAW_QPPS < limit + tuner.RUNAWAY_FACTOR


# ---------------------------------------------------------------------------
# The fit and the tracking measure
# ---------------------------------------------------------------------------


def synthetic_step(dead, tau, level, period=0.02, n=300, shift=0.0, rest=0.5):
    """A first-order lag behind a dead time, sampled as the tuner samples it.

    Each read lands 1-3 ms into its cycle, plus *shift*; counts are whole ticks, the encoder
    at rest *rest* of a tick past its last edge.
    """
    times, counts = [], []
    for k in range(n):
        t = k * period + 0.001 + 0.0005 * ((7 * k) % 5) + shift
        x = max(0.0, t - dead)
        times.append(t)
        counts.append(math.floor(level * (x - tau * (1.0 - math.exp(-x / tau))) + rest))
    return times, counts


STEPS = [(0.012, 0.022, 510.0), (0.005, 0.035, 145.0), (0.05, 0.08, 1200.0)]


@pytest.mark.parametrize("dead, tau, level", STEPS)
def test_fit_is_exact_on_an_exact_count(dead, tau, level):
    times, _ = synthetic_step(dead, tau, level)
    xs = [max(0.0, t - dead) for t in times]
    exact = [level * (x - tau * (1.0 - math.exp(-x / tau))) for x in xs]

    fit = tuner.fit_dead_time_and_tau(times, exact)

    assert fit.dead_time_s == pytest.approx(dead, abs=1e-4)
    assert fit.tau_s == pytest.approx(tau, rel=1e-3)
    assert fit.speed == pytest.approx(level, rel=1e-4)


@pytest.mark.parametrize(
    "dead, tau, level, sum_sd, dead_bias",
    [
        (0.012, 0.022, 510.0, 0.0009, 0.0011),
        (0.005, 0.035, 145.0, 0.0030, 0.0032),
        (0.05, 0.08, 1200.0, 0.0005, 0.0005),
    ],
)
def test_fit_in_whole_ticks_scatters_as_documented(dead, tau, level, sum_sd, dead_bias):
    """Over 24 read phases and rest positions: unbiased, the sum tighter than the split.

    The scatter grows as the level falls, as the fit's docstring states.
    """
    fits = []
    for k in range(24):
        shift, rest = 0.02 * k / 24, (7 * k % 24) / 24
        times, counts = synthetic_step(dead, tau, level, shift=shift, rest=rest)
        fits.append(tuner.fit_dead_time_and_tau(times, counts))
    sums = [f.dead_time_s + f.tau_s - dead - tau for f in fits]
    deads = [f.dead_time_s - dead for f in fits]

    assert abs(statistics.mean(sums)) < sum_sd / 2
    assert statistics.pstdev(sums) < sum_sd
    assert abs(statistics.mean(deads)) < dead_bias
    assert statistics.pstdev(deads) > statistics.pstdev(sums)
    assert all(f.speed == pytest.approx(level, rel=0.01) for f in fits)


def test_fit_ignores_the_count_after_the_rise():
    """Samples from 0.6 s on, the integral term's slow approach, do not move the fit."""
    times, counts = synthetic_step(0.012, 0.022, 510.0)
    crept = [c + (round(40 * (t - BOX_START_S) ** 2) if t >= BOX_START_S else 0)
             for t, c in zip(times, counts)]

    assert tuner.fit_dead_time_and_tau(times, crept) == tuner.fit_dead_time_and_tau(times, counts)


def test_fit_needs_motion():
    times, _ = synthetic_step(0.0, 0.02, 0.0)

    assert tuner.fit_dead_time_and_tau(times, [0] * len(times)) is None
    assert tuner.fit_dead_time_and_tau(times[:5], list(range(5))) is None


@pytest.mark.parametrize("past, fitted", [(0, False), (1, True)])
def test_a_count_a_tick_past_rest_is_fitted(past, fitted):
    """A count that moves REST_TICKS has not left rest, and gives no fit; a tick more has."""
    times, _ = synthetic_step(0.0, 0.02, 0.0)
    counts = [tuner.REST_TICKS + past if t >= BOX_START_S / 2 else 0 for t in times]

    assert (tuner.fit_dead_time_and_tau(times, counts) is not None) == fitted


def test_a_fit_needs_the_first_0_2_s():
    """MIN_FIT_SAMPLES is the samples of the first 0.2 s (the comment's): one fewer gives no fit."""
    times, counts = synthetic_step(*STEPS[0])
    n = round(0.2 / tuner.SAMPLE_PERIOD_S)

    assert tuner.fit_dead_time_and_tau(times[:n], counts[:n]) is not None
    assert tuner.fit_dead_time_and_tau(times[:n - 1], counts[:n - 1]) is None


def test_tracking_speed_is_the_count_over_the_window():
    times = [0.02 * k for k in range(301)]
    counts = [round(500 * t) if t >= 0.5 else 0 for t in times]

    assert tuner.tracking_speed(times, counts, 0.6, 6.0) == pytest.approx(500.0, rel=1e-3)
    assert tuner.tracking_speed(times, counts, 6.5, 7.0) is None


def test_count_delta_folds_the_wrap():
    assert tuner.count_delta(2**32 - 5, 3) == 8
    assert tuner.count_delta(3, 2**32 - 5) == -8
    assert tuner.count_delta(100, 40) == -60


def report(trace, hold_s, capsys):
    """(passed, printed lines) of the tuner's report on *trace*."""
    passed = tuner._report_step_result(trace, hold_s, False)
    return passed, capsys.readouterr().out


def trace_of(times, counts, target):
    trace = tuner.StepTrace(motor="M1", target=target)
    trace.times, trace.counts = list(times), list(counts)
    return trace


@pytest.mark.parametrize("dead, tau, flagged", [
    (0.4, 0.02, "Dead time:    300.0 ms (the fit's bound)"),
    (0.0, 0.002, "tau: 5.0 ms (the fit's bound)"),
])
def test_a_fit_at_its_bound_is_flagged_not_recorded(capsys, dead, tau, flagged):
    """A wheel that starts after the dead time's bound, or a lag shorter than tau's, gives no
    measurement to record."""
    _, out = report(trace_of(*synthetic_step(dead, tau, 535.0), 535), HOLD_S, capsys)

    assert flagged in out
    assert "The fit sits at a bound" in out


def test_a_lag_past_five_windows_is_flagged_not_recorded(capsys):
    """Past tau's bound, five windows (the comment's 3 s), a rise is a ramp inside the window: a
    lag of six, even on an exact count, fits at the bound and gives no measurement."""
    dead, tau, level = 0.012, 6 * BOX_START_S, tuner.STEP_VELOCITY_TICKS
    times, _ = synthetic_step(dead, tau, level)
    xs = [max(0.0, t - dead) for t in times]
    exact = [level * (x - tau * (1.0 - math.exp(-x / tau))) for x in xs]

    _, out = report(trace_of(times, exact, level), HOLD_S, capsys)

    assert "(the fit's bound)" in out and "The fit sits at a bound" in out


@pytest.mark.parametrize("times, counts", [([], []), ([0.0, tuner.SAMPLE_PERIOD_S], [0, 1])])
def test_a_fault_is_printed_with_a_step_short_of_samples(capsys, times, counts):
    """With no data, or too few samples to measure the tracking, what the check after the stop
    found is printed too."""
    trace = trace_of(times, counts, tuner.STEP_VELOCITY_TICKS)
    trace.fault = "the stop could not be checked"

    passed, out = report(trace, tuner.STEP_HOLD_S, capsys)

    assert not passed and trace.fault in out


# ---------------------------------------------------------------------------
# Two bench holds of the real wheels (data/step_holds/)
# ---------------------------------------------------------------------------

# scipy.optimize.least_squares on the same samples, continuous in all three parameters and
# started from 16 points: (dead time ms, tau ms) of the count before 0.6 s, and the tracking
# error (%) over [0.6, 6.0] s.
REFERENCE = {
    ("all535_r1", "F1"): (12.83, 22.34, -3.71),
    ("all535_r1", "F2"): (13.70, 20.96, -1.87),
    ("all535_r1", "R1"): (15.99, 18.57, -2.48),
    ("all535_r1", "R2"): (11.51, 28.12, -2.31),
    ("all178_r1", "F1"): (5.98, 32.58, -13.43),
    ("all178_r1", "F2"): (0.00, 30.82, -9.25),
    ("all178_r1", "R1"): (3.59, 39.49, -11.67),
    ("all178_r1", "R2"): (4.51, 37.80, -12.82),
}
# The bench's own summary of the 535 hold: each motor's mean reported speed over
# [0.6, 6.0] s against its target, %.
BENCH_MEAN_535 = {"F1": -3.8, "F2": -1.9, "R1": -2.5, "R2": -2.3}


@functools.lru_cache(maxsize=None)
def bench_trace(run, channel):
    """(times, counts) of one channel's drive rows, as the bench timed them.

    Times are tr_F or tr_R, when that controller's count reads returned, in s from the
    first command; counts are from the last count before it.
    """
    with open(os.path.join(TRACES, run + ".csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    key = "tr_" + channel[0]
    zero = [int(r["c_" + channel]) for r in rows if r["phase"] == "pre"][-1]
    drive = [r for r in rows if r["phase"] == "drive" and r[key] and r["c_" + channel]]
    times = tuple(float(r[key]) for r in drive)
    counts = tuple(tuner.count_delta(zero, int(r["c_" + channel])) for r in drive)
    return times, counts


def target(run):
    return int(re.search(r"all(\d+)_", run).group(1))


@pytest.mark.parametrize("run, channel", sorted(REFERENCE))
def test_fit_on_the_bench_holds(run, channel):
    dead_ms, tau_ms, _ = REFERENCE[(run, channel)]

    fit = tuner.fit_dead_time_and_tau(*bench_trace(run, channel))

    assert fit.dead_time_s * 1000 == pytest.approx(dead_ms, abs=0.1)
    assert fit.tau_s * 1000 == pytest.approx(tau_ms, abs=0.1)
    assert fit.rms_ticks < 0.5, "the model should follow the rise to within a tick"


@pytest.mark.parametrize("run, channel", sorted(REFERENCE))
def test_tracking_on_the_bench_holds(run, channel):
    """535 ticks/s (0.3 m/s) tracks inside the ±5 % box on every motor; 178 ticks/s does not."""
    _, _, error_pct = REFERENCE[(run, channel)]

    speed = tuner.tracking_speed(*bench_trace(run, channel), BOX_START_S, HOLD_S)
    error = speed / target(run) - 1.0

    assert error * 100 == pytest.approx(error_pct, abs=0.01)
    assert (abs(error) <= BOX_TOLERANCE) == (target(run) == 535)
    if target(run) == 535:
        # At the box's speed the count agrees with the bench's mean reported speed.
        assert error * 100 == pytest.approx(BENCH_MEAN_535[channel], abs=0.2)


@pytest.mark.parametrize("run, channel", sorted(REFERENCE))
def test_report_on_the_bench_holds(run, channel, capsys):
    """The printed verdict, dead time and tau are the window's and the rise's."""
    dead_ms, tau_ms, error_pct = REFERENCE[(run, channel)]

    passed, out = report(trace_of(*bench_trace(run, channel), target(run)), HOLD_S, capsys)

    error = float(re.search(r"error=([+-][\d.]+)%", out).group(1))
    assert error == pytest.approx(error_pct, abs=0.06)
    dead, tau = measured([out])
    assert dead * 1000 == pytest.approx(dead_ms, abs=0.06)
    assert tau * 1000 == pytest.approx(tau_ms, abs=0.06)
    assert passed == (target(run) == 535)


# ---------------------------------------------------------------------------
# The step, on the fake bus
# ---------------------------------------------------------------------------


class Clock:
    """time.monotonic and time.sleep for the tuner: sleeping advances the time.

    A run past ten simulated minutes fails the test, however the tool handles exceptions, so a
    step that never ends cannot hang the suite.
    """

    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)
        if self.now > 100.0 + 600.0:
            pytest.fail("the tuner ran for ten simulated minutes")


class Motor:
    """A wheel behind its speed target: a first-order lag behind a dead time.

    *gain* scales the speed it settles at; 0 is a dead encoder or a jammed wheel, and a
    negative gain turns it against its target.
    """

    def __init__(self, dead=0.012, tau=0.022, gain=1.0):
        self.dead, self.tau, self.gain = dead, tau, gain
        self.targets = [(-math.inf, 0)]  # (time, ticks/s)
        self.driven = False  # set once command 37 has reached it
        self.t = None
        self.speed = 0.0
        self.count = 0.0

    def command(self, now, ticks_per_s):
        self.targets.append((now, ticks_per_s))

    def advance(self, now, step=0.001):
        if self.t is None:
            self.t = now
        while self.t < now:
            dt = min(step, now - self.t)
            when = self.t - self.dead
            goal = self.gain * next(v for s, v in reversed(self.targets) if s <= when)
            decay = math.exp(-dt / self.tau)
            self.count += goal * dt + (self.speed - goal) * self.tau * (1.0 - decay)
            self.speed = goal + (self.speed - goal) * decay
            self.t += dt


class Drifting(Motor):
    """A wheel turning at *drift* ticks/s whatever its target: from the start, or once
    command 37 has reached its controller."""

    def __init__(self, drift, from_command=False, **kw):
        super().__init__(**kw)
        self.drift = drift
        self.driven = not from_command

    def advance(self, now, step=0.001):
        if self.driven and self.t is not None and now > self.t:
            self.count += self.drift * (now - self.t)
        super().advance(now, step)


class SteppedRoboClaw(FakeRoboClaw):
    """test_port_fence's fake with a simulated motor on each channel.

    Command 37 and the duty commands set the motors' targets (the tuner sends only duty 0);
    commands 16 and 17 read their counts. Each packet takes WIRE_S. The serial timeout is
    armed at 0.5 s unless given, and as on the controller, no packet for that long stops both
    motors; here any packet, answered or not, restarts it.
    """

    WIRE_S = 0.0005

    def __init__(self, address, clock, m1, m2, serial_timeout=5):
        super().__init__(address, serial_timeout=serial_timeout)
        self.clock = clock
        self.motors = {1: m1, 2: m2}
        self.count_reads = 0
        self.log = []  # (time, command) of every packet
        self.last_packet = None

    def respond(self, packet):
        self.clock.now += self.WIRE_S
        now = self.clock.now
        timeout = self.serial_timeout / 10
        if timeout and self.last_packet is not None and now - self.last_packet > timeout:
            for motor in self.motors.values():
                motor.command(self.last_packet + timeout, 0)
        self.last_packet = now
        for motor in self.motors.values():
            motor.advance(now)
        valid = (
            len(packet) >= 4
            and packet[0] in self.addresses
            and _crc16(packet[:-2]) == struct.unpack(">H", packet[-2:])[0]
        )
        command = packet[1] if valid else None
        self.log.append((now, command))
        if command == DRIVE_M1M2_SPEED:
            return self.drive(now, *struct.unpack(">ii", packet[2:10]))
        if command in DUTY_FORWARD:
            assert packet[2] == 0, "the tuner's only duty command is the stop"
            return self.stop(now, 1 if command == 0 else 2)
        if command in READ_COUNTS:
            self.count_reads += 1
            return self.read_count(packet, 1 if command == 16 else 2)
        return super().respond(packet)

    def drive(self, now, m1, m2):
        for motor, ticks_per_s in ((self.motors[1], m1), (self.motors[2], m2)):
            motor.command(now, ticks_per_s)
            motor.driven = True
        return b"\xff"

    def stop(self, now, channel):
        self.motors[channel].command(now, 0)
        return b"\xff"

    def read_count(self, packet, channel):
        payload = struct.pack(">I", round(self.motors[channel].count) % 2**32) + b"\x00"
        return payload + struct.pack(">H", _crc16(packet[:2] + payload))


FRONT_PORT, REAR_PORT = "/dev/roboclaw2", "/dev/roboclaw1"


def stepped_robot(clock, front=None, rear=None, front_cls=SteppedRoboClaw,
                  rear_cls=SteppedRoboClaw, present=("front", "rear"), **claw_kw):
    """The robot's serial devices, each RoboClaw with two simulated motors."""
    devices = {CAMERA_PORT: FakeCamera()}
    for name, tty, address, motors, cls in (
        ("front", "/dev/ttyACM2", FRONT, front, front_cls),
        ("rear", "/dev/ttyACM1", REAR, rear, rear_cls),
    ):
        if name not in present:
            continue
        claw = cls(address, clock, *(motors or (Motor(), Motor())), **claw_kw)
        devices[tty] = claw
        devices["/dev/roboclaw" + tty.removeprefix("/dev/ttyACM")] = claw
    return FakeBus(devices)


def run_tuner(clock, *argv, module=None, confirmed=True):
    """Run a fresh copy of the tuner's main() on the simulated clock; return its exit status.

    Its signal handlers go to a mock, not to this process. A step is run with --no-confirm
    unless *confirmed* is False.
    """
    module = module or load_tuner()
    if confirmed and {"--tune", "--step-response"} & set(argv):
        argv = (*argv, "--no-confirm")
    with patch.object(sys, "argv", ["tune_pid.py", *argv]), patch.object(
        module, "time", clock
    ), patch.object(module, "signal", MagicMock()):
        try:
            return module.main()
        except KeyboardInterrupt:
            pytest.fail("a KeyboardInterrupt escaped main()")


def gains_written(bus):
    """(D, P, I, QPPS) of every gains write (commands 28 and 29), on any port."""
    return [struct.unpack(">IIII", data[2:18]) for _, data in bus.packets() if data[1] in SET_PID]


def commands(bus, port):
    return [data[1] for _, data in bus.packets(port)]


def reports(out):
    """{(controller, motor): its report's lines}, in the order printed."""
    found, controller, motor = {}, None, None
    for line in out.splitlines():
        if m := re.search(r"(Front|Rear) RoboClaw \(0x", line):
            controller, motor = m.group(1), None
        elif m := re.fullmatch(r"\s+(M[12]):", line):
            motor = m.group(1)
            found[(controller, motor)] = []
        elif motor is not None and line.startswith("    "):
            found[(controller, motor)].append(line.strip())
    return found


def measured(lines):
    """(dead time s, tau s) from a motor's report."""
    m = re.search(r"Dead time:\s+([\d.]+) ms.*tau: ([\d.]+) ms", " ".join(lines))
    return float(m.group(1)) / 1000, float(m.group(2)) / 1000


def passes(lines):
    return any(line.startswith("PASS -- tracks") for line in lines)


def aborted(lines, text):
    return any(line.startswith("ABORT:") and text in line for line in lines)


def faulted(lines, text):
    """A step that ran its hold, measured, and then failed on *text*."""
    return any(line.startswith("Tracking:") for line in lines) and any(
        line.startswith("FAIL --") and text in line for line in lines
    )


def no_verdict(lines):
    return any(line.startswith("NO VERDICT --") for line in lines)


def summary(out):
    """The run's tracking verdict, as its summary prints it."""
    return re.search(r"^  Tracking: +(.+)$", out, re.M).group(1)


HOLD = "6"  # the default hold, the only one the box judges
CYCLES = 300  # a 6 s hold at the driver's 50 Hz
# Dead + tau: the step is timed from its first command 37's return, and the fake applies the
# command as it answers it.
PLANT_63 = 0.012 + 0.022


def test_tune_drives_with_37_reads_counts_and_reports_tau(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    rc = run_tuner(clock, "--tune", "--hold", HOLD)

    assert rc == 0
    for port in (FRONT_PORT, REAR_PORT):
        sent = commands(bus, port)
        assert CYCLES - 1 <= sent.count(DRIVE_M1M2_SPEED) <= CYCLES + 1
        assert not set(sent) & DRIVE_ONE_SPEED, "drives with command 37, as the driver does"
        assert not set(sent) & READ_SPEEDS, "measures the count, not the reported speed"
        assert READ_COUNTS <= set(sent)
        assert SET_PID <= set(sent), "--tune writes both motors' gains"
        drives = {
            struct.unpack(">ii", data[2:10]) for _, data in bus.packets(port) if data[1] == 37
        }
        assert drives == {(535, 535)}, "the default step is 0.3 m/s on both motors"
        for _, data in bus.packets(port):
            if data[1] in SET_PID:
                d, p, i, qpps = struct.unpack(">IIII", data[2:18])
                assert (p, i, d, qpps) == (
                    ROBOCLAW_PID_P,
                    ROBOCLAW_PID_I,
                    ROBOCLAW_PID_D,
                    ROBOCLAW_QPPS,
                )
        motor = [c for c in sent if c in MOTOR_COMMANDS]
        assert motor[-2:] == [0, 4], "the port ends in the duty-0 stop"

    out = capsys.readouterr().out
    found = reports(out)
    assert sorted(found) == [(c, m) for c in ("Front", "Rear") for m in ("M1", "M2")]
    for lines in found.values():
        assert passes(lines), lines
        dead, tau = measured(lines)
        # Up to one of the simulation's 1 ms steps.
        assert dead + tau == pytest.approx(PLANT_63, abs=0.001)
        assert dead == pytest.approx(0.012, abs=0.003)
        assert tau == pytest.approx(0.022, rel=0.12)
    assert "still moving" not in out


def test_the_step_stops_in_the_cycle_its_hold_ends(patched, capsys):
    """The step's own stop, before the settle reads and the controller's close."""
    clock = Clock()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0

    for port in (FRONT_PORT, REAR_PORT):
        sent = commands(bus, port)
        last37 = len(sent) - 1 - sent[::-1].index(DRIVE_M1M2_SPEED)
        between = sent[last37 + 1:sent.index(0, last37)]
        assert set(between) <= READ_COUNTS and len(between) <= 2
    assert "still moving" not in capsys.readouterr().out


def test_tune_writes_the_constants_themselves(patched):
    """The written gains follow strafer_shared.constants, not literals that equal them today."""
    clock = Clock()
    bus = patched(stepped_robot(clock))
    with patch("strafer_shared.constants.ROBOCLAW_QPPS", 2001), patch(
        "strafer_shared.constants.ROBOCLAW_PID_P", 15001
    ), patch("strafer_shared.constants.ROBOCLAW_PID_I", 751), patch(
        "strafer_shared.constants.ROBOCLAW_PID_D", 1
    ):
        module = load_tuner()
    run_tuner(clock, "--tune", "--hold", HOLD, module=module)

    written = gains_written(bus)
    assert len(written) == 4
    assert all((p, i, d, qpps) == (15001, 751, 1, 2001) for d, p, i, qpps in written)


@pytest.mark.parametrize("mode", [("--read",), ("--set", "--p", str(ROBOCLAW_PID_P))])
def test_each_controller_opened_is_stopped_before_anything_else(patched, mode):
    """Opening a controller sends both motors duty 0 before any command but the battery reads
    that find and open it, so --read and --set run with the motors stopped, as --stop does."""
    clock = Clock()
    bus = patched(stepped_robot(clock))

    run_tuner(clock, *mode)

    for port in (FRONT_PORT, REAR_PORT):
        sent = commands(bus, port)
        stop = sent.index(0)
        assert set(sent[:stop]) <= {READ_BATTERY} and sent[stop + 1] == 4, sent


def test_set_writes_the_constants_qpps(patched):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    run_tuner(clock, "--set", "--p", str(ROBOCLAW_PID_P))

    written = gains_written(bus)
    assert written and all(qpps == ROBOCLAW_QPPS for _, _, _, qpps in written)
    assert (tuner.DEFAULT_P, tuner.DEFAULT_I, tuner.DEFAULT_D) == (
        ROBOCLAW_PID_P,
        ROBOCLAW_PID_I,
        ROBOCLAW_PID_D,
    )
    assert not hasattr(tuner, "TARGET_TAU_S")


@pytest.mark.parametrize("mode", ["--step-response", "--tune"])
@pytest.mark.parametrize("short", [("Front", "M1"), ("Rear", "M2")])
def test_a_motor_short_of_its_target_fails_tracking(patched, capsys, mode, short):
    clock = Clock()
    motors = {"Front": [Motor(), Motor()], "Rear": [Motor(), Motor()]}
    motors[short[0]][int(short[1][1]) - 1] = Motor(gain=0.9)
    patched(stepped_robot(clock, front=motors["Front"], rear=motors["Rear"]))

    assert run_tuner(clock, mode, "--hold", HOLD) == 1

    found = reports(capsys.readouterr().out)
    failed = re.search(r"FAIL -- tracks ([\d.]+)% under target", " ".join(found[short]))
    assert failed and float(failed.group(1)) == pytest.approx(10.0, abs=0.3)
    assert measured(found[short])[1] == pytest.approx(0.022, rel=0.12), "tau is still reported"
    assert all(passes(lines) for key, lines in found.items() if key != short)


@pytest.mark.parametrize("gain, rc", [(0.94, 1), (0.96, 0)])
def test_the_box_edge(patched, gain, rc):
    clock = Clock()
    patched(stepped_robot(clock, rear=(Motor(), Motor(gain=gain))))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == rc


def test_a_reverse_step_tracks(patched, capsys):
    """Measured, with no verdict: the box judges only the default step, forward."""
    clock = Clock()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--hold", HOLD, "--speed", "-535") == 3

    drives = [struct.unpack(">ii", data[2:10]) for _, data in bus.packets() if data[1] == 37]
    assert set(drives) == {(-535, -535)}
    out = capsys.readouterr().out
    for lines in reports(out).values():
        assert no_verdict(lines) and measured(lines)[1] == pytest.approx(0.022, rel=0.12)
        assert re.search(r"error=[+-]0\.0%", " ".join(lines))
    assert "measurements, not targets" in out


@pytest.mark.parametrize("mode", ["--step-response", "--tune"])
@pytest.mark.parametrize("speed", ["535", "-535"])
def test_a_reversed_motor_stops_the_run(patched, capsys, speed, mode):
    clock = Clock()
    bus = patched(stepped_robot(clock, front=(Motor(gain=-1.0), Motor())))

    assert run_tuner(clock, mode, "--hold", HOLD, "--speed", speed) == 1

    sent = commands(bus, FRONT_PORT)
    assert sent.count(DRIVE_M1M2_SPEED) < 10, "the step stops within a few cycles"
    first_stop = next(k for k, c in enumerate(sent) if c == 0 and k > sent.index(DRIVE_M1M2_SPEED))
    assert DRIVE_M1M2_SPEED not in sent[first_stop:], "nothing drives after the stop"
    assert DRIVE_M1M2_SPEED not in commands(bus, REAR_PORT), "the run ends at the abort"
    out = capsys.readouterr().out
    found = reports(out)
    assert aborted(found[("Front", "M1")], "turns against its target")
    assert aborted(found[("Front", "M2")], "stopped with M1")
    assert "Not stepping the remaining controllers" in out


def test_a_runaway_motor_stops_the_run(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front=(Motor(gain=3.0), Motor())))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED) < 25
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "ran away")


class Jittered(SteppedRoboClaw):
    """M1's count read is answered a packet's time late on every other read, so successive reads
    come a packet's time more and less than a cycle apart, as a real cycle's do: a speed of a
    whole number of ticks a cycle then reads off the multiples of 1 tick / SAMPLE_PERIOD_S."""

    def respond(self, packet):
        if packet[1] == 16:
            self.late = not getattr(self, "late", False)
            if self.late:
                self.clock.now += self.WIRE_S
        return super().respond(packet)


@pytest.mark.parametrize("ticks, stopped", [(4, True), (2, False)])
def test_a_motor_turning_against_its_target_over_3_ticks_a_cycle_is_stopped(
    patched, capsys, ticks, stopped
):
    """The against rule's limit is its comment's 3 ticks a cycle, three times a count's jitter:
    a motor that turns against its target a tick a cycle faster (4) is stopped, and one a tick a
    cycle slower (2) runs its hold (and fails it). Its reads wander, so 4 ticks a cycle reads
    about 195 and 205 ticks/s: over the limit, and on alternate cycles under one a tick higher."""
    clock = Clock()
    gain = -ticks / tuner.SAMPLE_PERIOD_S / tuner.STEP_VELOCITY_TICKS
    patched(stepped_robot(clock, front=(Motor(gain=gain), Motor()), front_cls=Jittered))

    assert run_tuner(clock, "--step-response", "--controller", "front") == 1

    found = reports(capsys.readouterr().out)[("Front", "M1")]
    assert aborted(found, "turns against its target") == stopped


@pytest.mark.parametrize("over, stopped", [(1, True), (-1, False)])
def test_a_motor_a_tick_a_cycle_past_the_runaway_limit_is_stopped(patched, capsys, over, stopped):
    """The limit is RUNAWAY_FACTOR times the target plus the margin, its comment's 6 ticks a
    cycle: a motor a tick a cycle over it is stopped, and one a tick a cycle under it runs its
    hold (and fails it). Its reads wander as in the against rule's test, so neither reads on the
    limit's side of the other."""
    limit = tuner.RUNAWAY_FACTOR * tuner.STEP_VELOCITY_TICKS + 6 / tuner.SAMPLE_PERIOD_S
    speed = limit + over / tuner.SAMPLE_PERIOD_S
    clock = Clock()
    motor = Motor(gain=speed / tuner.STEP_VELOCITY_TICKS)
    patched(stepped_robot(clock, front=(motor, Motor()), front_cls=Jittered))

    assert run_tuner(clock, "--step-response", "--controller", "front") == 1

    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "ran away") == stopped


@pytest.mark.parametrize("over, stopped", [(1, True), (-1, False)])
def test_the_held_motor_a_tick_a_cycle_past_its_runaway_limit_is_stopped(
    patched, capsys, over, stopped
):
    """A motor held at 0 has its own runaway limit, RUNAWAY_FACTOR times 0 plus the margin, the
    margin's comment's 6 ticks a cycle: a held wheel drifting a tick a cycle over it stops the
    step, and one a tick a cycle under it does not (it fails the check after the stop). Its
    reads wander as in the stepped motor's test, so neither reads on the limit's side of the
    other."""
    limit = 6 / tuner.SAMPLE_PERIOD_S
    drift = limit + over / tuner.SAMPLE_PERIOD_S
    clock = Clock()
    patched(stepped_robot(clock, front=(Motor(), Drifting(drift, from_command=True)),
                          front_cls=Jittered))

    assert run_tuner(clock, "--step-response", "--controller", "front", "--motor", "m1") == 1

    found = reports(capsys.readouterr().out)[("Front", "M1")]
    assert aborted(found, "stopped with M2: M2 ran away") == stopped


class HeldRunsAway(SteppedRoboClaw):
    """M2's count moves *ticks* a cycle for *cycles* cycles from the *at*-th command 37, after the
    rise, then rests again."""

    at = round(tuner.TRACK_START_S / 2 / tuner.SAMPLE_PERIOD_S)
    cycles = ticks = 0

    def drive(self, now, m1, m2):
        self.drives = getattr(self, "drives", 0) + 1
        if self.at <= self.drives < self.at + self.cycles:
            self.motors[2].count += self.ticks
        return super().drive(now, m1, m2)


@pytest.mark.parametrize("sustained", [True, False])
def test_the_held_motor_runs_away_only_on_its_cycles_in_a_row(patched, capsys, sustained):
    """The held motor's runaway rule trips after its comment's 10 cycles in a row (0.2 s), as the
    stepped motor's does: a held wheel turning at twice the margin's 6 ticks a cycle for that many
    cycles stops the step, and for one fewer does not."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=HeldRunsAway))
    claw = bus.devices[FRONT_PORT]
    claw.ticks, claw.cycles = 2 * 6, 10 if sustained else 9

    assert run_tuner(clock, "--step-response", "--controller", "front", "--motor", "m1") == (
        1 if sustained else 0
    )

    found = reports(capsys.readouterr().out)[("Front", "M1")]
    assert aborted(found, "stopped with M2: M2 ran away") == sustained


def printed_counts(lines):
    """The counts a motor's report prints with -v, one a cycle."""
    return [int(m.group(1)) for line in lines if (m := re.fullmatch(r"[\d.]+\s+(-?\d+)", line))]


def test_a_reversed_motor_at_the_smallest_step_is_stopped_within_the_against_cycles(
    patched, capsys
):
    """MIN_STEP_TICKS is a tick a cycle over the against rule's limit: once a reversed motor is
    at the step, the rule sees it every cycle, however its reads wander, and stops it within
    AGAINST_SAMPLES cycles of the end of its rise (its dead time and five taus), at every phase
    of its count, here 25 phases a 25th of a tick apart."""
    phases = 25
    for phase in range(phases):
        clock = Clock()
        wheel = Motor(gain=-1.0)
        wheel.count = phase / phases
        rise = SteppedRoboClaw.WIRE_S + wheel.dead + 5 * wheel.tau
        risen = math.ceil(rise / tuner.SAMPLE_PERIOD_S)
        patched(stepped_robot(clock, front=(wheel, Motor()), front_cls=Jittered))

        assert run_tuner(
            clock, "--step-response", "--controller", "front", "--hold", f"{tuner.MIN_HOLD_S:g}",
            "--speed", str(tuner.MIN_STEP_TICKS), "-v",
        ) == 1, phase

        found = reports(capsys.readouterr().out)[("Front", "M1")]
        assert aborted(found, "turns against its target"), phase
        assert len(printed_counts(found)) <= risen + tuner.AGAINST_SAMPLES, phase


@pytest.mark.parametrize("mode", ["--step-response", "--tune"])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("under, refused", [(1, True), (0, False)])
def test_the_smallest_step_is_stepped_and_one_under_it_refused(
    patched, capsys, sign, under, refused, mode
):
    """The smallest step is a tick a cycle over the against rule's limit, as its comment gives
    it: a step a tick per second under it is refused, either way and in either mode, and the
    refusal says why."""
    smallest = round(tuner.AGAINST_TPS + 1 / tuner.SAMPLE_PERIOD_S)
    clock = Clock()
    bus = patched(stepped_robot(clock))
    argv = (mode, "--controller", "front", "--hold", f"{tuner.MIN_HOLD_S:g}",
            "--speed", str(sign * (smallest - under)))

    if refused:
        with pytest.raises(SystemExit) as exit_:
            run_tuner(clock, *argv)
        assert exit_.value.code == 2 and bus.packets() == []
        why = "below it the against abort cannot see a reversed motor every cycle"
        assert why in capsys.readouterr().err
    else:
        assert run_tuner(clock, *argv) == tuner.EXIT_NO_VERDICT


def test_the_help_gives_the_steps_the_aborts_guard_and_the_stops_check():
    text = " ".join(tuner.build_parser().format_help().split())
    low, high = tuner.MIN_STEP_TICKS, tuner.MAX_STEP_TICKS
    rest, still = tuner.REST_TICKS, f"{tuner.STILL_CHECK_S:g} s"

    assert (
        f"from {low}, so the against abort sees a reversed motor in every full-length cycle, to "
        f"{high}" in text
    )
    assert f"--speed is {low} to {high} ticks/s either way" in text
    assert (
        f"a stop is not acknowledged or errs, a read fails, or a wheel's count moves more than "
        f"{rest} ticks over {still} from {still} after the stop" in text
    )


class ScaledCount(SteppedRoboClaw):
    """M1's count moves *factor* times the wheel's travel for *cycles* cycles from the *at*-th
    command 37, after the rise and before the box opens, then follows the wheel again: -1 reads
    it reversed, as a reversed encoder would."""

    at = round(tuner.TRACK_START_S / 2 / tuner.SAMPLE_PERIOD_S)
    cycles, factor = 0, 1

    def drive(self, now, m1, m2):
        self.drives = getattr(self, "drives", 0) + 1
        return super().drive(now, m1, m2)

    def read_count(self, packet, channel):
        if channel != 1:
            return super().read_count(packet, channel)
        count = round(self.motors[1].count)
        travel, self.seen = count - getattr(self, "seen", count), count
        if self.at <= getattr(self, "drives", 0) < self.at + self.cycles:
            self.offset = getattr(self, "offset", 0) + (self.factor - 1) * travel
        payload = struct.pack(">I", round(count + getattr(self, "offset", 0)) % 2**32) + b"\x00"
        return payload + struct.pack(">H", _crc16(packet[:2] + payload))


@pytest.mark.parametrize("factor, cycles, reason", [
    (-1, 3, "turns against its target"),
    (tuner.RUNAWAY_FACTOR + 1, 10, "ran away"),
], ids=["against", "runaway"])
@pytest.mark.parametrize("sustained", [True, False])
def test_a_rule_trips_only_on_its_cycles_in_a_row(
    patched, capsys, factor, cycles, reason, sustained
):
    """The against rule stops the run after 3 cycles in a row (60 ms) and the runaway rule after
    10 (0.2 s), as their comments give them, so a glitch trips neither: M1's count read
    reversed, or running away at a multiple of the target past the limit, for that many cycles
    stops the run, and for one fewer does not."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=ScaledCount))
    claw = bus.devices[FRONT_PORT]
    claw.factor, claw.cycles = factor, cycles if sustained else cycles - 1

    assert run_tuner(clock, "--step-response", "--controller", "front") == (1 if sustained else 0)

    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], reason) == sustained


# The stall rule's window, as its comment gives it: 15 cycles, from 0.2 s in.
STALL_CYCLES, STALL_START_S = 15, 0.2


def test_a_count_that_never_moves_stops_the_run(patched, capsys):
    """A dead encoder: no abort fires on its speed, so the stall abort does, in the cycle that
    starts when the stall window, from 0.2 s in and 15 cycles long (the comment's), is full."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front=(Motor(gain=0.0), Motor())))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    full = STALL_START_S + STALL_CYCLES * tuner.SAMPLE_PERIOD_S
    cycle = round(full / tuner.SAMPLE_PERIOD_S) + 1
    assert commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED) == cycle
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "a dead encoder")


@pytest.mark.parametrize("ticks, stalls", [(-0.1, True), (1, False)])
def test_a_slow_wheel_stalls_under_a_tenth_of_its_target(patched, capsys, ticks, stalls):
    """The stall limit is a tenth (the comment's) of the target's travel over the window: a wheel
    that moves a tenth of a tick less in the window stalls, and one a tick more runs its hold
    (and fails it). The rule reads the window's count every cycle, so it resolves the limit to
    under a tick."""
    travel = tuner.STEP_VELOCITY_TICKS * STALL_CYCLES * tuner.SAMPLE_PERIOD_S
    clock = Clock()
    patched(stepped_robot(clock, front=(Motor(gain=(travel / 10 + ticks) / travel), Motor())))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "a dead encoder") == stalls


@pytest.mark.parametrize("run, channel", sorted(REFERENCE))
def test_the_stall_rule_passes_the_real_wheels(run, channel):
    trace = tuner.StepTrace(motor=channel, target=target(run))
    for t, c in zip(*bench_trace(run, channel)):
        trace.times.append(t)
        trace.counts.append(c)
        if t >= 0.5:
            assert tuner._stalled(trace, target(run)) is None, t


class Jamming(Motor):
    """A wheel that turns, then jams *after* s after its first command: its count stops there."""

    def __init__(self, after, **kw):
        super().__init__(**kw)
        self.after, self.jam_at = after, None

    def command(self, now, ticks_per_s):
        if self.jam_at is None and ticks_per_s:
            self.jam_at = now + self.after
        super().command(now, ticks_per_s)

    def advance(self, now, step=0.001):
        if self.jam_at is not None and now >= self.jam_at:
            super().advance(self.jam_at, step)
            self.t, self.speed = now, 0.0
            return
        super().advance(now, step)


class Slipping(SteppedRoboClaw):
    """Each cycle's M1 count read comes *slip* later in its cycle than the last cycle's. Reads at
    one point of each cycle put the read a whole stall window back on the window's edge, where
    the clock's rounding picks 15 cycles or 16; these put it inside."""

    slip = SteppedRoboClaw.WIRE_S / 100

    def drive(self, now, m1, m2):
        self.drives = getattr(self, "drives", 0) + 1
        return super().drive(now, m1, m2)

    def respond(self, packet):
        if packet[1] == 16:
            self.clock.now += self.slip * getattr(self, "drives", 0)
        return super().respond(packet)


def test_a_wheel_that_jams_mid_hold_stops_the_run(patched, capsys):
    """Nine tenths of a stall window after the jam, the window holds a tenth of its travel (the
    comment's 15 cycles, and a tenth): that falls mid-cycle, and the next cycle is stopped."""
    jam = 1.0
    clock = Clock()
    bus = patched(stepped_robot(clock, front=(Jamming(jam), Motor()), front_cls=Slipping))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    window = STALL_CYCLES * tuner.SAMPLE_PERIOD_S
    cycle = math.ceil((jam + 0.9 * window) / tuner.SAMPLE_PERIOD_S) + 1
    assert commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED) == cycle
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "a jammed wheel")


def test_the_motor_held_at_0_is_watched(patched, capsys):
    """With --motor m1, M2 is held at 0 by command 37, and still read and guarded."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front=(Motor(), Drifting(400.0, from_command=True))))

    assert run_tuner(clock, "--step-response", "--controller", "front", "--motor", "m1",
                     "--hold", HOLD) == 1

    sent = commands(bus, FRONT_PORT)
    first, last = sent.index(DRIVE_M1M2_SPEED), len(sent) - sent[::-1].index(DRIVE_M1M2_SPEED)
    assert sent[first:last].count(17) >= sent.count(DRIVE_M1M2_SPEED) - 1, "M2 read every cycle"
    found = reports(capsys.readouterr().out)
    assert sorted(found) == [("Front", "M1")]
    assert aborted(found[("Front", "M1")], "stopped with M2: M2 ran away")


def test_a_turning_wheel_is_not_stepped(patched, capsys):
    """The rest check catches a wheel turning at its comment's 20 ticks/s at every phase of its
    count, here 25 phases a 25th of a tick apart."""
    phases = 25
    for phase in range(phases):
        clock = Clock()
        wheel = Drifting(20.0)
        wheel.count = phase / phases
        bus = patched(stepped_robot(clock, front=(wheel, Motor())))

        assert run_tuner(clock, "--step-response", "--controller", "front") == 1

        assert DRIVE_M1M2_SPEED not in commands(bus, FRONT_PORT), phase
        found = reports(capsys.readouterr().out)
        assert aborted(found[("Front", "M1")], "not at rest"), phase
        assert aborted(found[("Front", "M2")], "stopped with M1")


class BumpedClaw(SteppedRoboClaw):
    """M1's count steps *bump* ticks at its 5th count read, mid-rest, as a settling wheel."""

    bump = 2

    def read_count(self, packet, channel):
        if channel == 1:
            self.m1_reads = getattr(self, "m1_reads", 0) + 1
            if self.m1_reads == 5:
                self.motors[1].count += self.bump
        return super().read_count(packet, channel)


def test_the_zero_is_the_last_count_at_rest(patched, capsys):
    clock = Clock()
    patched(stepped_robot(clock, front_cls=BumpedClaw))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0

    for key, lines in reports(capsys.readouterr().out).items():
        dead, tau = measured(lines)
        assert dead + tau == pytest.approx(PLANT_63, abs=0.0015), key


class BumpedPastAFlicker(BumpedClaw):
    bump = BumpedClaw.bump + 1


def test_a_count_a_tick_past_a_settling_wheels_step_is_not_at_rest(patched, capsys):
    """A count at rest may step a settling wheel's 2 ticks, a flicker with one to spare; a tick
    more is a turning wheel, and nothing is stepped."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=BumpedPastAFlicker))

    assert run_tuner(clock, "--step-response") == 1

    assert DRIVE_M1M2_SPEED not in commands(bus, FRONT_PORT)
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "not at rest")


class BumpedInTheLastRestCycle(SteppedRoboClaw):
    """M1's count steps a tick past a flicker at its first read in the *cycles*-th cycle from its
    first read."""

    cycles = 0

    def read_count(self, packet, channel):
        if channel == 1:
            self.first = getattr(self, "first", self.clock.now)
            due = self.clock.now - self.first >= (self.cycles - 1) * tuner.SAMPLE_PERIOD_S
            if due and not getattr(self, "bumped", False):
                self.bumped = True
                self.motors[1].count += BumpedPastAFlicker.bump
        return super().read_count(packet, channel)


def test_the_rest_check_watches_ten_cycles(patched, capsys):
    """PRE_S is ten cycles (the comment's): a count that steps a tick past a flicker in the tenth
    is not at rest, and nothing is stepped."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=BumpedInTheLastRestCycle))
    bus.devices[FRONT_PORT].cycles = 10

    assert run_tuner(clock, "--step-response", "--controller", "front") == 1

    assert DRIVE_M1M2_SPEED not in commands(bus, FRONT_PORT)
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "not at rest")


class SlowWire(SteppedRoboClaw):
    WIRE_S = 0.004


def test_reads_are_timed_when_they_return(patched, capsys):
    """Each read is timed when it returns, and the step from its first command 37's return: on a
    slow wire the count is read as it is latched, so the plant's dead time and tau come out as
    they are, not a packet's time short (reads timed when sent) or long (the step timed from its
    first command's start)."""
    clock = Clock()
    patched(stepped_robot(clock, front_cls=SlowWire, rear_cls=SlowWire))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0

    for key, lines in reports(capsys.readouterr().out).items():
        dead, tau = measured(lines)
        assert dead + tau == pytest.approx(PLANT_63, abs=0.0015), key


class DroppingClaw(SteppedRoboClaw):
    """Count reads go unanswered from the 40th on, about 10 cycles into the step."""

    def read_count(self, packet, channel):
        return b"" if self.count_reads >= 40 else super().read_count(packet, channel)


class Unacked37(SteppedRoboClaw):
    def drive(self, now, m1, m2):
        return b""


@pytest.mark.parametrize("cls, reason", [
    (DroppingClaw, "count reads failed"),
    (Unacked37, "command 37 unacknowledged"),
])
def test_a_link_that_stops_answering_stops_the_run(patched, capsys, cls, reason):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=cls))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED) < 20
    assert DRIVE_M1M2_SPEED not in commands(bus, REAR_PORT)
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], reason)


class MissesCycles(SteppedRoboClaw):
    """From the *at*-th command 37, *cycles* cycles in a row miss: their count reads go
    unanswered, or (*acks*) their command 37 is applied and its acknowledgement lost."""

    at, cycles, acks = CYCLES // 2, 0, False

    def drive(self, now, m1, m2):
        self.drives = getattr(self, "drives", 0) + 1
        reply = super().drive(now, m1, m2)
        return b"" if self.acks and self.missing() else reply

    def read_count(self, packet, channel):
        if not self.acks and self.missing():
            return b""
        return super().read_count(packet, channel)

    def missing(self):
        return self.at <= getattr(self, "drives", 0) < self.at + self.cycles


@pytest.mark.parametrize("acks, reason", [
    (False, "count reads failed"), (True, "command 37 unacknowledged"),
], ids=["reads", "acks"])
@pytest.mark.parametrize("cycles, stopped", [(2, True), (1, False)])
def test_two_missed_cycles_in_a_row_end_the_step(patched, capsys, acks, reason, cycles, stopped):
    """One missed cycle is let pass, and two in a row are a failing link, as the comment gives
    them: the step ends at the second, and no command 37 follows it."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=MissesCycles))
    claw = bus.devices[FRONT_PORT]
    claw.acks, claw.cycles = acks, cycles

    assert run_tuner(clock, "--step-response", "--controller", "front") == (1 if stopped else 0)

    sent = commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED)
    assert (sent == claw.at + cycles - 1) == stopped
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], reason) == stopped


class IgnoresStops(SteppedRoboClaw):
    """Acknowledges the duty-0 stop but keeps driving."""

    def stop(self, now, channel):
        return b"\xff"


def test_a_wheel_still_turning_after_the_stop_fails(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=IgnoresStops))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    out = capsys.readouterr().out
    assert faulted(reports(out)[("Front", "M1")], "still moving after the stop")
    sent = commands(bus, FRONT_PORT)
    tail = sent[len(sent) - sent[::-1].index(DRIVE_M1M2_SPEED):]
    assert tail.count(0) == tail.count(4) == 3, "the step's stop, the stop again, the close"
    assert DRIVE_M1M2_SPEED not in commands(bus, REAR_PORT), "the run ends at the fault"
    assert "Not stepping the remaining controllers" in out


@pytest.mark.parametrize("raw", [0, 11, 20])
def test_no_step_without_an_armed_serial_timeout(patched, capsys, raw):
    """0 disables the timeout; 1.1 and 2.0 s are past the range: neither controller is driven."""
    clock = Clock()
    bus = patched(stepped_robot(clock, serial_timeout=raw))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    for port in (FRONT_PORT, REAR_PORT):
        assert DRIVE_M1M2_SPEED not in commands(bus, port)
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "serial timeout")


@pytest.mark.parametrize("raw", [1, 10])
def test_the_serial_timeout_range_is_inclusive(patched, raw):
    clock = Clock()
    patched(stepped_robot(clock, serial_timeout=raw))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0


class NoTimeoutRead(SteppedRoboClaw):
    def respond(self, packet):
        return b"" if packet[1] == 15 else super().respond(packet)


def test_an_unread_serial_timeout_is_no_step(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=NoTimeoutRead))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert DRIVE_M1M2_SPEED not in [data[1] for _, data in bus.packets()]
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "serial timeout unread")


class StickyGains(SteppedRoboClaw):
    """Acknowledges commands 28 and 29 but keeps its old gains."""

    def respond(self, packet):
        kept = dict(self.gains)
        reply = super().respond(packet)
        self.gains = kept
        return reply


def test_tune_does_not_step_on_unconfirmed_gains(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=StickyGains))

    assert run_tuner(clock, "--tune", "--hold", HOLD) == 1

    for port in (FRONT_PORT, REAR_PORT):
        assert DRIVE_M1M2_SPEED not in commands(bus, port)
    assert "gains were not confirmed" in capsys.readouterr().out


class Interrupting(SteppedRoboClaw):
    """Ctrl-C arrives during a count read, about 10 cycles into the step."""

    def read_count(self, packet, channel):
        if self.count_reads == 40:
            raise KeyboardInterrupt
        return super().read_count(packet, channel)


def test_ctrl_c_stops_the_motors(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=Interrupting))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 130

    sent = commands(bus, FRONT_PORT)
    last37 = len(sent) - 1 - sent[::-1].index(DRIVE_M1M2_SPEED)
    assert {0, 4} <= set(sent[last37:]), "both motors are stopped after the interrupt"
    assert DRIVE_M1M2_SPEED not in commands(bus, REAR_PORT)
    assert "Interrupted" in capsys.readouterr().out


def test_a_hangup_a_kill_or_a_ctrl_z_takes_the_ctrl_c_path(patched):
    module = load_tuner()
    clock = Clock()
    bus = patched(stepped_robot(clock))
    with patch.object(sys, "argv", ["tune_pid.py", "--read"]), patch.object(
        module, "signal"
    ) as sig:
        opened = []
        sig.signal.side_effect = lambda signum, handler: (
            opened.append(len(bus.opens)) if handler is module._interrupt else None
        )
        module.main()

    assert opened == [0, 0, 0], "installed before any port opens"
    handlers = {
        call.args[0]: call.args[1]
        for call in sig.signal.call_args_list
        if call.args[1] is module._interrupt
    }
    assert set(handlers) == {sig.SIGTERM, sig.SIGHUP, sig.SIGTSTP}
    for handler in handlers.values():
        with pytest.raises(KeyboardInterrupt):
            handler(15, None)


@pytest.mark.parametrize("present", [(), ("front",)])
def test_a_missing_controller_fails_and_steps_nothing(patched, present):
    clock = Clock()
    bus = patched(stepped_robot(clock, present=present))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert DRIVE_M1M2_SPEED not in [data[1] for _, data in bus.packets()]


@pytest.mark.parametrize(
    "speed", ["0", str(tuner.MAX_STEP_TICKS + 1), str(-tuner.MAX_STEP_TICKS - 1)]
)
def test_a_step_the_aborts_cannot_guard_is_refused(patched, speed):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    with pytest.raises(SystemExit) as exit_:
        run_tuner(clock, "--step-response", "--hold", HOLD, "--speed", speed)

    assert exit_.value.code == 2
    assert bus.packets() == []


class AckOnRetry(SteppedRoboClaw):
    """Leaves every other duty-0 stop unacknowledged (and unapplied)."""

    def stop(self, now, channel):
        self.stops = getattr(self, "stops", 0) + 1
        return super().stop(now, channel) if self.stops % 2 == 0 else b""


def test_each_stop_is_retried_until_acknowledged(patched):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=AckOnRetry))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0

    sent = commands(bus, FRONT_PORT)
    last37 = len(sent) - 1 - sent[::-1].index(DRIVE_M1M2_SPEED)
    first_stop = sent.index(0, last37)
    assert sent[first_stop:first_stop + 4] == [0, 0, 4, 4], "each motor's stop sent again"


class NeverAcksStops(SteppedRoboClaw):
    def stop(self, now, channel):
        return b""


def test_a_controller_that_does_not_acknowledge_the_stop_is_not_stepped(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=NeverAcksStops))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert DRIVE_M1M2_SPEED not in [data[1] for _, data in bus.packets()]
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "stop went unacknowledged")


class StopsFailOnceDriven(SteppedRoboClaw):
    """Acknowledges stops until command 37 arrives, then neither applies nor acknowledges one."""

    def stop(self, now, channel):
        return b"" if self.motors[channel].driven else super().stop(now, channel)


def test_an_unacknowledged_final_stop_goes_silent_then_fails(patched, capsys):
    """No packet for 1.2 s, longer than any armed serial timeout, so the controller's timeout
    stops the wheels that ignored the stops; the run still fails on the unacknowledged stop."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=StopsFailOnceDriven))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    log = bus.devices[FRONT_PORT].log
    last37 = max(k for k, (_, command) in enumerate(log) if command == DRIVE_M1M2_SPEED)
    tries = [k for k in range(last37, len(log)) if log[k][1] in DUTY_FORWARD][:6]
    assert [log[k][1] for k in tries] == [0, 0, 0, 4, 4, 4], "each stop tried 3 times"
    # The next packet is the check's first read: the silence, then the check's own wait.
    quiet = log[tries[-1] + 1][0] - log[tries[-1]][0]
    assert tuner.SILENT_S + tuner.STILL_CHECK_S <= quiet < tuner.SILENT_S + tuner.STILL_CHECK_S + 0.01
    assert log[tries[-1] + 1][1] in READ_COUNTS
    out = capsys.readouterr().out
    assert "STOP NOT ACKNOWLEDGED: silent 1.2 s" in out
    found = reports(out)[("Front", "M1")]
    assert faulted(found, "stop went unacknowledged (silent 1.2 s)")
    assert "still moving" not in out, "the controller's timeout stopped them"


class IgnoresFirstStopOnceDriven(SteppedRoboClaw):
    """Acknowledges, but does not apply, the first stop after command 37."""

    def stop(self, now, channel):
        motor = self.motors[channel]
        if motor.driven and not getattr(motor, "ignored", False):
            motor.ignored = True
            return b"\xff"
        return super().stop(now, channel)


def test_a_wheel_the_second_stop_stops_still_fails(patched, capsys):
    clock = Clock()
    patched(stepped_robot(clock, front_cls=IgnoresFirstStopOnceDriven))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert faulted(reports(capsys.readouterr().out)[("Front", "M1")], "still moving after the stop")


class Halting(Motor):
    """A wheel that a duty-0 stop, once command 37 has reached it, halts *delay* s later and at
    once: it turns at its target until then, and its count stops dead."""

    def __init__(self, delay, **kw):
        super().__init__(**kw)
        self.delay, self.halt = delay, math.inf

    def command(self, now, ticks_per_s):
        if self.driven and not ticks_per_s:
            self.halt = min(self.halt, now + self.delay)
        else:
            super().command(now, ticks_per_s)

    def advance(self, now, step=0.001):
        super().advance(min(now, self.halt), step)
        if now > self.halt:
            self.t, self.speed = now, 0.0


@pytest.mark.parametrize("early", [True, False])
def test_the_check_after_the_stop_starts_a_settle_after_it(patched, capsys, early):
    """STILL_CHECK_S and SETTLE_TIME_S are both the time a lifted wheel stopped at duty 0 takes
    to come to rest. A wheel its stop halts at once, half a cycle before a settle, is at rest for
    the check and passes; one halted half a cycle after a settle turns into the check, and
    fails."""
    halt = tuner.SETTLE_TIME_S + (-1 if early else 1) * tuner.SAMPLE_PERIOD_S / 2
    clock = Clock()
    patched(stepped_robot(clock, front=(Halting(halt), Halting(halt))))

    assert run_tuner(clock, "--step-response", "--controller", "front") == (0 if early else 1)

    found = reports(capsys.readouterr().out)
    assert faulted(found[("Front", "M1")], "still moving after the stop") != early


class LeftTurning(SteppedRoboClaw):
    """Acknowledges each duty-0 stop, and leaves a motor that command 37 has reached turning at
    *rate* ticks/s."""

    rate = 0.0

    def stop(self, now, channel):
        motor = self.motors[channel]
        motor.command(now, self.rate if motor.driven else 0)
        return b"\xff"


@pytest.mark.parametrize("share, fails", [(1, True), (1 / 4, False)])
def test_a_wheel_the_stop_leaves_turning_at_20_ticks_a_second_fails(patched, capsys, share, fails):
    """The check after the stop catches a wheel turning at its comment's 20 ticks/s, and passes
    one at a quarter of that."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=LeftTurning))
    bus.devices[FRONT_PORT].rate = 20.0 * share

    assert run_tuner(clock, "--step-response", "--controller", "front") == (1 if fails else 0)

    found = reports(capsys.readouterr().out)
    assert faulted(found[("Front", "M1")], "still moving after the stop") == fails


@pytest.mark.parametrize("ticks, fails", [(1, True), (-1, False)])
def test_a_wheel_the_stop_leaves_a_tick_over_rest_fails(patched, capsys, ticks, fails):
    """The check after the step's stop, which watches both motors, fails wheels the stop leaves
    turning a tick over REST_TICKS in STILL_CHECK_S, and passes wheels a tick under, at every
    phase of their count, here 10 phases a tenth of a tick apart."""
    phases = 10
    for phase in range(phases):
        clock = Clock()
        wheels = (Motor(), Motor())
        for wheel in wheels:
            wheel.count = phase / phases
        bus = patched(stepped_robot(clock, front=wheels, front_cls=LeftTurning))
        bus.devices[FRONT_PORT].rate = (tuner.REST_TICKS + ticks) / tuner.STILL_CHECK_S

        assert run_tuner(
            clock, "--step-response", "--controller", "front", "--motor", "m1",
            "--hold", f"{tuner.MIN_HOLD_S:g}",
        ) == (1 if fails else tuner.EXIT_NO_VERDICT), phase

        found = reports(capsys.readouterr().out)
        assert faulted(found[("Front", "M1")], "still moving after the stop") == fails, phase


HELD = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGTSTP)


@pytest.fixture
def handlers_kept():
    """SIGTERM, SIGHUP and SIGTSTP take the Ctrl-C path, as main() sets them, and hard_stop
    leaves the handlers as it found them. The process's own handlers come back after the test."""
    before = {sig: signal.getsignal(sig) for sig in HELD}
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGTSTP):
        signal.signal(sig, tuner._interrupt)
    installed = {sig: signal.getsignal(sig) for sig in HELD}
    try:
        yield
        assert {sig: signal.getsignal(sig) for sig in HELD} == installed, "hard_stop restores them"
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)


def signalling_stops(*signals):
    """A controller whose M1 stop receives *signals*, from this process, as it is sent."""
    sent = []
    rc = MagicMock()

    def m1(duty):
        sent.append(1)
        for sig in signals:
            os.kill(os.getpid(), sig)
        return True

    rc.forward_m1.side_effect = m1
    rc.forward_m2.side_effect = lambda duty: sent.append(2) or True
    return rc, sent


@pytest.mark.parametrize("sig", HELD)
def test_hard_stop_holds_a_signal_until_both_stops_are_sent(handlers_kept, sig):
    """Python runs signal handlers in the main thread, so this holds with native threads too."""
    rc, sent = signalling_stops(sig)

    with pytest.raises(KeyboardInterrupt, match=f"signal {int(sig)}"):
        tuner.hard_stop(rc)

    assert sent == [1, 2]


def test_a_second_signal_abandons_the_stops(handlers_kept):
    """A write that hangs can still be interrupted: the controller's timeout stops the wheels."""
    rc, sent = signalling_stops(signal.SIGINT, signal.SIGINT)

    with pytest.raises(KeyboardInterrupt):
        tuner.hard_stop(rc)

    assert sent == [1]


def test_hard_stop_restores_the_handlers(handlers_kept):
    rc = MagicMock()
    rc.forward_m1.return_value = rc.forward_m2.return_value = True

    assert tuner.hard_stop(rc) is True


def test_hard_stop_off_the_main_thread_sends_both_stops(handlers_kept):
    rc = MagicMock()
    rc.forward_m1.return_value = rc.forward_m2.return_value = True
    result = []
    worker = threading.Thread(target=lambda: result.append(tuner.hard_stop(rc)))
    worker.start()
    worker.join()

    assert result == [True] and rc.forward_m2.called


def test_hard_stop_sends_m2_when_m1_fails(handlers_kept):
    rc = MagicMock()
    rc.forward_m1.side_effect = OSError("gone")
    rc.forward_m2.return_value = True

    assert tuner.hard_stop(rc) is False

    assert rc.forward_m1.call_count == 3 and rc.forward_m2.call_count == 1


def test_the_count_wraps_through_the_step(patched, capsys):
    clock = Clock()
    m1, m2 = Motor(), Motor()
    m1.count = m2.count = 2**32 - 60.0  # the 32-bit count wraps during the rise
    patched(stepped_robot(clock, front=(m1, m2)))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0

    for key, lines in reports(capsys.readouterr().out).items():
        dead, tau = measured(lines)
        assert dead + tau == pytest.approx(PLANT_63, abs=0.0015), key


class FlakyReads(SteppedRoboClaw):
    """Count reads go unanswered on every other cycle of the step, retries included."""

    def drive(self, now, m1, m2):
        self.flaky = not getattr(self, "flaky", True)
        return super().drive(now, m1, m2)

    def stop(self, now, channel):
        self.flaky = False
        return super().stop(now, channel)

    def read_count(self, packet, channel):
        return b"" if getattr(self, "flaky", False) else super().read_count(packet, channel)


def test_reads_failing_on_alternate_cycles_do_not_stop_the_step(patched, capsys):
    """Only MAX_MISSED failed cycles in a row stop it; one, then a good one, does not."""
    clock = Clock()
    patched(stepped_robot(clock, front_cls=FlakyReads))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0

    assert all(passes(lines) for lines in reports(capsys.readouterr().out).values())


class NoCounts(SteppedRoboClaw):
    def read_count(self, packet, channel):
        return b""


@pytest.mark.parametrize("mode", ["--step-response", "--tune"])
def test_an_error_ends_the_run(patched, capsys, mode):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=NoCounts))

    assert run_tuner(clock, mode, "--hold", HOLD) == 1

    assert DRIVE_M1M2_SPEED not in [data[1] for _, data in bus.packets()]
    out = capsys.readouterr().out
    assert "ERROR:" in out and "Not stepping the remaining controllers" in out


@pytest.mark.parametrize("hold", ["0.5", "0.99", "31", "nan", "inf"])
def test_a_hold_out_of_range_is_refused(patched, hold):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    with pytest.raises(SystemExit) as exit_:
        run_tuner(clock, "--step-response", "--hold", hold)

    assert exit_.value.code == 2
    assert bus.packets() == []


def test_stop_fails_on_an_unacknowledged_stop(patched, capsys):
    clock = Clock()
    patched(stepped_robot(clock, front_cls=NeverAcksStops))

    assert run_tuner(clock, "--stop") == 1

    assert "FAILED, unacknowledged" in capsys.readouterr().out


@pytest.mark.parametrize("motor, payload", [("m1", (535, 0)), ("m2", (0, 535))])
def test_one_motor_is_stepped_and_the_other_held_at_0(patched, capsys, motor, payload):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--controller", "front", "--motor", motor,
                     "--hold", HOLD) == 0

    drives = {struct.unpack(">ii", data[2:10]) for _, data in bus.packets() if data[1] == 37}
    assert drives == {payload}
    found = reports(capsys.readouterr().out)
    assert sorted(found) == [("Front", motor.upper())] and passes(found[("Front", motor.upper())])


@pytest.mark.parametrize("hold", ["1", "30"])
def test_the_hold_range_is_inclusive(patched, hold):
    clock = Clock()
    patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--controller", "front", "--hold", hold) == 3


def test_a_stop_ignores_the_step_options(patched):
    """An emergency stop is never refused over the step options, which it does not use."""
    clock = Clock()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--stop", "--hold", "0.5", "--speed", "0", "--settle", "-1") == 0

    assert {0, 4} <= set(commands(bus, FRONT_PORT)) and {0, 4} <= set(commands(bus, REAR_PORT))


def test_stop_fails_when_a_controller_is_missing(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, present=("rear",)))

    assert run_tuner(clock, "--stop") == 1

    assert {0, 4} <= set(commands(bus, REAR_PORT)), "the one found is still stopped"


# Commands 18 and 19 read a filtered speed: on the bench it decays after a stop with a time
# constant of about 0.1 s, while the count has stopped.
READING_LAG_S = 0.1


class LaggedReading(Motor):
    """A wheel whose reported speed follows its speed through a first-order lag of
    READING_LAG_S."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.reading = 0.0

    def advance(self, now, step=0.001):
        if self.t is None:
            super().advance(now, step)
        while self.t < now:
            start = self.t
            super().advance(min(now, start + step), step)
            self.reading += (self.speed - self.reading) * -math.expm1(
                (start - self.t) / READING_LAG_S
            )


class ReadsSpeeds(LeftTurning):
    """Commands 18 and 19 read each motor's speed, as a LaggedReading reports it."""

    def respond(self, packet):
        reply = super().respond(packet)
        if packet[1] in READ_SPEEDS:
            motor = self.motors[1 if packet[1] == 18 else 2]
            payload = struct.pack(">i", round(getattr(motor, "reading", motor.speed))) + b"\x00"
            reply = payload + struct.pack(">H", _crc16(packet[:2] + payload))
        return reply


def turning(motor=None):
    """A wheel turning at the default step when the tool starts, as command 37 left it."""
    motor = motor or Motor()
    motor.targets = [(-math.inf, tuner.STEP_VELOCITY_TICKS)]
    motor.speed = float(tuner.STEP_VELOCITY_TICKS)
    if isinstance(motor, LaggedReading):
        motor.reading = motor.speed
    motor.driven = True
    return motor


def stop_line(out, controller="Front"):
    return next(line for line in out.splitlines() if f"Stopping {controller}" in line)


def test_stop_passes_a_stopped_wheel_whose_reported_speed_lags(patched, capsys):
    """--stop judges the count, as the check after a step does: wheels the stop has stopped
    pass while their reported speed, which lags them, reads over the count's limit."""
    clock = Clock()
    wheels = (turning(LaggedReading()), turning(LaggedReading()))
    patched(stepped_robot(clock, front=wheels, front_cls=ReadsSpeeds))

    assert run_tuner(clock, "--stop", "--controller", "front") == 0

    line = stop_line(capsys.readouterr().out)
    assert "OK (reported speeds" in line
    reported = [int(speed) for speed in re.findall(r"M[12]=(-?\d+)", line)]
    assert len(reported) == 2, line
    assert all(speed * tuner.STILL_CHECK_S > tuner.REST_TICKS for speed in reported), line


@pytest.mark.parametrize("ticks, fails", [(1, True), (-1, False)])
def test_stop_fails_a_wheel_it_leaves_turning(patched, capsys, ticks, fails):
    """Wheels the stop leaves turning a tick over REST_TICKS in STILL_CHECK_S fail --stop, and the
    line says why; wheels a tick under pass. Both hold at every phase of the count, here 25
    phases a 25th of a tick apart."""
    phases = 25
    turning_after = f"FAILED, M1, M2 turning {tuner.STILL_CHECK_S:g} s after the stop"
    for phase in range(phases):
        clock = Clock()
        wheels = (turning(), turning())
        for wheel in wheels:
            wheel.count = phase / phases
        bus = patched(stepped_robot(clock, front=wheels, front_cls=ReadsSpeeds))
        bus.devices[FRONT_PORT].rate = (tuner.REST_TICKS + ticks) / tuner.STILL_CHECK_S

        assert run_tuner(clock, "--stop", "--controller", "front") == (1 if fails else 0), phase

        assert (turning_after in stop_line(capsys.readouterr().out)) == fails, phase


@pytest.mark.parametrize("early", [True, False])
def test_stop_checks_a_settle_after_the_stop(patched, capsys, early):
    """--stop's check starts STILL_CHECK_S after the stop, as the step's does: wheels the stop
    halts at once, half a cycle before a settle, pass; wheels halted half a cycle after it
    fail."""
    halt = tuner.SETTLE_TIME_S + (-1 if early else 1) * tuner.SAMPLE_PERIOD_S / 2
    clock = Clock()
    patched(stepped_robot(clock, front=(turning(Halting(halt)), turning(Halting(halt)))))

    assert run_tuner(clock, "--stop", "--controller", "front") == (0 if early else 1)

    turning_after = f"FAILED, M1, M2 turning {tuner.STILL_CHECK_S:g} s after the stop"
    assert (turning_after in stop_line(capsys.readouterr().out)) != early


@pytest.mark.parametrize("left", ["Front", "Rear"])
def test_stop_stops_every_controller_before_it_checks_any(patched, capsys, left):
    """An emergency stop: the rear's first duty-0 stop comes within the front's own stops of
    the front's first, not after the front's check. The check then runs on both at once,
    each window from STILL_CHECK_S after its controller's last stop, and fails wheels left
    turning a tick over REST_TICKS in STILL_CHECK_S and passes stopped ones, on either
    controller."""
    clock = Clock()
    bus = patched(stepped_robot(
        clock, front=(turning(), turning()), rear=(turning(), turning()),
        front_cls=LeftTurning, rear_cls=LeftTurning,
    ))
    claws = {"Front": bus.devices[FRONT_PORT], "Rear": bus.devices[REAR_PORT]}
    claws[left].rate = (tuner.REST_TICKS + 1) / tuner.STILL_CHECK_S

    assert run_tuner(clock, "--stop") == 1

    first = {
        name: next(t for t, c in claw.log if c in DUTY_FORWARD) for name, claw in claws.items()
    }
    # The open's stop and --stop's, each motor's tried up to STOP_TRIES times.
    front_stops = 2 * 2 * tuner.STOP_TRIES
    assert first["Rear"] - first["Front"] <= front_stops * SteppedRoboClaw.WIRE_S
    for name, claw in claws.items():
        read = next(t for t, c in claw.log if c in READ_COUNTS)
        last_stop = max(t for t, c in claw.log if c in DUTY_FORWARD and t < read)
        assert read - last_stop >= tuner.STILL_CHECK_S, name
        assert read - last_stop < tuner.STILL_CHECK_S + front_stops * SteppedRoboClaw.WIRE_S, name
    out = capsys.readouterr().out
    turning_after = f"FAILED, M1, M2 turning {tuner.STILL_CHECK_S:g} s after the stop"
    for name in claws:
        line = stop_line(out, name)
        assert (turning_after in line) == (name == left), line
        assert ("OK (reported speeds" in line) == (name != left), line


class Unplugging(FakeBus):
    """The robot's serial devices, with the front controller unplugged once detection has probed
    it: its port's next open fails."""

    def serial(self, *args, **kwargs):
        import serial

        port = kwargs.get("port", args[0] if args else None)
        if port == FRONT_PORT and port in self.opens:
            raise serial.SerialException(f"could not open port {port}")
        return super().serial(*args, **kwargs)


@pytest.mark.parametrize("where", ["open", "check"])
def test_an_error_on_one_controller_holds_back_no_other_stop(patched, capsys, where):
    """The front controller's port fails to open, or its counts go unread in the check: the
    rear is stopped and checked all the same, and the run fails on the front's error."""
    clock = Clock()
    robot = stepped_robot(clock, rear=(turning(), turning()),
                          front_cls=NoCounts if where == "check" else SteppedRoboClaw)
    bus = patched(Unplugging(robot.devices) if where == "open" else robot)

    assert run_tuner(clock, "--stop") == 1

    sent = commands(bus, REAR_PORT)
    assert DUTY_FORWARD <= set(sent) and READ_COUNTS <= set(sent)
    out = capsys.readouterr().out
    assert "ERROR: " in stop_line(out, "Front")
    assert "OK (reported speeds" in stop_line(out, "Rear")


class SignalsInItsStop(SteppedRoboClaw):
    """Sends *signals* to this process as its first duty-0 stop is applied."""

    signals = ()

    def stop(self, now, channel):
        signals, self.signals = self.signals, ()
        for signum in signals:
            os.kill(os.getpid(), signum)
        return super().stop(now, channel)


class SignalsAsItOpens(FakeBus):
    """The robot's serial devices, sending *signals* to this process as *port* is opened once
    *when* holds of the bus: detection's probes open each port first."""

    port, signals = FRONT_PORT, ()

    def when(self):
        return True

    def serial(self, *args, **kwargs):
        port = kwargs.get("port", args[0] if args else None)
        if port == self.port and self.signals and self.when():
            signals, self.signals = self.signals, ()
            for signum in signals:
                os.kill(os.getpid(), signum)
        return super().serial(*args, **kwargs)


# The opens a signal comes in: detection's probe of the front, the front's open for its stop,
# and the rear's open once the front has its stops.
OPENS = {
    "detection": (FRONT_PORT, lambda bus: FRONT_PORT not in bus.opens),
    "open": (FRONT_PORT, lambda bus: FRONT_PORT in bus.opens),
    "rear_open": (REAR_PORT, lambda bus: DUTY_FORWARD <= set(commands(bus, FRONT_PORT))),
}


def signalled_stop(clock, during, signals):
    """The robot with every wheel turning, on a bus that sends *signals* to this process
    *during* an open (OPENS) or the front's first stop ("stop")."""
    robot = stepped_robot(clock, front=(turning(), turning()), rear=(turning(), turning()),
                          front_cls=SignalsInItsStop)
    bus = SignalsAsItOpens(robot.devices)
    if during == "stop":
        bus.devices[FRONT_PORT].signals = signals
    else:
        bus.port, when = OPENS[during]
        bus.when, bus.signals = functools.partial(when, bus), signals
    return bus


def run_stop(clock):
    """main() with --stop on the simulated clock, under the process's own signal handlers."""
    with patch.object(sys, "argv", ["tune_pid.py", "--stop"]), patch.object(
        tuner, "time", clock
    ), patch.object(tuner, "_detected_ports", None):
        return tuner.main()


@pytest.mark.parametrize("sig", HELD)
@pytest.mark.parametrize("during", ["detection", "open", "stop", "rear_open"])
def test_an_interrupt_during_the_stops_holds_back_no_controllers_stop(
    patched, capsys, handlers_kept, during, sig
):
    """--stop sends every detected controller its stop before it acts on a Ctrl-C, Ctrl-Z,
    SIGTERM or SIGHUP: one during detection, as the front's port opens, during the front's first
    stop, or as the rear's port opens once the front has its stops, leaves both stopped, and the
    run exits 130. The signal is real, sent to this process, and main()'s handlers are the ones
    handlers_kept installs."""
    clock = Clock()
    bus = patched(signalled_stop(clock, during, (sig,)))

    assert run_stop(clock) == tuner.EXIT_INTERRUPTED

    for port in (FRONT_PORT, REAR_PORT):
        assert DUTY_FORWARD <= set(commands(bus, port)), port
    assert "Interrupted: stopping the motors." in capsys.readouterr().out
    assert not bus.signals and not bus.devices[FRONT_PORT].signals, "the signal was sent"


@pytest.mark.parametrize("during", ["open", "stop"])
def test_a_second_interrupt_abandons_that_controller_alone(
    patched, capsys, handlers_kept, during
):
    """A second signal raises at once, as in hard_stop, and abandons only the controller it
    comes on: as the front's port opens, the front's open (no stop reaches it); in the front's
    first stop, its other motor's stop, which main()'s stop on the way out then sends, since
    the front was opened. The rear is stopped either way, and the run exits 130."""
    clock = Clock()
    bus = patched(signalled_stop(clock, during, (signal.SIGINT, signal.SIGINT)))

    assert run_stop(clock) == tuner.EXIT_INTERRUPTED

    assert DUTY_FORWARD <= set(commands(bus, REAR_PORT))
    front = set(commands(bus, FRONT_PORT))
    assert DUTY_FORWARD <= front if during == "stop" else not DUTY_FORWARD & front, front
    assert "Interrupted: stopping the motors." in capsys.readouterr().out


def test_a_second_interrupt_in_a_later_stop_is_held_to_its_end(
    patched, capsys, handlers_kept
):
    """A second signal in a stop the first did not come in meets that stop's own hold, as in
    hard_stop, and is held to its end: one during detection, then one as the front's first stop
    is applied, leave that stop sent whole, M2's right after M1's, and the rear stopped, and the
    run exits 130."""
    clock = Clock()
    bus = patched(signalled_stop(clock, "detection", (signal.SIGINT,)))
    bus.devices[FRONT_PORT].signals = (signal.SIGINT,)

    assert run_stop(clock) == tuner.EXIT_INTERRUPTED

    front = commands(bus, FRONT_PORT)
    assert front[front.index(0) + 1] == 4, front
    assert DUTY_FORWARD <= set(commands(bus, REAR_PORT))
    assert not bus.signals and not bus.devices[FRONT_PORT].signals, "both signals were sent"


# ---------------------------------------------------------------------------
# The step's options, the constants behind them, and the box's verdict
# ---------------------------------------------------------------------------


def test_the_options_default_to_the_constants():
    """Each step option defaults to the constant that carries its reason, not to a literal."""
    with patch.object(tuner, "STEP_VELOCITY_TICKS", 534), patch.object(
        tuner, "STEP_HOLD_S", 5.5
    ), patch.object(tuner, "SETTLE_TIME_S", 0.3):
        args = tuner.build_parser().parse_args(["--tune"])

    assert (args.speed, args.hold, args.settle) == (534, 5.5, 0.3)


def test_the_silence_outlasts_every_timeout_a_step_accepts():
    """The bring-up's serial timeout is in the accepted range, and the silence after an
    unacknowledged stop is longer than any timeout in it."""
    low, high = tuner.SERIAL_TIMEOUT_RANGE_S

    assert low <= ROBOCLAW_SERIAL_TIMEOUT_S <= high < tuner.SILENT_S


def test_the_cycle_is_the_drivers():
    """The step's cycle is the driver's publish rate, which has no shared constant."""
    import yaml

    with open(os.path.join(os.path.dirname(__file__), "..", "config", "driver_params.yaml")) as f:
        rate = yaml.safe_load(f)["roboclaw_node"]["ros__parameters"]["publish_rate"]

    assert tuner.SAMPLE_PERIOD_S == pytest.approx(1.0 / rate)


@pytest.mark.parametrize("argv", [["--hold", "2"], ["--speed", "534"], ["--speed", "-535"]])
@pytest.mark.parametrize("mode", ["--step-response", "--tune"])
def test_only_the_default_step_gets_a_verdict(patched, capsys, mode, argv):
    """Off the default step the box does not judge: the tracking is printed, even for a wheel
    short of its target, and the run exits 3."""
    clock = Clock()
    patched(stepped_robot(clock, rear=(Motor(), Motor(gain=0.9))))

    assert run_tuner(clock, mode, *argv) == 3

    out = capsys.readouterr().out
    found = reports(out)
    assert all(no_verdict(lines) and not passes(lines) for lines in found.values())
    short = re.search(r"error=([+-][\d.]+)%", " ".join(found[("Rear", "M2")]))
    assert float(short.group(1)) == pytest.approx(-10.0, abs=0.3)
    assert "NO VERDICT -- the box is" in out and "has no verdict" in out


def test_a_step_off_the_default_fails_on_an_abort(patched, capsys):
    clock = Clock()
    patched(stepped_robot(clock, front=(Motor(gain=-1.0), Motor())))

    assert run_tuner(clock, "--step-response", "--hold", "2") == 1

    out = capsys.readouterr().out
    assert aborted(reports(out)[("Front", "M1")], "turns against its target")
    assert summary(out).startswith("FAIL -- the run failed above")
    assert "no verdict" not in summary(out)


def test_a_step_off_the_default_fails_on_a_wheel_turning_after_the_stop(patched, capsys):
    """The check after the stop fails any step, and the summary says the run failed, not that
    the step has no verdict."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=IgnoresStops))

    assert run_tuner(clock, "--step-response", "--hold", f"{tuner.MIN_HOLD_S:g}") == 1

    out = capsys.readouterr().out
    assert faulted(reports(out)[("Front", "M1")], "still moving after the stop")
    assert DRIVE_M1M2_SPEED not in commands(bus, REAR_PORT)
    assert summary(out).startswith("FAIL -- the run failed above")
    assert "no verdict" not in summary(out)


def test_the_box_judges_the_hold_the_gains_passed_on():
    """The integral term closes the last few percent over seconds: on the bench hold the
    front-left wheel misses the box over a 2 s hold and passes it over the 6 s one."""
    times, counts = bench_trace("all535_r1", "F1")
    error = {
        hold: tuner.tracking_speed(times, counts, tuner.TRACK_START_S, hold) / 535 - 1
        for hold in (2.0, tuner.STEP_HOLD_S)
    }

    assert error[2.0] < -tuner.TRACK_TOLERANCE < error[tuner.STEP_HOLD_S]


def test_a_dead_time_on_its_floor_is_flagged_not_recorded(capsys):
    """The bench's front-right wheel at 178 ticks/s: a fit on the 0 floor, whose split from tau
    is not measured."""
    _, out = report(trace_of(*bench_trace("all178_r1", "F2"), 178), HOLD_S, capsys)
    _, interior = report(trace_of(*bench_trace("all535_r1", "F1"), 535), HOLD_S, capsys)

    assert "(the fit's floor)" in out and "its split from tau is not measured" in out
    assert "Do not record these" in out
    assert "WARNING" not in interior


class SleepLog(Clock):
    """The simulated clock, keeping every sleep."""

    def __init__(self):
        super().__init__()
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        super().sleep(seconds)


def test_settle_is_the_wait_before_the_rest_check_not_the_still_check(patched):
    """--settle sets the wait after a stop, before each step and between controllers; the check
    that the wheels stopped keeps STILL_CHECK_S, twice per controller."""
    clock = SleepLog()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--hold", HOLD, "--settle", "0.7") == 0

    assert clock.sleeps.count(0.7) == 4
    assert clock.sleeps.count(tuner.STILL_CHECK_S) == 4
    log = bus.devices[FRONT_PORT].log
    first37 = next(k for k, (_, c) in enumerate(log) if c == DRIVE_M1M2_SPEED)
    stop = max(k for k in range(first37) if log[k][1] == 4)
    assert log[stop + 1][1] in READ_COUNTS and log[stop + 1][0] - log[stop][0] >= 0.7
    last37 = max(k for k, (_, c) in enumerate(log) if c == DRIVE_M1M2_SPEED)
    stop = next(k for k in range(last37, len(log)) if log[k][1] == 4)
    assert log[stop + 1][0] - log[stop][0] == pytest.approx(tuner.STILL_CHECK_S, abs=0.01)


@pytest.mark.parametrize("argv", [
    ["--settle", "-0.1"], ["--settle", "31"], ["--settle", "nan"], ["--settle", "inf"],
])
def test_an_option_out_of_range_is_refused(patched, argv):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    with pytest.raises(SystemExit) as exit_:
        run_tuner(clock, "--step-response", *argv)

    assert exit_.value.code == 2
    assert bus.packets() == []


@pytest.mark.parametrize("argv, rc", [
    (["--settle", "0"], 0), (["--settle", "30"], 0), (["--hold", "1"], 3), (["--hold", "30"], 3),
])
def test_an_option_at_its_bound_is_accepted(patched, argv, rc):
    """The settle is no part of the box: the default step is still judged."""
    clock = Clock()
    patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--controller", "front", *argv) == rc


def test_the_controllers_are_opened_as_the_driver_opens_them(patched):
    """The interface's own read timeout, retries and retry pause, as roboclaw_node opens it."""
    clock = Clock()
    patched(stepped_robot(clock))
    module = load_tuner()
    real, made = module.RoboClawInterface, []
    signature = inspect.signature(real)

    def spy(*args, **kwargs):
        made.append(signature.bind(*args, **kwargs).arguments)
        return real(*args, **kwargs)

    with patch.object(module, "RoboClawInterface", spy):
        assert run_tuner(clock, "--step-response", "--hold", HOLD, module=module) == 0

    tuned = {"read_timeout", "max_retries", "retry_pause"}
    assert made and not any(tuned & set(bound) for bound in made)


def test_the_help_gives_each_step_option_its_default_and_range():
    """--help renders (its %% escapes hold), with each default's reason and each bound."""
    text = " ".join(tuner.build_parser().format_help().split())

    for part in (
        "default: 535, 0.3 m/s at the wheel", "to 1045 either way", "10 % under full speed",
        "default: 6, the hold the driver's gains were chosen on", "from 1,", "to 30.",
        "default: 0.2: a lifted wheel stopped at duty 0", "Another hold gets no verdict",
        "about 1.8 m, and up to 17.6 m", "armed (0.1 to 1 s)", "±5 % of its target",
        "3 if the step was not the default one", "130 if interrupted (Ctrl-C, Ctrl-Z",
        "2 on a usage error",
    ):
        assert part in text, part


@pytest.mark.parametrize("mode", ["--tune", "--set"])
@pytest.mark.parametrize("qpps", ["0", "-1"])
def test_a_qpps_that_is_not_positive_is_refused(patched, mode, qpps):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    with pytest.raises(SystemExit) as exit_:
        run_tuner(clock, mode, "--p", "15000", "--qpps", qpps)

    assert exit_.value.code == 2
    assert bus.packets() == []


def test_a_gain_of_0_is_written_as_0(patched):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--tune", "--controller", "front", "--p", "0") == 0

    assert [p for _, p, _, _ in gains_written(bus)] == [0, 0]


def test_the_step_is_shown_as_a_share_of_the_qpps_read_back(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, present=("front",)))
    bus.devices[FRONT_PORT].gains = {1: (15000, 750, 0, 1070), 2: (15000, 750, 0, 1070)}

    assert run_tuner(clock, "--step-response", "--controller", "front") == 0

    assert "Step target: 535 ticks/s (50% of QPPS 1070)" in capsys.readouterr().out


def test_an_abort_prints_its_counts_with_verbose(patched, capsys):
    clock = Clock()
    patched(stepped_robot(clock, front=(Jamming(1.0), Motor())))

    assert run_tuner(clock, "--step-response", "-v") == 1

    lines = reports(capsys.readouterr().out)[("Front", "M1")]
    assert aborted(lines, "a jammed wheel") and "Time(ms) Count" in " ".join(" ".join(lines).split())


# ---------------------------------------------------------------------------
# A paused step, and the typed confirmation that the wheels are lifted
# ---------------------------------------------------------------------------


class Pausing(SteppedRoboClaw):
    """The host stalls for *pause* s while the *at*-th command 37 is sent, once the controller
    has heard it: 2 s into the step."""

    at, pause = 100, 1.0

    def drive(self, now, m1, m2):
        self.drives = getattr(self, "drives", 0) + 1
        if self.drives == self.at:
            self.clock.now += self.pause
        return super().drive(now, m1, m2)


def test_a_pause_past_the_serial_timeout_ends_the_step(patched, capsys):
    """The controller's timeout stopped the wheels during the pause; the step is not resumed."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=Pausing))

    assert run_tuner(clock, "--step-response") == 1

    assert commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED) == Pausing.at
    assert DRIVE_M1M2_SPEED not in commands(bus, REAR_PORT)
    out = capsys.readouterr().out
    assert aborted(reports(out)[("Front", "M1")], "the step is not resumed")
    assert "still moving" not in out


class PausedSleep(Clock):
    """The simulated clock, with the host stalled for *pause* s in the sleep after *claw*'s
    *at*-th command 37: a pause between cycles."""

    def __init__(self, at, pause):
        super().__init__()
        self.at, self.pause, self.claw = at, pause, None

    def sleep(self, seconds):
        if self.pause and [c for _, c in self.claw.log].count(DRIVE_M1M2_SPEED) == self.at:
            self.now, self.pause = self.now + self.pause, 0.0
        super().sleep(seconds)


@pytest.mark.parametrize("past_the_hold", [True, False])
def test_a_pause_between_cycles_ends_the_step(patched, capsys, past_the_hold):
    """A pause between cycles, from mid-hold, that makes the gap between them more than the
    serial timeout ends the step before another command 37, and the step is not resumed: one
    that carries the clock past the hold's end is no finished hold, and one that makes the gap
    half a packet's time over the timeout is caught too."""
    at = CYCLES // 2
    clock = PausedSleep(at, 0.0)
    bus = patched(stepped_robot(clock))
    clock.claw = bus.devices[FRONT_PORT]
    timeout, cycle = clock.claw.serial_timeout / 10, tuner.SAMPLE_PERIOD_S
    rest_of_hold = tuner.STEP_HOLD_S - at * cycle
    # The gap is the cycle the pause falls in, plus the pause.
    clock.pause = rest_of_hold + cycle if past_the_hold else timeout - cycle + clock.claw.WIRE_S / 2

    assert run_tuner(clock, "--step-response") == 1

    assert commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED) == at, "no command 37 after it"
    assert DRIVE_M1M2_SPEED not in commands(bus, REAR_PORT)
    out = capsys.readouterr().out
    found = reports(out)[("Front", "M1")]
    assert aborted(found, "s since the last cycle (or the rest check), over the")
    assert aborted(found, "the step is not resumed")
    assert aborted(found, "(the process was paused, or packets went unanswered)")
    assert "still moving" not in out


class PausedBeforeCommand(SteppedRoboClaw):
    """The host stalls for *pause* s between a cycle's clock read and its command 37 reaching
    the controller, at the *at*-th command 37."""

    at, pause = CYCLES // 2, 0.0

    def respond(self, packet):
        if packet[1] == DRIVE_M1M2_SPEED:
            self.drives = getattr(self, "drives", 0) + 1
            if self.drives == self.at:
                self.clock.now += self.pause
        return super().respond(packet)


# The ends of the accepted range, in command 15's 0.1 s steps, and the fake's own timeout.
RAW_TIMEOUTS = sorted(
    {round(10 * seconds) for seconds in tuner.SERIAL_TIMEOUT_RANGE_S}
    | {inspect.signature(SteppedRoboClaw).parameters["serial_timeout"].default}
)


@pytest.mark.parametrize("raw", RAW_TIMEOUTS)
@pytest.mark.parametrize("where", ["band_low", "band_high", "under"])
def test_a_pause_before_command_37_past_the_timeout_ends_the_step(patched, capsys, raw, where):
    """A pause between a cycle's clock read and its command 37. The controller hears nothing from
    the last cycle's last read, 3 packets into it, to the late command: a cycle less 2 packets,
    plus the pause. The next cycle's gap is the pause plus 3 packets. In the band between, the
    controller's timeout stops the wheels while that gap stays under it: the late command 37
    drives them again, and the stop follows it at once. Both edges of the band, half a packet's
    time inside it, are caught. Just under the band the controller hears the command in time,
    and the step runs to its verdict."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=PausedBeforeCommand, serial_timeout=raw))
    claw = bus.devices[FRONT_PORT]
    timeout, wire, cycle = claw.serial_timeout / 10, SteppedRoboClaw.WIRE_S, tuner.SAMPLE_PERIOD_S
    fires, next_gap_over = timeout - cycle + 2 * wire, timeout - 3 * wire
    claw.pause = {
        "band_low": fires + wire / 2,
        "band_high": next_gap_over - wire / 2,
        "under": fires - 4 * wire,
    }[where]
    in_band = where != "under"

    assert run_tuner(clock, "--step-response") == (1 if in_band else 0)

    log = claw.log
    late = [k for k, (_, c) in enumerate(log) if c == DRIVE_M1M2_SPEED][claw.at - 1]
    assert (log[late][0] - log[late - 1][0] > timeout) == in_band, "the controller's timeout"
    out = capsys.readouterr().out
    if in_band:
        after = [c for _, c in log[late + 1:]]
        assert after[:2] == [0, 4] and DRIVE_M1M2_SPEED not in after, "the stop, at once"
        found = reports(out)[("Front", "M1")]
        assert aborted(
            found, "s from the last cycle (or the rest check) to the end of this one's command 37"
        )
        assert aborted(found, "(the process was paused, or packets went unanswered)")
        assert aborted(found, "the step is not resumed")
        assert DRIVE_M1M2_SPEED not in commands(bus, REAR_PORT)
    else:
        assert all(passes(lines) for lines in reports(out).values())


# Several cycles, and under the fake controller's serial timeout.
LONG_PAUSE_S = 0.3


@pytest.mark.parametrize("where", ["sleep", "write", "sleep_then_write"])
def test_a_late_cycle_does_not_replay_the_missed_ones(patched, where):
    """A pause of several cycles under the serial timeout makes the next cycle late: in the sleep
    between cycles, or inside a command 37's write after the controller has heard it. The step
    goes on: the cycle after the late one comes one period after the late one's start, not back
    to back, and the cycles the pause took are not run. A shorter pause keeps the grid while the
    late cycle ends before the next grid point; one that starts half a cycle late and ends past
    that point, half a cycle more in its write, is followed one period after its start too."""
    cycle = tuner.SAMPLE_PERIOD_S
    sleep, write = {
        "sleep": (LONG_PAUSE_S, 0.0),
        "write": (0.0, LONG_PAUSE_S),
        "sleep_then_write": (cycle / 2, cycle / 2),
    }[where]
    clock = PausedSleep(Pausing.at, sleep)
    bus = patched(stepped_robot(clock, front_cls=Pausing))
    clock.claw = bus.devices[FRONT_PORT]
    # After a sleep's pause, the write's falls in the late cycle's command 37, the next one.
    clock.claw.at, clock.claw.pause = Pausing.at + (1 if sleep else 0), write

    assert run_tuner(clock, "--step-response") == 0

    sent = [t for t, c in clock.claw.log if c == DRIVE_M1M2_SPEED]
    gaps = [b - a for a, b in zip(sent, sent[1:])]
    late = gaps.index(max(gaps)) + 1
    assert gaps[late - 1] > max(sleep, write)
    assert gaps[late] == pytest.approx(cycle, abs=SteppedRoboClaw.WIRE_S)
    assert min(gaps) >= cycle - SteppedRoboClaw.WIRE_S, "one cycle at a time"
    missed = round((sleep + write) / cycle)
    assert abs(len(sent) - (CYCLES - missed)) <= 1, "the cycles the pause took are not run"


class WakesLate(Clock):
    """The simulated clock, waking from each sleep a packet's time late, as a host's sleep
    overshoots."""

    def sleep(self, seconds):
        super().sleep(seconds + SteppedRoboClaw.WIRE_S if seconds > 0 else seconds)


def test_a_late_wake_keeps_the_cycles_on_the_50_hz_grid(patched):
    """A sleep that overshoots does not stretch the cycle: each command 37 comes a period after
    the last, and the hold runs all its cycles."""
    clock = WakesLate()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--controller", "front") == 0

    sent = [t for t, c in bus.devices[FRONT_PORT].log if c == DRIVE_M1M2_SPEED]
    gaps = [b - a for a, b in zip(sent[1:], sent[2:])]
    assert max(abs(gap - tuner.SAMPLE_PERIOD_S) for gap in gaps) < SteppedRoboClaw.WIRE_S / 2
    assert abs(len(sent) - CYCLES) <= 1


def longest_read():
    """The longest the tries of a count read wait, through the interface as the tuner opens it:
    each try waits out the read timeout, with the retry pause between them."""
    rc = tuner.RoboClawInterface("unopened", FRONT)
    tries = 1 + rc.max_retries
    return tries * rc.read_timeout + (tries - 1) * rc.retry_pause


# The cycle whose count reads are the tracking window's first samples, by the at-th command 37.
WINDOW_OPENS = round(tuner.TRACK_START_S / tuner.SAMPLE_PERIOD_S) + 1


class PausedInRead(SteppedRoboClaw):
    """The host stalls for *pause* s inside the count read of the motor on *channel* after the
    *at*-th command 37, once the controller has answered it: the read returns, and its sample is
    timed, that much later."""

    at, pause, channel = WINDOW_OPENS, 0.0, 1

    def drive(self, now, m1, m2):
        self.drives = getattr(self, "drives", 0) + 1
        return super().drive(now, m1, m2)

    def read_count(self, packet, channel):
        reply = super().read_count(packet, channel)
        if channel == self.channel and getattr(self, "drives", 0) == self.at and self.pause:
            self.clock.now, self.pause = self.clock.now + self.pause, 0.0
        return reply


def read_taking(read):
    """How long a paused read takes: half a packet's time either side of the interface's tries,
    or of a cycle."""
    half = SteppedRoboClaw.WIRE_S / 2
    return {
        "over_the_longest": longest_read() + half,
        "under_the_longest": longest_read() - half,
        "over_a_cycle": tuner.SAMPLE_PERIOD_S + half,
        "under_a_cycle": tuner.SAMPLE_PERIOD_S - half,
    }[read]


@pytest.mark.parametrize("read, ends, channel, motor", [
    ("over_the_longest", True, 1, "both"), ("over_the_longest", True, 2, "both"),
    ("over_the_longest", True, 2, "m1"),
    ("under_the_longest", False, 1, "both"), ("under_the_longest", False, 2, "both"),
    ("over_a_cycle", False, 1, "both"), ("over_a_cycle", False, 2, "both"),
])
def test_a_pause_inside_a_count_read_past_its_tries_ends_the_step(
    patched, capsys, read, ends, channel, motor
):
    """A pause inside the tracking window's first count read of either motor, after the
    controller answered it. A read half a packet's time longer than the interface's tries can
    take ends the step, and no command 37 follows it, also in the read of a motor held at 0.
    One half a packet's time shorter, or half a packet's time over a cycle, goes on, and its
    sample opens no window: a wheel short of the box fails it by its own speed, though the pause
    would date the window's start late."""
    short = 1.2 * tuner.TRACK_TOLERANCE
    wheels = [Motor(), Motor()]
    wheels[channel - 1] = Motor(gain=1 - short)
    clock = Clock()
    bus = patched(stepped_robot(clock, front=wheels, front_cls=PausedInRead))
    claw = bus.devices[FRONT_PORT]
    claw.channel = channel
    claw.pause = read_taking(read) - claw.WIRE_S  # the read's own packet takes the rest

    assert run_tuner(clock, "--step-response", "--controller", "front", "--motor", motor) == 1

    paused = f"M{channel}"
    found = reports(capsys.readouterr().out)[("Front", "M1" if motor == "m1" else paused)]
    assert aborted(found, f"{paused}'s count read took") == ends
    assert aborted(found, "s its tries can wait (the process was paused)") == ends
    if ends:
        assert aborted(found, "the step is not resumed")
        assert commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED) == claw.at
    else:
        failed = re.search(r"FAIL -- tracks ([\d.]+)% under target", " ".join(found))
        assert failed and float(failed.group(1)) == pytest.approx(short * 100, abs=0.3), found


@pytest.mark.parametrize("channel", [1, 2])
@pytest.mark.parametrize("read, left_out", [
    ("over_a_cycle", True), ("under_the_longest", True), ("under_a_cycle", False),
])
def test_a_count_read_longer_than_a_cycle_is_left_out_of_the_fit(
    patched, capsys, read, left_out, channel
):
    """A pause inside either motor's count read in the rise, 4 cycles in, after the controller
    answered it. A read half a packet's time longer than a cycle, or within the interface's
    tries, cannot be placed within its cycle: the fit leaves its sample out, and the printed dead
    time and 63 % point stay within the fit's documented scatter (2 and 1 ms near 510 ticks/s) of
    the run without the pause. Half a packet's time under a cycle, the sample is fitted, dated
    late by the pause, and both move past it."""
    printed = []
    for pause in (0.0, read_taking(read) - SteppedRoboClaw.WIRE_S):
        clock = Clock()
        bus = patched(stepped_robot(clock, front_cls=PausedInRead))
        claw = bus.devices[FRONT_PORT]
        claw.at, claw.pause, claw.channel = 4, pause, channel

        assert run_tuner(clock, "--step-response", "--controller", "front") == 0

        printed.append(measured(reports(capsys.readouterr().out)[("Front", f"M{channel}")]))
    (dead, tau), (paused_dead, paused_tau) = printed
    assert (abs(paused_dead - dead) < 0.002) == left_out, printed
    assert (abs(paused_dead + paused_tau - dead - tau) < 0.001) == left_out, printed


class TimedReads(FakeBus):
    """The robot's serial devices, each port's short read waiting out its read timeout on the
    simulated clock, as pyserial's does."""

    def __init__(self, devices, clock):
        super().__init__(devices)
        self.clock = clock

    def serial(self, *args, **kwargs):
        port = super().serial(*args, **kwargs)
        read = port.read

        def timed(count=1):
            data = read(count)
            if len(data) < count:
                self.clock.now += kwargs["timeout"]
            return data

        port.read = timed
        return port


class RetriedRead(PausedInRead):
    """M1's count read after the *at*-th command 37 goes unanswered once, so the interface
    retries it."""

    missed = False

    def read_count(self, packet, channel):
        if channel == 1 and getattr(self, "drives", 0) == self.at and not self.missed:
            self.missed = True
            return b""
        return super().read_count(packet, channel)


def test_a_count_read_that_needed_its_retry_goes_on(patched, capsys):
    """The tracking window's first count read goes unanswered: the interface waits out its read
    timeout, pauses and tries again, which takes longer than a cycle. That is no pause: the step
    goes on, the sample left out of the tracking, and passes. M1's window opens at the cycle after
    it, and M2's at its own read in that cycle, which the retry delayed: both more than a cycle
    into the box, and the report prints them."""
    clock = Clock()
    robot = stepped_robot(clock, front_cls=RetriedRead)
    patched(TimedReads(robot.devices, clock))
    claw = robot.devices[FRONT_PORT]

    with patch("strafer_driver.roboclaw_interface.time", clock):
        assert run_tuner(clock, "--step-response", "--controller", "front") == 0

    assert claw.missed
    found = reports(capsys.readouterr().out)
    assert all(passes(lines) for lines in found.values())
    counted = re.search(r"Samples: +(\d+) \((\d+) read within a cycle, 1 left out\)",
                        " ".join(found[("Front", "M1")]))
    assert counted and int(counted.group(2)) == int(counted.group(1)) - 1
    assert not any("read within a cycle" in line for line in found[("Front", "M2")])
    for motor in ("M1", "M2"):
        opened = re.search(r"over \[([\d.]+), [\d.]+\] s", " ".join(found[("Front", motor)]))
        assert opened and float(opened.group(1)) > tuner.TRACK_START_S + tuner.SAMPLE_PERIOD_S


@pytest.mark.parametrize("left_out, window", [
    (None, "[0.6, 6]"), (0.6, "[0.61, 6.00]"), (6.0, "[0.60, 5.99]"), (3.0, "[0.6, 6]"),
])
def test_a_sample_left_out_at_a_window_end_prints_the_window_the_tracking_used(
    capsys, left_out, window
):
    """A sample left out at either end of the tracking window moves the window the tracking
    measured over to the next sample in, here half a cycle on: the report prints that window.
    One left out inside it, or none, leaves the window's ends, and the report prints the box's."""
    times = [round(k * tuner.SAMPLE_PERIOD_S / 2, 6) for k in range(601)]
    trace = trace_of(times, [535 * t for t in times], tuner.STEP_VELOCITY_TICKS)
    if left_out is not None:
        trace.slow = {times.index(left_out)}

    passed, out = report(trace, tuner.STEP_HOLD_S, capsys)

    assert passed
    assert f"Tracking:     535 ticks/s over {window} s" in out, out
    assert ("(600 read within a cycle, 1 left out)" in out) == (left_out is not None)


def window_trace(where, shift, left_out):
    """A wheel at the default step, read every cycle over the hold, whose tracking window's
    *where* end ("start" or "end") sits *shift* s inside the box's: the samples between are
    missing, or (*left_out*) there but read over a cycle."""
    cycle, start, hold = tuner.SAMPLE_PERIOD_S, tuner.TRACK_START_S, tuner.STEP_HOLD_S
    grid = [round(k * cycle, 6) for k in range(round(hold / cycle) + 1)]
    edge = start + shift if where == "start" else hold - shift
    gap = [t for t in grid if (start <= t < edge if where == "start" else edge < t <= hold)]
    times = sorted(set(grid) - set(gap) | {edge} | (set(gap) if left_out else set()))
    speed = tuner.STEP_VELOCITY_TICKS
    trace = trace_of(times, [speed * t for t in times], speed)
    trace.slow = {times.index(t) for t in gap} if left_out else set()
    trace.edge_s = tuner.SAMPLE_PERIOD_S + longest_read()
    return trace, edge


@pytest.mark.parametrize("left_out", [False, True])
@pytest.mark.parametrize("where", ["start", "end"])
@pytest.mark.parametrize("past", [True, False])
def test_a_window_an_end_more_than_a_cycle_and_a_read_inside_the_box_fails(
    capsys, where, past, left_out
):
    """Each end of the window the tracking uses may sit at most a cycle and one read's tries
    inside the box's. Half a cycle farther in, at either end, the motor fails for too few
    samples, with the samples left out or none come there as the reason; half a cycle nearer, it
    is measured over that window, which the report prints."""
    edge_s = tuner.SAMPLE_PERIOD_S + longest_read()
    shift = edge_s + (1 if past else -1) * tuner.SAMPLE_PERIOD_S / 2
    trace, edge = window_trace(where, shift, left_out)

    passed, out = report(trace, tuner.STEP_HOLD_S, capsys)

    assert passed != past, out
    if past:
        why = (
            "those whose read took longer than a cycle are left out" if left_out
            else "no sample came there (a pause, or reads lost)"
        )
        near = "FAIL -- too few samples near the ends"
        failed = [line for line in out.splitlines() if near in line]
        assert failed and why in failed[0] and "Tracking:" not in out, out
    else:
        window = f"[{edge:.2f}, 6.00]" if where == "start" else f"[0.60, {edge:.2f}]"
        assert f"Tracking:     535 ticks/s over {window} s" in out, out


@pytest.mark.parametrize("pause, printed", [
    (tuner.SAMPLE_PERIOD_S / 2, "[0.6, 6]"), (1.5 * tuner.SAMPLE_PERIOD_S, "[0.63, 5.99]"),
])
def test_the_window_used_is_printed_once_an_end_sits_a_cycle_inside_the_box(
    patched, capsys, pause, printed
):
    """With nothing left out, a pause across the box's start that moves the window's first sample
    half a cycle in leaves the box printed; one that moves it a cycle and a half in prints the
    window the tracking used."""
    clock = PausedSleep(round(tuner.TRACK_START_S / tuner.SAMPLE_PERIOD_S), pause)
    bus = patched(stepped_robot(clock, present=("front",)))
    clock.claw = bus.devices[FRONT_PORT]

    assert run_tuner(clock, "--step-response", "--controller", "front") == 0

    for lines in reports(capsys.readouterr().out).values():
        assert any(f"over {printed} s" in line for line in lines), lines


def test_a_fault_is_printed_with_a_window_short_of_the_box(capsys):
    """A fault from the check after the stop is printed on the fail for too few samples near the
    box's ends too."""
    trace, _ = window_trace("start", 0.3, True)
    trace.fault = "the stop could not be checked"

    passed, out = report(trace, tuner.STEP_HOLD_S, capsys)

    assert not passed and "inside the box's" in out and trace.fault in out, out


class MostReadsRetried(SteppedRoboClaw):
    """Every count read of M1 in the step needs its retry, but those after the *kept* commands
    37."""

    kept = ()

    def drive(self, now, m1, m2):
        self.drives, self.tried = getattr(self, "drives", 0) + 1, False
        return super().drive(now, m1, m2)

    def read_count(self, packet, channel):
        if channel == 1 and getattr(self, "drives", 0) not in (0, *self.kept) and not self.tried:
            self.tried = True
            return b""
        return super().read_count(packet, channel)


def test_samples_left_out_that_leave_a_short_window_fail_the_motor(patched, capsys):
    """Every M1 count read needs its retry but two adjacent ones, which are all the tracking
    could use: a cycle apart, where a wheel 5.2 % under its target reads +2.8 %. Their window's
    ends sit far inside the box's, so the motor fails for too few samples, with the samples left
    out as the reason, and the run exits 1."""
    clock = Clock()
    robot = stepped_robot(clock, front=(Motor(gain=0.948), Motor()), front_cls=MostReadsRetried)
    robot.devices[FRONT_PORT].kept = (11, 12)
    patched(TimedReads(robot.devices, clock))

    with patch("strafer_driver.roboclaw_interface.time", clock):
        assert run_tuner(clock, "--step-response", "--controller", "front", "--motor", "m1") == 1

    text = " ".join(reports(capsys.readouterr().out)[("Front", "M1")])
    assert "(2 read within a cycle," in text, text
    assert "FAIL -- too few samples near the ends of [0.6, 6] s" in text, text
    assert "those whose read took longer than a cycle are left out" in text, text


# A cycle and a read's tries: how far an end of the window the tracking uses may sit inside the
# box's.
EDGE = tuner.SAMPLE_PERIOD_S + longest_read()


@pytest.mark.parametrize("where, at, pause, passed", [
    ("start", 30, EDGE - tuner.SAMPLE_PERIOD_S / 2, True),
    ("start", 30, EDGE + tuner.SAMPLE_PERIOD_S / 4, False),
    ("start", 30, EDGE + tuner.SAMPLE_PERIOD_S / 2, False),
    ("start", 30, 0.4, False),
    ("end", 290, 0.3, True), ("end", 289, 0.3, False), ("end", 280, 0.42, False),
])
def test_a_pause_across_the_box_start_or_the_hold_end_moves_or_fails_the_window(
    patched, capsys, where, at, pause, passed
):
    """A pause between cycles under the serial timeout, across the box's start or across the
    hold's end, moves that end of the window the tracking uses: after the 30th command 37, by the
    pause (and its cycle's reads, a millisecond or two); the hold, ending at the first cycle past
    it, after the 290th's reads by 0.22 s and the 289th's by 0.24. Up to a cycle and a read's
    tries inside the box's (6 to 9 ms within it here), the report prints the window used; from
    6 ms past it, the motor fails for too few samples, none having come there."""
    clock = PausedSleep(at, pause)
    bus = patched(stepped_robot(clock, present=("front",)))
    clock.claw = bus.devices[FRONT_PORT]

    assert run_tuner(clock, "--step-response", "--controller", "front") == (0 if passed else 1)

    for lines in reports(capsys.readouterr().out).values():
        text = " ".join(lines)
        if passed:
            used = re.search(r"over \[([\d.]+), ([\d.]+)\] s", text)
            moved = float(used.group(1)) - 0.6 if where == "start" else 6.0 - float(used.group(2))
            assert used and tuner.SAMPLE_PERIOD_S < moved < tuner.SAMPLE_PERIOD_S + longest_read()
            assert passes(lines), text
        else:
            assert "no sample came there (a pause, or reads lost)" in text, text


class EveryReadRetried(SteppedRoboClaw):
    """Every count read's first try goes unanswered, so the interface retries each one."""

    def read_count(self, packet, channel):
        if self.count_reads % 2:
            return b""
        return super().read_count(packet, channel)


def test_a_link_whose_every_read_needs_its_retry_fails_on_the_samples_left_out(patched, capsys):
    """Every count read takes its retry, longer than a cycle, so no sample can be placed within
    its cycle: each motor's report says none was read within one, and its fail gives the samples
    left out as the reason there are too few."""
    clock = Clock()
    robot = stepped_robot(clock, front_cls=EveryReadRetried)
    patched(TimedReads(robot.devices, clock))

    with patch("strafer_driver.roboclaw_interface.time", clock):
        assert run_tuner(clock, "--step-response", "--controller", "front") == 1

    found = reports(capsys.readouterr().out)
    for motor in ("M1", "M2"):
        text = " ".join(found[("Front", motor)])
        counted = re.search(r"Samples: +(\d+) \(0 read within a cycle, (\d+) left out\)", text)
        assert counted and counted.group(1) == counted.group(2), text
        assert "FAIL -- too few samples in" in text, text
        assert "to measure tracking: those whose read took longer than a cycle are left out" in text


def test_the_rules_take_the_samples_left_out(patched, capsys):
    """Every count read needs its retry, so every sample is left out of the tracking and the fit,
    and the abort rules still take them: a reversed wheel is stopped by the against rule."""
    clock = Clock()
    robot = stepped_robot(clock, front=(Motor(gain=-1.0), Motor()), front_cls=EveryReadRetried)
    patched(TimedReads(robot.devices, clock))

    with patch("strafer_driver.roboclaw_interface.time", clock):
        assert run_tuner(clock, "--step-response", "--controller", "front") == 1

    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "turns against its target")


def test_the_read_bound_is_the_opened_interfaces(patched, capsys):
    """The read's bound, and the window's, come from the interface as the tool opens it: with a
    0.12 s read timeout and a 0.03 s retry pause, whose bound is 0.27 s, a read paused to 0.26 s
    in the window's first cycle, past what the defaults allow, or either of the two alone, is
    left out rather than ending the step, and the window it moves passes."""
    clock = Clock()
    bus = patched(stepped_robot(clock, present=("front",), front_cls=PausedInRead))
    claw = bus.devices[FRONT_PORT]
    claw.pause = 0.26 - claw.WIRE_S
    module = load_tuner()
    wide = functools.partial(module.RoboClawInterface, read_timeout=0.12, retry_pause=0.03)

    with patch.object(module, "RoboClawInterface", wide):
        assert run_tuner(clock, "--step-response", "--controller", "front", module=module) == 0

    lines = reports(capsys.readouterr().out)[("Front", "M1")]
    assert "1 left out)" in " ".join(lines) and passes(lines), lines


@pytest.mark.parametrize("samples, why", [(3, True), (2, False)])
def test_too_few_samples_give_the_slow_reads_as_the_reason_only_when_they_are(
    capsys, samples, why
):
    """Too few samples in the window to measure the tracking: the fail gives the samples whose
    read took longer than a cycle as the reason when the window would hold enough with them (the
    slow one opens it), and not when it would not (the slow one is before it)."""
    times = [0.0, tuner.TRACK_START_S, tuner.STEP_HOLD_S][:samples]
    trace = trace_of(times, range(samples), tuner.STEP_VELOCITY_TICKS)
    trace.slow = {samples - 2}

    passed, out = report(trace, tuner.STEP_HOLD_S, capsys)

    assert not passed and "FAIL -- too few samples in" in out
    assert ("those whose read took longer than a cycle are left out" in out) == why


class PausedAfterRest(Clock):
    """The simulated clock, with the host stalled for *pause* s in the sleep after *claw*'s last
    rest read, before its first command 37: the fake's rest reads come a cycle and two packets
    apart."""

    def __init__(self, pause):
        super().__init__()
        self.pause, self.claw = pause, None

    def sleep(self, seconds):
        rounds = math.ceil(tuner.PRE_S / (tuner.SAMPLE_PERIOD_S + 2 * self.claw.WIRE_S))
        if self.pause and self.claw.count_reads == 2 * rounds:
            self.now, self.pause = self.now + self.pause, 0.0
        super().sleep(seconds)


@pytest.mark.parametrize("where", ["after_the_rest_check", "before_command_37"])
def test_a_pause_before_the_first_command_37_past_the_timeout_ends_the_step(
    patched, capsys, where
):
    """The controller last hears a packet in the rest check's last read, so the first cycle's
    checks count from that read's start, as a later cycle's count from the last one's. A pause
    under the serial timeout, after the rest check or before the first command 37 reaches the
    controller, that leaves the controller silent past its timeout ends the step: before any
    command 37, or with the stop at once after the first."""
    clock = PausedAfterRest(0.0)
    bus = patched(stepped_robot(clock, front_cls=PausedBeforeCommand))
    claw = clock.claw = bus.devices[FRONT_PORT]
    timeout, cycle, wire = claw.serial_timeout / 10, tuner.SAMPLE_PERIOD_S, claw.WIRE_S
    pause = timeout - cycle + wire / 2
    if where == "after_the_rest_check":
        clock.pause = pause
    else:
        claw.at, claw.pause = 1, pause

    assert run_tuner(clock, "--step-response", "--controller", "front") == 1

    sent = [c for _, c in claw.log]
    found = reports(capsys.readouterr().out)[("Front", "M1")]
    assert aborted(found, "(the process was paused, or packets went unanswered)")
    assert aborted(found, "the step is not resumed")
    if where == "after_the_rest_check":
        assert DRIVE_M1M2_SPEED not in sent
        assert aborted(found, "s since the last cycle (or the rest check), over the")
    else:
        assert sent.count(DRIVE_M1M2_SPEED) == 1
        after = sent[sent.index(DRIVE_M1M2_SPEED) + 1:]
        assert after[:2] == [0, 4], "the stop, at once"


def test_a_pause_before_the_first_command_37_is_no_dead_time(patched, capsys):
    """The step is timed from its first command 37's return: a pause before the controller hears
    it, shorter than the fit's dead-time bound, which would not flag it, leaves the printed dead
    time, tau and 63 % point as they are without it."""
    printed = []
    for pause in (0.0, tuner.DEAD_TIME_MAX_S / 2):
        clock = Clock()
        bus = patched(stepped_robot(clock, front_cls=PausedBeforeCommand))
        claw = bus.devices[FRONT_PORT]
        claw.at, claw.pause = 1, pause

        assert run_tuner(clock, "--step-response", "--controller", "front") == 0

        found = reports(capsys.readouterr().out)
        printed.append([line for lines in found.values() for line in lines
                        if line.startswith("Dead time:")])
    assert len(printed[0]) == 2 and printed[1] == printed[0]


def test_a_pause_after_the_first_command_37_past_the_dead_time_flags_the_fit(patched, capsys):
    """A pause once the controller has heard the first command 37 times the step late by it: past
    the wheel's dead time the fit sits on its 0 floor, and is flagged not to be recorded."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=Pausing))
    claw = bus.devices[FRONT_PORT]
    claw.at, claw.pause = 1, 2 * Motor().dead

    assert run_tuner(clock, "--step-response", "--controller", "front") == 0

    for lines in reports(capsys.readouterr().out).values():
        assert "(the fit's floor)" in " ".join(lines) and "Do not record these" in " ".join(lines)


class FirstUnacked(SteppedRoboClaw):
    """The *at*-th command 37 goes unacknowledged: *loses* "command", and the controller never
    hears it; "acknowledgement", and it is applied and its acknowledgement lost; None, neither."""

    at, loses = 1, None

    def drive(self, now, m1, m2):
        self.drives = getattr(self, "drives", 0) + 1
        if self.drives == self.at and self.loses == "command":
            return b""
        reply = super().drive(now, m1, m2)
        return b"" if self.drives == self.at and self.loses else reply


UNACKED_FIRST = (
    "WARNING: The first command 37 went unacknowledged, so when the step started is not known. "
    "Do not record these."
)


@pytest.mark.parametrize("loses, at", [
    ("command", 1), ("acknowledgement", 1), (None, 1), ("acknowledgement", 2),
])
def test_an_unacknowledged_first_command_37_flags_the_fit(patched, capsys, loses, at):
    """The step is timed from its first command 37's return, which says nothing of when the
    controller heard it if it went unacknowledged: lost, the controller heard the next one, and
    the dead time reads a cycle long; its acknowledgement lost, the return came the read timeout
    late. Either way the fit is flagged not to be recorded, and the tracking passes. Acknowledged,
    or with only the second command 37's acknowledgement lost, it is not flagged."""
    clock = Clock()
    robot = stepped_robot(clock, front_cls=FirstUnacked)
    claw = robot.devices[FRONT_PORT]
    claw.loses, claw.at = loses, at
    patched(TimedReads(robot.devices, clock))

    assert run_tuner(clock, "--step-response", "--controller", "front") == 0

    for lines in reports(capsys.readouterr().out).values():
        text = " ".join(lines)
        assert passes(lines), text
        assert (UNACKED_FIRST in text) == (loses is not None and at == 1), text


def test_no_step_without_a_terminal_to_confirm_it(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--tune", confirmed=False) == 1

    assert bus.packets() == []
    assert "about 1.80 m of travel" in capsys.readouterr().out


@pytest.mark.parametrize("answer, rc", [("yes", 0), ("y", 1), (EOFError, 1)])
def test_only_a_typed_yes_steps(patched, answer, rc):
    clock = Clock()
    bus = patched(stepped_robot(clock))
    module = load_tuner()
    reply = {"side_effect": answer} if answer is EOFError else {"return_value": answer}

    with patch.object(module, "sys") as fake_sys, patch.object(module, "termios"), patch(
        "builtins.input", **reply
    ):
        fake_sys.stdin.isatty.return_value = True
        assert run_tuner(clock, "--step-response", module=module, confirmed=False) == rc

    assert (DRIVE_M1M2_SPEED in [data[1] for _, data in bus.packets()]) == (rc == 0)


class LosesCountsInTheCheck(SteppedRoboClaw):
    """Answers count reads until the check after its last stop has read once, then none."""

    def respond(self, packet):
        reply = super().respond(packet)
        if packet[1] in DUTY_FORWARD and self.motors[1].driven:
            self.stopped_at = self.clock.now
        return reply

    def read_count(self, packet, channel):
        stopped_at = getattr(self, "stopped_at", None)
        if stopped_at is not None and self.clock.now > stopped_at + 1.5 * tuner.STILL_CHECK_S:
            return b""
        return super().read_count(packet, channel)


@pytest.mark.parametrize("mode", ["--step-response", "--stop"])
def test_counts_lost_in_the_checks_second_read_fail_it(patched, capsys, mode):
    clock = Clock()
    wheels = (turning(), turning()) if mode == "--stop" else (Motor(), Motor())
    patched(stepped_robot(clock, front=wheels, front_cls=LosesCountsInTheCheck))

    assert run_tuner(clock, mode, "--controller", "front") == 1

    out = capsys.readouterr().out
    if mode == "--stop":
        assert "ERROR: " in stop_line(out)
    else:
        assert faulted(reports(out)[("Front", "M1")], "the stop could not be checked")
