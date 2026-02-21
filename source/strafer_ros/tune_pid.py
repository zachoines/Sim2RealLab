#!/usr/bin/env python3
"""PID tuning script for RoboClaw velocity controllers.

Programmatically sets PID/QPPS values and runs step-response tests to
measure the actual time constant, targeting a ~50 ms first-order response
to match the Isaac Lab simulation motor model.

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
    and --motor m1|m2|both (default: both)
"""

from __future__ import annotations
from strafer_shared.constants import (
    ENCODER_PPR_OUTPUT_SHAFT,
    MAX_WHEEL_ANGULAR_VEL,
    MOTOR_MAX_RPM,
    ROBOCLAW_BAUD_RATE,
    ROBOCLAW_FRONT_ADDRESS,
    ROBOCLAW_REAR_ADDRESS,
)
from strafer_driver.roboclaw_interface import RoboClawInterface

import argparse
import atexit
import math
import sys
import time

# Add parent so we can import the driver package directly
sys.path.insert(0, "strafer_driver")


# Global list of open controllers for cleanup on exit
_open_controllers: list[RoboClawInterface] = []

# ---------------------------------------------------------------------------
# Derived constants for PID tuning
# ---------------------------------------------------------------------------

# QPPS = max ticks per second at full speed
QPPS = int(MOTOR_MAX_RPM * ENCODER_PPR_OUTPUT_SHAFT / 60.0)  # ~2796

# Target time constant from sim: 50 ms first-order system
TARGET_TAU_S = 0.05

# Step-response test parameters
STEP_VELOCITY_TICKS = QPPS // 2  # 50% of max speed for safety
SAMPLE_PERIOD_S = 0.005           # 5 ms sample period (200 Hz)
STEP_DURATION_S = 0.5             # 500 ms capture window
SETTLE_TIME_S = 0.2               # Wait after stopping before next test

# Abort step-response if speed starts >50% of target on first 5 samples
# (indicates reversed motor/encoder direction)
REVERSED_DETECT_SAMPLES = 5
REVERSED_DETECT_THRESHOLD = 0.5  # fraction of target speed

# Default starting PID gains (conservative).
# RoboClaw PID is integer fixed-point. Typical ranges:
#   P: 1000 - 65535, I: 0 - 5000, D: 0 - 10000
# These are starting values for a 312 RPM motor with ~2796 QPPS.
DEFAULT_P = 15000
DEFAULT_I = 750
DEFAULT_D = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_first_order_tau(times: list[float], speeds: list[float], target: float) -> float:
    """Estimate the first-order time constant from step-response data.

    Fits y(t) = target * (1 - exp(-t/tau)) by finding tau that minimises
    the sum of squared errors using a simple grid + refine search.

    Returns tau in seconds, or -1.0 if fitting fails.
    """
    if not times or target == 0:
        return -1.0

    best_tau = 0.01
    best_err = float("inf")

    # Coarse grid: 5 ms to 500 ms
    for tau_ms in range(5, 501):
        tau = tau_ms / 1000.0
        err = sum(
            (s - target * (1.0 - math.exp(-t / tau))) ** 2
            for t, s in zip(times, speeds)
        )
        if err < best_err:
            best_err = err
            best_tau = tau

    # Fine grid: ±5 ms around best, 0.1 ms steps
    lo = max(0.001, best_tau - 0.005)
    hi = best_tau + 0.005
    step = 0.0001
    tau = lo
    while tau <= hi:
        err = sum(
            (s - target * (1.0 - math.exp(-t / tau))) ** 2
            for t, s in zip(times, speeds)
        )
        if err < best_err:
            best_err = err
            best_tau = tau
        tau += step

    return best_tau


def hard_stop(rc: RoboClawInterface) -> None:
    """Force-stop both motors using duty-cycle commands (bypasses PID).

    This sends forward_m1(0) and forward_m2(0) which switch the RoboClaw
    out of velocity PID mode into duty-cycle mode with 0% duty = stopped.
    This is critical because drive_m*_speed(0) only sets the PID target
    to 0, and with accumulated I-term or reversed motors, the PID may
    continue driving.
    """
    try:
        rc.forward_m1(0)
        rc.forward_m2(0)
    except Exception:
        pass  # Best-effort during cleanup


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
    """Open and return a RoboClawInterface for the given port."""
    rc = RoboClawInterface(
        port=port,
        address=address,
        baud_rate=ROBOCLAW_BAUD_RATE,
        read_timeout=0.1,
        max_retries=2,
    )
    rc.open()
    hard_stop(rc)  # Ensure motors are stopped on connect
    _open_controllers.append(rc)
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


def get_controllers(which: str) -> list[tuple[str, str, int]]:
    """Return list of (label, port, address) based on --controller arg."""
    controllers = []
    if which in ("front", "both"):
        controllers.append(("Front", "/dev/roboclaw_front", ROBOCLAW_FRONT_ADDRESS))
    if which in ("rear", "both"):
        controllers.append(("Rear", "/dev/roboclaw_rear", ROBOCLAW_REAR_ADDRESS))
    return controllers


def get_motor_list(which: str) -> list[str]:
    """Return list of motor names based on --motor arg."""
    if which == "m1":
        return ["M1"]
    if which == "m2":
        return ["M2"]
    return ["M1", "M2"]


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
                f"  M1 PID: P={pid_m1['P']:>6}  I={pid_m1['I']:>6}  D={pid_m1['D']:>6}  QPPS={pid_m1['QPPS']}")
            print(
                f"  M2 PID: P={pid_m2['P']:>6}  I={pid_m2['I']:>6}  D={pid_m2['D']:>6}  QPPS={pid_m2['QPPS']}")
        except Exception as e:
            print(f"  ERROR: {e}")


def cmd_set(args: argparse.Namespace) -> None:
    """Set PID values on selected controllers/motors."""
    p = args.p
    i = args.i
    d = args.d
    qpps = args.qpps or QPPS

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
                f"  Readback M1: P={pid_m1['P']} I={pid_m1['I']} D={pid_m1['D']} QPPS={pid_m1['QPPS']}")
            print(
                f"  Readback M2: P={pid_m2['P']} I={pid_m2['I']} D={pid_m2['D']} QPPS={pid_m2['QPPS']}")
            rc.close()
        except Exception as e:
            print(f"  ERROR: {e}")


def run_step_response(
    rc: RoboClawInterface,
    motor: str,
    speed_ticks: int,
) -> tuple[list[float], list[float], float, bool]:
    """Run a step-response test on one motor.

    Returns:
        (times, speeds, steady_state_speed, reversed_detected)
        times: seconds from step onset
        speeds: ticks/sec at each sample (absolute values)
        steady_state_speed: mean of last 20% of samples
        reversed_detected: True if motor direction appears reversed
    """
    reversed_detected = False

    # Hard-stop both motors before test (duty-cycle, bypasses PID)
    hard_stop(rc)
    time.sleep(SETTLE_TIME_S)

    # Verify motor is actually stopped
    try:
        if motor == "M1":
            pre_spd, _ = rc.read_speed_m1()
        else:
            pre_spd, _ = rc.read_speed_m2()
        if abs(pre_spd) > 10:
            print(f"    WARNING: {motor} still moving at {pre_spd} ticks/s before test")
            hard_stop(rc)
            time.sleep(SETTLE_TIME_S)
    except Exception:
        pass

    times: list[float] = []
    speeds: list[float] = []
    signed_speeds: list[float] = []

    # Command the step
    t0 = time.monotonic()
    if motor == "M1":
        rc.drive_m1_speed(speed_ticks)
    else:
        rc.drive_m2_speed(speed_ticks)

    # Sample velocity
    next_sample = t0 + SAMPLE_PERIOD_S
    while True:
        now = time.monotonic()
        elapsed = now - t0
        if elapsed > STEP_DURATION_S:
            break

        # Wait for next sample time
        sleep_time = next_sample - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        next_sample += SAMPLE_PERIOD_S

        # Read speed
        try:
            if motor == "M1":
                spd, direction = rc.read_speed_m1()
            else:
                spd, direction = rc.read_speed_m2()
            sample_t = time.monotonic() - t0
            times.append(sample_t)
            # direction: 0=forward, 1=backward
            signed_spd = float(-abs(spd) if direction else abs(spd))
            signed_speeds.append(signed_spd)
            speeds.append(float(abs(spd)))
        except Exception:
            pass  # Skip missed samples

        # Detect reversed motor: if early samples show speed decaying from
        # a high initial value, the motor was already moving (PID fighting
        # reversed direction) or if speed is consistently negative
        if len(signed_speeds) == REVERSED_DETECT_SAMPLES:
            neg_count = sum(1 for s in signed_speeds if s < 0)
            avg_early = sum(speeds[:REVERSED_DETECT_SAMPLES]) / REVERSED_DETECT_SAMPLES
            if neg_count >= 3 or avg_early > speed_ticks * REVERSED_DETECT_THRESHOLD:
                reversed_detected = True
                print(f"    ABORT: {motor} appears to have REVERSED motor/encoder direction!")
                print(f"           Early speeds: {[int(s) for s in signed_speeds[:5]]}")
                print(f"           Expected: ramp from 0 toward +{speed_ticks}")
                print(
                    f"           Fix: swap M{'1' if motor == 'M1' else '2'}A / M{'1' if motor == 'M1' else '2'}B wires")
                # Hard-stop immediately
                hard_stop(rc)
                time.sleep(SETTLE_TIME_S)
                return times, speeds, 0.0, True

    # Hard-stop motor (duty-cycle, not PID)
    hard_stop(rc)
    time.sleep(0.05)  # Brief pause

    # Verify motor stopped
    try:
        if motor == "M1":
            post_spd, _ = rc.read_speed_m1()
        else:
            post_spd, _ = rc.read_speed_m2()
        if abs(post_spd) > 10:
            print(f"    WARNING: {motor} still at {post_spd} ticks/s after stop, retrying...")
            hard_stop(rc)
            time.sleep(SETTLE_TIME_S)
    except Exception:
        pass

    # Steady-state = mean of last 20% of samples
    if speeds:
        tail = speeds[int(len(speeds) * 0.8):]
        steady = sum(tail) / len(tail) if tail else 0.0
    else:
        steady = 0.0

    return times, speeds, steady, reversed_detected


def _report_step_result(
    motor: str, times: list[float], speeds: list[float],
    steady: float, speed: int, reversed_detected: bool,
    verbose: bool, p: int | None = None,
) -> None:
    """Print step-response analysis for one motor."""
    if reversed_detected:
        return  # Already printed abort message

    if not times:
        print(f"    No data collected -- check motor/encoder connection")
        return

    tau = fit_first_order_tau(times, speeds, steady)
    error_pct = (tau - TARGET_TAU_S) / TARGET_TAU_S * 100.0

    print(f"    Samples:      {len(times)}")
    print(f"    Steady-state: {steady:.0f} ticks/s (target={speed})")
    print(
        f"    Measured tau:  {tau*1000:.1f} ms (target={TARGET_TAU_S*1000:.0f} ms, error={error_pct:+.1f}%)")

    if steady < speed * 0.1:
        print(f"    WARNING: Steady-state speed is <10% of target.")
        print(f"             Motor may be stalled, disconnected, or wheels loaded.")
        print(f"             Lift the robot so wheels spin freely for tuning.")
    elif steady < speed * 0.5:
        print(f"    WARNING: Steady-state speed is <50% of target.")
        print(f"             Check motor wiring and ensure wheels are unloaded.")

    if abs(error_pct) < 20:
        print(f"    PASS -- within 20% of target tau")
    elif tau > TARGET_TAU_S:
        hint = f" (currently {p})" if p is not None else ""
        print(f"    SLOW -- consider increasing P{hint} or decreasing D")
    else:
        hint = f" (currently {p})" if p is not None else ""
        print(f"    FAST -- consider decreasing P{hint} or increasing D")

    if verbose:
        print(f"\n    {'Time(ms)':>10}  {'Speed':>10}")
        for t, s in zip(times, speeds):
            print(f"    {t*1000:10.1f}  {s:10.0f}")


def cmd_step_response(args: argparse.Namespace) -> None:
    """Run step-response test and report measured time constant."""
    motors = get_motor_list(args.motor)
    speed = args.speed or STEP_VELOCITY_TICKS

    for label, port, addr in get_controllers(args.controller):
        print(f"\n{'='*50}")
        print(f"  {label} RoboClaw (0x{addr:02X}) -- Step Response")
        print(f"{'='*50}")

        try:
            rc = open_controller(port, addr)

            # Show current PID
            pid_m1 = rc.read_velocity_pid_m1()
            pid_m2 = rc.read_velocity_pid_m2()
            print(
                f"  Current M1 PID: P={pid_m1['P']} I={pid_m1['I']} D={pid_m1['D']} QPPS={pid_m1['QPPS']}")
            print(
                f"  Current M2 PID: P={pid_m2['P']} I={pid_m2['I']} D={pid_m2['D']} QPPS={pid_m2['QPPS']}")
            print(f"  Step target: {speed} ticks/s ({speed / QPPS * 100:.0f}% of max)")
            print()

            for motor in motors:
                print(f"  Testing {motor}...")
                times, speeds, steady, reversed_detected = run_step_response(rc, motor, speed)
                _report_step_result(motor, times, speeds, steady, speed,
                                    reversed_detected, args.verbose)
                time.sleep(SETTLE_TIME_S)

            close_controller(rc)
        except Exception as e:
            print(f"  ERROR: {e}")


def cmd_tune(args: argparse.Namespace) -> None:
    """Set PID values and immediately run step-response test."""
    p = args.p or DEFAULT_P
    i_val = args.i if args.i is not None else DEFAULT_I
    d = args.d if args.d is not None else DEFAULT_D
    qpps = args.qpps or QPPS
    motors = get_motor_list(args.motor)
    speed = args.speed or STEP_VELOCITY_TICKS

    for label, port, addr in get_controllers(args.controller):
        print(f"\n{'='*60}")
        print(f"  {label} RoboClaw (0x{addr:02X}) -- Tune + Step Response")
        print(f"{'='*60}")

        try:
            rc = open_controller(port, addr)

            # Set PID
            for motor in motors:
                if motor == "M1":
                    ok = rc.set_velocity_pid_m1(p, i_val, d, qpps)
                else:
                    ok = rc.set_velocity_pid_m2(p, i_val, d, qpps)
                status = "OK" if ok else "FAILED"
                print(f"  {motor} SET P={p} I={i_val} D={d} QPPS={qpps} -> {status}")

            # Verify
            for motor in motors:
                if motor == "M1":
                    pid = rc.read_velocity_pid_m1()
                else:
                    pid = rc.read_velocity_pid_m2()
                print(
                    f"  {motor} readback: P={pid['P']} I={pid['I']} D={pid['D']} QPPS={pid['QPPS']}")

            print(f"\n  Step target: {speed} ticks/s ({speed / QPPS * 100:.0f}% of max)")
            print()

            # Run step-response
            for motor in motors:
                print(f"  Testing {motor}...")
                times, speeds, steady, reversed_detected = run_step_response(rc, motor, speed)
                _report_step_result(motor, times, speeds, steady, speed,
                                    reversed_detected, args.verbose, p)
                time.sleep(SETTLE_TIME_S)

            close_controller(rc)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print("  Tuning Summary")
    print(f"{'='*60}")
    print(f"  Target tau:  {TARGET_TAU_S*1000:.0f} ms (from sim motor_time_constant_s)")
    print(f"  QPPS:        {qpps} ({MOTOR_MAX_RPM:.0f} RPM * {ENCODER_PPR_OUTPUT_SHAFT} PPR / 60)")
    print()
    print("  If all motors show PASS, these PID values are good.")
    print("  If SLOW: increase P by 50%, or add small I (e.g., 500).")
    print("  If FAST/oscillating: decrease P by 30%, or add D (e.g., 5000).")
    print()
    print("  Iterate by running:  python3 tune_pid.py --tune --p <new_P>")
    print("  PID values persist until power cycle (stored in RAM, not EEPROM).")


def cmd_stop(args: argparse.Namespace) -> None:
    """Emergency stop all motors."""
    for label, port, addr in get_controllers(args.controller):
        print(f"  Stopping {label} (0x{addr:02X})...", end=" ")
        try:
            rc = open_controller(port, addr)
            # hard_stop is called by open_controller, but call again explicitly
            hard_stop(rc)
            # Also read back speed to verify
            spd1, _ = rc.read_speed_m1()
            spd2, _ = rc.read_speed_m2()
            close_controller(rc)
            print(f"OK (M1={spd1}, M2={spd2} ticks/s)")
        except Exception as e:
            print(f"ERROR: {e}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboClaw velocity PID tuning for Strafer chassis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tune_pid.py --read                          # Show current PID
  python3 tune_pid.py --tune                          # Set defaults + test
  python3 tune_pid.py --tune --p 20000 --i 500        # Custom PID + test
  python3 tune_pid.py --set --p 15000 --i 750 --d 0   # Set PID only
  python3 tune_pid.py --step-response                  # Test only
  python3 tune_pid.py --step-response -v               # Test with raw data
  python3 tune_pid.py --tune --controller front        # Front only
  python3 tune_pid.py --tune --motor m1                # M1 only (all controllers)
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--read", action="store_true", help="Read current PID values")
    mode.add_argument("--tune", action="store_true", help="Set PID + run step-response")
    mode.add_argument("--set", action="store_true", help="Set PID values only")
    mode.add_argument("--step-response", action="store_true", help="Run step-response only")
    mode.add_argument("--stop", action="store_true", help="Emergency stop all motors")

    parser.add_argument("--controller", choices=["front", "rear", "both"], default="both",
                        help="Which controller(s) to operate on (default: both)")
    parser.add_argument("--motor", choices=["m1", "m2", "both"], default="both",
                        help="Which motor(s) to operate on (default: both)")

    parser.add_argument("--p", type=int, default=None,
                        help=f"P gain (default for --tune: {DEFAULT_P})")
    parser.add_argument("--i", type=int, default=None,
                        help=f"I gain (default for --tune: {DEFAULT_I})")
    parser.add_argument("--d", type=int, default=None,
                        help=f"D gain (default for --tune: {DEFAULT_D})")
    parser.add_argument("--qpps", type=int, default=None, help=f"QPPS (default: {QPPS})")
    parser.add_argument("--speed", type=int, default=None,
                        help=f"Step-response target speed in ticks/s (default: {STEP_VELOCITY_TICKS})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print raw step-response data")

    args = parser.parse_args()

    # Validate --set requires at least one PID value
    if args.set and args.p is None and args.i is None and args.d is None and args.qpps is None:
        parser.error("--set requires at least one of --p, --i, --d, --qpps")

    if args.stop:
        cmd_stop(args)
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
    elif args.step_response:
        cmd_step_response(args)
    elif args.tune:
        cmd_tune(args)


if __name__ == "__main__":
    main()
