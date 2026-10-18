#!/usr/bin/env python3
"""Quick diagnostic for RoboClaw -- reads each controller, and with --drive tries duty-cycle
commands instead of PID velocity.

Also reads each controller's packet-serial timeout, and sets it on request.
"""

from __future__ import annotations

import argparse
import decimal
import struct
import sys
import termios
import time

from strafer_shared.constants import (
    ROBOCLAW_FRONT_ADDRESS,
    ROBOCLAW_REAR_ADDRESS,
    ROBOCLAW_BAUD_RATE,
    ROBOCLAW_SERIAL_TIMEOUT_S,
)
from strafer_driver.roboclaw_interface import (
    RoboClawInterface,
    RoboClawError,
    _crc16,
    detect_roboclaws,
    is_udev_roboclaw_port,
    same_device,
)

# Additional command IDs for PID reads
CMD_READ_M1_VEL_PID = 55
CMD_READ_M2_VEL_PID = 56

# The flash commands live here only: the node never touches NVM.
CMD_WRITE_SETTINGS_NVM = 94
CMD_READ_SETTINGS_NVM = 95
NVM_WRITE_KEY = 0xE22EAB7A

DUTY = 32
DUTY_HOLD_S = 1.0
# M1 and M2 of each controller, as the node maps them.
WHEELS = {ROBOCLAW_FRONT_ADDRESS: ("FL", "FR"), ROBOCLAW_REAR_ADDRESS: ("RL", "RR")}
SERIAL_ERRORS = (RoboClawError, OSError, termios.error)
# Inexact is trapped so a rounded product cannot pass for a whole number of tenths.
EXACT = decimal.Context(traps=[decimal.Inexact, decimal.InvalidOperation])


def read_velocity_pid(claw: RoboClawInterface, motor: int) -> dict:
    """Read velocity PID constants for a motor.

    Returns dict with P, I, D, QPPS values.
    """
    cmd = CMD_READ_M1_VEL_PID if motor == 1 else CMD_READ_M2_VEL_PID
    # Response: P(4) + I(4) + D(4) + QPPS(4) = 16 bytes
    header = bytes([claw.address, cmd])
    packet = header
    crc = _crc16(packet)
    packet += struct.pack(">H", crc)

    with claw._lock:
        claw._serial.reset_input_buffer()
        claw._serial.write(packet)
        response = claw._serial.read(18)  # 16 data + 2 CRC

    if len(response) < 18:
        return {"error": f"Short response ({len(response)} bytes)"}

    p = struct.unpack(">I", response[0:4])[0]
    i = struct.unpack(">I", response[4:8])[0]
    d = struct.unpack(">I", response[8:12])[0]
    qpps = struct.unpack(">I", response[12:16])[0]

    return {"P": p, "I": i, "D": d, "QPPS": qpps}


def write_settings_nvm(claw: RoboClawInterface) -> bool:
    """Save every active setting to flash (command 94); the controller then resets."""
    with claw._lock:
        return claw._send_with_ack(
            CMD_WRITE_SETTINGS_NVM, struct.pack(">I", NVM_WRITE_KEY)
        )


def reload_settings_nvm(claw: RoboClawInterface) -> bool:
    """Reload the saved settings into active memory (command 95), discarding unsaved changes."""
    with claw._lock:
        return claw._send_with_ack(CMD_READ_SETTINGS_NVM)


def describe_serial_timeout(seconds: float) -> str:
    if seconds == ROBOCLAW_SERIAL_TIMEOUT_S:
        return f"{seconds:.1f} s"
    disabled = "0 = disabled; " if seconds == 0 else ""
    return f"{seconds:.1f} s ({disabled}the bring-up expects {ROBOCLAW_SERIAL_TIMEOUT_S:.1f} s)"


def test_duty_drive(
    claw: RoboClawInterface, motor: int, duty: int = 32, duration: float = 1.0
):
    """Drive a motor with simple duty-cycle command (no PID needed)."""
    motor_label = f"M{motor}"
    forward = claw.forward_m1 if motor == 1 else claw.forward_m2
    print(f"  Driving {motor_label} with duty={duty}/127 for {duration}s ...")

    try:
        forward(duty)

        time.sleep(duration)

        # Read encoder speed while running
        if motor == 1:
            speed, _ = claw.read_speed_m1()
            enc, _ = claw.read_encoder_m1()
        else:
            speed, _ = claw.read_speed_m2()
            enc, _ = claw.read_encoder_m2()

        print(f"    Speed reading: {speed} ticks/s, Encoder: {enc}")
        return speed

    except SERIAL_ERRORS as e:
        print(f"    ERROR: {e}")
        return None

    finally:
        try:
            forward(0)
        except Exception:
            pass


def diagnose_roboclaw(
    port: str,
    address: int,
    label: str,
    set_timeout_s: float | None = None,
    write_nvm: bool = False,
    drive: bool = False,
) -> int:
    """Run diagnostics on a single RoboClaw; return 1 if it was not reached or a write failed."""
    print(f"\n{'='*60}")
    print(f"Diagnosing RoboClaw '{label}' at {port} (address 0x{address:02x})")
    print(f"{'='*60}")

    claw = RoboClawInterface(port, address, ROBOCLAW_BAUD_RATE)

    try:
        claw.open()
        print(f"  Serial port opened.")
    except Exception as e:
        print(f"  FATAL: Cannot open port: {e}")
        return 1

    # Battery voltage: the address probe. Nothing else is written without an answer.
    try:
        v = claw.read_main_battery()
        print(f"  Battery voltage: {v:.1f}V")
    except SERIAL_ERRORS as e:
        print(f"  Battery read failed: {e}. No RoboClaw answered; stopping here.")
        claw.close()
        return 1

    # Temperature
    try:
        t = claw.read_temperature()
        print(f"  Temperature: {t:.1f}°C")
    except RoboClawError as e:
        print(f"  Temperature read failed: {e}")

    if write_nvm:
        try:
            reloaded = reload_settings_nvm(claw)
        except SERIAL_ERRORS as e:
            print(f"  Command 95 failed: {e}")
            reloaded = False
        if not reloaded:
            print(
                "  The controller did not reload its saved settings (cmd 95), so nothing "
                "is set or saved."
            )
            claw.close()
            return 1
        print("  Reloaded the saved settings (cmd 95); unsaved changes are discarded.")

    # Serial timeout
    try:
        timeout_s = claw.read_serial_timeout()
        print(f"  Serial timeout (cmd 15): {describe_serial_timeout(timeout_s)}")
    except SERIAL_ERRORS as e:
        timeout_s = None
        print(f"  Serial timeout read failed (cmd 15): {e}")
        if set_timeout_s is not None:
            print("  Not setting it: the controller did not answer the read.")
            claw.close()
            return 1

    if set_timeout_s is not None:
        target = round(set_timeout_s * 10) / 10
        if write_nvm and timeout_s == target:
            print(f"  It already reads {target:.1f} s, so nothing is set or saved.")
            claw.close()
            return 0
        try:
            acked = claw.set_serial_timeout(set_timeout_s)
            timeout_s = claw.read_serial_timeout()
        except SERIAL_ERRORS as e:
            print(f"  Serial timeout set to {target:.1f} s (cmd 14) or its read-back failed: {e}")
            claw.close()
            return 1
        print(
            f"  Serial timeout set to {target:.1f} s (cmd 14)"
            f"{'' if acked else ' was not acknowledged'}; read back {timeout_s:.1f} s (cmd 15)"
        )
        if not acked or timeout_s != target:
            print("  The set did not take; nothing more is written to this port.")
            claw.close()
            return 1

        if write_nvm:
            print("  Command 94 saves every active setting to flash and resets the controller.")
            try:
                saved = write_settings_nvm(claw)
            except SERIAL_ERRORS as e:
                print(f"  Command 94 failed: {e}")
                saved = False
            if saved:
                print(
                    "  Saved. The controller is resetting, and its USB port may re-enumerate; "
                    "once /dev/roboclaw* is back, the same command run again reloads the saved "
                    "settings (cmd 95) and reads them."
                )
            else:
                print(
                    "  No ack for command 94, so whether anything was saved is unknown; "
                    "once /dev/roboclaw* is back, run the same command again, not --read-only: "
                    "it reloads the saved settings (cmd 95) first, so it reads what was saved."
                )
            claw.close()
            return 0 if saved else 1

    # Read PID values
    print("\n  Velocity PID settings:")
    for m in (1, 2):
        pid = read_velocity_pid(claw, m)
        if "error" in pid:
            print(f"    M{m}: {pid['error']}")
        else:
            print(
                f"    M{m}: P={pid['P']}  I={pid['I']}  D={pid['D']}  QPPS={pid['QPPS']}"
            )
            if pid["P"] == 0 and pid["I"] == 0 and pid["D"] == 0:
                print(
                    f"    *** M{m} PID is all zeros -- velocity commands will NOT work! ***"
                )
                print(
                    f"    *** Use Motion Studio to auto-tune PID, or set QPPS manually ***"
                )

    if not drive:
        print("\n  No motor command: the duty-cycle drive runs only with --drive.")
        claw.close()
        return 0

    # Test with duty-cycle commands (no PID needed)
    print("\n  Testing duty-cycle drive (bypasses PID):")
    if timeout_s is not None and 0 < timeout_s <= DUTY_HOLD_S:
        print(
            f"  Note: the duty test sends nothing while it holds each motor for "
            f"{DUTY_HOLD_S:.1f} s, so the controller's serial timeout stops the motor after "
            f"{timeout_s:.1f} s and the speed read at {DUTY_HOLD_S:.1f} s comes out low or "
            "zero; that is the timeout working."
        )
    for m in (1, 2):
        test_duty_drive(claw, m, duty=DUTY, duration=DUTY_HOLD_S)
        time.sleep(0.5)

    for forward in (claw.forward_m1, claw.forward_m2):
        try:
            forward(0)
        except Exception:
            pass
    claw.stop_motors()
    claw.close()
    return 0


def serial_timeout_seconds(text: str) -> float:
    try:
        tenths = EXACT.multiply(decimal.Decimal(text), 10)
    except decimal.DecimalException:
        tenths = None
    if tenths is None or not tenths.is_finite() or not 0 <= tenths <= 255 or tenths % 1:
        raise argparse.ArgumentTypeError(
            f"{text} s is not a whole number of 0.1 s steps from 0 to 25.5 s "
            "(command 14 takes 0-255 in 100 ms units)"
        )
    return int(tenths) / 10


def main(argv: list[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(
        description="RoboClaw diagnostic tool. It only reads, unless --drive or "
        "--set-serial-timeout is given."
    )
    parser.add_argument("--front", default=None)
    parser.add_argument("--rear", default=None)
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--front-only", action="store_true", help="Only test front")
    only.add_argument("--rear-only", action="store_true", help="Only test rear")
    parser.add_argument(
        "--allow-unverified-ports",
        action="store_true",
        help="Bench only: accept --front/--rear paths that are not /dev/roboclaw* names",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--read-only",
        action="store_true",
        help="Read battery, temperature, serial timeout and velocity PID only, as a run "
        "without --drive or --set-serial-timeout does; no motor command is sent",
    )
    mode.add_argument(
        "--drive",
        action="store_true",
        help=f"After the reads, run the duty-cycle test: each motor of each controller "
        f"tested (FL, FR on 0x80; RL, RR on 0x81) at duty {DUTY}/127 for {DUTY_HOLD_S:g} s. "
        "The wheels must be lifted. Without it no motor command is sent",
    )
    mode.add_argument(
        "--set-serial-timeout",
        type=serial_timeout_seconds,
        metavar="SECONDS",
        help="Set each controller's packet-serial timeout (command 14; 0 to 25.5 s in "
        "0.1 s steps, 0 disables it) and read it back; no motor command is sent. Sets "
        "the active value only: a power-up loads the saved one",
    )
    parser.add_argument(
        "--write-nvm",
        action="store_true",
        help="With --set-serial-timeout, first reload the saved settings (command 95), "
        "then set the timeout and save every setting to flash (command 94), which resets "
        "the controller. Run it with base stopped",
    )
    args = parser.parse_args(argv)
    if args.write_nvm and args.set_serial_timeout is None:
        parser.error("--write-nvm needs --set-serial-timeout")

    unverified = [p for p in (args.front, args.rear) if p and not is_udev_roboclaw_port(p)]
    if unverified and not args.allow_unverified_ports:
        print(
            f"Refusing {', '.join(unverified)}: not /dev/roboclaw* names (`make udev`). "
            "On a bench, pass --allow-unverified-ports."
        )
        return 2

    # Auto-detect ports if not explicitly given
    if args.front is None or args.rear is None:
        detected = detect_roboclaws(
            ROBOCLAW_FRONT_ADDRESS, ROBOCLAW_REAR_ADDRESS, ROBOCLAW_BAUD_RATE
        )
        if args.front is None:
            args.front = detected.get(ROBOCLAW_FRONT_ADDRESS)
        if args.rear is None:
            args.rear = detected.get(ROBOCLAW_REAR_ADDRESS)
        if detected:
            print(
                f"Auto-detected: {', '.join(f'0x{a:02X}->{p}' for a, p in detected.items())}"
            )
        else:
            print(
                "Auto-detect: no RoboClaws found on /dev/roboclaw* (the only names "
                "scanned; `make udev`, or pass --front/--rear)."
            )

    both = not (args.front_only or args.rear_only)
    if both and args.front and args.rear and same_device(args.front, args.rear):
        print(
            f"Refusing: front and rear name one device ({args.front}, {args.rear}); each "
            "controller needs its own port. To test it alone, pass --front-only or --rear-only."
        )
        return 2

    flags = dict(
        set_timeout_s=args.set_serial_timeout,
        write_nvm=args.write_nvm,
        drive=args.drive,
    )
    set_requested = args.set_serial_timeout is not None
    plan = []
    if not args.rear_only:
        plan.append((args.front, ROBOCLAW_FRONT_ADDRESS, "Front"))
    if not args.front_only:
        plan.append((args.rear, ROBOCLAW_REAR_ADDRESS, "Rear"))
    wheels = [w for port, address, _ in plan if port for w in WHEELS[address]]
    if args.drive and wheels:
        print(
            f"\nDriving {', '.join(wheels)} at duty {DUTY}/127 for {DUTY_HOLD_S:g} s each: "
            "the wheels must be lifted."
        )
    failed = False
    for port, address, label in plan:
        if port:
            failed |= diagnose_roboclaw(port, address, label, **flags) != 0
        else:
            print(f"\n{label} RoboClaw (0x{address:02X}): not detected")
            failed |= set_requested
    if failed:
        return 1


if __name__ == "__main__":
    sys.exit(main())
