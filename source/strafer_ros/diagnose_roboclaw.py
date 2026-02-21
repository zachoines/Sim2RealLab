#!/usr/bin/env python3
"""Quick diagnostic for RoboClaw -- tries duty-cycle commands instead of PID velocity.
"""

from __future__ import annotations
from strafer_shared.constants import (
    ROBOCLAW_FRONT_ADDRESS,
    ROBOCLAW_REAR_ADDRESS,
    ROBOCLAW_BAUD_RATE,
)
from roboclaw_interface import RoboClawInterface, RoboClawError, _crc16

import struct
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parent / "strafer_driver" / "strafer_driver"))


# Additional command IDs for PID reads
CMD_READ_M1_VEL_PID = 55
CMD_READ_M2_VEL_PID = 56


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


def test_duty_drive(claw: RoboClawInterface, motor: int, duty: int = 32, duration: float = 1.0):
    """Drive a motor with simple duty-cycle command (no PID needed)."""
    motor_label = f"M{motor}"
    print(f"  Driving {motor_label} with duty={duty}/127 for {duration}s ...")

    try:
        if motor == 1:
            claw.forward_m1(duty)
        else:
            claw.forward_m2(duty)

        time.sleep(duration)

        # Read encoder speed while running
        if motor == 1:
            speed, _ = claw.read_speed_m1()
            enc, _ = claw.read_encoder_m1()
        else:
            speed, _ = claw.read_speed_m2()
            enc, _ = claw.read_encoder_m2()

        print(f"    Speed reading: {speed} ticks/s, Encoder: {enc}")

        # Stop
        if motor == 1:
            claw.forward_m1(0)
        else:
            claw.forward_m2(0)

        return speed

    except RoboClawError as e:
        print(f"    ERROR: {e}")
        try:
            if motor == 1:
                claw.forward_m1(0)
            else:
                claw.forward_m2(0)
        except Exception:
            pass
        return None


def diagnose_roboclaw(port: str, address: int, label: str):
    """Run diagnostics on a single RoboClaw."""
    print(f"\n{'='*60}")
    print(f"Diagnosing RoboClaw '{label}' at {port} (address 0x{address:02x})")
    print(f"{'='*60}")

    claw = RoboClawInterface(port, address, ROBOCLAW_BAUD_RATE)

    try:
        claw.open()
        print(f"  Serial port opened.")
    except Exception as e:
        print(f"  FATAL: Cannot open port: {e}")
        return

    # Battery voltage
    try:
        v = claw.read_main_battery()
        print(f"  Battery voltage: {v:.1f}V")
    except RoboClawError as e:
        print(f"  Battery read failed: {e}")

    # Temperature
    try:
        t = claw.read_temperature()
        print(f"  Temperature: {t:.1f}°C")
    except RoboClawError as e:
        print(f"  Temperature read failed: {e}")

    # Read PID values
    print("\n  Velocity PID settings:")
    for m in (1, 2):
        pid = read_velocity_pid(claw, m)
        if "error" in pid:
            print(f"    M{m}: {pid['error']}")
        else:
            print(f"    M{m}: P={pid['P']}  I={pid['I']}  D={pid['D']}  QPPS={pid['QPPS']}")
            if pid['P'] == 0 and pid['I'] == 0 and pid['D'] == 0:
                print(f"    *** M{m} PID is all zeros -- velocity commands will NOT work! ***")
                print(f"    *** Use Motion Studio to auto-tune PID, or set QPPS manually ***")

    # Test with duty-cycle commands (no PID needed)
    print("\n  Testing duty-cycle drive (bypasses PID):")
    for m in (1, 2):
        test_duty_drive(claw, m, duty=32, duration=1.0)
        time.sleep(0.5)

    claw.stop_motors()
    claw.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RoboClaw diagnostic tool")
    parser.add_argument("--front", default=None)
    parser.add_argument("--rear", default=None)
    parser.add_argument("--front-only", action="store_true", help="Only test front")
    parser.add_argument("--rear-only", action="store_true", help="Only test rear")
    args = parser.parse_args()

    # Auto-detect ports if not explicitly given
    if args.front is None or args.rear is None:
        from roboclaw_interface import detect_roboclaws
        detected = detect_roboclaws(ROBOCLAW_FRONT_ADDRESS,
                                    ROBOCLAW_REAR_ADDRESS, ROBOCLAW_BAUD_RATE)
        if args.front is None:
            args.front = detected.get(ROBOCLAW_FRONT_ADDRESS)
        if args.rear is None:
            args.rear = detected.get(ROBOCLAW_REAR_ADDRESS)
        if detected:
            print(f"Auto-detected: {', '.join(f'0x{a:02X}->{p}' for a, p in detected.items())}")
        else:
            print("Auto-detect: no RoboClaws found on any port.")

    if not args.rear_only:
        if args.front:
            diagnose_roboclaw(args.front, ROBOCLAW_FRONT_ADDRESS, "Front")
        else:
            print("\nFront RoboClaw (0x80): not detected")
    if not args.front_only:
        if args.rear:
            diagnose_roboclaw(args.rear, ROBOCLAW_REAR_ADDRESS, "Rear")
        else:
            print("\nRear RoboClaw (0x81): not detected")


if __name__ == "__main__":
    main()
