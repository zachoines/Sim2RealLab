#!/usr/bin/env python3
"""Quick diagnostic for RoboClaw -- reads each controller, and with --drive tries duty-cycle
commands instead of PID velocity.

Also reads each controller's packet-serial timeout, and sets it on request. On request it
writes the driver's velocity gains, and with --write-nvm saves either setting to flash.

Every run that reaches the gains also reads the settings: firmware (21), status (90), the S-pin
modes (75), the main battery settings (59), the default duty acceleration (81), the standard
config (99), the current limits (135, 136), the PWM mode (149) and the idle modes (161). A
reply whose length depends on the firmware (75, 81, 90) is taken at the length that
arrived, if its CRC matches: 75's length is always printed, and a short 81 or 90 says so. Reading
90 clears its warning bits and the E-Stop bit, so it is read once, never retried, and with base
stopped the tool is their only reader.
"""

from __future__ import annotations

import argparse
import decimal
import glob
import struct
import sys
import termios
import time

from strafer_shared.constants import (
    ROBOCLAW_FRONT_ADDRESS,
    ROBOCLAW_REAR_ADDRESS,
    ROBOCLAW_BAUD_RATE,
    ROBOCLAW_PID_D,
    ROBOCLAW_PID_I,
    ROBOCLAW_PID_P,
    ROBOCLAW_QPPS,
    ROBOCLAW_SERIAL_TIMEOUT_S,
)
from strafer_driver.roboclaw_interface import (
    RoboClawChecksumError,
    RoboClawInterface,
    RoboClawError,
    RoboClawTimeoutError,
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

# The settings reads, all read-only (the RoboClaw user manual, Rev 6.0, pages 72-90).
CMD_READ_FIRMWARE = 21
CMD_READ_MAIN_BATTERY_SETTINGS = 59
CMD_READ_PIN_MODES = 75
CMD_READ_DEFAULT_DUTY_ACCEL = 81
CMD_READ_STATUS = 90
CMD_READ_STANDARD_CONFIG = 99
CMD_READ_M1_CURRENT_LIMIT = 135
CMD_READ_M2_CURRENT_LIMIT = 136
CMD_READ_PWM_MODE = 149
CMD_READ_IDLE = 161
FIRMWARE_MAX_BYTES = 48
SETTINGS_GAP_S = 0.012  # between the settings reads
# Command 90's bits (four-byte status). Warnings clear when read; errors persist until a reset,
# but for the E-Stop, which clears on read.
STATUS_BITS = {
    0x00000001: ("error", "Emergency Stop"),
    0x00000002: ("error", "Temperature"),
    0x00000004: ("error", "Temperature 2"),
    0x00000010: ("error", "Logic Battery High"),
    0x00000020: ("error", "Logic Battery Low"),
    0x00000040: ("error", "Motor 1 Fault"),
    0x00000080: ("error", "Motor 2 Fault"),
    0x00000100: ("error", "Motor 1 Speed"),
    0x00000200: ("error", "Motor 2 Speed"),
    0x00000400: ("error", "Motor 1 Position"),
    0x00000800: ("error", "Motor 2 Position"),
    0x00001000: ("error", "Motor Current 1"),
    0x00002000: ("error", "Motor Current 2"),
    0x00010000: ("warning", "Over Current 1"),
    0x00020000: ("warning", "Over Current 2"),
    0x00040000: ("warning", "Main Battery High"),
    0x00080000: ("warning", "Main Battery Low"),
    0x00100000: ("warning", "Temperature"),
    0x00200000: ("warning", "Temperature 2"),
    0x00400000: ("warning", "S4 Limit Switch"),
    0x00800000: ("warning", "S5 Limit Switch"),
    0x01000000: ("warning", "Idle Motor 1"),
    0x02000000: ("warning", "Idle Motor 2"),
    0x20000000: ("warning", "Reset"),
    0x40000000: ("warning", "Over Regen Motor 1"),
    0x80000000: ("warning", "Over Regen Motor 2"),
}
# Command 74/75's pin modes (the 5-byte frame). Motion Studio's Invert option is not among them.
PIN_MODES = {
    0x00: "Default/Disabled",
    0xC1: "E-Stop (Latching)",
    0x14: "Voltage Clamp",
    0x24: "RS485 Direction",
    0x80: "Encoder Toggle",
    0x04: "Brake",
    0xE2: "Home (Auto)",
    0x62: "Home (User)",
    0xF2: "Home (Auto)/Limit (Fwd)",
    0x72: "Home (User)/Limit (Fwd)",
    0x12: "Limit (Fwd)",
    0x22: "Limit (Rev)",
    0x32: "Limit (Both)",
    0x11: "Stop M1",
    0x21: "Stop M2",
    0x41: "E-Stop",
    0x84: "User Output",
    0x94: "Main Batt Out",
    0xA4: "Logic Batt Out",
    0xB4: "M1 Current Out",
    0xC4: "M2 Current Out",
    0xD4: "Temperature 1 Out",
    0xE4: "Temperature 2 Out",
    0x40: "Flip Switch",
    0x34: "Main Batt Warning",
    0x44: "Logic Batt Warning",
}
PIN_MODE_UNLISTED = "(not in the manual's table)"
# Command 98/99's low bits: the control mode and the battery mode.
CONFIG_MODES = {0x0000: "RC mode", 0x0001: "analog mode", 0x0002: "simple serial mode",
                0x0003: "packet serial mode"}
CONFIG_BATTERY_MODES = {0x0000: "off", 0x0004: "auto", 0x0008: "2 cell", 0x000C: "3 cell",
                        0x0010: "4 cell", 0x0014: "5 cell", 0x0018: "6 cell", 0x001C: "7 cell"}
# Command 148/149's PWM modes.
PWM_MODES = {0: "inductive", 1: "resistive, 5 % blanking", 2: "resistive, 10 % blanking",
             3: "resistive, 15 % blanking", 4: "resistive, 20 % blanking"}

DUTY = 32
DUTY_HOLD_S = 1.0
# The gains roboclaw_node writes to both motors of both controllers at every start.
DRIVER_GAINS = {
    "P": ROBOCLAW_PID_P,
    "I": ROBOCLAW_PID_I,
    "D": ROBOCLAW_PID_D,
    "QPPS": ROBOCLAW_QPPS,
}
# A flash write resets the controller: its USB port drops, then comes back, maybe renamed. The
# polls are counted as well as timed, so the waits stay bounded when sleeps are skipped.
RESET_GONE_POLL_S = 0.1
RESET_GONE_WAIT_S = 5.0
RESET_BACK_POLL_S = 0.5
RESET_BACK_WAIT_S = 15.0
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


def format_gains(gains: dict) -> str:
    return f"P={gains['P']} I={gains['I']} D={gains['D']} QPPS={gains['QPPS']}"


def read_gains(claw: RoboClawInterface) -> list[dict]:
    """Both motors' velocity gains (commands 55 and 56)."""
    return [claw.read_velocity_pid_m1(), claw.read_velocity_pid_m2()]


def set_driver_gains(claw: RoboClawInterface) -> bool:
    """Write DRIVER_GAINS to both motors (commands 28 and 29); True if both were acknowledged."""
    g = DRIVER_GAINS
    m1 = claw.set_velocity_pid_m1(g["P"], g["I"], g["D"], g["QPPS"])
    m2 = claw.set_velocity_pid_m2(g["P"], g["I"], g["D"], g["QPPS"])
    return bool(m1 and m2)


def poll(check, period: float, limit: float):
    """Call *check* every *period* s until it returns a truthy value or *limit* s pass."""
    t0 = time.monotonic()
    polls = round(limit / period)
    for n in range(polls + 1):
        found = check()
        if found or n == polls or time.monotonic() - t0 >= limit:
            return found, time.monotonic() - t0
        time.sleep(period)


def verify_saved_gains(port: str, address: int, label: str, saved_timeout_s: float | None) -> bool:
    """See the controller reset after a flash write, find it again by address, read what it loaded.

    The gains (55, 56) and serial timeout (15) read after the reset are the ones it loaded from
    flash, so they must be the driver's gains and the timeout that was saved before.
    """
    gone, waited = poll(lambda: not glob.glob(port), RESET_GONE_POLL_S, RESET_GONE_WAIT_S)
    if not gone:
        print(
            f"  {port} was still there {waited:.1f} s after the save, so no reset was seen and "
            "the save is unverified."
        )
        return False
    print(f"  {port} dropped after {waited:.1f} s: the controller is resetting.")
    port_back, waited = poll(
        lambda: detect_roboclaws(
            ROBOCLAW_FRONT_ADDRESS, ROBOCLAW_REAR_ADDRESS, ROBOCLAW_BAUD_RATE
        ).get(address),
        RESET_BACK_POLL_S,
        RESET_BACK_WAIT_S,
    )
    if port_back is None:
        print(
            f"  {label} did not answer on /dev/roboclaw* within {waited:.1f} s, so the save is "
            "unverified; once it is back, run the same command again, not --read-only."
        )
        return False
    print(f"  {label} answers on {port_back}, {waited:.1f} s later.")
    claw = RoboClawInterface(port_back, address, ROBOCLAW_BAUD_RATE)
    try:
        claw.open()
        loaded = read_gains(claw)
        loaded_timeout_s = claw.read_serial_timeout()
    except SERIAL_ERRORS as e:
        print(f"  Reading what it loaded failed: {e}; the save is unverified.")
        return False
    finally:
        claw.close()
    print(
        f"  Loaded at the reset (cmd 55, 56, 15): M1 {format_gains(loaded[0])}, "
        f"M2 {format_gains(loaded[1])}; serial timeout {loaded_timeout_s:.1f} s"
    )
    ok = True
    if loaded != [DRIVER_GAINS, DRIVER_GAINS]:
        print(f"  Those gains are not the driver's {format_gains(DRIVER_GAINS)}.")
        ok = False
    if saved_timeout_s is not None and loaded_timeout_s != saved_timeout_s:
        print(f"  The serial timeout changed from the saved {saved_timeout_s:.1f} s.")
        ok = False
    if ok:
        print("  Saved and verified.")
    return ok


def describe_serial_timeout(seconds: float) -> str:
    if seconds == ROBOCLAW_SERIAL_TIMEOUT_S:
        return f"{seconds:.1f} s"
    disabled = "0 = disabled; " if seconds == 0 else ""
    return f"{seconds:.1f} s ({disabled}the bring-up expects {ROBOCLAW_SERIAL_TIMEOUT_S:.1f} s)"


# ---------------------------------------------------------------------------
# The settings reads: firmware, status, pins, battery limits, acceleration, config, current
# limits, PWM and idle modes. Each only reads. Formats from the RoboClaw user manual, Rev 6.0; the
# shorter 75, 81 and 90 replies are an older firmware's.
# ---------------------------------------------------------------------------


def _read_payload(claw: RoboClawInterface, command: int, lengths: tuple[int, ...]) -> bytes:
    header = bytes([claw.address, command])
    claw._serial.reset_input_buffer()
    claw._serial.write(header + struct.pack(">H", _crc16(header)))
    response = claw._serial.read(max(lengths) + 2)
    payload, crc = response[:-2], response[-2:]
    if len(payload) not in lengths:
        raise RoboClawTimeoutError(
            f"{len(response)} bytes on the wire, expected "
            f"{' or '.join(str(n) for n in lengths)} data bytes and a 2-byte CRC"
        )
    if struct.unpack(">H", crc)[0] != _crc16(header + payload):
        raise RoboClawChecksumError("CRC mismatch")
    return payload


def read_reply(claw: RoboClawInterface, command: int, *lengths: int) -> bytes:
    """Send read *command* and return its payload, one of *lengths* bytes long.

    Some replies' length depends on the firmware: the longest is read, and a shorter one ends at
    the port's read timeout. The reply is taken at the length that arrived, if its CRC (over the
    address, the command and the payload) matches.
    """
    return claw._retry(_read_payload, claw, command, lengths)


def _read_firmware(claw: RoboClawInterface) -> str:
    header = bytes([claw.address, CMD_READ_FIRMWARE])
    claw._serial.reset_input_buffer()
    claw._serial.write(header + struct.pack(">H", _crc16(header)))
    text = b""
    while not text.endswith(b"\n\x00"):
        if len(text) >= FIRMWARE_MAX_BYTES + 2:
            raise RoboClawError(f"no line feed and null in {len(text)} bytes")
        byte = claw._serial.read(1)
        if not byte:
            raise RoboClawTimeoutError(f"reply ended after {len(text)} bytes")
        text += byte
    crc = claw._serial.read(2)
    if len(crc) != 2 or struct.unpack(">H", crc)[0] != _crc16(header + text):
        raise RoboClawChecksumError("CRC mismatch")
    return text[:-2].decode("ascii", "replace").strip()


def read_firmware(claw: RoboClawInterface) -> str:
    """The product name and firmware version (command 21): a string ending in LF and NUL."""
    return claw._retry(_read_firmware, claw)


def describe_status(claw: RoboClawInterface) -> str:
    # Read once, never retried: the read clears the warning bits, so a retry after a lost reply
    # would read them cleared.
    try:
        with claw._lock:
            payload = _read_payload(claw, CMD_READ_STATUS, (4, 2))
    except SERIAL_ERRORS as e:
        raise RoboClawError(f"{e}; the failed read may have cleared the warning bits") from e
    if len(payload) == 2:
        value = struct.unpack(">H", payload)[0]
        return (
            f"0x{value:04X} (2-byte reply: an older firmware's status, whose bits the manual's "
            "table does not describe; not decoded)"
        )
    value = struct.unpack(">I", payload)[0]
    if value == 0:
        return "0x00000000, normal"
    names = [f"{name} ({kind})" for bit, (kind, name) in STATUS_BITS.items() if value & bit]
    unknown = value & ~sum(STATUS_BITS)
    if unknown:
        names.append(f"unlisted 0x{unknown:08X}")
    return f"0x{value:08X}: {', '.join(names)}"


def describe_pin_modes(claw: RoboClawInterface) -> str:
    payload = read_reply(claw, CMD_READ_PIN_MODES, 5, 3)
    if len(payload) == 3:
        # An older firmware's frame: Rev 6.0's mode table is not known to hold for its values.
        modes = ", ".join(f"{pin} 0x{mode:02X}" for pin, mode in zip(("S3", "S4", "S5"), payload))
        return f"{modes} (3-byte reply: S3, S4, S5, an older firmware's frame; modes not named)"
    modes = ", ".join(
        f"{pin} 0x{mode:02X} {PIN_MODES.get(mode, PIN_MODE_UNLISTED)}"
        for pin, mode in zip(("S3", "S4", "S5", "CTRL1", "CTRL2"), payload)
    )
    return f"{modes} (5-byte reply: S3, S4, S5, CTRL1, CTRL2)"


def describe_battery_settings(claw: RoboClawInterface) -> str:
    low, high, offset = struct.unpack(">HHB", read_reply(claw, CMD_READ_MAIN_BATTERY_SETTINGS, 5))
    return f"min {low / 10:.1f} V, max {high / 10:.1f} V, auto offset {offset}"


def describe_duty_acceleration(claw: RoboClawInterface) -> str:
    payload = read_reply(claw, CMD_READ_DEFAULT_DUTY_ACCEL, 16, 8)
    if len(payload) == 8:
        m1, m2 = struct.unpack(">II", payload)
        return f"M1 {m1}, M2 {m2} (8-byte reply: one value per motor)"
    m1a, m1d, m2a, m2d = struct.unpack(">IIII", payload)
    return f"M1 accel {m1a} decel {m1d}, M2 accel {m2a} decel {m2d}"


def describe_standard_config(claw: RoboClawInterface) -> str:
    (config,) = struct.unpack(">H", read_reply(claw, CMD_READ_STANDARD_CONFIG, 2))
    mode = CONFIG_MODES[config & 0x0003]
    battery = CONFIG_BATTERY_MODES[config & 0x001C]
    return f"0x{config:04X} ({mode}, battery mode {battery})"


def describe_current_limits(claw: RoboClawInterface) -> str:
    """Both motors' limits; one motor's failed read does not keep the other's from being read."""
    parts = []
    for motor, command in ((1, CMD_READ_M1_CURRENT_LIMIT), (2, CMD_READ_M2_CURRENT_LIMIT)):
        try:
            high, low = struct.unpack(">II", read_reply(claw, command, 8))
        except SERIAL_ERRORS as e:
            parts.append(f"M{motor} read failed (cmd {command}): {e}")
            continue
        minimum = f" min {low / 100:.2f} A" if low else ""
        parts.append(f"M{motor} max {high / 100:.2f} A{minimum}")
    return ", ".join(parts)


def describe_pwm_modes(claw: RoboClawInterface) -> str:
    payload = read_reply(claw, CMD_READ_PWM_MODE, 2)
    return ", ".join(
        f"M{motor} {PWM_MODES.get(mode, f'unlisted {mode}')}"
        for motor, mode in enumerate(payload, start=1)
    )


def describe_idle_modes(claw: RoboClawInterface) -> str:
    payload = read_reply(claw, CMD_READ_IDLE, 2)
    parts = []
    for motor, byte in enumerate(payload, start=1):
        delay = (byte & 0x7F) / 10
        if byte & 0x80:
            parts.append(f"M{motor} braking at zero (its {delay:.1f} s delay unused)")
        else:
            parts.append(f"M{motor} braking for {delay:.1f} s, then freewheel")
    return ", ".join(parts)


# (label, commands, describer), in the order printed.
SETTINGS_READS = (
    ("Firmware", "21", read_firmware),
    ("Status", "90", describe_status),
    ("S-pin modes", "75", describe_pin_modes),
    ("Main battery settings", "59", describe_battery_settings),
    ("Default duty acceleration", "81", describe_duty_acceleration),
    ("Standard config", "99", describe_standard_config),
    ("Current limits", "135, 136", describe_current_limits),
    ("PWM mode", "149", describe_pwm_modes),
    ("Idle modes", "161", describe_idle_modes),
)


def print_settings_reads(claw: RoboClawInterface) -> None:
    """Each settings read on its own line; a failed read is said so, and the rest go on."""
    print("\n  Settings:")
    for label, commands, describe in SETTINGS_READS:
        try:
            print(f"    {label} (cmd {commands}): {describe(claw)}")
        except SERIAL_ERRORS as e:
            print(f"    {label} (cmd {commands}): read failed: {e}")
        # A read's CRC byte can fall in 0x80-0x87, which over USB starts a packet; the controller
        # drops a partial packet after 10 ms, so a gap keeps it off the next read.
        time.sleep(SETTINGS_GAP_S)
    print(
        "    Reading command 90 clears its warning bits and the E-Stop bit: with base stopped, "
        "this tool is their only reader."
    )


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
    set_gains: bool = False,
    write_nvm: bool = False,
    drive: bool = False,
) -> int:
    """Run diagnostics on a single RoboClaw.

    Return 1 if it was not reached, a write failed or a save could not be verified.
    """
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

    if write_nvm and set_gains:
        try:
            active = read_gains(claw)
        except SERIAL_ERRORS as e:
            print(f"  Velocity gains read failed (cmd 55, 56): {e}. Not setting them.")
            claw.close()
            return 1
        print(
            f"  Velocity gains in active memory (cmd 55, 56): M1 {format_gains(active[0])}, "
            f"M2 {format_gains(active[1])}"
        )

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

    if set_gains:
        try:
            before = read_gains(claw)
        except SERIAL_ERRORS as e:
            print(f"  Velocity gains read failed (cmd 55, 56): {e}. Not setting them.")
            claw.close()
            return 1
        print(
            f"  Velocity gains{' as saved' if write_nvm else ''} (cmd 55, 56): "
            f"M1 {format_gains(before[0])}, M2 {format_gains(before[1])}"
        )
        try:
            acked = set_driver_gains(claw)
            after = read_gains(claw)
        except SERIAL_ERRORS as e:
            print(f"  Setting the gains (cmd 28, 29) or reading them back failed: {e}")
            claw.close()
            return 1
        print(
            f"  Velocity gains set to {format_gains(DRIVER_GAINS)} (cmd 28, 29)"
            f"{'' if acked else ', not acknowledged'}; read back M1 {format_gains(after[0])}, "
            f"M2 {format_gains(after[1])}"
        )
        if not acked or after != [DRIVER_GAINS, DRIVER_GAINS]:
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
            claw.close()
            if not saved:
                print(
                    "  No ack for command 94, so whether anything was saved is unknown; "
                    "once /dev/roboclaw* is back, run the same command again, not --read-only: "
                    "it saves again and checks what the controller loads."
                )
                return 1
            return 0 if verify_saved_gains(port, address, label, timeout_s) else 1

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
                    "    *** --set-gains --write-nvm writes and saves the driver's gains ***"
                )

    print_settings_reads(claw)

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
        description="RoboClaw diagnostic tool. It only reads, unless --drive, "
        "--set-serial-timeout or --set-gains is given."
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
        help="Read only, as a run without --drive, --set-serial-timeout or --set-gains does: "
        "battery (24), temperature (82), serial timeout (15), velocity PID (55, 56), firmware "
        "(21), status (90, which clears its warning bits and the E-Stop bit), S-pin modes (75), "
        "main battery "
        "settings (59), default duty acceleration (81), standard config (99), current limits "
        "(135, 136), PWM mode (149) and idle modes (161); no motor command is sent",
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
    mode.add_argument(
        "--set-gains",
        action="store_true",
        help=f"Write the driver's velocity gains ({format_gains(DRIVER_GAINS)}, from "
        "strafer_shared.constants) to both motors of each controller (commands 28, 29) and "
        "read them back (55, 56); no motor command is sent. Sets the active values only: a "
        "power-up loads the saved ones",
    )
    parser.add_argument(
        "--write-nvm",
        action="store_true",
        help="With --set-serial-timeout or --set-gains, first reload the saved settings "
        "(command 95), then set and save every setting to flash (command 94), which resets "
        "the controller. With --set-gains it then waits to see the port drop, finds the "
        "controller again by its address, and checks the gains and timeout it loaded. After a "
        "failure the next controller is not written. Run it with base stopped",
    )
    args = parser.parse_args(argv)
    if args.write_nvm and args.set_serial_timeout is None and not args.set_gains:
        parser.error("--write-nvm needs --set-serial-timeout or --set-gains")

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
        set_gains=args.set_gains,
        write_nvm=args.write_nvm,
        drive=args.drive,
    )
    set_requested = args.set_serial_timeout is not None or args.set_gains
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
        if failed and args.write_nvm:
            print(f"\n{label} RoboClaw (0x{address:02X}): not written, after the failure above.")
            break
        if port:
            failed |= diagnose_roboclaw(port, address, label, **flags) != 0
        else:
            print(f"\n{label} RoboClaw (0x{address:02X}): not detected")
            failed |= set_requested
    if failed:
        return 1


if __name__ == "__main__":
    sys.exit(main())
