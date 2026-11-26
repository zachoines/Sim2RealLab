"""Tests for the settings reads of ``diagnose_roboclaw.py`` (its --read-only run).

Besides the battery (24), temperature (82), serial timeout (15) and gains (55, 56), the tool
reads the firmware (21), status (90), S-pin modes (75), main battery settings (59), default
duty acceleration (81), standard config (99), current limits (135, 136), PWM mode (149) and
idle modes (161), and writes nothing. A reply whose length depends on the firmware (75, 81,
90) is taken at the length its CRC confirms. The serial layer is test_port_fence's fake.
"""

import struct

import pytest

from strafer_driver.roboclaw_interface import _crc16

# test/ has no __init__.py, so pytest puts it on sys.path; the imported autouse fixtures apply here.
from test_port_fence import (  # noqa: F401
    CAMERA_PORT,
    FRONT,
    FakeBus,
    FakeCamera,
    FakeRoboClaw,
    _shutdown_rclpy,
    patched,
    run_diagnose,
)

PORT = "/dev/roboclaw2"
FRONT_ONLY = ("--front", PORT, "--front-only", "--read-only")
# The manual's numbers, written out as in test_port_fence.
EARLIER_READS = {24, 82, 15, 55, 56}
SETTINGS_READS = {21, 90, 75, 59, 81, 99, 135, 136, 149, 161}

REPLIES = {
    21: b"USB Roboclaw 2x45a v4.4.3\n\x00",
    90: struct.pack(">I", 0x20040000),
    75: bytes([0x00, 0x00, 0x14, 0x00, 0x00]),
    59: struct.pack(">HHB", 60, 120, 20),
    81: struct.pack(">IIII", 655360, 655361, 655362, 655363),
    99: struct.pack(">H", 0x0007),
    135: struct.pack(">II", 3000, 0),
    136: struct.pack(">II", 2500, 0),
    149: bytes([0, 1]),
    161: bytes([0x05, 0x85]),
}
LINES = {
    21: "Firmware (cmd 21): USB Roboclaw 2x45a v4.4.3",
    90: "Status (cmd 90): 0x20040000: Main Battery High (warning), Reset (warning)",
    75: (
        "S-pin modes (cmd 75): S3 0x00 Default/Disabled, S4 0x00 Default/Disabled, "
        "S5 0x14 Voltage Clamp, CTRL1 0x00 Default/Disabled, CTRL2 0x00 Default/Disabled "
        "(5-byte reply: S3, S4, S5, CTRL1, CTRL2)"
    ),
    59: "Main battery settings (cmd 59): min 6.0 V, max 12.0 V, auto offset 20",
    81: (
        "Default duty acceleration (cmd 81): M1 accel 655360 decel 655361, "
        "M2 accel 655362 decel 655363"
    ),
    99: "Standard config (cmd 99): 0x0007 (packet serial mode, battery mode auto)",
    135: "Current limits (cmd 135, 136): M1 max 30.00 A, M2 max 25.00 A",
    149: "PWM mode (cmd 149): M1 inductive, M2 resistive, 5 % blanking",
    161: (
        "Idle modes (cmd 161): M1 braking for 0.5 s, then freewheel, "
        "M2 braking at zero (its 0.5 s delay unused)"
    ),
}


class ReadingRoboClaw(FakeRoboClaw):
    """test_port_fence's fake, also answering the settings reads with *replies*' payloads.

    A command in *bad_crc* gets a reply whose CRC is off by one bit; one left out of *replies*
    gets the base fake's bare acknowledgement, as an unanswered read would get nothing usable.
    """

    def __init__(self, replies=None, bad_crc=(), bad_first=()):
        super().__init__(FRONT, serial_timeout=5)
        self.replies = dict(REPLIES if replies is None else replies)
        self.bad_crc = set(bad_crc)
        self.bad_first = set(bad_first)  # only the first reply's CRC is off

    def respond(self, packet):
        valid = (
            len(packet) == 4
            and packet[0] in self.addresses
            and _crc16(packet[:2]) == struct.unpack(">H", packet[2:])[0]
        )
        if valid and packet[1] in self.replies:
            command = packet[1]
            payload = self.reply(command)
            spoil = command in self.bad_crc or command in self.bad_first
            self.bad_first.discard(command)
            crc = _crc16(packet[:2] + payload) ^ (1 if spoil else 0)
            return payload + struct.pack(">H", crc)
        return super().respond(packet)

    def reply(self, command):
        return self.replies[command]


class ClearOnRead(ReadingRoboClaw):
    """Command 90 as on the controllers: each read returns the status, then clears the warnings."""

    def __init__(self, status, **kw):
        super().__init__(**kw)
        self.status = status

    def reply(self, command):
        if command != 90:
            return super().reply(command)
        value, self.status = self.status, self.status & ~0xFFFF0000 & ~0x00000001
        return struct.pack(">I", value)


def read_only(patched, capsys, device=None, sleep=None, **claw):
    """Run the tool's --read-only on one front controller; (exit status, output, packets)."""
    device = device or ReadingRoboClaw(**claw)
    bus = patched(FakeBus({CAMERA_PORT: FakeCamera(), "/dev/ttyACM2": device, PORT: device}))
    rc = run_diagnose(*FRONT_ONLY, sleep=sleep)
    return rc, capsys.readouterr().out, [data for _, data in bus.packets(PORT)]


# 135 and 136 share one line.
LINE_OF = {**{command: command for command in LINES}, 136: 135}


def setting(out, command):
    """The printed line of the settings read that *command* is part of."""
    prefix = LINES[LINE_OF[command]].split(":")[0] + ":"
    lines = (line.strip() for line in out.splitlines())
    line = next((line for line in lines if line.startswith(prefix)), None)
    assert line is not None, f"no {prefix!r} line in:\n{out}"
    return line


@pytest.mark.parametrize("command", sorted(LINES))
def test_each_settings_read_is_decoded(patched, capsys, command):
    rc, out, _ = read_only(patched, capsys)

    assert rc in (None, 0)
    assert setting(out, command) == LINES[command]


def test_the_run_only_reads(patched, capsys):
    """Every packet is a bare read: address, command, CRC. Each settings read is sent once."""
    _, _, packets = read_only(patched, capsys)

    assert all(len(packet) == 4 for packet in packets), [p for p in packets if len(p) != 4]
    sent = [packet[1] for packet in packets]
    assert set(sent) == EARLIER_READS | SETTINGS_READS
    assert all(sent.count(command) == 1 for command in SETTINGS_READS)


def test_reading_the_status_is_said_to_clear_it(patched, capsys):
    _, out, _ = read_only(patched, capsys)

    assert (
        "Reading command 90 clears its warning bits and the E-Stop bit: with base stopped, "
        "this tool is their only reader."
    ) in out


def test_a_three_byte_pin_mode_reply(patched, capsys):
    """An older firmware's frame: S3, S4 and S5 only, and its length kept."""
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, 75: bytes([0x00, 0x00, 0x14])})

    assert setting(out, 75) == (
        "S-pin modes (cmd 75): S3 0x00, S4 0x00, S5 0x14 "
        "(3-byte reply: S3, S4, S5, an older firmware's frame; modes not named)"
    )


def test_a_pin_mode_outside_the_manuals_table(patched, capsys):
    """The Invert option's encoding is undocumented, so an unknown byte is shown, not named."""
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, 75: bytes([0, 0, 0x15, 0, 0])})

    assert "S5 0x15 (not in the manual's table)" in setting(out, 75)


@pytest.mark.parametrize("value, shown", [
    (0, "Status (cmd 90): 0x00000000, normal"),
    (
        0x40000001,
        "Status (cmd 90): 0x40000001: Emergency Stop (error), Over Regen Motor 1 (warning)",
    ),
    (0x00000008, "Status (cmd 90): 0x00000008: unlisted 0x00000008"),
])
def test_the_status_bits_are_named(patched, capsys, value, shown):
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, 90: struct.pack(">I", value)})

    assert setting(out, 90) == shown


def test_a_two_byte_status_is_shown_not_decoded(patched, capsys):
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, 90: struct.pack(">H", 0x0040)})

    assert setting(out, 90).startswith("Status (cmd 90): 0x0040 (2-byte reply")
    assert "not decoded" in setting(out, 90)


def test_an_eight_byte_duty_acceleration(patched, capsys):
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, 81: struct.pack(">II", 7, 8)})

    assert setting(out, 81) == (
        "Default duty acceleration (cmd 81): M1 7, M2 8 (8-byte reply: one value per motor)"
    )


def test_a_nonzero_minimum_current_is_shown(patched, capsys):
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, 136: struct.pack(">II", 2500, 150)})

    assert setting(out, 135).endswith("M2 max 25.00 A min 1.50 A")


@pytest.mark.parametrize("command", sorted(SETTINGS_READS))
def test_a_bad_crc_fails_that_read_alone(patched, capsys, command):
    rc, out, packets = read_only(patched, capsys, bad_crc={command})

    assert rc in (None, 0)
    assert "read failed" in setting(out, command) and "CRC mismatch" in setting(out, command)
    for other in sorted(set(LINES) - {LINE_OF[command]}):
        assert setting(out, other) == LINES[other], other
    assert SETTINGS_READS <= {packet[1] for packet in packets}, "every read is still sent"


@pytest.mark.parametrize("command", sorted(SETTINGS_READS))
def test_an_unanswered_read_fails_that_read_alone(patched, capsys, command):
    replies = {k: v for k, v in REPLIES.items() if k != command}
    rc, out, packets = read_only(patched, capsys, replies=replies)

    assert rc in (None, 0)
    assert "read failed" in setting(out, command)
    for other in sorted(set(LINES) - {LINE_OF[command]}):
        assert setting(out, other) == LINES[other], other
    assert SETTINGS_READS <= {packet[1] for packet in packets}, "every read is still sent"


@pytest.mark.parametrize("failed, kept", [(135, "M2 max 25.00 A"), (136, "M1 max 30.00 A")])
def test_one_motors_current_limit_failing_keeps_the_other(patched, capsys, failed, kept):
    _, out, _ = read_only(patched, capsys, bad_crc={failed})

    line = setting(out, 135)
    assert f"read failed (cmd {failed}): CRC mismatch" in line and kept in line


def test_a_firmware_string_without_its_end_fails(patched, capsys):
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, 21: b"x" * 60})

    assert setting(out, 21) == "Firmware (cmd 21): read failed: no line feed and null in 50 bytes"
    assert setting(out, 90) == LINES[90]


# Command 90's table, written out from the manual (Rev 6.0, p. 76): (mask, kind, name).
MANUAL_STATUS = [
    (0x00000001, "error", "Emergency Stop"), (0x00000002, "error", "Temperature"),
    (0x00000004, "error", "Temperature 2"), (0x00000010, "error", "Logic Battery High"),
    (0x00000020, "error", "Logic Battery Low"), (0x00000040, "error", "Motor 1 Fault"),
    (0x00000080, "error", "Motor 2 Fault"), (0x00000100, "error", "Motor 1 Speed"),
    (0x00000200, "error", "Motor 2 Speed"), (0x00000400, "error", "Motor 1 Position"),
    (0x00000800, "error", "Motor 2 Position"), (0x00001000, "error", "Motor Current 1"),
    (0x00002000, "error", "Motor Current 2"), (0x00010000, "warning", "Over Current 1"),
    (0x00020000, "warning", "Over Current 2"), (0x00040000, "warning", "Main Battery High"),
    (0x00080000, "warning", "Main Battery Low"), (0x00100000, "warning", "Temperature"),
    (0x00200000, "warning", "Temperature 2"), (0x00400000, "warning", "S4 Limit Switch"),
    (0x00800000, "warning", "S5 Limit Switch"), (0x01000000, "warning", "Idle Motor 1"),
    (0x02000000, "warning", "Idle Motor 2"), (0x20000000, "warning", "Reset"),
    (0x40000000, "warning", "Over Regen Motor 1"), (0x80000000, "warning", "Over Regen Motor 2"),
]


@pytest.mark.parametrize("mask, kind, name", MANUAL_STATUS)
def test_every_status_bit_has_the_manuals_name(patched, capsys, mask, kind, name):
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, 90: struct.pack(">I", mask)})

    assert setting(out, 90) == f"Status (cmd 90): 0x{mask:08X}: {name} ({kind})"


def test_the_status_is_read_once_so_a_lost_reply_cannot_hide_its_warnings(patched, capsys):
    """The first read clears the warnings: a retry would read them cleared, as 'normal'."""
    device = ClearOnRead(0x20040000, bad_first={90})
    _, out, packets = read_only(patched, capsys, device=device)

    line = setting(out, 90)
    assert "read failed: CRC mismatch" in line and "may have cleared the warning bits" in line
    assert [packet[1] for packet in packets].count(90) == 1


@pytest.mark.parametrize("command", [21, 59, 75, 149])
def test_a_lost_first_reply_is_read_again(patched, capsys, command):
    _, out, packets = read_only(patched, capsys, bad_first={command})

    assert setting(out, command) == LINES[command]
    assert [packet[1] for packet in packets].count(command) == 2


@pytest.mark.parametrize("command, payload", [
    (75, bytes([0x00, 0x00, 0x14])),
    (81, struct.pack(">II", 7, 8)),
    (90, struct.pack(">H", 0x0040)),
])
def test_a_short_reply_with_a_bad_crc_fails(patched, capsys, command, payload):
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, command: payload}, bad_crc={command})

    assert "read failed" in setting(out, command) and "CRC mismatch" in setting(out, command)


def test_a_reply_of_another_length_fails(patched, capsys):
    """A 4-byte 75 reply is neither frame, even with a good CRC."""
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, 75: bytes([0, 0, 0x14, 0])})

    assert setting(out, 75) == (
        "S-pin modes (cmd 75): read failed: "
        "6 bytes on the wire, expected 5 or 3 data bytes and a 2-byte CRC"
    )


def test_the_pin_modes_have_the_manuals_names(patched, capsys):
    modes = bytes([0xC1, 0x41, 0x14, 0x04, 0x84])
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, 75: modes})

    assert setting(out, 75) == (
        "S-pin modes (cmd 75): S3 0xC1 E-Stop (Latching), S4 0x41 E-Stop, S5 0x14 Voltage Clamp, "
        "CTRL1 0x04 Brake, CTRL2 0x84 User Output (5-byte reply: S3, S4, S5, CTRL1, CTRL2)"
    )


def test_a_four_cell_battery_mode(patched, capsys):
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, 99: struct.pack(">H", 0x0013)})

    assert setting(out, 99) == (
        "Standard config (cmd 99): 0x0013 (packet serial mode, battery mode 4 cell)"
    )


def test_an_unlisted_pwm_mode_is_shown(patched, capsys):
    rc, out, _ = read_only(patched, capsys, replies={**REPLIES, 149: bytes([0, 7])})

    assert rc in (None, 0)
    assert setting(out, 149) == "PWM mode (cmd 149): M1 inductive, M2 unlisted 7"


def test_the_longest_idle_delay(patched, capsys):
    _, out, _ = read_only(patched, capsys, replies={**REPLIES, 161: bytes([0x7F, 0xFF])})

    assert setting(out, 161) == (
        "Idle modes (cmd 161): M1 braking for 12.7 s, then freewheel, "
        "M2 braking at zero (its 12.7 s delay unused)"
    )


def test_the_settings_reads_are_spaced(patched, capsys):
    """A gap after each read, so a partial packet left by a CRC byte in 0x80-0x87 is dropped."""
    slept = []
    read_only(patched, capsys, sleep=slept.append)

    assert slept.count(0.012) == len(LINES)
