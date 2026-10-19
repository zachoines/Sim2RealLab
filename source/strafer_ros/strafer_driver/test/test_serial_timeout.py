"""Tests for the RoboClaw serial timeout (commands 14 and 15) and the flash commands (94, 95).

``diagnose_roboclaw.py`` reads each controller's timeout and sets it only behind
``--set-serial-timeout``; ``--write-nvm`` reloads the saved settings (95) before
the set and saves them after it (94). It sends a motor command only behind
``--drive``. The node never sends any of the four.
The serial layer is test_port_fence's fake.
"""

import pytest
import serial

from strafer_driver.roboclaw_interface import RoboClawInterface

# test/ has no __init__.py, so pytest puts it on sys.path; the imported autouse fixtures apply here.
from test_port_fence import (  # noqa: F401
    CAMERA_PORT,
    CMD_READ_MAIN_BATTERY,
    CMD_READ_SERIAL_TIMEOUT,
    CMD_READ_SETTINGS_NVM,
    CMD_SET_SERIAL_TIMEOUT,
    CMD_WRITE_SETTINGS_NVM,
    DUTY_COMMANDS,
    FRONT,
    NVM_WRITE_KEY,
    FakeRoboClaw,
    _shutdown_rclpy,
    build_node,
    one_device_for_both,
    patched,
    robot,
    run_diagnose,
)

TIMEOUT_COMMANDS = (
    CMD_SET_SERIAL_TIMEOUT,
    CMD_READ_SERIAL_TIMEOUT,
    CMD_WRITE_SETTINGS_NVM,
    CMD_READ_SETTINGS_NVM,
)
SET_OR_SAVE = (CMD_SET_SERIAL_TIMEOUT, CMD_WRITE_SETTINGS_NVM)
M1_FORWARD, M2_FORWARD = DUTY_COMMANDS
# The manual's numbers, written out as in test_port_fence.
READ_SPEED_M1 = 18
DRIVE_M1M2_SPEED = 37
READ_COMMANDS = {CMD_READ_MAIN_BATTERY, 82, CMD_READ_SERIAL_TIMEOUT, 55, 56}
DUTY_HOLD_S = 1.0
NOTICE = "at duty 32/127 for 1 s each: the wheels must be lifted."

BOTH = ("--front", "/dev/roboclaw2", "--rear", "/dev/roboclaw1")
FRONT_ONLY = ("--front", "/dev/roboclaw2", "--front-only")
SET_AND_SAVE = ("--set-serial-timeout", "0.5", "--write-nvm")


def two_controllers():
    return robot(front_tty="/dev/ttyACM2", rear_tty="/dev/ttyACM1")


def front_controller(claw=None):
    """The camera, plus *claw* (a plain front RoboClaw by default) at ttyACM2 and roboclaw2."""
    bus = robot()
    if claw is None:
        claw = FakeRoboClaw(FRONT)
    bus.devices["/dev/ttyACM2"] = bus.devices["/dev/roboclaw2"] = claw
    return bus


def commands(bus, port):
    return [data[1] for _, data in bus.packets(port)]


class StickyTimeout(FakeRoboClaw):
    """Acknowledges command 14 but keeps its old value."""

    def respond(self, packet):
        kept = self.serial_timeout
        reply = super().respond(packet)
        self.serial_timeout = kept
        return reply


class Unanswered(FakeRoboClaw):
    """Acts on *silent* commands as FakeRoboClaw does, but sends no reply to them."""

    def __init__(self, address, *silent):
        super().__init__(address)
        self.silent = silent

    def respond(self, packet):
        reply = super().respond(packet)
        return b"" if packet[1] in self.silent else reply


class Transcript(list):
    """A bus's write log that also keeps what the tool printed before each packet."""

    def __init__(self, capsys):
        super().__init__()
        self.capsys = capsys
        self.printed = []

    def append(self, item):
        self.printed.append(self.capsys.readouterr().out)
        super().append(item)


class Raises(FakeRoboClaw):
    """Raises SerialException on *command*: *times* times, or every time."""

    def __init__(self, address, command, times=None):
        super().__init__(address)
        self.command = command
        self.times = times

    def respond(self, packet):
        if packet[1] == self.command and self.times != 0:
            if self.times is not None:
                self.times -= 1
            raise serial.SerialException("device reports readiness to read but returned no data")
        return super().respond(packet)


class TestDiagnoseRead:
    def test_prints_each_controllers_timeout(self, patched, capsys):
        """Without flags: command 15 on each port, its value printed, nothing set or saved."""
        bus = patched(two_controllers())
        bus.devices["/dev/roboclaw1"].serial_timeout = 5

        rc = run_diagnose(*BOTH)

        out = capsys.readouterr().out
        assert rc in (None, 0)
        assert "Serial timeout (cmd 15): 0.0 s (0 = disabled; the bring-up expects 0.5 s)" in out
        assert "Serial timeout (cmd 15): 0.5 s\n" in out
        for port in ("/dev/roboclaw2", "/dev/roboclaw1"):
            assert CMD_READ_SERIAL_TIMEOUT in commands(bus, port)
        sent = {data[1] for _, data in bus.writes}
        assert not {*SET_OR_SAVE, CMD_READ_SETTINGS_NVM} & sent

    @pytest.mark.parametrize("raw, noted", [(0, False), (5, True), (10, True), (11, False)])
    def test_duty_note_up_to_the_hold(self, patched, capsys, raw, noted):
        """The duty test notes the timeout for 0 < timeout <= its 1.0 s hold, 1.0 s included."""
        patched(front_controller(FakeRoboClaw(FRONT, serial_timeout=raw)))

        run_diagnose(*FRONT_ONLY, "--drive")

        assert ("that is the timeout working" in capsys.readouterr().out) is noted

    @pytest.mark.parametrize("flags", [(), ("--read-only",)], ids=["default", "read-only"])
    def test_reads_only_without_drive(self, patched, capsys, flags):
        """No --drive: battery, temperature, timeout and PID reads on each port; nothing else."""
        bus = patched(two_controllers())

        rc = run_diagnose(*BOTH, *flags)

        out = capsys.readouterr().out
        assert rc in (None, 0)
        for port in ("/dev/roboclaw2", "/dev/roboclaw1"):
            assert set(commands(bus, port)) == READ_COMMANDS
        assert out.count("Serial timeout (cmd 15): 0.0 s (0 = disabled") == 2
        assert out.count("the duty-cycle drive runs only with --drive") == 2
        assert NOTICE not in out


class TestDiagnoseDrive:
    def test_drive_runs_the_duty_test_after_one_notice(self, patched, capsys):
        """--drive: one line naming the four wheels, printed before the first motor packet."""
        bus = patched(two_controllers())
        bus.writes = Transcript(capsys)

        rc = run_diagnose(*BOTH, "--drive")

        first = min(i for i, (_, data) in enumerate(bus.writes) if data[1] in DUTY_COMMANDS)
        before = "".join(bus.writes.printed[: first + 1])
        out = "".join(bus.writes.printed) + capsys.readouterr().out
        assert rc in (None, 0)
        assert f"Driving FL, FR, RL, RR {NOTICE}" in before
        assert out.count(NOTICE) == 1
        for port in ("/dev/roboclaw2", "/dev/roboclaw1"):
            sent = [data[1:3] for _, data in bus.packets(port)]
            assert bytes([M1_FORWARD, 32]) in sent and bytes([M2_FORWARD, 32]) in sent

    @pytest.mark.parametrize(
        "only, wheels", [("--front-only", "FL, FR"), ("--rear-only", "RL, RR")]
    )
    def test_notice_names_the_tested_wheels(self, patched, capsys, only, wheels):
        patched(two_controllers())

        run_diagnose(*BOTH, only, "--drive")

        assert f"Driving {wheels} {NOTICE}" in capsys.readouterr().out

    def test_notice_names_only_the_detected_controller(self, patched, capsys):
        """Auto-detect finds the rear alone: the notice names RL and RR, the wheels it drives."""
        patched(robot(rear_tty="/dev/ttyACM1"))

        run_diagnose("--drive")

        assert f"Driving RL, RR {NOTICE}" in capsys.readouterr().out

    def test_no_notice_when_nothing_is_detected(self, patched, capsys):
        patched(robot())

        run_diagnose("--drive")

        assert NOTICE not in capsys.readouterr().out

    def test_help_names_the_drive(self, patched, capsys):
        """--drive's own help names the wheels, the duty and the hold, and the lifted wheels."""
        bus = patched(two_controllers())

        with pytest.raises(SystemExit) as exit_info:
            run_diagnose("--help")

        out = " ".join(capsys.readouterr().out.split())
        start = out.index("--drive After the reads")
        drive_help = out[start : out.index("--set-serial-timeout SECONDS", start)]
        assert exit_info.value.code == 0
        assert (
            "(FL, FR on 0x80; RL, RR on 0x81) at duty 32/127 for 1 s. The wheels must be lifted."
            in drive_help
        )
        assert bus.opens == []


class TestDiagnoseUnreached:
    @pytest.mark.parametrize(
        "flags", [(), ("--read-only",), ("--drive",)], ids=["default", "read-only", "drive"]
    )
    def test_silent_port_exits_non_zero(self, patched, flags):
        """An explicit port that opens but never answers: battery reads only, and exit 1."""
        bus = patched(robot())

        rc = run_diagnose(
            "--front", CAMERA_PORT, "--front-only", "--allow-unverified-ports", *flags
        )

        sent = bus.packets(CAMERA_PORT)
        assert rc == 1
        assert sent and all(data[1] == CMD_READ_MAIN_BATTERY for _, data in sent), sent

    def test_port_that_does_not_open_exits_non_zero(self, patched, capsys):
        """Both ports named, so nothing is detected: the front's open fails, and exit 1."""
        bus = patched(two_controllers())

        rc = run_diagnose(
            "--front", "/dev/roboclaw9", "--rear", "/dev/roboclaw1", "--front-only", "--read-only"
        )

        assert rc == 1
        assert "FATAL: Cannot open port" in capsys.readouterr().out
        assert bus.writes == []


class TestDiagnoseSet:
    def test_flagged_set_writes_and_reads_back(self, patched, capsys):
        """--set-serial-timeout 0.5: 14 with 5 on each port and a 15 read; no motor command."""
        bus = patched(two_controllers())

        rc = run_diagnose(*BOTH, "--set-serial-timeout", "0.5")

        out = capsys.readouterr().out
        assert rc in (None, 0)
        for port in ("/dev/roboclaw2", "/dev/roboclaw1"):
            sent = [data for _, data in bus.packets(port)]
            (i,) = [n for n, data in enumerate(sent) if data[1] == CMD_SET_SERIAL_TIMEOUT]
            assert sent[i][2] == 5
            assert CMD_READ_SERIAL_TIMEOUT in [data[1] for data in sent[i + 1 :]]
            assert set(commands(bus, port)) == READ_COMMANDS | {CMD_SET_SERIAL_TIMEOUT}
            assert bus.devices[port].serial_timeout == 5
        assert out.count("Serial timeout set to 0.5 s (cmd 14); read back 0.5 s (cmd 15)") == 2
        assert out.count("the duty-cycle drive runs only with --drive") == 2
        assert CMD_WRITE_SETTINGS_NVM not in {data[1] for _, data in bus.writes}

    def test_set_writes_whole_tenths(self, patched):
        """2.3 s goes out as 23, not the 22 that float 2.3 / 0.1 truncates to."""
        bus = patched(front_controller())

        rc = run_diagnose(*FRONT_ONLY, "--set-serial-timeout", "2.3")

        (packet,) = [data for _, data in bus.packets() if data[1] == CMD_SET_SERIAL_TIMEOUT]
        assert rc in (None, 0)
        assert packet[2] == 23

    @pytest.mark.parametrize("write_nvm", [False, True])
    def test_read_back_mismatch_fails(self, patched, capsys, write_nvm):
        """A controller that acks 14 but keeps its value: non-zero, nothing saved or driven."""
        claw = StickyTimeout(FRONT)
        bus = patched(front_controller(claw))
        flags = ["--set-serial-timeout", "0.5"] + (["--write-nvm"] if write_nvm else [])

        rc = run_diagnose(*FRONT_ONLY, *flags)

        sent = commands(bus, "/dev/roboclaw2")
        assert rc not in (None, 0)
        assert "read back 0.0 s (cmd 15)" in capsys.readouterr().out
        assert sent[-2:] == [CMD_SET_SERIAL_TIMEOUT, CMD_READ_SERIAL_TIMEOUT]
        assert CMD_WRITE_SETTINGS_NVM not in sent and claw.nvm_writes == 0
        assert not set(DUTY_COMMANDS) & set(sent)

    @pytest.mark.parametrize("write_nvm", [False, True])
    def test_unacknowledged_set_fails(self, patched, capsys, write_nvm):
        """No ack for 14, though 15 then reads the target: non-zero, nothing saved or driven."""
        claw = Unanswered(FRONT, CMD_SET_SERIAL_TIMEOUT)
        claw.serial_timeout = 5  # the target already; under --write-nvm the reload restores 0
        bus = patched(front_controller(claw))
        flags = ["--set-serial-timeout", "0.5"] + (["--write-nvm"] if write_nvm else [])

        rc = run_diagnose(*FRONT_ONLY, *flags)

        sent = commands(bus, "/dev/roboclaw2")
        assert rc not in (None, 0)
        assert "(cmd 14) was not acknowledged; read back 0.5 s" in capsys.readouterr().out
        assert sent[-2:] == [CMD_SET_SERIAL_TIMEOUT, CMD_READ_SERIAL_TIMEOUT]
        assert CMD_WRITE_SETTINGS_NVM not in sent and claw.nvm_writes == 0
        assert not set(DUTY_COMMANDS) & set(sent)

    @pytest.mark.parametrize("found", ["front", "rear"])
    def test_missing_controller_fails_the_set(self, patched, capsys, found):
        """Auto-detect finds one controller: the set reaches it, and the run still fails."""
        if found == "front":
            bus, missing = patched(robot(front_tty="/dev/ttyACM2")), "Rear RoboClaw (0x81)"
        else:
            bus, missing = patched(robot(rear_tty="/dev/ttyACM1")), "Front RoboClaw (0x80)"

        rc = run_diagnose("--set-serial-timeout", "0.5")

        assert rc not in (None, 0)
        assert f"{missing}: not detected" in capsys.readouterr().out
        assert CMD_SET_SERIAL_TIMEOUT in {data[1] for _, data in bus.writes}

    def test_set_error_fails(self, patched, capsys):
        """14 raising SerialException: non-zero, nothing driven."""
        bus = patched(front_controller(Raises(FRONT, CMD_SET_SERIAL_TIMEOUT)))

        rc = run_diagnose(*FRONT_ONLY, "--set-serial-timeout", "0.5")

        sent = commands(bus, "/dev/roboclaw2")
        assert rc not in (None, 0)
        assert "(cmd 14) or its read-back failed" in capsys.readouterr().out
        assert not set(DUTY_COMMANDS) & set(sent)

    def test_unanswered_timeout_read_sets_nothing(self, patched, capsys):
        """A controller that answers the battery read but not command 15 gets no 14: non-zero."""
        bus = patched(front_controller(Unanswered(FRONT, CMD_READ_SERIAL_TIMEOUT)))

        rc = run_diagnose(*FRONT_ONLY, "--set-serial-timeout", "0.5")

        sent = commands(bus, "/dev/roboclaw2")
        assert rc not in (None, 0)
        assert "Not setting it: the controller did not answer the read." in capsys.readouterr().out
        assert CMD_READ_SERIAL_TIMEOUT in sent
        assert not {CMD_SET_SERIAL_TIMEOUT, *DUTY_COMMANDS} & set(sent)

    def test_silent_port_fails_the_set(self, patched):
        """Under the override, a port that does not answer gets only battery reads: non-zero."""
        bus = patched(robot())

        rc = run_diagnose(
            "--front",
            CAMERA_PORT,
            "--front-only",
            "--allow-unverified-ports",
            "--set-serial-timeout",
            "0.5",
        )

        sent = bus.packets(CAMERA_PORT)
        assert rc not in (None, 0)
        assert sent, "the battery read was not sent"
        assert all(data[1] == CMD_READ_MAIN_BATTERY for _, data in sent), sent


class TestDiagnoseWriteNvm:
    def test_set_then_write_nvm(self, patched, capsys):
        """95, 15, 14(5), 15, then 94 with the key; one reload, one flash write, nothing after."""
        bus = patched(front_controller())
        claw = bus.devices["/dev/roboclaw2"]

        rc = run_diagnose(*FRONT_ONLY, *SET_AND_SAVE)

        sent = [data for _, data in bus.packets("/dev/roboclaw2")]
        assert rc in (None, 0)
        assert [data[1] for data in sent if data[1] in TIMEOUT_COMMANDS] == [
            CMD_READ_SETTINGS_NVM,
            CMD_READ_SERIAL_TIMEOUT,
            CMD_SET_SERIAL_TIMEOUT,
            CMD_READ_SERIAL_TIMEOUT,
            CMD_WRITE_SETTINGS_NVM,
        ]
        assert [data[1] for data in sent[-3:]] == [
            CMD_SET_SERIAL_TIMEOUT,
            CMD_READ_SERIAL_TIMEOUT,
            CMD_WRITE_SETTINGS_NVM,
        ]
        assert sent[-3][2] == 5
        assert sent[-1][2:6] == NVM_WRITE_KEY == bytes([0xE2, 0x2E, 0xAB, 0x7A])
        assert (claw.reloads, claw.nvm_writes, claw.saved_serial_timeout) == (1, 1, 5)
        assert bus.writes[-1] == ("/dev/roboclaw2", sent[-1])
        out = capsys.readouterr().out
        assert "the same command run again reloads the saved settings (cmd 95)" in out
        assert "with --read-only" not in out

    @pytest.mark.parametrize(
        "make_claw",
        [
            lambda: Unanswered(FRONT, CMD_READ_SETTINGS_NVM),
            lambda: Raises(FRONT, CMD_READ_SETTINGS_NVM),
        ],
        ids=["no-ack", "serial-error"],
    )
    def test_failed_reload_sets_nothing(self, patched, capsys, make_claw):
        """95 not acknowledged, or failing: non-zero before any 14 or 94."""
        bus = patched(front_controller(make_claw()))

        rc = run_diagnose(*FRONT_ONLY, *SET_AND_SAVE)

        sent = commands(bus, "/dev/roboclaw2")
        assert rc not in (None, 0)
        assert "did not reload its saved settings (cmd 95)" in capsys.readouterr().out
        assert sent.count(CMD_READ_SETTINGS_NVM) == 1
        assert not {*SET_OR_SAVE, *DUTY_COMMANDS} & set(sent)

    def test_reload_discards_an_unsaved_value(self, patched, capsys):
        """Active 0.3 s over a saved 0: 15 reads 0 after the reload, so 0.3 s is set and saved."""
        claw = FakeRoboClaw(FRONT)
        claw.serial_timeout = 3
        bus = patched(front_controller(claw))

        rc = run_diagnose(*FRONT_ONLY, "--set-serial-timeout", "0.3", "--write-nvm")

        sent = commands(bus, "/dev/roboclaw2")
        assert rc in (None, 0)
        assert "Serial timeout (cmd 15): 0.0 s (0 = disabled" in capsys.readouterr().out
        assert sent.index(CMD_READ_SETTINGS_NVM) < sent.index(CMD_SET_SERIAL_TIMEOUT)
        assert sent.count(CMD_WRITE_SETTINGS_NVM) == 1
        assert (claw.serial_timeout, claw.saved_serial_timeout) == (3, 3)

    def test_saved_value_already_the_target_sets_nothing(self, patched, capsys):
        """The reloaded value is the target: zero, with no 14, no 94 and no drive."""
        claw = FakeRoboClaw(FRONT)
        claw.saved_serial_timeout = 5
        bus = patched(front_controller(claw))

        rc = run_diagnose(*FRONT_ONLY, *SET_AND_SAVE)

        sent = commands(bus, "/dev/roboclaw2")
        assert rc in (None, 0)
        assert "already reads 0.5 s, so nothing is set or saved" in capsys.readouterr().out
        assert claw.reloads == 1
        assert not {*SET_OR_SAVE, *DUTY_COMMANDS} & set(sent)

    def test_unacknowledged_flash_write_fails(self, patched, capsys):
        bus = patched(front_controller(Unanswered(FRONT, CMD_WRITE_SETTINGS_NVM)))

        rc = run_diagnose(*FRONT_ONLY, *SET_AND_SAVE)

        out = capsys.readouterr().out
        assert rc not in (None, 0)
        assert "No ack for command 94" in out
        assert "run the same command again, not --read-only" in out
        assert commands(bus, "/dev/roboclaw2").count(CMD_WRITE_SETTINGS_NVM) == 1

    def test_flash_write_error_is_not_retried(self, patched, capsys):
        """94 raising SerialException once: exactly one 94 packet, and non-zero."""
        claw = Raises(FRONT, CMD_WRITE_SETTINGS_NVM, times=1)
        bus = patched(front_controller(claw))

        rc = run_diagnose(*FRONT_ONLY, *SET_AND_SAVE)

        assert rc not in (None, 0)
        assert "Command 94 failed" in capsys.readouterr().out
        assert commands(bus, "/dev/roboclaw2").count(CMD_WRITE_SETTINGS_NVM) == 1
        assert claw.nvm_writes == 0


class TestDiagnoseDutyStops:
    def test_failed_speed_read_still_stops_the_motor(self, patched, capsys):
        """SerialException on M1's speed read in the hold: duty 0 to M1 before M2 is driven."""
        bus = patched(front_controller(Raises(FRONT, READ_SPEED_M1)))

        run_diagnose(*FRONT_ONLY, "--drive")

        sent = [data for _, data in bus.packets("/dev/roboclaw2")]
        failed = max(i for i, data in enumerate(sent) if data[1] == READ_SPEED_M1)
        m2_driven = min(i for i, data in enumerate(sent) if data[1] == M2_FORWARD)
        assert [data[:3] for data in sent[failed + 1 : m2_driven]] == [
            bytes([FRONT, M1_FORWARD, 0])
        ]
        assert "ERROR" in capsys.readouterr().out

    def test_interrupted_hold_stops_the_motor(self, patched):
        """Ctrl-C during M1's hold: duty 0 to M1, then the interrupt carries on."""
        bus = patched(front_controller())

        def sleep(seconds):
            if seconds == DUTY_HOLD_S:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run_diagnose(*FRONT_ONLY, "--drive", sleep=sleep)

        sent = [data[:3] for _, data in bus.packets("/dev/roboclaw2")]
        assert sent[-2:] == [bytes([FRONT, M1_FORWARD, 32]), bytes([FRONT, M1_FORWARD, 0])]

    def test_failed_stops_are_swallowed(self, patched):
        """Every duty packet to M1 fails: M2 is still driven, and both final stops still go out."""
        bus = patched(front_controller(Raises(FRONT, M1_FORWARD)))

        rc = run_diagnose(*FRONT_ONLY, "--drive")

        sent = [data for _, data in bus.packets("/dev/roboclaw2")]
        assert rc in (None, 0)
        assert bytes([FRONT, M2_FORWARD, 32]) in [data[:3] for data in sent]
        assert [data[:3] for data in sent[-3:-1]] == [
            bytes([FRONT, M1_FORWARD, 0]),
            bytes([FRONT, M2_FORWARD, 0]),
        ]
        assert sent[-1][1] == DRIVE_M1M2_SPEED

    def test_final_stop_sends_duty_zero_to_both_motors(self, patched):
        """After both holds: duty 0 to M1 and M2, then the velocity stop."""
        bus = patched(front_controller())

        run_diagnose(*FRONT_ONLY, "--drive")

        sent = [data for _, data in bus.packets("/dev/roboclaw2")]
        assert [data[:3] for data in sent[-3:-1]] == [
            bytes([FRONT, M1_FORWARD, 0]),
            bytes([FRONT, M2_FORWARD, 0]),
        ]
        assert sent[-1][1] == DRIVE_M1M2_SPEED


class TestDiagnoseOneDevice:
    @pytest.mark.parametrize(
        "argv",
        [(), ("--front", "/dev/roboclaw1", "--rear", "/dev/roboclaw1"), SET_AND_SAVE],
        ids=["detected", "explicit", "set-and-save"],
    )
    def test_one_device_for_both_refused(self, patched, capsys, argv):
        """Both passes on one device: exit 2, with nothing sent but detection's probes."""
        bus = patched(one_device_for_both(robot(), "/dev/ttyACM1", "/dev/roboclaw1"))

        rc = run_diagnose(*argv)

        assert rc == 2
        assert "name one device (/dev/roboclaw1, /dev/roboclaw1)" in capsys.readouterr().out
        assert all(data[1] == CMD_READ_MAIN_BATTERY for _, data in bus.writes), bus.writes

    def test_aliased_paths_refused(self, patched, capsys, tmp_path):
        """Two paths that resolve to one device (a symlink): exit 2 before any open."""
        target = tmp_path / "claw"
        target.touch()
        alias = tmp_path / "alias"
        alias.symlink_to(target)
        bus = patched(one_device_for_both(robot(), str(target), str(alias)))

        rc = run_diagnose("--front", str(target), "--rear", str(alias), "--allow-unverified-ports")

        assert rc == 2
        assert "name one device" in capsys.readouterr().out
        assert bus.opens == []

    def test_one_device_alone_is_tested(self, patched):
        """--front-only on that device: its single pass runs."""
        bus = patched(one_device_for_both(robot(), "/dev/ttyACM1", "/dev/roboclaw1"))

        rc = run_diagnose("--front-only", "--drive")

        assert rc in (None, 0)
        assert set(DUTY_COMMANDS) <= set(commands(bus, "/dev/roboclaw1"))


class TestDiagnoseUsage:
    def test_write_nvm_needs_set(self, patched, capsys):
        """--write-nvm alone is a usage error, before detection opens anything."""
        bus = patched(two_controllers())

        with pytest.raises(SystemExit) as exit_info:
            run_diagnose("--write-nvm")

        assert exit_info.value.code == 2
        assert "--write-nvm needs --set-serial-timeout" in capsys.readouterr().err
        assert bus.opens == []

    @pytest.mark.parametrize(
        "value",
        ["25.6", "-0.1", "0.05", "0.25", "1.23", "0.1" + "0" * 30 + "1", "nan", "abc"],
    )
    def test_timeout_off_the_tenths_refused(self, patched, capsys, value):
        """Command 14 takes whole tenths from 0 to 25.5 s; anything else is a usage error."""
        bus = patched(two_controllers())

        with pytest.raises(SystemExit) as exit_info:
            run_diagnose(f"--set-serial-timeout={value}")

        assert exit_info.value.code == 2
        assert "not a whole number of 0.1 s steps from 0 to 25.5 s" in capsys.readouterr().err
        assert bus.opens == []

    @pytest.mark.parametrize(
        "argv, refusal",
        [
            (
                ("--read-only", "--set-serial-timeout", "0.5"),
                "not allowed with argument --read-only",
            ),
            (("--read-only", "--drive"), "not allowed with argument --read-only"),
            (
                ("--drive", "--set-serial-timeout", "0.5"),
                "not allowed with argument --drive",
            ),
            (("--front-only", "--rear-only"), "not allowed with argument --front-only"),
        ],
        ids=[
            "read-only-and-set",
            "read-only-and-drive",
            "drive-and-set",
            "front-only-and-rear-only",
        ],
    )
    def test_exclusive_flags_refused(self, patched, capsys, argv, refusal):
        bus = patched(two_controllers())

        with pytest.raises(SystemExit) as exit_info:
            run_diagnose(*argv)

        assert exit_info.value.code == 2
        assert refusal in capsys.readouterr().err
        assert bus.opens == []


def test_node_never_sends_timeout_or_flash_commands(patched):
    """The node runs its loop and a reconnect without sending 14, 15, 94 or 95."""
    bus = patched(two_controllers())
    node, refusal = build_node(auto_detect="true")
    assert refusal is None, repr(refusal)
    try:
        for _ in range(5):
            node._timer_callback()
        node._in_error_state = True
        node._last_reconnect_time = 0.0
        node._timer_callback()
        assert not node._in_error_state
    finally:
        node.destroy_node()

    sent = [data for _, data in bus.writes]
    assert len(sent) > 10
    assert not [data for data in sent if data[1] in TIMEOUT_COMMANDS]


def test_interface_round_trip(patched):
    """set_serial_timeout writes whole tenths and reads them back."""
    bus = patched(robot(front_tty="/dev/ttyACM2"))
    fake = bus.devices["/dev/roboclaw2"]
    claw = RoboClawInterface("/dev/roboclaw2", FRONT)
    claw.open()

    assert claw.set_serial_timeout(0.5) is True
    assert fake.serial_timeout == 5
    assert claw.read_serial_timeout() == 0.5
    for seconds, raw in ((2.3, 23), (0.1 + 0.2, 3), (1 - 0.9, 1), (25.5, 255), (0, 0)):
        assert claw.set_serial_timeout(seconds) is True
        assert fake.serial_timeout == raw, seconds
    claw.close()


@pytest.mark.parametrize("seconds", [0.05, 2.25, 1.23, 25.6, -0.1, float("inf"), float("nan")])
def test_interface_refuses_off_the_tenths(patched, seconds):
    """A value that is not a whole number of tenths from 0 to 25.5 s raises and sends nothing."""
    bus = patched(robot(front_tty="/dev/ttyACM2"))
    claw = RoboClawInterface("/dev/roboclaw2", FRONT)
    claw.open()

    with pytest.raises(ValueError, match="0.1 s steps"):
        claw.set_serial_timeout(seconds)

    assert bus.writes == []
    claw.close()
