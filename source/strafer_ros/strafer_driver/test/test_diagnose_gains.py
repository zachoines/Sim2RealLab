"""Tests for ``diagnose_roboclaw.py --set-gains`` and its flash save.

``--set-gains`` writes the driver's velocity gains (strafer_shared.constants) to both motors
of each controller (commands 28 and 29) and reads them back (55 and 56). With ``--write-nvm``
it reloads the saved settings (95) first, saves (94) only after an exact read-back, then finds
the controller again by its address, reloads (95) and reads the saved gains. It sends no motor
command. The serial layer is test_port_fence's fake.
"""

import struct

import pytest
import serial

from strafer_shared.constants import (
    ROBOCLAW_PID_D,
    ROBOCLAW_PID_I,
    ROBOCLAW_PID_P,
    ROBOCLAW_QPPS,
)

# test/ has no __init__.py, so pytest puts it on sys.path; the imported autouse fixtures apply here.
from test_port_fence import (  # noqa: F401
    CMD_READ_MAIN_BATTERY,
    CMD_READ_SERIAL_TIMEOUT,
    CMD_READ_SETTINGS_NVM,
    CMD_SET_SERIAL_TIMEOUT,
    CMD_WRITE_SETTINGS_NVM,
    DUTY_COMMANDS,
    FRONT,
    NVM_WRITE_KEY,
    REAR,
    FakeRoboClaw,
    _shutdown_rclpy,
    patched,
    robot,
    run_diagnose,
)
from test_serial_timeout import Raises, Unanswered, commands, front_controller, two_controllers

# The manual's numbers, written out as in test_port_fence.
SET_M1_PID, SET_M2_PID = 28, 29
READ_M1_PID, READ_M2_PID = 55, 56
SPEED_COMMANDS = {35, 36, 37}
MOTOR_COMMANDS = {*DUTY_COMMANDS, 1, 5, *SPEED_COMMANDS}
GAINS = (ROBOCLAW_PID_P, ROBOCLAW_PID_I, ROBOCLAW_PID_D, ROBOCLAW_QPPS)  # P, I, D, QPPS
OLD = (15000, 750, 0, 2796)
assert OLD != GAINS, "the tests need saved gains that differ from the driver's"

BOTH = ("--front", "/dev/roboclaw2", "--rear", "/dev/roboclaw1")
FRONT_ONLY = ("--front", "/dev/roboclaw2", "--front-only")
SAVE = ("--set-gains", "--write-nvm")


def saved_front(gains=OLD):
    """A front RoboClaw whose flash holds *gains*, loaded into RAM as at power-up."""
    claw = FakeRoboClaw(FRONT)
    claw.saved_gains = {1: gains, 2: gains}
    claw.gains = dict(claw.saved_gains)
    return claw


def gains_packets(bus, port):
    """(command, (P, I, D, QPPS)) for each gains write on *port*; packets carry D, P, I, QPPS."""
    out = []
    for _, data in bus.packets(port):
        if data[1] in (SET_M1_PID, SET_M2_PID):
            d, p, i, qpps = struct.unpack(">IIII", data[2:18])
            out.append((data[1], (p, i, d, qpps)))
    return out


class StickyGains(FakeRoboClaw):
    """Acknowledges commands 28 and 29 but keeps its old gains."""

    def respond(self, packet):
        kept = dict(self.gains)
        reply = super().respond(packet)
        self.gains = kept
        return reply


class StickyM2(FakeRoboClaw):
    """Acknowledges command 29 but keeps M2's old gains."""

    def respond(self, packet):
        kept = self.gains[2]
        reply = super().respond(packet)
        self.gains[2] = kept
        return reply


class NoQpps(FakeRoboClaw):
    """Takes P, I and D from commands 28 and 29 but keeps its QPPS."""

    def respond(self, packet):
        kept = {m: g[3] for m, g in self.gains.items()}
        reply = super().respond(packet)
        self.gains = {m: (*g[:3], kept[m]) for m, g in self.gains.items()}
        return reply


class Resetting(FakeRoboClaw):
    """A RoboClaw (the front by default) whose acknowledged flash write resets it.

    The reset loads the flash into RAM (all of it, or with *lose_m2* M2's previous saved gains;
    with *keeps_flash* false the write saved nothing). After it, *silent* commands go unanswered
    and *raising* ones raise.
    """

    def __init__(
        self, gains=OLD, keeps_flash=True, lose_m2=False, silent=(), raising=(), address=FRONT
    ):
        super().__init__(address)
        self.saved_gains = {1: gains, 2: gains}
        self.gains = dict(self.saved_gains)
        self.keeps_flash, self.lose_m2 = keeps_flash, lose_m2
        self.silent, self.raising = silent, raising
        self.on_reset = None
        self.was_reset = False

    def respond(self, packet):
        if self.was_reset and packet[1:2] and packet[1] in self.raising:
            raise serial.SerialException("device reports readiness to read but returned no data")
        saved_before = dict(self.saved_gains)
        reply = super().respond(packet)
        if len(packet) > 1 and packet[1] == CMD_WRITE_SETTINGS_NVM and reply == b"\xff":
            if not self.keeps_flash:
                self.saved_gains = saved_before
            elif self.lose_m2:
                self.saved_gains[2] = saved_before[2]
            self.gains = dict(self.saved_gains)
            self.was_reset = True
            if self.on_reset:
                self.on_reset()
        if self.was_reset and packet[1:2] and packet[1] in self.silent:
            return b""
        return reply


def resets_on(bus, claw, old="2", new="2", back_after=1):
    """Put *claw* on *bus* as ttyACM<old>; its reset drops it, and it re-enumerates as ttyACM<new>.

    It is back once the bus has been listed *back_after* times since the reset.
    """
    names = ("/dev/ttyACM", "/dev/roboclaw")
    for prefix in names:
        bus.devices[prefix + old] = claw
    state = {"gone": False, "looks": 0}
    listing = bus.glob

    def drop():
        for prefix in names:
            bus.devices.pop(prefix + old, None)
        state["gone"] = True

    def glob(pattern):
        if state["gone"]:
            state["looks"] += 1
            if state["looks"] > back_after:
                for prefix in names:
                    bus.devices[prefix + new] = claw
                state["gone"] = False
        return listing(pattern)

    claw.on_reset = drop
    bus.glob = glob
    return claw


def front_that_resets(patched, **kwargs):
    """A robot with only a resetting front at ttyACM2; returns (bus, claw)."""
    bus = patched(robot())
    back = {k: kwargs.pop(k) for k in ("new", "back_after") if k in kwargs}
    return bus, resets_on(bus, Resetting(**kwargs), **back)


class TestSetGains:
    def test_writes_the_driver_gains_and_reads_them_back(self, patched, capsys):
        """28 and 29 carry the constants to both controllers, then read back; no flash or motor."""
        bus = patched(two_controllers())

        rc = run_diagnose(*BOTH, "--set-gains")

        out = capsys.readouterr().out
        assert rc in (None, 0)
        for port in ("/dev/roboclaw2", "/dev/roboclaw1"):
            sent = commands(bus, port)
            assert gains_packets(bus, port) == [(SET_M1_PID, GAINS), (SET_M2_PID, GAINS)]
            last_set = max(i for i, c in enumerate(sent) if c in (SET_M1_PID, SET_M2_PID))
            assert {READ_M1_PID, READ_M2_PID} <= set(sent[last_set + 1 :])
            assert bus.devices[port].gains == {1: GAINS, 2: GAINS}
            assert not {CMD_WRITE_SETTINGS_NVM, CMD_READ_SETTINGS_NVM} & set(sent)
            assert not MOTOR_COMMANDS & set(sent)
        p, i, d, qpps = GAINS
        line = f"Velocity gains set to P={p} I={i} D={d} QPPS={qpps} (cmd 28, 29); read back"
        assert out.count(line) == 2

    def test_tool_gains_are_the_constants(self):
        import importlib.util

        from test_port_fence import DIAGNOSE

        spec = importlib.util.spec_from_file_location("diagnose_roboclaw", DIAGNOSE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert tuple(module.DRIVER_GAINS[k] for k in ("P", "I", "D", "QPPS")) == GAINS

    @pytest.mark.parametrize("write_nvm", [False, True])
    def test_read_back_mismatch_fails(self, patched, capsys, write_nvm):
        """A controller that acks 28 and 29 but keeps its gains: non-zero, nothing saved."""
        claw = StickyGains(FRONT)
        bus = patched(front_controller(claw))

        rc = run_diagnose(*FRONT_ONLY, "--set-gains", *(["--write-nvm"] if write_nvm else []))

        sent = commands(bus, "/dev/roboclaw2")
        assert rc not in (None, 0)
        out = capsys.readouterr().out
        assert "The set did not take; nothing more is written to this port." in out
        assert sent[-2:] == [READ_M1_PID, READ_M2_PID]
        assert CMD_WRITE_SETTINGS_NVM not in sent and claw.nvm_writes == 0
        assert not MOTOR_COMMANDS & set(sent)

    @pytest.mark.parametrize("command", [SET_M1_PID, SET_M2_PID])
    def test_unacknowledged_set_fails(self, patched, capsys, command):
        bus = patched(front_controller(Unanswered(FRONT, command)))

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        sent = commands(bus, "/dev/roboclaw2")
        assert rc not in (None, 0)
        assert "not acknowledged" in capsys.readouterr().out
        assert CMD_WRITE_SETTINGS_NVM not in sent

    def test_failed_gains_read_sets_nothing(self, patched, capsys):
        claw = Raises(FRONT, READ_M1_PID)
        bus = patched(front_controller(claw))

        rc = run_diagnose(*FRONT_ONLY, "--set-gains")

        sent = commands(bus, "/dev/roboclaw2")
        assert rc not in (None, 0)
        assert "Not setting them." in capsys.readouterr().out
        assert not {SET_M1_PID, SET_M2_PID, CMD_WRITE_SETTINGS_NVM} & set(sent)

    def test_missing_controller_fails(self, patched, capsys):
        """Only the front detected: the front is set, and the run exits non-zero for the rear."""
        bus = patched(robot(front_tty="/dev/ttyACM2"))

        rc = run_diagnose("--set-gains")

        assert rc not in (None, 0)
        assert "Rear RoboClaw (0x81): not detected" in capsys.readouterr().out
        assert bus.devices["/dev/roboclaw2"].gains == {1: GAINS, 2: GAINS}

    def test_m2_read_back_mismatch_fails_before_the_save(self, patched):
        bus = patched(front_controller(StickyM2(FRONT)))

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        assert rc not in (None, 0)
        assert CMD_WRITE_SETTINGS_NVM not in commands(bus, "/dev/roboclaw2")

    def test_qpps_not_taken_fails_before_the_save(self, patched):
        bus = patched(front_controller(NoQpps(FRONT)))

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        assert rc not in (None, 0)
        assert CMD_WRITE_SETTINGS_NVM not in commands(bus, "/dev/roboclaw2")

    def test_set_error_fails_before_the_save(self, patched, capsys):
        bus = patched(front_controller(Raises(FRONT, SET_M2_PID)))

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        assert rc not in (None, 0)
        assert "or reading them back failed" in capsys.readouterr().out
        assert CMD_WRITE_SETTINGS_NVM not in commands(bus, "/dev/roboclaw2")

    def test_set_alone_always_writes(self, patched):
        """Without --write-nvm, gains already the driver's are written again."""
        bus = patched(front_controller(saved_front(GAINS)))

        rc = run_diagnose(*FRONT_ONLY, "--set-gains")

        assert rc in (None, 0)
        assert gains_packets(bus, "/dev/roboclaw2") == [(SET_M1_PID, GAINS), (SET_M2_PID, GAINS)]


class TestSaveGains:
    def test_reload_set_save_then_check_what_the_reset_loaded(self, patched, capsys):
        """55/56, 95, 55/56, 28/29, 55/56, 94; after the reset, 55, 56 and 15 with no 95."""
        bus, claw = front_that_resets(patched)

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        sent = commands(bus, "/dev/roboclaw2")
        flow = [c for c in sent if c not in (CMD_READ_MAIN_BATTERY, 82)]
        assert rc in (None, 0)
        assert flow == [
            READ_M1_PID, READ_M2_PID,
            CMD_READ_SETTINGS_NVM,
            CMD_READ_SERIAL_TIMEOUT,
            READ_M1_PID, READ_M2_PID,
            SET_M1_PID, SET_M2_PID,
            READ_M1_PID, READ_M2_PID,
            CMD_WRITE_SETTINGS_NVM,
            READ_M1_PID, READ_M2_PID, CMD_READ_SERIAL_TIMEOUT,
        ]
        packets = bus.packets("/dev/roboclaw2")
        (key,) = [data[2:6] for _, data in packets if data[1] == CMD_WRITE_SETTINGS_NVM]
        assert key == NVM_WRITE_KEY
        assert (claw.reloads, claw.nvm_writes) == (1, 1)
        assert claw.saved_gains == {1: GAINS, 2: GAINS}
        out = capsys.readouterr().out
        assert "/dev/roboclaw2 dropped after" in out
        assert "Front answers on /dev/roboclaw2" in out
        assert "Saved and verified." in out
        assert not MOTOR_COMMANDS & set(sent)

    def test_prints_active_then_saved_gains(self, patched, capsys):
        """The driver's gains in RAM over old saved ones: both are printed, the old after 95."""
        bus, claw = front_that_resets(patched)
        claw.gains = {1: GAINS, 2: GAINS}

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        out = capsys.readouterr().out
        assert rc in (None, 0)
        p, i, d, qpps = GAINS
        active = out.index(f"in active memory (cmd 55, 56): M1 P={p} I={i} D={d} QPPS={qpps}")
        p, i, d, qpps = OLD
        saved = out.index(f"Velocity gains as saved (cmd 55, 56): M1 P={p} I={i} D={d} QPPS={qpps}")
        assert active < saved
        assert claw.saved_gains == {1: GAINS, 2: GAINS}

    def test_saves_even_when_the_saved_gains_look_right(self, patched, capsys):
        """No shortcut: saved gains already the driver's are set and saved again, and checked."""
        bus, claw = front_that_resets(patched, gains=GAINS)

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        assert rc in (None, 0)
        assert commands(bus, "/dev/roboclaw2").count(CMD_WRITE_SETTINGS_NVM) == 1
        assert "Saved and verified." in capsys.readouterr().out

    def test_only_m2_unsaved_is_saved(self, patched):
        bus, claw = front_that_resets(patched)
        claw.saved_gains = {1: GAINS, 2: OLD}
        claw.gains = dict(claw.saved_gains)

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        assert rc in (None, 0)
        assert claw.saved_gains == {1: GAINS, 2: GAINS}

    def test_the_timeout_is_saved_as_it_was(self, patched):
        """With the reload first, the save carries the saved timeout, not an unsaved one."""
        bus, claw = front_that_resets(patched)
        claw.saved_serial_timeout, claw.serial_timeout = 5, 3

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        assert rc in (None, 0)
        assert claw.saved_serial_timeout == 5
        assert CMD_SET_SERIAL_TIMEOUT not in commands(bus, "/dev/roboclaw2")

    @pytest.mark.parametrize(
        "make_claw",
        [
            lambda: Unanswered(FRONT, CMD_READ_SETTINGS_NVM),
            lambda: Raises(FRONT, CMD_READ_SETTINGS_NVM),
        ],
        ids=["no-ack", "serial-error"],
    )
    def test_failed_reload_sets_nothing(self, patched, capsys, make_claw):
        bus = patched(front_controller(make_claw()))

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        sent = commands(bus, "/dev/roboclaw2")
        assert rc not in (None, 0)
        assert "did not reload its saved settings (cmd 95)" in capsys.readouterr().out
        assert not {SET_M1_PID, SET_M2_PID, CMD_WRITE_SETTINGS_NVM} & set(sent)

    def test_unacknowledged_flash_write_fails_unverified(self, patched, capsys):
        claw = Unanswered(FRONT, CMD_WRITE_SETTINGS_NVM)
        claw.saved_gains = {1: OLD, 2: OLD}
        bus = patched(front_controller(claw))

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        sent = commands(bus, "/dev/roboclaw2")
        out = capsys.readouterr().out
        assert rc not in (None, 0)
        assert "No ack for command 94" in out
        assert "run the same command again, not --read-only" in out
        assert sent.count(CMD_WRITE_SETTINGS_NVM) == 1
        assert sent[-1] == CMD_WRITE_SETTINGS_NVM, "nothing follows an unacknowledged save"

    def test_flash_write_error_is_not_retried(self, patched, capsys):
        claw = Raises(FRONT, CMD_WRITE_SETTINGS_NVM, times=1)
        claw.saved_gains = {1: OLD, 2: OLD}
        bus = patched(front_controller(claw))

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        assert rc not in (None, 0)
        assert "Command 94 failed" in capsys.readouterr().out
        assert commands(bus, "/dev/roboclaw2").count(CMD_WRITE_SETTINGS_NVM) == 1
        assert claw.nvm_writes == 0

    def test_no_reset_seen_fails_unverified(self, patched, capsys):
        """94 acked but the port never drops: no reset was seen, so non-zero, nothing read."""
        claw = saved_front()
        bus = patched(front_controller(claw))
        slept = []

        rc = run_diagnose(*FRONT_ONLY, *SAVE, sleep=slept.append)

        sent = commands(bus, "/dev/roboclaw2")
        assert rc not in (None, 0)
        assert "so no reset was seen and the save is unverified" in capsys.readouterr().out
        assert sent[-1] == CMD_WRITE_SETTINGS_NVM
        assert sum(s for s in slept if s == 0.1) <= 5.0 + 1e-9  # the drop's bounded wait

    def test_port_closed_before_the_wait(self, patched):
        """No serial handle is open while the tool waits for the reset."""
        bus, claw = front_that_resets(patched, back_after=3)
        opened, real = [], bus.serial

        def tracked(*a, **kw):
            handle = real(*a, **kw)
            opened.append(handle)
            return handle

        bus.serial = tracked
        open_at_sleep = []

        run_diagnose(*FRONT_ONLY, *SAVE, sleep=lambda s: open_at_sleep.append(
            [h.port for h in opened if h.is_open]) if s >= 0.1 else None)

        assert open_at_sleep and all(ports == [] for ports in open_at_sleep), open_at_sleep

    def test_found_again_under_a_new_name(self, patched, capsys):
        """The reset re-enumerates the front as ttyACM3: found there by address, and checked."""
        bus = patched(robot(rear_tty="/dev/ttyACM1"))
        claw = resets_on(bus, Resetting(), new="3")

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        out = capsys.readouterr().out
        assert rc in (None, 0)
        assert "Front answers on /dev/roboclaw3" in out
        after = commands(bus, "/dev/roboclaw3")
        assert after[-3:] == [READ_M1_PID, READ_M2_PID, CMD_READ_SERIAL_TIMEOUT]
        assert claw.saved_gains == {1: GAINS, 2: GAINS}
        rear = set(commands(bus, "/dev/roboclaw1"))
        assert rear <= {CMD_READ_MAIN_BATTERY}, "the rear only answers detection's probes"

    def test_found_again_after_several_polls(self, patched, capsys):
        bus, claw = front_that_resets(patched, new="3", back_after=7)

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        assert rc in (None, 0)
        assert "Front answers on /dev/roboclaw3" in capsys.readouterr().out
        assert claw.saved_gains == {1: GAINS, 2: GAINS}

    def test_not_found_again_fails_after_a_bounded_wait(self, patched, capsys):
        bus = patched(robot(rear_tty="/dev/ttyACM1"))
        resets_on(bus, Resetting(), back_after=10**6)
        slept = []

        rc = run_diagnose(*FRONT_ONLY, *SAVE, sleep=slept.append)

        out = capsys.readouterr().out
        assert rc not in (None, 0)
        assert "did not answer on /dev/roboclaw* within" in out
        assert "the save is unverified; once it is back, run the same command again" in out
        assert sum(s for s in slept if s == 0.5) <= 15.0
        rear = set(commands(bus, "/dev/roboclaw1"))
        assert rear <= {CMD_READ_MAIN_BATTERY}

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"keeps_flash": False}, "Those gains are not the driver's"),
            ({"lose_m2": True}, "Those gains are not the driver's"),
            ({"raising": (READ_M1_PID,)}, "the save is unverified"),
            ({"silent": (CMD_READ_SERIAL_TIMEOUT,)}, "the save is unverified"),
        ],
        ids=["flash-kept-nothing", "m2-not-kept", "read-raises", "timeout-unanswered"],
    )
    def test_what_the_reset_loaded_must_check_out(self, patched, capsys, kwargs, message):
        bus, claw = front_that_resets(patched, **kwargs)

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        out = capsys.readouterr().out
        assert rc not in (None, 0)
        assert message in out
        assert "Saved and verified." not in out

    def test_a_changed_timeout_fails(self, patched, capsys):
        """The reset loads a timeout other than the saved one: non-zero."""
        bus, claw = front_that_resets(patched)
        claw.saved_serial_timeout = claw.serial_timeout = 5
        drop = claw.on_reset

        def reset_with_new_timeout():
            claw.serial_timeout = 7
            drop()

        claw.on_reset = reset_with_new_timeout

        rc = run_diagnose(*FRONT_ONLY, *SAVE)

        assert rc not in (None, 0)
        assert "The serial timeout changed from the saved 0.5 s." in capsys.readouterr().out

    def test_front_is_saved_and_checked_before_the_rear_is_written(self, patched):
        bus = patched(robot())
        front = resets_on(bus, Resetting())
        rear = resets_on(bus, Resetting(address=REAR), old="1", new="1")

        rc = run_diagnose(*SAVE)

        assert rc in (None, 0)
        marks = (SET_M1_PID, CMD_WRITE_SETTINGS_NVM, CMD_READ_SERIAL_TIMEOUT)
        order = [(p[-1], d[1]) for p, d in bus.writes if d[1] in marks]
        rear_set = order.index(("1", SET_M1_PID))
        assert order[:rear_set] == [
            ("2", CMD_READ_SERIAL_TIMEOUT),
            ("2", SET_M1_PID),
            ("2", CMD_WRITE_SETTINGS_NVM),
            ("2", CMD_READ_SERIAL_TIMEOUT),
            ("1", CMD_READ_SERIAL_TIMEOUT),
        ]
        assert front.saved_gains == rear.saved_gains == {1: GAINS, 2: GAINS}

    def test_a_failed_front_save_leaves_the_rear_unwritten(self, patched, capsys):
        bus = patched(robot(rear_tty="/dev/ttyACM1"))
        resets_on(bus, Resetting(), back_after=10**6)

        rc = run_diagnose(*SAVE)

        out = capsys.readouterr().out
        assert rc not in (None, 0)
        assert "Rear RoboClaw (0x81): not written, after the failure above." in out
        rear = set(commands(bus, "/dev/roboclaw1"))
        assert not {SET_M1_PID, SET_M2_PID, CMD_WRITE_SETTINGS_NVM, CMD_READ_SETTINGS_NVM} & rear


class TestUsage:
    def test_write_nvm_needs_a_set(self, patched, capsys):
        bus = patched(two_controllers())

        with pytest.raises(SystemExit) as exit_info:
            run_diagnose("--write-nvm")

        assert exit_info.value.code == 2
        assert "--write-nvm needs --set-serial-timeout or --set-gains" in capsys.readouterr().err
        assert bus.opens == []

    @pytest.mark.parametrize(
        "other", [("--read-only",), ("--drive",), ("--set-serial-timeout", "0.5")],
        ids=["read-only", "drive", "set-serial-timeout"],
    )
    def test_set_gains_is_exclusive(self, patched, capsys, other):
        bus = patched(two_controllers())

        with pytest.raises(SystemExit) as exit_info:
            run_diagnose("--set-gains", *other)

        assert exit_info.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err
        assert bus.opens == []

    def test_help_names_the_gains(self, patched, capsys):
        with pytest.raises(SystemExit):
            run_diagnose("--help")

        out = " ".join(capsys.readouterr().out.split())
        p, i, d, qpps = GAINS
        help_text = f"--set-gains Write the driver's velocity gains (P={p} I={i} D={d} QPPS={qpps}"
        assert help_text in out
