"""Tests for the driver's port fence.

The node may write only to a port that answered a RoboClaw address probe, and
auto-detection considers only the udev names ``/dev/roboclaw*``. On the robot
the D555 enumerates a CDC serial interface at ``/dev/ttyACM0``; these tests put
a silent device there and check that nothing opens or writes to it.

The serial layer is faked end to end: an autouse fixture patches
``serial.Serial`` and ``glob.glob`` for every test, so no test can reach a real
device.
"""

import fnmatch
import struct
from unittest.mock import patch

import pytest
import rclpy

from strafer_driver import roboclaw_node
from strafer_driver.roboclaw_interface import (
    CMD_READ_MAIN_BATTERY,
    CMD_SET_M1_VEL_PID,
    CMD_SET_M2_VEL_PID,
    _crc16,
    detect_roboclaws,
)

CAMERA_PORT = "/dev/ttyACM0"
FRONT = 0x80
REAR = 0x81
PID_COMMANDS = (CMD_SET_M1_VEL_PID, CMD_SET_M2_VEL_PID)


# ---------------------------------------------------------------------------
# Fake serial layer
# ---------------------------------------------------------------------------


class FakeRoboClaw:
    """Answers packet-serial commands sent to its own address."""

    def __init__(self, address):
        self.address = address

    def respond(self, packet):
        if len(packet) < 4 or packet[0] != self.address:
            return b""
        if _crc16(packet[:-2]) != struct.unpack(">H", packet[-2:])[0]:
            return b""
        command = packet[1]
        if command == CMD_READ_MAIN_BATTERY:
            payload = struct.pack(">H", 120)
        elif command in (16, 17, 18, 19):  # encoder counts and speeds
            payload = struct.pack(">i", 0) + b"\x00"
        else:
            return b"\xff"  # write commands are acknowledged
        return payload + struct.pack(">H", _crc16(packet[:2] + payload))


class FakeCamera:
    """The D555's CDC interface: accepts writes, never answers."""

    def respond(self, packet):
        return b""


class FakeBus:
    """Device paths as the host shows them, plus a log of opens and writes.

    A udev symlink is a second path to the same device object.
    """

    def __init__(self, devices):
        self.devices = devices
        self.opens = []
        self.writes = []

    def glob(self, pattern):
        return sorted(p for p in self.devices if fnmatch.fnmatch(p, pattern))

    def serial(self, *args, **kwargs):
        port = kwargs.get("port", args[0] if args else None)
        return FakeSerial(self, port)

    def touched(self, port):
        return port in self.opens or any(p == port for p, _ in self.writes)

    def packets(self, port=None):
        return [(p, data) for p, data in self.writes if port is None or p == port]


class FakeSerial:
    def __init__(self, bus, port):
        import serial

        if port not in bus.devices:
            raise serial.SerialException(f"could not open port {port}")
        bus.opens.append(port)
        self._bus = bus
        self._device = bus.devices[port]
        self._rx = b""
        self.port = port
        self.is_open = True

    def write(self, data):
        self._bus.writes.append((self.port, bytes(data)))
        self._rx += self._device.respond(bytes(data))
        return len(data)

    def read(self, count=1):
        out, self._rx = self._rx[:count], self._rx[count:]
        return out

    def reset_input_buffer(self):
        self._rx = b""

    def close(self):
        self.is_open = False


def robot(front_tty=None, rear_tty=None):
    """The robot's serial devices: the camera at ttyACM0, plus the given RoboClaws.

    Each RoboClaw gets its kernel name and the ``/dev/roboclaw<N>`` symlink
    that ``99-strafer.rules`` gives it (``%n`` is the kernel number).
    """
    devices = {CAMERA_PORT: FakeCamera()}
    for tty, address in ((front_tty, FRONT), (rear_tty, REAR)):
        if tty is None:
            continue
        claw = FakeRoboClaw(address)
        devices[tty] = claw
        devices["/dev/roboclaw" + tty.removeprefix("/dev/ttyACM")] = claw
    return FakeBus(devices)


@pytest.fixture(autouse=True)
def patched():
    """Route serial.Serial and glob.glob to a bus the test installs.

    A test that opens a port without installing a bus fails on a KeyError.
    """
    holder = {}

    def install(bus):
        holder["bus"] = bus
        return bus

    with patch(
        "strafer_driver.roboclaw_interface.serial.Serial",
        side_effect=lambda *a, **kw: holder["bus"].serial(*a, **kw),
    ), patch(
        "strafer_driver.roboclaw_interface.glob.glob",
        side_effect=lambda pattern: holder["bus"].glob(pattern),
    ):
        yield install


def ros_args(**params):
    args = ["--ros-args"]
    for name, value in params.items():
        args += ["-p", f"{name}:={value}"]
    return args


# Bare kernel names; on the robot ttyACM0 is the camera's.
TTY_PORTS = {"front_port": "/dev/ttyACM0", "rear_port": "/dev/ttyACM1"}


def build_node(**params):
    """Construct the node; return ``(node, None)`` or ``(None, refusal)``."""
    rclpy.init(args=ros_args(**params))
    try:
        return roboclaw_node.RoboClawNode(), None
    except Exception as exc:  # the fence's refusal, checked by the caller
        return None, exc


@pytest.fixture(autouse=True)
def _shutdown_rclpy():
    yield
    rclpy.try_shutdown()


def assert_refused(node, refusal, bus):
    if node is not None:
        node.destroy_node()
    assert refusal is not None, (
        f"RoboClawNode started; opened {bus.opens}, "
        f"wrote to {sorted({p for p, _ in bus.writes})}"
    )
    assert type(refusal).__name__ == "PortFenceError", repr(refusal)


def pid_ports(bus):
    return sorted({p for p, data in bus.writes if data[1] in PID_COMMANDS})


# ---------------------------------------------------------------------------
# detect_roboclaws
# ---------------------------------------------------------------------------


class TestDetectCandidates:
    def test_no_udev_names_probes_nothing(self, patched):
        """With no /dev/roboclaw*, detection finds nothing and opens no ttyACM port."""
        bus = patched(robot(rear_tty="/dev/ttyACM1"))
        del bus.devices["/dev/roboclaw1"]  # the rule is not installed

        assert detect_roboclaws(FRONT, REAR) == {}
        assert bus.opens == []

    def test_scans_udev_names_only(self, patched):
        bus = patched(robot(front_tty="/dev/ttyACM2", rear_tty="/dev/ttyACM1"))

        assert detect_roboclaws(FRONT, REAR) == {
            FRONT: "/dev/roboclaw2",
            REAR: "/dev/roboclaw1",
        }
        assert all(p.startswith("/dev/roboclaw") for p in bus.opens)


# ---------------------------------------------------------------------------
# RoboClawNode with auto_detect (the default)
# ---------------------------------------------------------------------------


class TestAutoDetectFence:
    def test_no_controllers_refuses_and_opens_nothing(self, patched, capfd):
        """The camera alone: the node refuses before opening any port."""
        bus = patched(robot())

        node, refusal = build_node(auto_detect="true", **TTY_PORTS)

        assert_refused(node, refusal, bus)
        assert bus.opens == []
        assert bus.writes == []
        assert "0/2" in str(refusal)
        assert "0x80" in str(refusal) and "0x81" in str(refusal)
        # The reason reaches the node's log, not only the exception.
        captured = capfd.readouterr()
        assert "Missing addresses: 0x80, 0x81" in captured.err + captured.out

    def test_one_controller_refuses(self, patched):
        """Rear only: the configured front port is the camera's."""
        bus = patched(robot(rear_tty="/dev/ttyACM1"))

        node, refusal = build_node(auto_detect="true", **TTY_PORTS)

        assert_refused(node, refusal, bus)
        assert not bus.touched(CAMERA_PORT)
        assert pid_ports(bus) == []
        # The only traffic is detection's probe of the udev-named port.
        assert {p for p, _ in bus.writes} <= {"/dev/roboclaw1"}
        assert all(data[1] == CMD_READ_MAIN_BATTERY for _, data in bus.writes)
        assert "1/2" in str(refusal) and "0x80" in str(refusal)

    def test_two_controllers_start_on_detected_ports(self, patched):
        """Both present, on any kernel numbers: the node starts on them."""
        bus = patched(robot(front_tty="/dev/ttyACM2", rear_tty="/dev/ttyACM1"))

        node, refusal = build_node(auto_detect="true", **TTY_PORTS)

        assert refusal is None, repr(refusal)
        try:
            assert node._front.port == "/dev/roboclaw2"
            assert node._rear.port == "/dev/roboclaw1"
            assert not bus.touched(CAMERA_PORT)
            assert pid_ports(bus) == ["/dev/roboclaw1", "/dev/roboclaw2"]
        finally:
            node.destroy_node()
        assert not bus.touched(CAMERA_PORT)

    def test_main_exits_non_zero_on_refusal(self, patched):
        """The entry point returns non-zero, so the process exits non-zero."""
        bus = patched(robot())

        with patch.object(roboclaw_node.rclpy, "spin", side_effect=KeyboardInterrupt):
            rc = roboclaw_node.main(args=ros_args(auto_detect="true", **TTY_PORTS))

        assert rc not in (None, 0)
        assert bus.opens == []


# ---------------------------------------------------------------------------
# RoboClawNode with auto_detect false
# ---------------------------------------------------------------------------


class TestConfiguredPortFence:
    def test_tty_port_refused(self, patched):
        """A configured port that is not a udev name is refused."""
        bus = patched(robot(rear_tty="/dev/ttyACM1"))

        node, refusal = build_node(auto_detect="false", **TTY_PORTS)

        assert_refused(node, refusal, bus)
        assert bus.opens == []
        assert "/dev/ttyACM0" in str(refusal)

    def test_tty_port_allowed_with_override(self, patched):
        """The bench override admits a tty port that answers the probe."""
        bus = patched(robot(front_tty="/dev/ttyACM2", rear_tty="/dev/ttyACM1"))

        node, refusal = build_node(
            auto_detect="false",
            allow_unverified_ports="true",
            front_port="/dev/ttyACM2",
            rear_port="/dev/ttyACM1",
        )

        assert refusal is None, repr(refusal)
        try:
            assert (node._front.port, node._rear.port) == ("/dev/ttyACM2", "/dev/ttyACM1")
            assert not bus.touched(CAMERA_PORT)
        finally:
            node.destroy_node()

    def test_one_tty_port_refused(self, patched):
        """Either configured port alone is enough to refuse."""
        bus = patched(robot(front_tty="/dev/ttyACM2", rear_tty="/dev/ttyACM1"))

        node, refusal = build_node(
            auto_detect="false", front_port="/dev/roboclaw2", rear_port="/dev/ttyACM0"
        )

        assert_refused(node, refusal, bus)
        assert bus.opens == []
        assert "/dev/ttyACM0" in str(refusal) and "/dev/roboclaw2" not in str(refusal)

    def test_udev_ports_accepted(self, patched):
        bus = patched(robot(front_tty="/dev/ttyACM2", rear_tty="/dev/ttyACM1"))

        node, refusal = build_node(
            auto_detect="false", front_port="/dev/roboclaw2", rear_port="/dev/roboclaw1"
        )

        assert refusal is None, repr(refusal)
        node.destroy_node()
        assert not bus.touched(CAMERA_PORT)


# ---------------------------------------------------------------------------
# PID writes follow an answered probe
# ---------------------------------------------------------------------------


def assert_every_pid_write_probed(bus):
    """Each PID packet on a port follows an answered probe of that address there."""
    answered = set()
    for port, data in bus.writes:
        address, command = data[0], data[1]
        if command == CMD_READ_MAIN_BATTERY and bus.devices[port].respond(data):
            answered.add((port, address))
        if command in PID_COMMANDS:
            assert (port, address) in answered, (
                f"PID packet for 0x{address:02X} written to {port} before any "
                "answered probe there"
            )


class TestNoPidWithoutProbe:
    def test_configured_ports_probed_before_pid(self, patched):
        """With auto_detect false, PID constants follow an answered probe."""
        bus = patched(robot(front_tty="/dev/ttyACM2", rear_tty="/dev/ttyACM1"))

        node, refusal = build_node(
            auto_detect="false", front_port="/dev/roboclaw2", rear_port="/dev/roboclaw1"
        )

        assert refusal is None, repr(refusal)
        node.destroy_node()
        assert pid_ports(bus) == ["/dev/roboclaw1", "/dev/roboclaw2"]
        assert_every_pid_write_probed(bus)

    def test_silent_port_gets_no_pid(self, patched):
        """Under the override, a port that does not answer gets no PID packet."""
        bus = patched(robot(rear_tty="/dev/ttyACM1"))

        node, refusal = build_node(
            auto_detect="false", allow_unverified_ports="true", **TTY_PORTS
        )

        assert_refused(node, refusal, bus)
        assert CAMERA_PORT not in pid_ports(bus)
        assert pid_ports(bus) == []
        assert_every_pid_write_probed(bus)
        assert "0x80" in str(refusal) and CAMERA_PORT in str(refusal)

    def test_wrong_address_gets_no_pid(self, patched):
        """A udev-named port holding the other controller is not configured."""
        bus = patched(robot(front_tty="/dev/ttyACM2", rear_tty="/dev/ttyACM1"))

        node, refusal = build_node(
            auto_detect="false", front_port="/dev/roboclaw1", rear_port="/dev/roboclaw2"
        )

        assert_refused(node, refusal, bus)
        assert pid_ports(bus) == []


# ---------------------------------------------------------------------------
# Reconnect
# ---------------------------------------------------------------------------


class TestReconnectProbe:
    def test_reopened_path_must_answer(self, patched):
        """A reopened path that no longer answers keeps the node in its error state."""
        bus = patched(robot(front_tty="/dev/ttyACM2", rear_tty="/dev/ttyACM1"))
        node, refusal = build_node(auto_detect="true")
        assert refusal is None, repr(refusal)
        try:
            # The front controller is gone and its path now leads to a silent device.
            bus.devices["/dev/roboclaw2"] = FakeCamera()
            node._in_error_state = True
            node._last_reconnect_time = 0.0
            start = len(bus.writes)

            node._timer_callback()  # the reconnect attempt
            node._timer_callback()

            assert node._in_error_state
            sent = [data for port, data in bus.writes[start:] if port == "/dev/roboclaw2"]
            assert sent and all(data[1] == CMD_READ_MAIN_BATTERY for data in sent)
        finally:
            node.destroy_node()

    def test_answering_path_recovers(self, patched):
        bus = patched(robot(front_tty="/dev/ttyACM2", rear_tty="/dev/ttyACM1"))
        node, refusal = build_node(auto_detect="true")
        assert refusal is None, repr(refusal)
        try:
            node._in_error_state = True
            node._last_reconnect_time = 0.0

            node._timer_callback()

            assert not node._in_error_state
        finally:
            node.destroy_node()


# ---------------------------------------------------------------------------
# Launch defaults
# ---------------------------------------------------------------------------


def test_driver_launch_defaults_are_udev_names():
    import importlib.util
    import os

    from launch.actions import DeclareLaunchArgument

    path = os.path.join(os.path.dirname(__file__), "..", "launch", "driver.launch.py")
    spec = importlib.util.spec_from_file_location("driver_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    defaults = {
        entity.name: "".join(s.text for s in entity.default_value)
        for entity in module.generate_launch_description().entities
        if isinstance(entity, DeclareLaunchArgument) and entity.name.endswith("_port")
    }

    assert defaults == {"front_port": "/dev/roboclaw0", "rear_port": "/dev/roboclaw1"}
