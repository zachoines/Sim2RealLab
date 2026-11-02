"""Tests for three ways the driver could keep or resume driving when it should not.

- A message that waited through a stall of the executor must not re-arm a stopped drive.
- A controller that stays on USB but stops answering must stop both controllers within the
  watchdog window, not after ten slow failures.
- A reconnect after a power cycle must re-apply the velocity gains, and read them back, before
  the driver leaves its error state.

The serial layer is test_port_fence's fake.
"""

import struct
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor

from strafer_driver import roboclaw_node
from strafer_driver.roboclaw_interface import _crc16

# test/ has no __init__.py, so pytest puts it on sys.path; the imported autouse fixtures apply here.
from test_port_fence import (  # noqa: F401
    FRONT,
    REAR,
    FakeRoboClaw,
    _shutdown_rclpy,
    build_node,
    patched,
    robot,
)

# The manual's numbers, written out as in test_port_fence.
DRIVE_M1M2_SPEED = 37
READ_PID = {55: 1, 56: 2}
READ_TIMEOUT = 0.1  # RoboClawInterface's default read_timeout
GAINS = (15000, 750, 0, 2657)  # P, I, D, QPPS, as strafer_shared.constants sets them
FRONT_PORT, REAR_PORT = "/dev/roboclaw2", "/dev/roboclaw1"

SPIN = Twist()
SPIN.angular.z = 0.8


def bus_with(front, rear):
    """The camera at ttyACM0, *front* at ttyACM2 (roboclaw2) and *rear* at ttyACM1 (roboclaw1)."""
    bus = robot()
    bus.devices["/dev/ttyACM2"] = bus.devices[FRONT_PORT] = front
    bus.devices["/dev/ttyACM1"] = bus.devices[REAR_PORT] = rear
    return bus


def drives(bus, start=0, port=None):
    """The (M1, M2) speeds of every drive command written since *start*, optionally to one port."""
    return [
        struct.unpack(">ii", data[2:10])
        for p, data in bus.writes[start:]
        if data[1] == DRIVE_M1M2_SPEED and (port is None or p == port)
    ]


def node_on(bus_fixture, bus):
    bus_fixture(bus)
    node, refusal = build_node(auto_detect="true")
    assert refusal is None, repr(refusal)
    return node


# ---------------------------------------------------------------------------
# A message that waited through a stall of the executor
# ---------------------------------------------------------------------------


@pytest.fixture
def local_dds(monkeypatch):
    """These tests publish motion commands: keep them on this host, off the stack's domain.

    The domain is fixed, so two runs of this suite at once on one host would cross-talk.
    """
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "77")


def spin_for(executor, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        executor.spin_once(timeout_sec=0.005)


def publish_for(pub, seconds, executor=None, period=0.05):
    """Publish the spin every *period*, spinning the driver between sends unless it is stalled."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pub.publish(SPIN)
        if executor is None:
            time.sleep(period)
        else:
            spin_for(executor, period)


@pytest.fixture
def driver_and_talker(patched, local_dds):
    bus = patched(robot(front_tty="/dev/ttyACM2", rear_tty="/dev/ttyACM1"))
    node, refusal = build_node(auto_detect="true")
    assert refusal is None, repr(refusal)
    talker = rclpy.create_node("stall_test_talker")
    pub = talker.create_publisher(Twist, "/strafer/cmd_vel", 10)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    deadline = time.monotonic() + 10.0
    while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        spin_for(executor, 0.05)
    assert pub.get_subscription_count() == 1
    yield bus, node, pub, executor
    executor.shutdown()
    talker.destroy_node()
    node.destroy_node()


@pytest.mark.parametrize("catch_up_tick", [False, True], ids=["take-after-overdue-tick", "catch-up-tick-first"])
def test_a_message_that_waited_through_a_stall_does_not_rearm(driver_and_talker, catch_up_tick):
    bus, node, pub, executor = driver_and_talker
    publish_for(pub, 0.5, executor)
    assert any(s != (0, 0) for s in drives(bus)), "the spin never reached the controllers"

    publish_for(pub, 0.5)  # the executor stalls while the publisher sends on
    time.sleep(1.0)  # then the publisher falls silent, and its last message ages
    mark = len(bus.writes)
    if catch_up_tick:
        # As on the robot: the overdue tick, then the executor's own catch-up tick
        # milliseconds later, and only then the take.
        node._timer_callback()
    spin_for(executor, 0.8)

    after = drives(bus, mark)
    assert after and after[0] == (0, 0), "the first tick after the stall is the watchdog's zero"
    assert all(s == (0, 0) for s in after), "a message that waited through the stall was driven"


def test_a_live_publisher_drives_again_within_one_period(driver_and_talker):
    bus, node, pub, executor = driver_and_talker
    publish_for(pub, 0.5, executor)
    publish_for(pub, 0.8)  # a stall, with the publisher live throughout
    mark = len(bus.writes)
    for period in range(6):
        pub.publish(SPIN)
        spin_for(executor, 0.05)
        if any(s != (0, 0) for s in drives(bus, mark)):
            break
    else:
        pytest.fail("the live spin never came back")
    assert period <= 1, f"the spin came back only after {period + 1} periods"


# ---------------------------------------------------------------------------
# A controller that stays on USB and stops answering
# ---------------------------------------------------------------------------


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Silenced(FakeRoboClaw):
    """A RoboClaw that can stop answering while it stays on USB.

    Each packet it leaves unanswered costs the caller one read timeout on the test's clock.
    """

    def __init__(self, address, clock):
        super().__init__(address)
        self.clock = clock
        self.silent = False

    def respond(self, packet):
        if self.silent and packet and packet[0] in self.addresses:
            self.clock.t += READ_TIMEOUT
            return b""
        return super().respond(packet)


@pytest.fixture
def silenceable(patched, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(roboclaw_node, "time", SimpleNamespace(monotonic=clock))
    front, rear = Silenced(FRONT, clock), Silenced(REAR, clock)
    bus = bus_with(front, rear)
    node = node_on(patched, bus)
    yield bus, node, rear, clock
    node.destroy_node()


def tick(node, clock):
    """One timer tick with the spin commanded, as from a live publisher."""
    node._cmd_vel_callback(SPIN)
    clock.t += 0.02
    node._timer_callback()


def test_a_silent_controller_stops_both_within_the_watchdog_window(silenceable):
    bus, node, rear, clock = silenceable
    tick(node, clock)
    assert drives(bus, port=FRONT_PORT)[-1] != (0, 0), "the spin never reached the front"

    rear.silent = True
    failing = 0
    while not node._in_error_state and failing < 12:
        tick(node, clock)
        failing += 1
    assert node._in_error_state
    assert failing <= 2, f"the error state came only after {failing} failing ticks"
    assert drives(bus, port=FRONT_PORT)[-1] == (0, 0), "the front was not sent a stop"


def test_one_failed_tick_between_good_ones_does_not_escalate(silenceable):
    bus, node, rear, clock = silenceable
    tick(node, clock)
    rear.silent = True
    tick(node, clock)
    rear.silent = False
    tick(node, clock)
    assert not node._in_error_state
    assert node._consecutive_failures == 0


# ---------------------------------------------------------------------------
# A reconnect after a power cycle
# ---------------------------------------------------------------------------


class GainsClaw(FakeRoboClaw):
    """A RoboClaw whose power cycle reloads its saved gains over the ones written to RAM.

    With *stuck* set, it reads those gains back whatever was written. While *deaf*, it ignores
    the gain writes, unanswered.
    """

    def __init__(self, address, saved=(0, 0, 0, 0), stuck=None, deaf=False):
        super().__init__(address)
        self.saved = {1: saved, 2: saved}
        self.gains = dict(self.saved)
        self.stuck = stuck
        self.deaf = deaf

    def power_cycle(self):
        self.gains = dict(self.saved)

    def respond(self, packet):
        if self.deaf and packet[:1] and packet[0] in self.addresses and packet[1] in (28, 29):
            return b""
        if self.stuck and packet[:1] and packet[0] in self.addresses and packet[1] in READ_PID:
            payload = struct.pack(">IIII", *self.stuck)
            return payload + struct.pack(">H", _crc16(packet[:2] + payload))
        return super().respond(packet)


def power_cycle_and_reconnect(node, *claws):
    """Main power cut and back: each controller reloads its saved gains; the driver reconnects."""
    for claw in claws:
        claw.power_cycle()
    node._in_error_state = True
    node._last_reconnect_time = 0.0
    node._timer_callback()


def test_a_reconnect_reapplies_the_gains_before_driving(patched):
    # The robot's saved settings: the front's from an earlier tune, the rear's all zero.
    front, rear = GainsClaw(FRONT, saved=(14999, 749, 0, 2796)), GainsClaw(REAR)
    node = node_on(patched, bus_with(front, rear))
    try:
        assert rear.gains == {1: GAINS, 2: GAINS}, "the driver writes its gains at start"
        with patch.object(node, "get_logger") as logger:
            power_cycle_and_reconnect(node, front, rear)
        assert not node._in_error_state
        assert front.gains == {1: GAINS, 2: GAINS}
        assert rear.gains == {1: GAINS, 2: GAINS}, "the rear drives on its saved, zero gains"
        infos = [str(call) for call in logger.return_value.info.call_args_list]
        assert any("Rear gains read back" in i for i in infos), infos
    finally:
        node.destroy_node()


def test_a_disagreeing_read_back_keeps_the_error_state(patched):
    front, rear = GainsClaw(FRONT), GainsClaw(REAR, stuck=(0, 0, 0, 0))
    node = node_on(patched, bus_with(front, rear))
    try:
        with patch.object(node, "get_logger") as logger:
            power_cycle_and_reconnect(node, front, rear)
        assert node._in_error_state
        errors = [str(call) for call in logger.return_value.error.call_args_list]
        assert any("Rear gains read back" in e for e in errors), errors
    finally:
        node.destroy_node()


def test_unconfirmed_gains_at_startup_hold_the_error_state(patched):
    rear = GainsClaw(REAR, deaf=True)
    bus = bus_with(GainsClaw(FRONT), rear)
    node = node_on(patched, bus)
    try:
        assert node._in_error_state, "the rear never took its gains, yet the driver would drive"
        mark = len(bus.writes)
        for _ in range(3):
            node._timer_callback()
        assert not drives(bus, mark), "a drive command went out on unconfirmed gains"

        rear.deaf = False
        node._last_reconnect_time = 0.0
        node._timer_callback()
        assert not node._in_error_state
        assert rear.gains == {1: GAINS, 2: GAINS}
        node._timer_callback()
        assert drives(bus, mark)
    finally:
        node.destroy_node()
