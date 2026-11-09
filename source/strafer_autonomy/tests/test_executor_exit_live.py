"""The executor's real exit, run as a subprocess while its camera and odometry topics carry traffic.

A daemon spin thread woken inside rclpy's wait during interpreter finalization aborts the
process with ``terminate called without an active exception``. A sample on the ROS client's
subscriptions, or the context's shutdown, wakes it, and the startup-failure exit aborts only
while samples flow, so every case waits until the colour-image publisher counts this
executor's subscription.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_ros

ABORT_LINE = "terminate called without an active exception"
READY_LINE = "Autonomy executor ready"
HEALTH_LINE = "Running parallel health checks"
COUNT_LINE = re.compile(r"^subscribers (\d+)$")
HEALTH_ERROR = re.compile(
    r"^strafer_autonomy\.clients\.(vlm_client\.GroundingServiceUnavailable|"
    r"planner_client\.PlannerServiceUnavailable): .*health check failed"
)

PUBLISHER = r"""
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

rclpy.init()
node = Node("executor_exit_traffic")
color = node.create_publisher(Image, "/d555/color/image_sync", 10)
depth = node.create_publisher(Image, "/d555/aligned_depth_to_color/image_sync", 10)
info = node.create_publisher(CameraInfo, "/d555/color/camera_info_sync", 10)
odom = node.create_publisher(Odometry, "/strafer/odom", 10)
color_msg = Image(height=360, width=640, encoding="rgb8", step=1920, data=bytes(640 * 360 * 3))
depth_msg = Image(height=360, width=640, encoding="16UC1", step=1280, data=bytes(640 * 360 * 2))
info_msg = CameraInfo(height=360, width=640)
odom_msg = Odometry()
seen = [-1]

def frame():
    color.publish(color_msg)
    depth.publish(depth_msg)
    info.publish(info_msg)
    count = color.get_subscription_count()
    if count != seen[0]:
        seen[0] = count
        print(f"subscribers {count}", flush=True)

node.create_timer(1 / 30, frame)
node.create_timer(1 / 50, lambda: odom.publish(odom_msg))
print("publishing", flush=True)
rclpy.spin(node)
"""


class _Health(http.server.BaseHTTPRequestHandler):
    healthy = True
    release: threading.Event | None = None

    def do_GET(self) -> None:
        if self.release is not None:
            self.release.wait(30.0)
        body = json.dumps({"model_loaded": self.healthy}).encode()
        self.send_response(200 if self.healthy else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


class _HealthServer:
    def __init__(self, *, healthy: bool, release: threading.Event | None = None) -> None:
        handler = type("Health", (_Health,), {"healthy": healthy, "release": release})
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class _Output:
    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self.lines: list[str] = []
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            self.lines.append(line)

    def wait_for(self, text: str, timeout_s: float, *, after: int = 0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if any(text in line for line in self.lines[after:]):
                return True
            if self.proc.poll() is not None:
                self._thread.join(2.0)
                return any(text in line for line in self.lines[after:])
            time.sleep(0.05)
        return False

    def text(self) -> str:
        self._thread.join(2.0)
        return "".join(self.lines)


def _counts(out: _Output, *, after: int = 0) -> list[int]:
    return [int(m.group(1)) for line in out.lines[after:] if (m := COUNT_LINE.match(line.strip()))]


def _wait_for_count_above(out: _Output, base: int, timeout_s: float, *, after: int) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if any(count > base for count in _counts(out, after=after)):
            return True
        time.sleep(0.02)
    return any(count > base for count in _counts(out, after=after))


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(10)


@pytest.fixture(scope="module")
def ros_env():
    pytest.importorskip("rclpy")
    pytest.importorskip("sensor_msgs.msg")
    pytest.importorskip("nav_msgs.msg")
    pytest.importorskip("strafer_msgs.action")
    import strafer_autonomy

    package_root = str(Path(strafer_autonomy.__file__).resolve().parents[1])
    domain = uuid.uuid4().int % 100 + 1
    env = dict(os.environ)
    env.pop("CYCLONEDDS_URI", None)
    env.update({
        "ROS_DOMAIN_ID": str(domain if domain != 42 else 43),
        "ROS_LOCALHOST_ONLY": "1",
        "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join([package_root, env.get("PYTHONPATH", "")]),
    })
    return env, package_root


@pytest.fixture(scope="module")
def traffic(ros_env):
    env, cwd = ros_env
    proc = subprocess.Popen(
        [sys.executable, "-c", PUBLISHER], env=env, cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    out = _Output(proc)
    try:
        if not out.wait_for("publishing", 30.0):
            pytest.fail(f"traffic publisher did not start:\n{out.text()}")
        yield out
    finally:
        _kill(proc)


def _start_executor(ros_env, url: str) -> tuple[subprocess.Popen, _Output]:
    base_env, cwd = ros_env
    env = dict(base_env, VLM_URL=url, PLANNER_URL=url)
    proc = subprocess.Popen(
        [sys.executable, "-m", "strafer_autonomy.executor.main",
         "--ros-args", "-p", "use_sim_time:=false"],
        env=env, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc, _Output(proc)


@pytest.mark.parametrize("run", [1, 2])
def test_a_startup_health_check_failure_exits_1_with_its_own_error(ros_env, traffic, run):
    release = threading.Event()
    server = _HealthServer(healthy=False, release=release)
    base = (_counts(traffic) or [0])[-1]
    mark = len(traffic.lines)
    proc, out = _start_executor(ros_env, server.url)
    matched = alive = False
    try:
        if out.wait_for(HEALTH_LINE, 60.0):
            matched = _wait_for_count_above(traffic, base, 3.0, after=mark)
            alive = proc.poll() is None
            time.sleep(0.5)
        release.set()
        rc = proc.wait(60.0)
    finally:
        release.set()
        _kill(proc)
        server.close()
    text = out.text()
    assert matched, f"the publisher never counted this executor's subscription:\n{text}"
    assert alive, text
    assert ABORT_LINE not in text, text
    assert rc == 1, (rc, text)
    last = text.rstrip().splitlines()[-1]
    assert HEALTH_ERROR.match(last) and "503 Server Error" in last, text


@pytest.mark.parametrize("run", [1, 2])
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
def test_a_handled_signal_exits_0(ros_env, traffic, signum, run):
    server = _HealthServer(healthy=True)
    base = (_counts(traffic) or [0])[-1]
    mark = len(traffic.lines)
    proc, out = _start_executor(ros_env, server.url)
    ready = matched = alive = False
    rc = None
    try:
        ready = out.wait_for(READY_LINE, 60.0)
        matched = ready and _wait_for_count_above(traffic, base, 30.0, after=mark)
        alive = proc.poll() is None
        if ready and matched and alive:
            time.sleep(1.0)
            proc.send_signal(signum)
            rc = proc.wait(30.0)
    finally:
        _kill(proc)
        server.close()
    text = out.text()
    assert ready, text
    assert matched, f"the publisher never counted this executor's subscription:\n{text}"
    assert alive, text
    assert ABORT_LINE not in text, text
    assert rc == 0, (rc, text)
