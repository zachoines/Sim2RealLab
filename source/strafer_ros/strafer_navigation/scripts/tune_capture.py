#!/usr/bin/env python3
"""Passive capture harness for MPPI critic-tuning sweeps.

Subscribes to ``/cmd_vel`` (Twist) and ``/strafer/odom`` (Odometry) for a
configurable duration and emits a markdown summary of the captured
velocities — peak instantaneous, best 1 s sustained, and median 1 s
sustained — on each axis. The summary pastes directly into the per-pass
run-table in the PR description for ``mppi-critic-tuning-for-sim-envelope``
without further reshaping.

The harness is a passive observer. The operator triggers the mission
separately (executor, rqt_robot_steering, etc.) any time after the
"Recording…" line prints.

CRITICAL: Source ``env_sim_in_the_loop.env`` in the harness terminal
before running, otherwise this harness lands on the default DDS
(rmw_fastrtps_cpp + domain 0) while the sim stack is on
rmw_cyclonedds_cpp + domain 42 — discovery fails silently and the
capture returns zero samples. The startup banner flags this.

Usage (sim — honor /clock):

    source ~/workspaces/Sim2RealLab/source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env
    python3 tune_capture.py --duration 12 --label translate-3m-baseline \\
        --ros-args -p use_sim_time:=true

Usage (real-robot — wall clock):

    python3 tune_capture.py --duration 12 --label rotate-90-tuned
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args


# Sim stack convention from
# source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env.
_EXPECTED_RMW = "rmw_cyclonedds_cpp"
_EXPECTED_DOMAIN = "42"


@dataclass
class Sample:
    t: float  # seconds (sim-time-aware when use_sim_time:=true)
    vx: float
    vy: float
    wz: float


@dataclass
class SeriesStats:
    peak: float            # max |x| instantaneous
    sustained_max: float   # max over all 1 s sliding-window medians
    sustained_median: float  # median over all 1 s sliding-window medians
    n_samples: int


def _summarize(samples: List[Sample], axis: str, window_s: float = 1.0) -> SeriesStats:
    """Compute peak / 1 s sustained-max / 1 s sustained-median for one axis.

    1 s sustained is computed by sliding a window across the captured
    timestamps and taking the median |value| inside each near-full window
    (windows shorter than ``0.8 * window_s`` are skipped so transient
    short windows at start/end don't pollute the stats). The reported
    sustained_max is the best 1 s in the run; sustained_median is the
    typical 1 s. Together they distinguish a one-off spike from genuine
    sustained motion.
    """
    if not samples:
        return SeriesStats(0.0, 0.0, 0.0, 0)
    values = [abs(getattr(s, axis)) for s in samples]
    times = [s.t for s in samples]
    peak = max(values)

    window_medians: List[float] = []
    j = 0
    for i in range(len(samples)):
        while times[i] - times[j] > window_s and j < i:
            j += 1
        if times[i] - times[j] >= window_s * 0.8:
            window_medians.append(statistics.median(values[j : i + 1]))

    if not window_medians:
        # Capture too short for a full 1 s window — surface peak as the
        # only honest signal.
        return SeriesStats(peak, peak, peak, len(samples))
    return SeriesStats(
        peak=peak,
        sustained_max=max(window_medians),
        sustained_median=statistics.median(window_medians),
        n_samples=len(samples),
    )


class CaptureNode(Node):
    """Subscribes to /cmd_vel and /strafer/odom, accumulates samples."""

    def __init__(self) -> None:
        super().__init__("mppi_tune_capture")
        # Both topics are reliable QoS in the strafer stack; depth 200
        # comfortably absorbs ~10 s at 20 Hz (cmd) / 50 Hz (odom).
        qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)
        self._cmd: List[Sample] = []
        self._odom: List[Sample] = []
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, qos)
        self.create_subscription(Odometry, "/strafer/odom", self._on_odom, qos)

    def _t_now(self) -> float:
        # Sim-time aware via use_sim_time parameter.
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_cmd(self, msg: Twist) -> None:
        # Twist has no header → stamp at receive time. With
        # use_sim_time:=true that's /clock-aware.
        self._cmd.append(Sample(t=self._t_now(),
                                vx=msg.linear.x,
                                vy=msg.linear.y,
                                wz=msg.angular.z))

    def _on_odom(self, msg: Odometry) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        v = msg.twist.twist
        self._odom.append(Sample(t=t,
                                 vx=v.linear.x,
                                 vy=v.linear.y,
                                 wz=v.angular.z))

    def cmd_samples(self) -> List[Sample]:
        return list(self._cmd)

    def odom_samples(self) -> List[Sample]:
        return list(self._odom)


def _format_table(label: str,
                  duration_s: float,
                  cmd_stats: Dict[str, SeriesStats],
                  odom_stats: Dict[str, SeriesStats]) -> str:
    out: List[str] = []
    out.append(f"### Capture: {label}")
    out.append("")
    out.append(f"- Duration: {duration_s:.1f} s")
    out.append(f"- /cmd_vel samples: {cmd_stats['vx'].n_samples}")
    out.append(f"- /strafer/odom samples: {odom_stats['vx'].n_samples}")
    out.append("")
    out.append("| topic | axis | peak inst | 1 s sustained max | 1 s sustained median |")
    out.append("|---|---|---|---|---|")
    for axis in ("vx", "vy", "wz"):
        s = cmd_stats[axis]
        out.append(
            f"| /cmd_vel | {axis} | {s.peak:.3f} | "
            f"{s.sustained_max:.3f} | {s.sustained_median:.3f} |"
        )
    for axis in ("vx", "vy", "wz"):
        s = odom_stats[axis]
        out.append(
            f"| /strafer/odom | {axis} | {s.peak:.3f} | "
            f"{s.sustained_max:.3f} | {s.sustained_median:.3f} |"
        )
    return "\n".join(out) + "\n"


def _print_env_banner() -> None:
    """Surface DDS / domain mismatch — the silent zero-sample failure mode.

    The sim stack runs on rmw_cyclonedds_cpp + ROS_DOMAIN_ID=42; an
    unsourced harness terminal lands on the FastDDS+domain-0 default and
    silently sees no traffic. This banner makes that mismatch obvious
    before the user sits through a full capture window for nothing.
    """
    rmw = os.environ.get("RMW_IMPLEMENTATION", "<unset, defaults to rmw_fastrtps_cpp>")
    domain = os.environ.get("ROS_DOMAIN_ID", "<unset, defaults to 0>")
    print(f"  RMW_IMPLEMENTATION = {rmw}", flush=True)
    print(f"  ROS_DOMAIN_ID      = {domain}", flush=True)
    mismatch = []
    if _EXPECTED_RMW not in rmw:
        mismatch.append(f"RMW != {_EXPECTED_RMW}")
    if domain != _EXPECTED_DOMAIN:
        mismatch.append(f"DOMAIN != {_EXPECTED_DOMAIN}")
    if mismatch:
        print(
            f"  WARNING: {', '.join(mismatch)} — sim stack is unreachable on these settings.",
            file=sys.stderr, flush=True,
        )
        print(
            "  Source env_sim_in_the_loop.env in this terminal first:",
            file=sys.stderr, flush=True,
        )
        print(
            "    source ~/workspaces/Sim2RealLab/source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env",
            file=sys.stderr, flush=True,
        )


def main() -> None:
    rclpy.init(args=sys.argv)
    cleaned = remove_ros_args(args=sys.argv)
    parser = argparse.ArgumentParser(description="MPPI tuning capture harness")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Capture duration in WALL-CLOCK seconds. The "
                             "outer recording window is wall time even when "
                             "use_sim_time:=true; the 1 s sliding-window "
                             "stats over the captured samples are still "
                             "sim-time-aware via odom message stamps.")
    parser.add_argument("--label", type=str, required=True,
                        help="Label for this run (e.g. 'translate-3m-baseline').")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to write the markdown summary "
                             "(stdout always prints it).")
    args = parser.parse_args(cleaned[1:])

    _print_env_banner()
    node = CaptureNode()
    print(f"Recording '{args.label}' for {args.duration:.1f} s wall — "
          f"trigger the mission now.", flush=True)

    # Use wall-clock (time.monotonic) for the recording window deadline.
    # We deliberately do NOT use node.get_clock().now() here even though
    # it would be sim-time-aware: when use_sim_time:=true and the harness
    # subscribes to /clock, the first /clock message can arrive AFTER we
    # capture `start`, with sim-time already advanced by minutes (the sim
    # has been running awhile from the operator's perspective). The
    # subsequent now-start delta would then exceed any reasonable
    # --duration on the very first iteration and exit the loop with zero
    # samples. Wall-clock is immune to that race. The 1 s sliding-window
    # stats inside _summarize() use the per-message timestamps (sim-time
    # via odom header.stamp), so the analysis frame remains correct.
    start = time.monotonic()
    deadline = start + args.duration
    next_progress = start + 2.0
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.monotonic()
        # Periodic 2 s progress so silent "no samples" failures are loud.
        if now >= next_progress:
            n_cmd = len(node.cmd_samples())
            n_odom = len(node.odom_samples())
            print(
                f"  t+{now - start:5.1f}s  cmd_vel={n_cmd:4d}  "
                f"strafer/odom={n_odom:4d}",
                flush=True,
            )
            next_progress += 2.0
        if now >= deadline:
            break

    cmd = node.cmd_samples()
    odom = node.odom_samples()

    # If we got nothing, distinguish "DDS/domain mismatch" from
    # "publishers exist but no mission was triggered" by counting the
    # publishers actually visible on each topic. ros2 topic list /
    # node.get_topic_names_and_types() also reflects the harness's own
    # subscriptions, so they're not a reliable signal here — only
    # remote-publisher count is.
    if not cmd or not odom:
        cmd_pubs = node.count_publishers("/cmd_vel")
        odom_pubs = node.count_publishers("/strafer/odom")
        print(
            f"\nWARNING: zero samples on "
            f"{'/cmd_vel' if not cmd else ''}"
            f"{' and ' if not cmd and not odom else ''}"
            f"{'/strafer/odom' if not odom else ''}.",
            file=sys.stderr, flush=True,
        )
        print(
            f"  Remote publishers seen — /cmd_vel: {cmd_pubs}, "
            f"/strafer/odom: {odom_pubs}",
            file=sys.stderr, flush=True,
        )
        if cmd_pubs == 0 and odom_pubs == 0:
            print(
                "  No publishers visible — almost certainly a DDS / domain "
                "mismatch (re-source env_sim_in_the_loop.env) or the sim "
                "stack is not running.",
                file=sys.stderr, flush=True,
            )
        else:
            print(
                "  Publishers ARE visible — discovery worked but the mission "
                "did not produce traffic during this window. Trigger the "
                "mission AFTER the harness banner prints, or extend "
                "--duration so the mission's active phase is in scope.",
                file=sys.stderr, flush=True,
            )

    cmd_stats = {axis: _summarize(cmd, axis) for axis in ("vx", "vy", "wz")}
    odom_stats = {axis: _summarize(odom, axis) for axis in ("vx", "vy", "wz")}

    table = _format_table(args.label, args.duration, cmd_stats, odom_stats)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table)
        print(f"(wrote {args.out})")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
