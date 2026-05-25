"""Pure-logic watchdog for the inference loop's six-source freshness check.

Kept rclpy-free for direct unit testing. The node feeds its monotonic
receive times in and acts on the returned list of stale sources.

The sources and their thresholds match the brief's Phase 3
6-source-watchdog spec: goal, IMU, joint_states, odom, depth, and the
``map → base_link`` TF lookup age. The TF threshold is checked against
the transform's wall-clock stamp age, not its receive time, because
``tf2_ros.Buffer.lookup_transform`` returns the latest cached value
regardless of how long ago the listener last cached it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class WatchdogTimeouts:
    """Per-source freshness thresholds in seconds."""

    goal: float
    imu: float
    joint_states: float
    odom: float
    depth: float
    tf: float

    def __post_init__(self) -> None:
        for name in ("goal", "imu", "joint_states", "odom", "depth", "tf"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(
                    f"watchdog timeout {name}={value} must be > 0; "
                    "non-positive disables the watchdog source which "
                    "is exactly the silent-failure mode the watchdog "
                    "exists to prevent."
                )


def stale_sources(
    *,
    now_monotonic_s: float,
    last_goal_rx_t: Optional[float],
    last_imu_rx_t: Optional[float],
    last_joint_states_rx_t: Optional[float],
    last_odom_rx_t: Optional[float],
    last_depth_rx_t: Optional[float],
    tf_age_s: Optional[float],
    timeouts: WatchdogTimeouts,
) -> list[str]:
    """Return the names of the watchdog sources that are stale or absent.

    A ``None`` rx time / TF age is treated as stale — the source has
    never produced a sample. Returned list order matches the
    brief's enumeration so log messages read consistently.
    """
    stale: list[str] = []
    if last_goal_rx_t is None or now_monotonic_s - last_goal_rx_t > timeouts.goal:
        stale.append("goal")
    if last_imu_rx_t is None or now_monotonic_s - last_imu_rx_t > timeouts.imu:
        stale.append("imu")
    if (
        last_joint_states_rx_t is None
        or now_monotonic_s - last_joint_states_rx_t > timeouts.joint_states
    ):
        stale.append("joint_states")
    if last_odom_rx_t is None or now_monotonic_s - last_odom_rx_t > timeouts.odom:
        stale.append("odom")
    if last_depth_rx_t is None or now_monotonic_s - last_depth_rx_t > timeouts.depth:
        stale.append("depth")
    if tf_age_s is None or tf_age_s > timeouts.tf:
        stale.append("tf")
    return stale
