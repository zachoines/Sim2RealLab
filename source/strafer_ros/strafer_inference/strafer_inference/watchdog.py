"""Pure-logic freshness watchdog for the inference loop.

Kept rclpy-free for direct unit testing. The node feeds per-source
freshness inputs in and acts on the returned list of stale sources.

Sources: goal, IMU, joint_states, odom, depth, the ``map → base_link``
TF lookup age, and (hybrid variants) the rolling subgoal. Most are
streamed and checked as monotonic receive-time age against a per-source
threshold. Two exceptions:

- ``goal`` is presence-keyed: an executing ``navigate_to_pose`` action
  goal keeps it fresh (the goal is latched for the mission, not
  streamed); the receive-time check applies only to topic-driven goals.
- The TF threshold is checked against the transform's wall-clock stamp
  age, not its receive time, because ``tf2_ros.Buffer.lookup_transform``
  returns the latest cached value regardless of how long ago the
  listener last cached it.
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
    # Inference half of the split stale-plan budget (~1.0 s here + ~1.0 s in
    # the generator ~ 2.0 s end-to-end); defaulted so non-hybrid callers omit it.
    path: float = 1.0

    def __post_init__(self) -> None:
        for name in ("goal", "imu", "joint_states", "odom", "depth", "tf", "path"):
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
    depth_enabled: bool = True,
    last_subgoal_rx_t: Optional[float] = None,
    subgoal_enabled: bool = False,
    goal_active: bool = False,
) -> list[str]:
    """Return the names of the watchdog sources that are stale or absent.

    A ``None`` rx time / TF age is treated as stale — the source has
    never produced a sample. Returned list order matches the
    enumeration so log messages read consistently.

    ``depth_enabled`` is ``False`` for no-camera variants (whose policy
    declares no depth field); the depth source is then dropped entirely
    rather than tripping forever on a topic the variant never subscribes.

    ``subgoal_enabled`` is ``True`` for rolling-subgoal (hybrid) variants;
    it adds a ``subgoal`` source that zero-twists ``/cmd_vel`` when the
    ``/strafer/subgoal`` stream goes stale (generator died, or its upstream
    ``/plan`` went stale and the generator suppressed output). It is the
    inference-side half of the hybrid plan-freshness guard.

    ``goal_active`` is ``True`` while a ``navigate_to_pose`` action goal
    is executing. The action goal is latched for the whole mission —
    nothing restamps it — so the ``goal`` source is fresh when a goal is
    active OR a ``/strafer/goal`` topic message arrived within
    ``timeouts.goal`` (the topic path serves live goal updates and the
    mid-mission reset).
    """
    stale: list[str] = []
    if not goal_active and (
        last_goal_rx_t is None or now_monotonic_s - last_goal_rx_t > timeouts.goal
    ):
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
    if depth_enabled and (
        last_depth_rx_t is None or now_monotonic_s - last_depth_rx_t > timeouts.depth
    ):
        stale.append("depth")
    if tf_age_s is None or tf_age_s > timeouts.tf:
        stale.append("tf")
    if subgoal_enabled and (
        last_subgoal_rx_t is None
        or now_monotonic_s - last_subgoal_rx_t > timeouts.path
    ):
        stale.append("subgoal")
    return stale
