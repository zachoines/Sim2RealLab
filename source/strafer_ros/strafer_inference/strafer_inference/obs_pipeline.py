"""Pure helpers for the DEPTH observation pipeline.

Kept rclpy-free so unit tests can exercise the math without spinning
up a ROS node or a TF buffer. The node wires these to its sensor
caches and the result feeds ``strafer_shared.policy_interface.assemble_observation``.
"""

from __future__ import annotations

import math

import numpy as np

from strafer_shared.constants import (
    DEPTH_HEIGHT,
    DEPTH_MAX,
    DEPTH_MIN,
    DEPTH_NEARFIELD_FILL,
    DEPTH_WIDTH,
    PERCEPTION_HEIGHT,
    PERCEPTION_WIDTH,
    WHEEL_JOINT_NAMES,
)
from strafer_shared.mecanum_kinematics import wheel_vels_to_ticks_per_sec


_BLOCK_H = PERCEPTION_HEIGHT // DEPTH_HEIGHT  # 6
_BLOCK_W = PERCEPTION_WIDTH // DEPTH_WIDTH    # 8

assert PERCEPTION_HEIGHT == _BLOCK_H * DEPTH_HEIGHT, (
    "Block-average requires PERCEPTION_HEIGHT to be an integer multiple "
    "of DEPTH_HEIGHT; got non-integer ratio."
)
assert PERCEPTION_WIDTH == _BLOCK_W * DEPTH_WIDTH, (
    "Block-average requires PERCEPTION_WIDTH to be an integer multiple "
    "of DEPTH_WIDTH; got non-integer ratio."
)


def downsample_depth(
    depth_meters: np.ndarray,
    *,
    max_depth: float = DEPTH_MAX,
    nearfield_clip: float = DEPTH_MIN,
    nearfield_fill: float = DEPTH_NEARFIELD_FILL,
) -> np.ndarray:
    """640×360 raw depth meters → 4800-dim flat normalized [0, 1].

    Mirrors mdp/observations.py:depth_image, with the noise step
    skipped (inference adds no noise of its own) and an area-resize
    inserted because the bridged stream is the 640×360 perception
    camera, not the 80×60 policy camera that exists only in sim.

    Block-average is exact-integer (640/80=8, 360/60=6) so it
    matches cv2.INTER_AREA to within float roundoff for the integer
    ratio case.
    """
    depth = np.asarray(depth_meters, dtype=np.float32)
    if depth.shape != (PERCEPTION_HEIGHT, PERCEPTION_WIDTH):
        raise ValueError(
            f"Expected raw depth shape ({PERCEPTION_HEIGHT}, "
            f"{PERCEPTION_WIDTH}); got {depth.shape}"
        )

    depth = np.where(np.isfinite(depth), depth, np.float32(max_depth))
    depth = depth.reshape(
        DEPTH_HEIGHT, _BLOCK_H, DEPTH_WIDTH, _BLOCK_W
    ).mean(axis=(1, 3))
    depth = np.where(
        depth < nearfield_clip, np.float32(nearfield_fill), depth
    )
    depth = np.clip(depth, 0.0, max_depth)
    depth = depth * np.float32(1.0 / max_depth)
    return depth.reshape(-1).astype(np.float32, copy=False)


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """ZYX yaw from a unit quaternion. Matches tf_transformations'
    euler_from_quaternion(..., axes='sxyz')[2] result.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def body_frame_goal(
    *,
    goal_map_xy: tuple[float, float],
    base_in_map_xy: tuple[float, float],
    base_in_map_yaw: float,
) -> tuple[np.ndarray, float, float]:
    """Map-frame goal → body-frame (rel_xy, distance, heading_to_goal).

    The training env computes the same triplet in base_link frame; if
    inference returned map-frame values the policy would turn the
    wrong way on the real robot.
    """
    gx, gy = float(goal_map_xy[0]), float(goal_map_xy[1])
    bx, by = float(base_in_map_xy[0]), float(base_in_map_xy[1])
    yaw = float(base_in_map_yaw)

    dx_map = gx - bx
    dy_map = gy - by
    cos_y = math.cos(-yaw)
    sin_y = math.sin(-yaw)
    dx_body = cos_y * dx_map - sin_y * dy_map
    dy_body = sin_y * dx_map + cos_y * dy_map

    rel = np.array([dx_body, dy_body], dtype=np.float32)
    dist = float(math.hypot(dx_body, dy_body))
    heading = float(math.atan2(dy_body, dx_body))
    return rel, dist, heading


def joint_state_to_wheel_vels(
    names: list[str],
    velocities: list[float],
) -> np.ndarray:
    """Pick out the four wheel velocities (rad/s) in [FL, FR, RL, RR]
    order regardless of how the publisher ordered them.
    """
    if len(names) != len(velocities):
        raise ValueError(
            f"JointState name/velocity length mismatch: "
            f"{len(names)} vs {len(velocities)}"
        )
    lookup = dict(zip(names, velocities))
    missing = [n for n in WHEEL_JOINT_NAMES if n not in lookup]
    if missing:
        raise KeyError(
            f"JointState missing wheel joints: {missing}; "
            f"got {list(names)}"
        )
    return np.array(
        [lookup[n] for n in WHEEL_JOINT_NAMES], dtype=np.float64
    )


def build_raw_obs_dict(
    *,
    imu_accel: tuple[float, float, float],
    imu_gyro: tuple[float, float, float],
    wheel_vels_rad_s: np.ndarray,
    goal_relative_xy: np.ndarray,
    goal_distance: float,
    goal_heading_to_goal: float,
    body_velocity_xy: tuple[float, float],
    last_action: np.ndarray,
    depth_flat_normalized: np.ndarray,
) -> dict[str, np.ndarray]:
    """Marshal pre-extracted sensor scalars into the raw dict shape
    that ``assemble_observation(PolicyVariant.DEPTH)`` consumes.
    """
    encoder_ticks = wheel_vels_to_ticks_per_sec(
        np.asarray(wheel_vels_rad_s, dtype=np.float64)
    )
    return {
        "imu_accel": np.asarray(imu_accel, dtype=np.float32),
        "imu_gyro": np.asarray(imu_gyro, dtype=np.float32),
        "encoder_vels_ticks": encoder_ticks.astype(np.float32),
        "goal_relative": np.asarray(goal_relative_xy, dtype=np.float32),
        "goal_distance": np.asarray([goal_distance], dtype=np.float32),
        "goal_heading_to_goal": np.asarray(
            [goal_heading_to_goal], dtype=np.float32
        ),
        "body_velocity_xy": np.asarray(body_velocity_xy, dtype=np.float32),
        "last_action": np.asarray(last_action, dtype=np.float32),
        "depth_image": np.asarray(
            depth_flat_normalized, dtype=np.float32
        ),
    }
