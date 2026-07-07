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
from strafer_shared.mecanum_kinematics import (
    l1_clamp_twist as l1_clamp_velocity,
    wheel_vels_to_ticks_per_sec,
)
from strafer_shared.policy_interface import PolicyVariant


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
    """640×360 raw depth meters → 4800-dim flat, in raw meters [0, max_depth].

    Returns raw meters, not normalized: the single 1/max_depth normalization
    is applied once downstream by ``assemble_observation``'s ``DEPTH_SCALE``,
    matching the sim ``ObsTerm(func=depth_image, scale=DEPTH_SCALE)``. The
    noise step is skipped (inference adds none) and an area-resize maps the
    640×360 perception stream to the 80×60 policy resolution that exists only
    in sim.

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
    return depth.reshape(-1).astype(np.float32, copy=False)


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """ZYX yaw from a unit quaternion. Matches tf_transformations'
    euler_from_quaternion(..., axes='sxyz')[2] result.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def quat_apply_inverse_xy(
    quat_xyzw: tuple[float, float, float, float],
    delta_xy: tuple[float, float],
) -> np.ndarray:
    """Rotate the planar world displacement ``(dx, dy, 0)`` into the body
    frame by the INVERSE of the base orientation quaternion (XYZW),
    returning local ``(x, y)``.

    Mirrors ``isaaclab.utils.math.quat_apply_inverse`` so the deployed
    ``*_relative`` obs field is consistent with the training
    ``goal_position_relative`` term (full 3-D rotation, not yaw-only). The
    two agree only at zero roll/pitch; the real-robot/sim ``map -> base_link``
    TF carries tilt, so the yaw-only shortcut breaks obs parity there.
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)  # (x, y, z, w)
    xyz = q[:3]
    w = q[3]
    vec = np.array([float(delta_xy[0]), float(delta_xy[1]), 0.0], dtype=np.float64)
    t = 2.0 * np.cross(xyz, vec)
    rel = vec - w * t + np.cross(xyz, t)
    return rel[:2].astype(np.float32)


def body_frame_goal(
    *,
    goal_map_xy: tuple[float, float],
    base_in_map_xy: tuple[float, float],
    base_in_map_quat: tuple[float, float, float, float],
) -> tuple[np.ndarray, float, float]:
    """Map-frame goal → body-frame (rel_xy, distance, heading_to_goal).

    ``rel_xy`` uses the FULL quaternion-inverse rotation, matching the
    training ``goal_position_relative`` term (and TF2 on the real robot).
    ``distance`` and ``heading_to_goal`` are yaw-only / 2-D, matching the
    training ``goal_distance`` and ``goal_heading_to_goal`` terms exactly —
    a yaw-only rotation preserves both the 2-D magnitude and the relative
    bearing, so those two must NOT switch to the 3-D path.

    Args:
        base_in_map_quat: base orientation as ``(x, y, z, w)`` (XYZW), the
            ordering of both ROS ``geometry_msgs/Quaternion`` and the
            training ``root_quat_w``.
    """
    gx, gy = float(goal_map_xy[0]), float(goal_map_xy[1])
    bx, by = float(base_in_map_xy[0]), float(base_in_map_xy[1])
    dx_map = gx - bx
    dy_map = gy - by

    # rel_xy: full quaternion-inverse of the planar displacement.
    rel = quat_apply_inverse_xy(base_in_map_quat, (dx_map, dy_map))

    # distance + heading: yaw-only / 2-D (these already match training).
    yaw = quaternion_to_yaw(*base_in_map_quat)
    cos_y = math.cos(-yaw)
    sin_y = math.sin(-yaw)
    dx_body = cos_y * dx_map - sin_y * dy_map
    dy_body = sin_y * dx_map + cos_y * dy_map
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
    variant: PolicyVariant,
    imu_accel: tuple[float, float, float],
    imu_gyro: tuple[float, float, float],
    wheel_vels_rad_s: np.ndarray,
    goal_relative_xy: np.ndarray,
    goal_distance: float,
    goal_heading_to_goal: float,
    body_velocity_xy: tuple[float, float],
    last_action: np.ndarray,
    depth_flat_meters: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Marshal pre-extracted sensor values into the raw dict that
    ``assemble_observation(variant)`` consumes.

    Variant-agnostic by design: the goal-shaped triplet is emitted under
    whatever key names the variant declares -- ``goal_*`` for goal-referent
    variants, ``subgoal_*`` for rolling-subgoal variants -- and
    ``depth_image`` is emitted only when the variant has a depth field. The
    ``goal_*`` argument names denote the body-frame referent triplet
    regardless of which referent the variant actually tracks; the caller
    transforms the right pose (final goal or rolling subgoal) before calling.

    Every value here is the raw, pre-scale sensor reading — ``assemble_observation``
    applies each field's normalization once. ``depth_flat_meters`` is therefore
    depth in metres from :func:`downsample_depth`, not pre-normalized to [0, 1].
    """
    encoder_ticks = wheel_vels_to_ticks_per_sec(
        np.asarray(wheel_vels_rad_s, dtype=np.float64)
    )
    raw: dict[str, np.ndarray] = {
        "imu_accel": np.asarray(imu_accel, dtype=np.float32),
        "imu_gyro": np.asarray(imu_gyro, dtype=np.float32),
        "encoder_vels_ticks": encoder_ticks.astype(np.float32),
        "body_velocity_xy": np.asarray(body_velocity_xy, dtype=np.float32),
        "last_action": np.asarray(last_action, dtype=np.float32),
    }
    referent_relative = np.asarray(goal_relative_xy, dtype=np.float32)
    referent_distance = np.asarray([goal_distance], dtype=np.float32)
    referent_heading = np.asarray([goal_heading_to_goal], dtype=np.float32)

    for field in variant.fields:
        key = field.key
        if key in raw:
            continue
        if key.endswith("_relative"):
            raw[key] = referent_relative
        elif key.endswith("_distance"):
            raw[key] = referent_distance
        elif "heading" in key:
            raw[key] = referent_heading
        elif key == "depth_image":
            if depth_flat_meters is None:
                raise ValueError(
                    f"variant {variant.name} declares a depth_image field but "
                    "depth_flat_meters was not provided"
                )
            raw[key] = np.asarray(depth_flat_meters, dtype=np.float32)
    return raw
