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


_BLOCK_H = PERCEPTION_HEIGHT // DEPTH_HEIGHT  # 8
_BLOCK_W = PERCEPTION_WIDTH // DEPTH_WIDTH    # 8

assert PERCEPTION_HEIGHT == _BLOCK_H * DEPTH_HEIGHT, (
    "Block-average requires PERCEPTION_HEIGHT to be an integer multiple "
    "of DEPTH_HEIGHT; got non-integer ratio."
)
assert PERCEPTION_WIDTH == _BLOCK_W * DEPTH_WIDTH, (
    "Block-average requires PERCEPTION_WIDTH to be an integer multiple "
    "of DEPTH_WIDTH; got non-integer ratio."
)


# Encodings the inference node decodes on its depth subscription. Z16 is what
# realsense2_camera publishes (uint16 millimetres, 0 = no return); 32FC1 is the
# Isaac bridge's (float32 metres, +inf = no return).
DEPTH_ENCODINGS_Z16 = frozenset({"16UC1", "mono16"})
DEPTH_ENCODING_F32 = "32FC1"
Z16_METRES_PER_UNIT = 0.001

# Invalid pixels in a block at or above this count put the block median on the
# far clamp (they are max_depth after the validity rescue, and they hold both
# middle ranks of the 64). Exactly half is not enough: the even-count median
# then averages the largest valid value with max_depth.
MAJORITY_INVALID_MIN = _BLOCK_H * _BLOCK_W // 2 + 1  # 33


class DepthDecodeError(ValueError):
    """A depth frame the node cannot use. ``reason`` is ``"encoding"`` or
    ``"shape"``, naming the counter the node charges it to."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _unpack_rows(
    data, *, height: int, width: int, step: int, dtype: np.dtype
) -> np.ndarray:
    """Bytes of a ``sensor_msgs/Image`` → ``(height, width)`` array of ``dtype``.

    Packed rows (``len(data) == height * width * itemsize``) are accepted
    whatever ``step`` says, as the 32FC1 path always has; padded rows are
    accepted when ``len(data) == height * step``.
    """
    if len(data) == 0:
        raise DepthDecodeError("shape", f"Empty depth frame ({height}x{width})")
    buf = np.frombuffer(data, dtype=np.uint8)
    row_bytes = width * dtype.itemsize
    if buf.size == height * row_bytes:
        return np.frombuffer(data, dtype=dtype).reshape(height, width)
    if step > row_bytes and buf.size == height * step:
        rows = buf.reshape(height, step)[:, :row_bytes]
        return np.ascontiguousarray(rows).view(dtype).reshape(height, width)
    raise DepthDecodeError(
        "shape",
        f"Depth frame data length {buf.size} B does not match "
        f"{height}x{width} at {dtype.itemsize} B/px (step={step})",
    )


def decode_depth_image(
    *,
    encoding: str,
    data,
    height: int,
    width: int,
    step: int = 0,
    is_bigendian: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Decode a depth ``sensor_msgs/Image`` by its declared encoding.

    Returns ``(depth_meters, valid_mask)``:

    - ``16UC1`` / ``mono16`` (Z16): uint16 millimetres → float32 metres, with
      ``valid_mask = raw != 0``. Z16 marks "no return" with 0, which is finite
      and would otherwise pass :func:`downsample_depth`'s non-finite rescue
      and turn into the nearfield fill, the opposite of training's meaning.
    - ``32FC1``: float32 metres, ``valid_mask = None``. The sim path, unchanged:
      its invalids are +inf/NaN, which the non-finite rescue already handles.

    Raises :class:`DepthDecodeError` for any other encoding or a buffer whose
    length fits neither packed nor ``step``-padded rows.
    """
    order = ">" if is_bigendian else "<"
    if encoding in DEPTH_ENCODINGS_Z16:
        raw = _unpack_rows(
            data, height=height, width=width, step=step,
            dtype=np.dtype(order + "u2"),
        )
        valid = raw != 0
        meters = raw.astype(np.float32) * np.float32(Z16_METRES_PER_UNIT)
        return meters, valid
    if encoding == DEPTH_ENCODING_F32:
        meters = _unpack_rows(
            data, height=height, width=width, step=step,
            dtype=np.dtype(order + "f4"),
        )
        if meters.dtype != np.float32:  # big-endian on a little-endian host
            meters = meters.astype(np.float32)
        return meters, None
    raise DepthDecodeError(
        "encoding",
        f"Dropping depth frame with encoding={encoding!r}; expected "
        f"{DEPTH_ENCODING_F32} or one of {sorted(DEPTH_ENCODINGS_Z16)}",
    )


def count_majority_invalid_cells(valid_mask: np.ndarray) -> int:
    """Policy cells whose 8×8 block is majority-invalid, i.e. the cells the
    validity mask forces to the far clamp."""
    invalid = ~np.asarray(valid_mask, dtype=bool)
    per_block = invalid.reshape(
        DEPTH_HEIGHT, _BLOCK_H, DEPTH_WIDTH, _BLOCK_W
    ).sum(axis=(1, 3))
    return int(np.count_nonzero(per_block >= MAJORITY_INVALID_MIN))


def downsample_depth(
    depth_meters: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    max_depth: float = DEPTH_MAX,
    nearfield_clip: float = DEPTH_MIN,
    nearfield_fill: float = DEPTH_NEARFIELD_FILL,
) -> np.ndarray:
    """640×360 raw depth meters → 3600-dim flat, in raw meters [0, max_depth].

    Returns raw meters, not normalized: the single 1/max_depth normalization
    is applied once downstream by ``assemble_observation``'s ``DEPTH_SCALE``,
    matching the sim ``ObsTerm(func=depth_image, scale=DEPTH_SCALE)``. The
    noise step is skipped (inference adds none).

    The block reduction is a MEDIAN over the exact 8×8 integer ratio. The
    training camera renders one ray per policy pixel; a mean over a block
    that straddles the far-clip validity boundary returns a depth that is on
    no surface in the scene, where the median returns the majority one.

    Not a stride, either: the policy pixel's centre maps to the corner
    BETWEEN source pixels ``8c+3`` and ``8c+4``, so no single source pixel
    sits on the training ray and the two that bracket it disagree.

    ``valid_mask`` (bool, same shape) is the explicit validity of each source
    pixel; the Z16 decode supplies ``raw != 0``. Invalid pixels take
    ``max_depth`` BEFORE the median, the training convention
    (``mdp/observations.py:depth_image`` rescues non-finite depth to
    ``max_depth`` ahead of the same reduction, then fills the nearfield after
    it). So a majority-invalid block (>= 33 of 64) reads ``max_depth``, an
    exactly-half block takes numpy's even-count median exactly as training
    would, and a genuine sub-``nearfield_clip`` return still takes
    ``nearfield_fill``. ``None`` is the 32FC1 path, byte-identical to before
    the mask existed: non-finite alone marks invalid.
    """
    depth = np.asarray(depth_meters, dtype=np.float32)
    if depth.shape != (PERCEPTION_HEIGHT, PERCEPTION_WIDTH):
        raise ValueError(
            f"Expected raw depth shape ({PERCEPTION_HEIGHT}, "
            f"{PERCEPTION_WIDTH}); got {depth.shape}"
        )

    # +inf (frustum cull) and NaN must read as out-of-range before the
    # reduction rather than poisoning it; so must any pixel the mask marks
    # invalid (Z16's finite 0).
    if valid_mask is None:
        depth = np.where(np.isfinite(depth), depth, np.float32(max_depth))
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != depth.shape:
            raise ValueError(
                f"valid_mask shape {valid.shape} does not match depth "
                f"shape {depth.shape}"
            )
        depth = np.where(
            valid & np.isfinite(depth), depth, np.float32(max_depth)
        )
    depth = np.median(
        depth.reshape(DEPTH_HEIGHT, _BLOCK_H, DEPTH_WIDTH, _BLOCK_W),
        axis=(1, 3),
    )
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
