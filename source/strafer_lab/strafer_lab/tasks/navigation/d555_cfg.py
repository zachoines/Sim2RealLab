"""Shared Intel RealSense D555 sensor configuration.

This keeps the camera mount, orientation, and intrinsics aligned across
the training environments and the dedicated depth-noise integration
scene.

Two camera configurations coexist:

- **Policy camera** (`make_d555_camera_cfg`, prim path ``d555_camera``):
  80x60 RGB + depth. Consumed by the RL policy's observation pipeline.
  This is the original config and is used by every ``StraferSceneCfg*``
  scene class that feeds the navigation policy.

- **Perception camera** (`make_d555_perception_camera_cfg`, prim path
  ``d555_camera_perception``): 640x360 RGB + depth. Used ONLY by
  perception data collection, Replicator bbox extraction, and the
  Isaac Sim ROS2 bridge. Never instantiated in an RL training env —
  at 640x360 Isaac Sim can only render 1-8 parallel envs, so mixing
  it with the RL policy's large env counts would wreck throughput.

Source of truth for real hardware specs: :mod:`strafer_shared.constants`.
Every value that mirrors a real-world D555 property — native capture
resolution (``PERCEPTION_WIDTH`` / ``PERCEPTION_HEIGHT``), lens specs
(``D555_FOCAL_LENGTH_MM`` / ``D555_HORIZONTAL_APERTURE_MM``), frame rate
(``CAMERA_HZ`` / ``CAMERA_UPDATE_PERIOD_S``), IMU rate (``IMU_HZ`` /
``IMU_UPDATE_PERIOD_S``), hardware clip limits (``DEPTH_CLIP_FAR``), and
mount offset (``CAMERA_OFFSET_X/Y/Z``) — lives in ``strafer_shared`` so
the Jetson real-robot driver and the Isaac Sim cameras share one
authoritative specification and can never drift against each other.
What stays local to this module is sim-only: Isaac Sim USD prim-path
strings, the Isaac Sim → ROS camera-frame quaternion, and the Isaac Sim
``data_types`` channel names.

Both cameras share the same physical parameters, so a single physical
D555 at deployment serves both pipelines without recalibration. They
differ ONLY in resolution and prim path.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import ImuCfg, TiledCameraCfg

from strafer_shared.constants import (
    CAMERA_OFFSET_X,
    CAMERA_OFFSET_Y,
    CAMERA_OFFSET_Z,
    CAMERA_UPDATE_PERIOD_S,
    D555_FOCAL_LENGTH_MM,
    D555_HORIZONTAL_APERTURE_MM,
    DEPTH_CLIP_FAR,
    DEPTH_HEIGHT,
    DEPTH_SIM_CLIP_NEAR,
    DEPTH_WIDTH,
    IMU_UPDATE_PERIOD_S,
    PERCEPTION_HEIGHT,
    PERCEPTION_WIDTH,
)

# ---------------------------------------------------------------------------
# Sim-only constants — these are Isaac Sim / USD naming conventions or
# Isaac-Sim-specific coordinate-frame transforms and do NOT correspond to
# any real-world hardware property. Everything that IS a real hardware
# spec (focal length, aperture, frame rate, native capture resolution,
# mount offset, IMU rate) lives in ``strafer_shared.constants`` so the
# Jetson real-robot code and the Isaac Sim envs share one source of truth.
# ---------------------------------------------------------------------------

D555_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/strafer/body_link/d555_camera"
D555_PERCEPTION_CAMERA_PRIM_PATH = (
    "{ENV_REGEX_NS}/Robot/strafer/body_link/d555_camera_perception"
)
D555_IMU_PRIM_PATH = "{ENV_REGEX_NS}/Robot/strafer/body_link"

# Mount offset tuple — assembled from the three shared hardware constants
# so strafer_lab scene configs can pass it directly to OffsetCfg(pos=...).
D555_CAMERA_OFFSET = (CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z)

# Isaac Sim camera-frame quaternion: rotates Isaac Sim's default camera
# axes onto the ROS REP-103 optical frame (X-right, Y-down, Z-forward).
# This is a sim-side transform, not a hardware property — the Jetson
# executor's RealSense driver publishes frames directly in the ROS frame
# convention, so no equivalent rotation is needed on the real robot.
# Values use the XYZW convention required by Isaac Lab 3.0.
D555_CAMERA_ROT_ROS = (-0.5, 0.5, -0.5, 0.5)
D555_IMU_ROT = (0.0, 0.0, 0.0, 1.0)  # identity, XYZW

# Clipping range mixes the sim-only near override (sim renders below the
# D555's 0.4 m stereo blind zone so objects don't pop out of view next to
# the robot) with the real hardware far limit.
D555_CAMERA_CLIPPING_RANGE = (DEPTH_SIM_CLIP_NEAR, DEPTH_CLIP_FAR)

# Data types for the perception camera spawn. These are Isaac Sim sensor
# output channel names — NOT D555 hardware config. RGB is mandatory (the
# Replicator ``bounding_box_2d_tight`` annotator reads it); depth is
# needed by ``project_detection_to_goal_pose`` and by the ROS2 bridge.
D555_PERCEPTION_DATA_TYPES: tuple[str, ...] = ("rgb", "distance_to_image_plane")


def make_d555_camera_cfg(*, data_types: tuple[str, ...]) -> TiledCameraCfg:
    """Create the standard Strafer D555 camera config (80x60, policy input)."""

    return TiledCameraCfg(
        prim_path=D555_CAMERA_PRIM_PATH,
        update_period=CAMERA_UPDATE_PERIOD_S,
        height=DEPTH_HEIGHT,
        width=DEPTH_WIDTH,
        data_types=list(data_types),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=D555_FOCAL_LENGTH_MM,
            horizontal_aperture=D555_HORIZONTAL_APERTURE_MM,
            clipping_range=D555_CAMERA_CLIPPING_RANGE,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=D555_CAMERA_OFFSET,
            rot=D555_CAMERA_ROT_ROS,
            convention="ros",
        ),
    )


def make_d555_perception_camera_cfg() -> TiledCameraCfg:
    """Create the Strafer D555 perception camera config (640x360, RGB + depth).

    Used for Replicator bbox extraction, perception data collection, and the
    Isaac Sim ROS2 bridge. NOT for RL training — the higher resolution caps
    parallel env count at ~1-8.

    All physical parameters (focal length, aperture, clipping range, mount
    offset, update rate) are imported from :mod:`strafer_shared.constants`
    and therefore identical to the policy camera's spec, which guarantees a
    single physical D555 at deployment serves both pipelines without
    recalibration.
    """

    return TiledCameraCfg(
        prim_path=D555_PERCEPTION_CAMERA_PRIM_PATH,
        update_period=CAMERA_UPDATE_PERIOD_S,
        height=PERCEPTION_HEIGHT,
        width=PERCEPTION_WIDTH,
        data_types=list(D555_PERCEPTION_DATA_TYPES),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=D555_FOCAL_LENGTH_MM,
            horizontal_aperture=D555_HORIZONTAL_APERTURE_MM,
            clipping_range=D555_CAMERA_CLIPPING_RANGE,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=D555_CAMERA_OFFSET,
            rot=D555_CAMERA_ROT_ROS,
            convention="ros",
        ),
    )


def make_d555_imu_cfg() -> ImuCfg:
    """Create the standard Strafer D555 IMU config."""

    return ImuCfg(
        prim_path=D555_IMU_PRIM_PATH,
        update_period=IMU_UPDATE_PERIOD_S,
        offset=ImuCfg.OffsetCfg(
            pos=D555_CAMERA_OFFSET,
            rot=D555_IMU_ROT,
        ),
    )
