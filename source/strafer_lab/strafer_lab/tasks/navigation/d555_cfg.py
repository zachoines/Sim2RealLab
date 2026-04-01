"""Shared Intel RealSense D555 sensor configuration.

This keeps the camera mount, orientation, and intrinsics aligned across the
training environments and the dedicated depth-noise integration scene.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import ImuCfg, TiledCameraCfg

from strafer_shared.constants import (
    CAMERA_OFFSET_X,
    CAMERA_OFFSET_Y,
    CAMERA_OFFSET_Z,
    DEPTH_CLIP_FAR,
    DEPTH_HEIGHT,
    DEPTH_SIM_CLIP_NEAR,
    DEPTH_WIDTH,
)

D555_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/strafer/body_link/d555_camera"
D555_IMU_PRIM_PATH = "{ENV_REGEX_NS}/Robot/strafer/body_link"

D555_CAMERA_OFFSET = (CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z)
D555_CAMERA_ROT_ROS = (0.5, -0.5, 0.5, -0.5)
D555_IMU_ROT = (1.0, 0.0, 0.0, 0.0)

D555_CAMERA_UPDATE_PERIOD = 1.0 / 30.0
D555_IMU_UPDATE_PERIOD = 1.0 / 200.0

D555_CAMERA_FOCAL_LENGTH = 1.93
D555_CAMERA_HORIZONTAL_APERTURE = 3.68
D555_CAMERA_CLIPPING_RANGE = (DEPTH_SIM_CLIP_NEAR, DEPTH_CLIP_FAR)


def make_d555_camera_cfg(*, data_types: tuple[str, ...]) -> TiledCameraCfg:
    """Create the standard Strafer D555 camera config."""

    return TiledCameraCfg(
        prim_path=D555_CAMERA_PRIM_PATH,
        update_period=D555_CAMERA_UPDATE_PERIOD,
        height=DEPTH_HEIGHT,
        width=DEPTH_WIDTH,
        data_types=list(data_types),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=D555_CAMERA_FOCAL_LENGTH,
            horizontal_aperture=D555_CAMERA_HORIZONTAL_APERTURE,
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
        update_period=D555_IMU_UPDATE_PERIOD,
        offset=ImuCfg.OffsetCfg(
            pos=D555_CAMERA_OFFSET,
            rot=D555_IMU_ROT,
        ),
        gravity_bias=(0.0, 0.0, 9.81),
    )
