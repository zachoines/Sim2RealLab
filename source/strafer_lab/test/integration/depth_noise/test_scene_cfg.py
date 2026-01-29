# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Dedicated test scene configuration for depth noise integration tests.

This module provides a controlled test scene with known geometry:
- A single wall in front of the robot at a fixed distance
- Ground plane and robot
"""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import TiledCameraCfg, ImuCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils

# Import robot configuration
from strafer_lab.assets import STRAFER_CFG

# =============================================================================
# Test Scene Geometry Constants
# =============================================================================

# Wall placement
TEST_WALL_DISTANCE = 2.0   # meters - distance from camera to wall surface

# Wall dimensions - sized to fill camera FOV
# At 2.0m distance with 87° horizontal FOV:
#   visible_width = 2 * 2.0 * tan(87°/2) ≈ 4.0m
TEST_WALL_WIDTH = 4.0
TEST_WALL_HEIGHT = 2.0
TEST_WALL_THICKNESS = 0.1

# Camera mount offset from robot body_link
# NOTE: These values are scaled differently than main scene due to USD unit handling.
# The offset values appear to need 100x scaling to achieve the desired physical offset.
# -20.0 achieves ~20cm forward offset, 25.0 achieves ~25cm up offset.
CAMERA_Y_OFFSET = -20.
CAMERA_Z_OFFSET = 25.

# Environment spacing must be large enough to prevent wall overlap
# Each wall is 4m wide, so spacing should be > 4m
# Also account for wall distance (2m) in front of each robot
ENV_SPACING = 8.0


@configclass
class DepthNoiseTestSceneCfg(InteractiveSceneCfg):
    """Test scene with a wall at known distance for depth noise validation.
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = STRAFER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
    )

    # Single wall in front of the camera
    # The camera faces -Y direction from its mount on the robot
    # Wall is positioned so its front surface is at TEST_WALL_DISTANCE from camera
    #
    # Camera world position when robot at origin facing -Y:
    #   camera_Y = robot_Y (0) + CAMERA_Y_OFFSET (-0.15) = -0.15
    # Wall front surface should be TEST_WALL_DISTANCE away from camera:
    #   wall_front_Y = camera_Y - TEST_WALL_DISTANCE = -0.15 - 2.0 = -2.15
    # Wall center (accounting for thickness):
    #   wall_center_Y = wall_front_Y - TEST_WALL_THICKNESS/2 = -2.15 - 0.05 = -2.2
    test_wall: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestWall",
        spawn=sim_utils.CuboidCfg(
            size=(TEST_WALL_WIDTH, TEST_WALL_THICKNESS, TEST_WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),  # Gray wall
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                0.0,
                # Wall center Y = camera_Y - wall_distance - thickness/2
                (CAMERA_Y_OFFSET / 100) - TEST_WALL_DISTANCE - TEST_WALL_THICKNESS / 2,
                TEST_WALL_HEIGHT / 2,
            ),
        ),
    )

    # Intel RealSense D555 depth camera (87° FOV, 0.4-6m range)
    # IMPORTANT: Camera config must match strafer_env_cfg.py StraferSceneCfg
    d555_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link/d555_camera",
        update_period=1.0 / 30.0,
        height=60,
        width=80,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,
            horizontal_aperture=3.68,
            clipping_range=(0.4, 6.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, CAMERA_Y_OFFSET, CAMERA_Z_OFFSET),
            # ROS camera convention: Z-forward, X-right, Y-down
            # Rotation to point camera in robot's forward direction (-Y in USD)
            # (0, 0, 0.707, -0.707) is 90° rotation around Z axis
            rot=(0.0, 0.0, 0.707, -0.707),
            convention="ros",
        ),
    )

    # Intel RealSense D555 IMU (BMI055)
    # IMPORTANT: IMU config must match strafer_env_cfg.py StraferSceneCfg
    d555_imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=1.0 / 200.0,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, CAMERA_Y_OFFSET, CAMERA_Z_OFFSET),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        gravity_bias=(0.0, 0.0, 9.81),
    )
