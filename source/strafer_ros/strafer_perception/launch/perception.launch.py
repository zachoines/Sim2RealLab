"""Launch file for the Strafer perception stack.

Launches:
  1. RealSense D555 camera node (depth + color + IMU if available)
  2. IMU orientation filter (Madgwick) — fuses accel+gyro → quaternion
  3. Timestamp fixer — shifts HW clock timestamps to system time
  4. Depth downsampler node (full-res → 80x60 for policy input)

The base_link → d555_link static TF is published by strafer_description
from the URDF (with constants from strafer_shared).  Do not add a
duplicate static_transform_publisher here.

Published topics (matching the sim-to-real perception contract):
  /d555/color/image_raw       - sensor_msgs/Image   @ 30 Hz
  /d555/depth/image_rect_raw  - sensor_msgs/Image   @ 30 Hz
  /d555/imu                   - sensor_msgs/Imu     @ 200 Hz (raw, no orientation)
  /d555/imu/filtered          - sensor_msgs/Imu     @ 200 Hz (with orientation quaternion)
  /d555/depth/downsampled     - sensor_msgs/Image   @ 30 Hz (80x60 32FC1)

NOTE: IMU requires the
hid-sensor-hub kernel modules (see /etc/modules-load.d/hid-sensor-imu.conf).
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    realsense_dir = get_package_share_directory("realsense2_camera")

    return LaunchDescription([

        # ── RealSense D555 camera node ──────────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(realsense_dir, "launch", "rs_launch.py")
            ),
            launch_arguments={
                "camera_name": "d555",
                "camera_namespace": "",

                # Streams
                "enable_depth": "true",
                "enable_color": "true",
                "enable_infra1": "false",
                "enable_infra2": "false",
                "enable_gyro": "true",
                "enable_accel": "true",

                # Stream profiles: WIDTHxHEIGHTxFPS
                # D555 native resolutions: 640x360 (not 640x480)
                "depth_module.depth_profile": "640x360x30",
                "rgb_camera.color_profile": "640x360x30",

                # Combine accel + gyro into single /d555/imu topic
                "unite_imu_method": "2",

                # Align depth frame to color frame
                "align_depth.enable": "true",

                # Reset USB device on startup (helps with reconnections)
                # NOTE: Disabled — causes tegra-xusb transfer errors on
                # re-enumeration.  Camera state is clean at module load.
                "initial_reset": "false",

                # Global-time maps HW timestamps to system clock.
                # Broken on Jetson (Tegra USB SOF timing) — falls back
                # to system time when disabled.
                "depth_module.global_time_enabled": "false",
                "rgb_camera.global_time_enabled": "false",
                "motion_module.global_time_enabled": "false",
            }.items(),
        ),

        # ── IMU orientation filter (Madgwick) ────────────────────────────
        # RealSense D555 publishes raw accel+gyro on /d555/imu but no
        # orientation quaternion.  Madgwick fuses these into a proper
        # orientation published on /d555/imu/filtered for RTAB-Map.
        Node(
            package="imu_filter_madgwick",
            executable="imu_filter_madgwick_node",
            name="imu_filter",
            output="screen",
            parameters=[{
                "use_mag": False,
                "publish_tf": False,
                "world_frame": "enu",
            }],
            remappings=[
                ("imu/data_raw", "/d555/imu"),
                ("imu/data", "/d555/imu/filtered"),
            ],
        ),

        # ── Timestamp fixer (HW clock → system clock) ───────────────────
        # RealSense D555 on Jetson uses hardware timestamps that are offset
        # from system time.  This node shifts camera message stamps so they
        # match wheel odometry for RTAB-Map's approximate sync.
        Node(
            package="strafer_perception",
            executable="timestamp_fixer",
            name="timestamp_fixer",
            output="screen",
        ),

        # ── Depth downsampler (full-res → 80x60 for policy) ────────────
        Node(
            package="strafer_perception",
            executable="depth_downsampler",
            name="depth_downsampler",
            output="screen",
        ),
    ])
