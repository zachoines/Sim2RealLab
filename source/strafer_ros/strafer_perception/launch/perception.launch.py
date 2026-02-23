"""Launch file for the Strafer perception stack.

Launches:
  1. RealSense D555 camera node (depth + color + IMU if available)
  2. Static TF: base_link → d555_link (camera mounting position)
  3. Depth downsampler node (full-res → 80x60 for policy input)

Published topics (matching sim contract in CLAUDE.md):
  /d555/color/image_raw       - sensor_msgs/Image   @ 30 Hz
  /d555/depth/image_rect_raw  - sensor_msgs/Image   @ 30 Hz
  /d555/imu                   - sensor_msgs/Imu     @ 200 Hz (if IMU available)
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
            }.items(),
        ),

        # ── Static TF: base_link → d555_link ───────────────────────────
        # Camera mount position matching sim config:
        #   20cm forward (+X), 0cm lateral, 25cm up (+Z) from body center
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="d555_static_tf",
            arguments=[
                "--x", "0.20",
                "--y", "0.0",
                "--z", "0.25",
                "--roll", "0.0",
                "--pitch", "0.0",
                "--yaw", "0.0",
                "--frame-id", "base_link",
                "--child-frame-id", "d555_link",
            ],
        ),

        # ── Depth downsampler (full-res → 80x60 for policy) ────────────
        Node(
            package="strafer_perception",
            executable="depth_downsampler",
            name="depth_downsampler",
            output="screen",
        ),
    ])
