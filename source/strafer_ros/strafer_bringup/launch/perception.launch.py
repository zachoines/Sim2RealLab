"""Strafer perception launch — base + RealSense camera stack.

Launches:
  - base.launch.py:       driver + description
  - strafer_perception:   RealSense D555 (depth + color + IMU) + depth downsampler

NOTE: strafer_description (via base) already publishes the base_link → d555_link
TF from the URDF. strafer_perception also publishes a static TF for d555_link.
Both transforms are identical, so the duplicate is harmless. The URDF version
is the authoritative source; the static TF in perception exists for standalone
use without description.

Usage:
  ros2 launch strafer_bringup perception.launch.py
  ros2 launch strafer_bringup perception.launch.py front_port:=/dev/roboclaw_front
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    bringup_dir = get_package_share_directory("strafer_bringup")
    perception_dir = get_package_share_directory("strafer_perception")

    front_port = LaunchConfiguration("front_port")
    rear_port = LaunchConfiguration("rear_port")

    return LaunchDescription([
        DeclareLaunchArgument(
            "front_port", default_value="/dev/ttyACM0",
            description="Serial port for front RoboClaw (0x80).",
        ),
        DeclareLaunchArgument(
            "rear_port", default_value="/dev/ttyACM1",
            description="Serial port for rear RoboClaw (0x81).",
        ),

        # ── Base (driver + description) ─────────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_dir, "launch", "base.launch.py")
            ),
            launch_arguments={
                "front_port": front_port,
                "rear_port": rear_port,
            }.items(),
        ),

        # ── Perception (RealSense D555 + downsampler) ──────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(perception_dir, "launch", "perception.launch.py")
            ),
        ),
    ])
