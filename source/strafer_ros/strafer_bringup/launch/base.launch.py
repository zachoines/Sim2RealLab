"""Strafer base launch — driver + robot description (TF tree).

This is the minimal configuration for driving the robot.
Launches:
  - strafer_driver:      RoboClaw motor control, odom, joint_states
  - strafer_description: robot_state_publisher (URDF → TF tree)

Usage:
  ros2 launch strafer_bringup base.launch.py
  ros2 launch strafer_bringup base.launch.py front_port:=/dev/roboclaw_front
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    driver_dir = get_package_share_directory("strafer_driver")
    description_dir = get_package_share_directory("strafer_description")

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

        # ── Robot description (URDF → TF tree) ─────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(description_dir, "launch", "description.launch.py")
            ),
        ),

        # ── Motor driver (RoboClaw → odom + joint_states) ──────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(driver_dir, "launch", "driver.launch.py")
            ),
            launch_arguments={
                "front_port": front_port,
                "rear_port": rear_port,
            }.items(),
        ),
    ])
