"""Strafer SLAM launch — perception + RTAB-Map SLAM.

Launches:
  - perception.launch.py:  base + RealSense
  - strafer_slam:          depthimage_to_laserscan + RTAB-Map

Provides:
  - /scan              virtual 2D laser scan from depth
  - /rtabmap/map       occupancy grid
  - map → odom TF      from RTAB-Map loop closure

Usage:
  ros2 launch strafer_bringup slam.launch.py
  ros2 launch strafer_bringup slam.launch.py localization:=true
  ros2 launch strafer_bringup slam.launch.py rtabmap_args:="-d"   # delete DB on start
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    bringup_dir = get_package_share_directory("strafer_bringup")
    slam_dir = get_package_share_directory("strafer_slam")

    front_port = LaunchConfiguration("front_port")
    rear_port = LaunchConfiguration("rear_port")
    localization = LaunchConfiguration("localization")
    database_path = LaunchConfiguration("database_path")
    rtabmap_args = LaunchConfiguration("rtabmap_args")
    rtabmap_viz = LaunchConfiguration("rtabmap_viz")

    return LaunchDescription([
        # ── Forwarded arguments ─────────────────────────────────────────
        DeclareLaunchArgument(
            "front_port", default_value="/dev/ttyACM0",
            description="Serial port for front RoboClaw (0x80).",
        ),
        DeclareLaunchArgument(
            "rear_port", default_value="/dev/ttyACM1",
            description="Serial port for rear RoboClaw (0x81).",
        ),

        # ── SLAM arguments ─────────────────────────────────────────────
        DeclareLaunchArgument(
            "localization", default_value="false",
            description="Run in localization mode against a saved map.",
        ),
        DeclareLaunchArgument(
            "database_path", default_value="~/.ros/rtabmap.db",
            description="RTAB-Map database file path.",
        ),
        DeclareLaunchArgument(
            "rtabmap_args", default_value="",
            description="Extra RTAB-Map args (e.g. '-d' to delete DB on start).",
        ),
        DeclareLaunchArgument(
            "rtabmap_viz", default_value="false",
            description="Launch RTAB-Map's built-in visualizer.",
        ),

        # ── Perception (base + RealSense) ──────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_dir, "launch", "perception.launch.py")
            ),
            launch_arguments={
                "front_port": front_port,
                "rear_port": rear_port,
            }.items(),
        ),

        # ── SLAM (depthimage_to_laserscan + RTAB-Map) ──────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam_dir, "launch", "slam.launch.py")
            ),
            launch_arguments={
                "localization": localization,
                "database_path": database_path,
                "rtabmap_args": rtabmap_args,
                "rtabmap_viz": rtabmap_viz,
            }.items(),
        ),
    ])
