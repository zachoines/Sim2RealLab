"""Strafer navigation launch — SLAM + Nav2.

Launches the full autonomy stack:
  - slam.launch.py:          base + perception + SLAM
  - strafer_navigation:      Nav2 (MPPI controller, costmaps, behavior trees)

Usage:
  ros2 launch strafer_bringup navigation.launch.py
  ros2 launch strafer_bringup navigation.launch.py localization:=true
  ros2 launch strafer_bringup navigation.launch.py nav_log_level:=info
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    bringup_dir = get_package_share_directory("strafer_bringup")
    navigation_dir = get_package_share_directory("strafer_navigation")

    front_port = LaunchConfiguration("front_port")
    rear_port = LaunchConfiguration("rear_port")
    localization = LaunchConfiguration("localization")
    database_path = LaunchConfiguration("database_path")
    rtabmap_args = LaunchConfiguration("rtabmap_args")
    rtabmap_viz = LaunchConfiguration("rtabmap_viz")
    nav_log_level = LaunchConfiguration("nav_log_level")

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
        DeclareLaunchArgument(
            "nav_log_level", default_value="warn",
            description="Log level for Nav2 nodes.",
        ),
        DeclareLaunchArgument(
            "donut_warmup", default_value="false",
            description=(
                "Opt-in 360 deg rotation at bringup to fill in the "
                "camera blind-spot donut around base_link. Default off "
                "on real-robot: in steady-state ops the rtabmap.db "
                "carries the donut fill-in from prior sessions, and a "
                "startup spin has chassis/safety implications on real "
                "hardware. Pass donut_warmup:=true on a cold-start "
                "session (fresh rtabmap.db or new environment) where "
                "the first mission would otherwise oscillate."
            ),
        ),

        # ── SLAM (base + perception + RTAB-Map) ────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_dir, "launch", "slam.launch.py")
            ),
            launch_arguments={
                "front_port": front_port,
                "rear_port": rear_port,
                "localization": localization,
                "database_path": database_path,
                "rtabmap_args": rtabmap_args,
                "rtabmap_viz": rtabmap_viz,
            }.items(),
        ),

        # ── Navigation (Nav2) ──────────────────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(navigation_dir, "launch", "navigation.launch.py")
            ),
            launch_arguments={
                "log_level": nav_log_level,
            }.items(),
        ),

        # ── Opt-in donut warmup (real-robot, default off) ──────────────
        # Off by default — see the launch arg description for the
        # rationale. The node spins, then exits; can't fire on
        # subsequent goals.
        Node(
            package="strafer_bringup",
            executable="donut_warmup",
            name="donut_warmup",
            output="screen",
            condition=IfCondition(LaunchConfiguration("donut_warmup")),
        ),
    ])
