"""Sim-in-the-loop launch — autonomy-consuming side of the stack.

Brings up ONLY the nodes that consume remote-published topics from the
DGX Isaac Sim ROS2 bridge. No real driver, no real camera, no hardware
watchdogs that would fail without the USB bus. The DGX-side sim publishes
on the same topic names the robot's hardware normally produces, so the
autonomy pipeline can run against simulated sensors without code changes.

Started here (subscribes to DGX-published topics):
  - strafer_description robot_state_publisher (URDF → base_link TF tree)
  - timestamp_fixer                           (/d555/color/image_sync etc.)
  - depthimage_to_laserscan                    (/scan from /d555 depth)
  - RTAB-Map                                   (map → odom, /rtabmap/map)
  - Nav2                                       (/navigate_to_pose)
  - goal_projection_node                       (bbox → map-frame goal)
  - strafer-executor                           (autonomy command server)

Not started (hardware-only):
  - strafer_driver (RoboClaw — no motors)
  - realsense_node (D555 — no camera; DGX bridge publishes frames instead)
  - imu_filter_madgwick (no IMU — DGX bridge has no ROS2PublishImu node)

Prerequisites
-------------
Source the env file first on BOTH hosts so DDS discovery works:

    source source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env
    ros2 launch strafer_bringup bringup_sim_in_the_loop.launch.py
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _launch_setup(context, *args, **kwargs):
    description_dir = get_package_share_directory("strafer_description")
    navigation_dir = get_package_share_directory("strafer_navigation")
    slam_dir = get_package_share_directory("strafer_slam")

    vlm_url = LaunchConfiguration("vlm_url").perform(context)
    planner_url = LaunchConfiguration("planner_url").perform(context)
    nav_log_level = LaunchConfiguration("nav_log_level").perform(context)
    localization = LaunchConfiguration("localization").perform(context)
    database_path = LaunchConfiguration("database_path").perform(context)
    rtabmap_args = LaunchConfiguration("rtabmap_args").perform(context)
    rtabmap_viz = LaunchConfiguration("rtabmap_viz").perform(context)

    nodes = [
        # ── Robot description (URDF → TF tree) ─────────────────────────
        # TF from base_link → d555_link. Odom TF comes from the DGX bridge;
        # map → odom comes from RTAB-Map.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(description_dir, "launch", "description.launch.py")
            ),
        ),

        # ── Timestamp fixer ─────────────────────────────────────────────
        # Re-stamps camera frames from the DGX bridge so message sync
        # across odom / depth / color works in RTAB-Map and Nav2.
        #
        # In sim the D555 is a single co-registered RGB+D sensor, so the
        # bridge's raw depth is already aligned to color. The real-robot
        # driver publishes under /d555/aligned_depth_to_color/* because
        # alignment happens at the RealSense driver; these remaps hand
        # the bridge's /d555/depth/* topics in under the same contract.
        Node(
            package="strafer_perception",
            executable="timestamp_fixer",
            name="timestamp_fixer",
            output="screen",
            # restamp=False keeps the bridge's sim-time stamps so the
            # cameras stay synchronised with the bridge-published odom,
            # which does not pass through this node.
            parameters=[{"restamp": False}],
            remappings=[
                ("/d555/aligned_depth_to_color/image_raw",
                 "/d555/depth/image_rect_raw"),
                ("/d555/aligned_depth_to_color/camera_info",
                 "/d555/depth/camera_info"),
            ],
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

        # ── Nav2 ───────────────────────────────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(navigation_dir, "launch", "navigation.launch.py")
            ),
            launch_arguments={
                "log_level": nav_log_level,
            }.items(),
        ),

        # ── Goal projection service (VLM bbox → map-frame pose) ────────
        Node(
            package="strafer_perception",
            executable="goal_projection",
            name="goal_projection",
            output="screen",
        ),

        # ── Autonomy executor ─────────────────────────────────────────
        ExecuteProcess(
            cmd=["strafer-executor"],
            name="strafer_executor",
            output="screen",
            additional_env={
                "VLM_URL": vlm_url,
                "PLANNER_URL": planner_url,
                # Surfaces in ros_client startup: skip hardware watchdogs
                # that would false-positive without the real USB bus.
                "HARDWARE_PRESENT": "false",
            },
        ),
    ]
    return nodes


def generate_launch_description():
    default_vlm = os.environ.get("VLM_URL", "")
    default_planner = os.environ.get("PLANNER_URL", "")

    return LaunchDescription([
        DeclareLaunchArgument(
            "vlm_url", default_value=default_vlm,
            description="VLM grounding service URL on DGX (e.g. http://192.168.50.196:8100).",
        ),
        DeclareLaunchArgument(
            "planner_url", default_value=default_planner,
            description="LLM planner service URL on DGX (e.g. http://192.168.50.196:8200).",
        ),
        DeclareLaunchArgument(
            "localization", default_value="false",
            description="Run RTAB-Map in localization mode against a saved database.",
        ),
        DeclareLaunchArgument(
            "database_path", default_value="~/.ros/rtabmap.db",
            description="RTAB-Map database file path.",
        ),
        DeclareLaunchArgument(
            "rtabmap_args", default_value="",
            description="Extra RTAB-Map CLI args (e.g. '-d' to delete DB on start).",
        ),
        DeclareLaunchArgument(
            "rtabmap_viz", default_value="false",
            description="Launch the RTAB-Map visualizer.",
        ),
        DeclareLaunchArgument(
            "nav_log_level", default_value="warn",
            description="Log level for Nav2 nodes.",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
