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
  - foxglove_bridge                            (WebSocket :8765, viewer:=true)

Not started (hardware-only):
  - strafer_driver (RoboClaw — no motors)
  - realsense_node (D555 — no camera; DGX bridge publishes frames instead)
  - imu_filter_madgwick (no IMU — DGX bridge has no ROS2PublishImu node)

Prerequisites
-------------
Source the env file first on BOTH hosts so DDS discovery works:

    source source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env
    ros2 launch strafer_bringup bringup_sim_in_the_loop.launch.py

A ``foxglove_bridge`` node is brought up on TCP 8765 by default so an
operator on a remote workstation can SSH-tunnel back and inspect the
robot's camera, depth, TF, and map state in Foxglove Studio. Disable
with ``viewer:=false``. End-to-end setup walkthrough lives in
``docs/INTEGRATION_SIM_IN_THE_LOOP.md``.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
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
    viewer_port = LaunchConfiguration("viewer_port").perform(context)

    # The DGX bridge publishes /clock, so every node in this launch runs
    # on sim time. Pins the whole chain (TF, approx_sync, freshness
    # checks) to a single monotonic source.
    sim_time = "true"

    nodes = [
        # ── Robot description (URDF → TF tree) ─────────────────────────
        # TF from base_link → d555_link. Odom TF comes from the DGX bridge;
        # map → odom comes from RTAB-Map.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(description_dir, "launch", "description.launch.py")
            ),
            launch_arguments={"use_sim_time": sim_time}.items(),
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
            parameters=[{"use_sim_time": True}],
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
                "use_sim_time": sim_time,
            }.items(),
        ),

        # ── Nav2 ───────────────────────────────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(navigation_dir, "launch", "navigation.launch.py")
            ),
            launch_arguments={
                "log_level": nav_log_level,
                "use_sim_time": sim_time,
            }.items(),
        ),

        # ── Goal projection service (VLM bbox → map-frame pose) ────────
        Node(
            package="strafer_perception",
            executable="goal_projection",
            name="goal_projection",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),

        # ── Autonomy executor ─────────────────────────────────────────
        ExecuteProcess(
            cmd=["strafer-executor", "--ros-args", "-p", "use_sim_time:=true"],
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

        # ── Headless visual debugger (Foxglove WebSocket) ─────────────
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            condition=IfCondition(LaunchConfiguration("viewer")),
            parameters=[{
                "use_sim_time": True,
                "port": int(viewer_port),
            }],
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
        DeclareLaunchArgument(
            "viewer", default_value="true",
            description="Start foxglove_bridge on `viewer_port` for remote inspection.",
        ),
        DeclareLaunchArgument(
            "viewer_port", default_value="8765",
            description="Foxglove WebSocket TCP port.",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
