"""Launch RTAB-Map SLAM for the Strafer robot.

Requires (launched separately or via strafer_bringup):
  - strafer_driver:   /strafer/odom, /strafer/joint_states
  - strafer_perception: /d555/color/image_sync, /d555/…/image_sync, /d555/imu/filtered
  - strafer_description: robot_state_publisher (TF tree)

This launch file:
  1. Starts depthimage_to_laserscan (depth → /scan for occupancy grid)
  2. Starts RTAB-Map in mapping or localization mode

Modes:
  - Mapping (default):  ros2 launch strafer_slam slam.launch.py
  - Localization:       ros2 launch strafer_slam slam.launch.py localization:=true
  - Delete & remap:     ros2 launch strafer_slam slam.launch.py rtabmap_args:=-d

The map database is saved to ~/.ros/rtabmap.db by default.
"""

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from strafer_shared.constants import DEPTH_MIN, DEPTH_MAX, MAP_RESOLUTION


def _launch_setup(context, *args, **kwargs):
    """Resolve launch arguments, load RTAB-Map params, and build nodes."""
    pkg_dir = get_package_share_directory("strafer_slam")

    # Resolve launch arguments
    localization = LaunchConfiguration("localization").perform(context) == "true"
    database_path = LaunchConfiguration("database_path").perform(context)
    rtabmap_args = LaunchConfiguration("rtabmap_args").perform(context)
    show_viz = LaunchConfiguration("rtabmap_viz").perform(context) == "true"

    # ── Load & patch RTAB-Map parameters ────────────────────────────────
    rtabmap_params_path = os.path.join(pkg_dir, "config", "rtabmap_params.yaml")
    with open(rtabmap_params_path) as f:
        rtabmap_cfg = yaml.safe_load(f)

    # Override from strafer_shared.constants (single source of truth)
    rtabmap_cfg["Grid/RangeMin"] = str(DEPTH_MIN)
    rtabmap_cfg["Grid/RangeMax"] = str(DEPTH_MAX)
    rtabmap_cfg["Grid/CellSize"] = str(MAP_RESOLUTION)

    # Localization mode: freeze the map, load all nodes into WM
    if localization:
        rtabmap_cfg["Mem/IncrementalMemory"] = "false"
        rtabmap_cfg["Mem/InitWMWithAllNodes"] = "true"

    depthimage_params_path = os.path.join(
        pkg_dir, "config", "depthimage_to_laserscan.yaml"
    )

    # Parse user's extra RTAB-Map CLI args (e.g. "-d")
    extra_args = rtabmap_args.split() if rtabmap_args.strip() else []

    # ── Nodes ───────────────────────────────────────────────────────────
    nodes = [
        # Virtual 2D laser scan from aligned depth image
        Node(
            package="depthimage_to_laserscan",
            executable="depthimage_to_laserscan_node",
            name="depthimage_to_laserscan",
            output="screen",
            parameters=[
                depthimage_params_path,
                {
                    "range_min": DEPTH_MIN,
                    "range_max": DEPTH_MAX,
                    "output_frame": "d555_link",
                },
            ],
            remappings=[
                ("depth", "/d555/aligned_depth_to_color/image_sync"),
                ("depth_camera_info", "/d555/aligned_depth_to_color/camera_info_sync"),
                ("scan", "/scan"),
            ],
        ),

        # RTAB-Map SLAM — launched directly (not via upstream launch file)
        # so we can inject rtabmap_params.yaml with constants overrides.
        # Visual/ICP odometry nodes are not needed (wheel odom from driver).
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            namespace="rtabmap",
            parameters=[
                rtabmap_cfg,
                {
                    "subscribe_depth": True,
                    "subscribe_rgbd": False,
                    "subscribe_rgb": False,
                    "subscribe_stereo": False,
                    "subscribe_scan": True,
                    "subscribe_scan_cloud": False,
                    "subscribe_user_data": False,
                    "subscribe_odom_info": False,
                    "frame_id": "base_link",
                    "map_frame_id": "map",
                    "odom_frame_id": "",
                    "publish_tf": True,
                    "database_path": database_path,
                    "approx_sync": True,
                    "topic_queue_size": 100,
                    "sync_queue_size": 100,
                    "qos_image": 1,
                    "qos_scan": 1,
                    "qos_odom": 1,
                    "qos_camera_info": 1,
                    "qos_imu": 1,
                    "wait_for_transform": 0.2,
                },
            ],
            remappings=[
                ("rgb/image", "/d555/color/image_sync"),
                ("depth/image", "/d555/aligned_depth_to_color/image_sync"),
                ("rgb/camera_info", "/d555/color/camera_info_sync"),
                ("scan", "/scan"),
                ("odom", "/strafer/odom"),
                ("imu", "/d555/imu/filtered"),
            ],
            arguments=extra_args,
        ),
    ]

    if show_viz:
        nodes.append(
            Node(
                package="rtabmap_viz",
                executable="rtabmap_viz",
                name="rtabmap_viz",
                output="screen",
                namespace="rtabmap",
                parameters=[
                    rtabmap_cfg,
                    {
                        "subscribe_depth": True,
                        "subscribe_scan": True,
                        "subscribe_odom_info": False,
                        "frame_id": "base_link",
                        "approx_sync": True,
                        "qos_image": 1,
                        "qos_scan": 1,
                        "qos_odom": 1,
                        "qos_camera_info": 1,
                    },
                ],
                remappings=[
                    ("rgb/image", "/d555/color/image_sync"),
                    ("depth/image", "/d555/aligned_depth_to_color/image_sync"),
                    ("rgb/camera_info", "/d555/color/camera_info_sync"),
                    ("scan", "/scan"),
                    ("odom", "/strafer/odom"),
                ],
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "localization", default_value="false",
            description="If true, run in localization mode against saved map.",
        ),
        DeclareLaunchArgument(
            "database_path", default_value="~/.ros/rtabmap.db",
            description="Path to the RTAB-Map database file.",
        ),
        DeclareLaunchArgument(
            "rtabmap_args", default_value="",
            description="Extra RTAB-Map args (e.g. '-d' to delete DB on start).",
        ),
        DeclareLaunchArgument(
            "rtabmap_viz", default_value="false",
            description="Launch RTAB-Map's built-in visualizer.",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
