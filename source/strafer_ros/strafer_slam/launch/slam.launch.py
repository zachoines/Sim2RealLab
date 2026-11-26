"""Launch RTAB-Map SLAM for the Strafer robot.

Requires (launched separately or via strafer_bringup):
  - strafer_driver:   /strafer/odom, /strafer/joint_states
  - strafer_perception: /d555/color/image_sync, /d555/…/image_sync, /d555/imu/filtered
  - strafer_description: robot_state_publisher (TF tree)

This launch file:
  1. Projects the timestamp-fixed depth image into a PointCloud2 via
     depth_image_proc::point_cloud_xyz_node. Works for both lanes —
     real D555 (perception.launch.py) and the sim bridge feed the
     same `_sync` depth + camera_info topics.
  2. Runs pointcloud_to_laserscan on that cloud to publish /scan with
     a base_link Z-axis filter that excludes floor + above-body returns.
  3. Starts RTAB-Map in mapping or localization mode.

Modes:
  - Mapping (default):  ros2 launch strafer_slam slam.launch.py
  - Localization:       ros2 launch strafer_slam slam.launch.py localization:=true
  - Delete & remap:     ros2 launch strafer_slam slam.launch.py rtabmap_args:=-d

Scene keying
------------
RTAB-Map reloads its persisted database on start, which is right for the one
persistent scene the real robot maps and wrong for a procedurally regenerated
sim scene, where the reloaded grid describes a different layout and every Nav2
``/plan`` derived from it is meaningless. It is wrong as well for a map made
from another camera mount, whose features sit against that mount's camera pose.

``task_id`` / ``scene_key`` (env ``STRAFER_SLAM_TASK_ID`` /
``STRAFER_SLAM_SCENE_TOKEN``) form a scene key. A database's key is that scene
key, ``@`` and the camera mount (``CAMERA_OFFSET_X/Y/Z`` in millimetres, the
URDF's ``d555_mount``), so an empty scene key keys the database to the mount
alone. Left at its default, ``database_path`` becomes
``~/.ros/rtabmap_<key>.db``. A ``<db>.scene.json`` sidecar records the key, and
an existing database loads only under the key it records: a launch that finds
another key, or none, aborts. An explicit ``database_path:=`` chooses the file,
not that check. The TF publisher must run on the same constants.
"""

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from strafer_shared.constants import DEPTH_MIN, DEPTH_MAX, MAP_RESOLUTION
from strafer_slam import database


def _launch_setup(context, *args, **kwargs):
    """Resolve launch arguments, load RTAB-Map params, and build nodes."""
    pkg_dir = get_package_share_directory("strafer_slam")

    # Resolve launch arguments
    localization = LaunchConfiguration("localization").perform(context) == "true"
    database_path = LaunchConfiguration("database_path").perform(context)
    rtabmap_args = LaunchConfiguration("rtabmap_args").perform(context)
    show_viz = LaunchConfiguration("rtabmap_viz").perform(context) == "true"
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context) == "true"
    task_id = LaunchConfiguration("task_id").perform(context)
    scene_key_arg = LaunchConfiguration("scene_key").perform(context)

    # ── Scene and mount keying + guard ──────────────────────────────────
    # The key joins the scene key to the camera mount, so a database reloads
    # only under the mount it was made from. The path must resolve before it
    # can be claimed, since the key may derive it. A refusal raises out of
    # this OpaqueFunction and aborts the launch.
    scene = database.scene_key(task_id, scene_key_arg)
    key = database.database_key(scene, database.CAMERA_MOUNT)
    database_path = database.resolve_database_path(database_path, key)
    extra_args = rtabmap_args.split() if rtabmap_args.strip() else []
    database.claim_database(database_path, key, deleting="-d" in extra_args)

    notices = [LogInfo(msg=f"[strafer_slam] database={database_path} key={key}")]
    if use_sim_time and not scene:
        # Warned rather than refused: restarting this container alone is a
        # supported way to recover from a /clock hitch, and it arrives with no
        # scene key.
        notices.append(LogInfo(msg=(
            "[strafer_slam] WARNING: use_sim_time is true with NO scene key, so "
            f"this run reloads whatever map is already at {database_path}. A "
            "procedurally regenerated sim scene may be a different layout, "
            "which would make /rtabmap/map and every Nav2 /plan derived from it "
            "meaningless. Set STRAFER_SLAM_SCENE_TOKEN (or scene_key:=) per sim "
            "run for a database of its own."
        )))

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

    pc_to_scan_params_path = os.path.join(
        pkg_dir, "config", "pointcloud_to_laserscan.yaml"
    )

    # ── Nodes ───────────────────────────────────────────────────────────
    nodes = notices + [
        # Project depth + intrinsics into a PointCloud2. Both lanes feed
        # /d555/aligned_depth_to_color/image_sync — real D555 via the
        # realsense driver + timestamp_fixer, sim via the sim bridge +
        # timestamp_fixer's remap — so the same projector works for both.
        Node(
            package="depth_image_proc",
            executable="point_cloud_xyz_node",
            name="depth_to_pointcloud",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
            remappings=[
                ("image_rect", "/d555/aligned_depth_to_color/image_sync"),
                ("camera_info", "/d555/aligned_depth_to_color/camera_info_sync"),
                ("points", "/d555/aligned_depth_to_color/points"),
            ],
        ),
        # Virtual 2D laser scan from the projected depth cloud. base_link
        # Z filter drops floor + above-body returns so the scan doesn't
        # contain a phantom arc tracking the camera's downward FOV cone.
        Node(
            package="pointcloud_to_laserscan",
            executable="pointcloud_to_laserscan_node",
            name="pointcloud_to_laserscan",
            output="screen",
            parameters=[
                pc_to_scan_params_path,
                {
                    "range_min": DEPTH_MIN,
                    "range_max": DEPTH_MAX,
                    "use_sim_time": use_sim_time,
                },
            ],
            remappings=[
                ("cloud_in", "/d555/aligned_depth_to_color/points"),
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
                    # Wider than the default (0.0 = unrestricted but
                    # message_filters' policy struggles when one topic
                    # is at ~1 Hz with 0.5 s jitter while others are
                    # faster). Sim-in-the-loop bridge clocking sees the
                    # tail; real-robot rates are higher and the wider
                    # window is a no-op.
                    "approx_sync_max_interval": 2.0,
                    "qos_image": 1,
                    # pointcloud_to_laserscan hard-codes SensorDataQoS
                    # (BEST_EFFORT) on its publisher. Match it on the
                    # sub side; the prior RELIABLE setting silently
                    # discarded every scan via incompatible-QoS.
                    "qos_scan": 2,
                    "qos_odom": 1,
                    "qos_camera_info": 1,
                    "qos_imu": 1,
                    "wait_for_transform": 0.2,
                    "use_sim_time": use_sim_time,
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
                        "use_sim_time": use_sim_time,
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
            "database_path", default_value=database.DEFAULT_DB_ARG,
            description=(
                "Path to the RTAB-Map database file. This default, or an empty "
                "value, derives ~/.ros/rtabmap_<key>.db from the key "
                "(scene key @ camera mount); any other value is the path, with ~ "
                "expanded. Either way an existing database loads only under the "
                "key its sidecar records."
            ),
        ),
        DeclareLaunchArgument(
            "task_id", default_value=os.environ.get(database.TASK_ID_ENV, ""),
            description=(
                "Scene-key half 1: the task/environment the map belongs to, "
                "e.g. the Isaac task id. Empty on the real robot."
            ),
        ),
        DeclareLaunchArgument(
            "scene_key",
            default_value=os.environ.get(database.SCENE_TOKEN_ENV, ""),
            description=(
                "Scene-key half 2: per-sim-run token. Bump it on every sim "
                "restart — a procedural scene regenerates its layout, and a map "
                "from the previous run silently corrupts /plan."
            ),
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
            "use_sim_time", default_value="false",
            description="Set true when /clock is published upstream (sim-in-the-loop).",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
