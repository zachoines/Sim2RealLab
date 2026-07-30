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
``/plan`` derived from it is meaningless.

``task_id`` / ``scene_key`` (env ``STRAFER_SLAM_TASK_ID`` /
``STRAFER_SLAM_SCENE_TOKEN``) form a scene key that derives the database path
when ``database_path`` is left at its default, and a ``<db>.scene.json`` sidecar
records which key a database belongs to. A launch whose key disagrees with the
sidecar aborts. An empty key keeps the unkeyed ``~/.ros/rtabmap.db``, and an
explicit ``database_path:=`` overrides both.
"""

import datetime
import json
import os
import re
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from strafer_shared.constants import DEPTH_MIN, DEPTH_MAX, MAP_RESOLUTION

# Every caller in strafer_bringup spells its database_path default exactly like
# this, so receiving this value means nobody chose a path and scene keying may
# derive one; any other value is honoured verbatim.
_UNKEYED_DEFAULT_DB = "~/.ros/rtabmap.db"

# Sits beside the database so deleting the db (or its volume) takes both.
_SCENE_SIDECAR_SUFFIX = ".scene.json"


def _slug(value: str) -> str:
    """Filename-safe form of a task id / scene token."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")


def _scene_key(task_id: str, scene_token: str) -> str:
    """Combined scene key; empty when neither half is set."""
    return "_".join(p for p in (_slug(task_id), _slug(scene_token)) if p)


def _resolve_database_path(database_path: str, key: str) -> str:
    """Expanded db path, keyed to the scene when the caller chose no path."""
    if key and database_path.strip() in ("", _UNKEYED_DEFAULT_DB):
        database_path = f"~/.ros/rtabmap_{key}.db"
    return os.path.expanduser(database_path.strip() or _UNKEYED_DEFAULT_DB)


def _claim_database_for_scene(db_path: str, key: str, *, deleting: bool) -> None:
    """Refuse a database recorded under a different scene; else claim it.

    ``deleting`` is True when ``rtabmap_args`` carries ``-d`` (rtabmap wipes the
    db at start), so there is no foreign map left to protect against.
    """
    sidecar = db_path + _SCENE_SIDECAR_SUFFIX
    recorded = None
    if os.path.isfile(sidecar):
        try:
            recorded = json.loads(open(sidecar).read()).get("key")
        except (OSError, ValueError):
            recorded = None            # unreadable -> re-adopt below

    if (
        not deleting
        and os.path.isfile(db_path)
        and recorded is not None
        and recorded != key
    ):
        raise RuntimeError(
            f"RTAB-Map database {db_path!r} was recorded under scene key "
            f"{recorded!r} but this launch has scene key {key!r}. Loading it "
            "would silently publish a map of a DIFFERENT scene, and every Nav2 "
            "/plan derived from it would be meaningless. Refusing to start. "
            "Pick one:\n"
            f"  - new scene:     STRAFER_SLAM_SCENE_TOKEN=<token>   (db becomes "
            "~/.ros/rtabmap_<key>.db)\n"
            f"  - explicit db:   database_path:=<path>\n"
            f"  - wipe & remap:  rtabmap_args:=-d\n"
            f"  - it really IS scene {recorded!r}: set the matching task_id / "
            "scene_key, or delete "
            f"{sidecar!r} to re-adopt the database under the current key."
        )

    # Claim it so the next mismatch is caught. Best-effort: an unwritable
    # sidecar loses the guard but must not stop SLAM.
    record = {
        "key": key,
        "task_id": os.environ.get("STRAFER_SLAM_TASK_ID", ""),
        "scene_token": os.environ.get("STRAFER_SLAM_SCENE_TOKEN", ""),
        "database_path": db_path,
        "claimed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "written_by": "strafer_slam/slam.launch.py",
    }
    try:
        os.makedirs(os.path.dirname(sidecar) or ".", exist_ok=True)
        with open(sidecar, "w") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
    except OSError as exc:
        print(
            f"[strafer_slam] WARNING: could not write the scene sidecar "
            f"{sidecar!r} ({exc}); the wrong-scene guard is INACTIVE for this "
            "database.",
            flush=True,
        )


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

    # ── Scene keying + wrong-scene guard ────────────────────────────────
    # The path must resolve before it can be claimed, since the key may derive
    # it. A mismatch raises out of this OpaqueFunction and aborts the launch.
    key = _scene_key(task_id, scene_key_arg)
    database_path = _resolve_database_path(database_path, key)
    extra_args = rtabmap_args.split() if rtabmap_args.strip() else []
    _claim_database_for_scene(database_path, key, deleting="-d" in extra_args)

    notices = [LogInfo(msg=(
        f"[strafer_slam] database={database_path} scene_key="
        f"{key or '<none>'}"
    ))]
    if use_sim_time and not key:
        # Warned rather than refused: restarting this container alone is a
        # supported way to recover from a /clock hitch, and it arrives unkeyed.
        notices.append(LogInfo(msg=(
            "[strafer_slam] WARNING: use_sim_time is true with NO scene key, so "
            f"this run reloads whatever map is already at {database_path}. A "
            "procedurally regenerated sim scene may be a different layout, "
            "which would make /rtabmap/map and every Nav2 /plan derived from it "
            "meaningless. Set STRAFER_SLAM_SCENE_TOKEN (or scene_key:=) per sim "
            "run for a keyed database and the wrong-scene guard."
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
            "database_path", default_value=_UNKEYED_DEFAULT_DB,
            description=(
                "Path to the RTAB-Map database file. Left at this default, a "
                "non-empty scene key derives ~/.ros/rtabmap_<key>.db instead; "
                "any other value is honoured verbatim."
            ),
        ),
        DeclareLaunchArgument(
            "task_id", default_value=os.environ.get("STRAFER_SLAM_TASK_ID", ""),
            description=(
                "Scene-key half 1: the task/environment the map belongs to, "
                "e.g. the Isaac task id. Empty on the real robot."
            ),
        ),
        DeclareLaunchArgument(
            "scene_key",
            default_value=os.environ.get("STRAFER_SLAM_SCENE_TOKEN", ""),
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
