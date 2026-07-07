"""Sim-in-the-loop launch — autonomy-consuming side of the stack.

Brings up ONLY the nodes that consume remote-published topics from the
sim host's Isaac Sim ROS2 bridge. No real driver, no real camera, no
hardware watchdogs that would fail without the USB bus. The sim host
publishes on the same topic names the robot's hardware normally produces,
so the autonomy pipeline can run against simulated sensors without code
changes.

Started here (subscribes to sim-published topics):
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
  - realsense_node (D555 — no camera; the sim bridge publishes frames instead)
  - imu_filter_madgwick (no IMU — the sim bridge has no ROS2PublishImu node)

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

import logging
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

# navigate_to_pose backends that need the RL inference node; the
# subgoal generator is needed only for the hybrid one.
_POLICY_BACKENDS = {"strafer_direct", "hybrid_nav2_strafer"}


def _launch_setup(context, *args, **kwargs):
    description_dir = get_package_share_directory("strafer_description")
    navigation_dir = get_package_share_directory("strafer_navigation")
    slam_dir = get_package_share_directory("strafer_slam")
    inference_dir = get_package_share_directory("strafer_inference")

    vlm_url = LaunchConfiguration("vlm_url").perform(context)
    planner_url = LaunchConfiguration("planner_url").perform(context)
    nav_log_level = LaunchConfiguration("nav_log_level").perform(context)
    localization = LaunchConfiguration("localization").perform(context)
    database_path = LaunchConfiguration("database_path").perform(context)
    rtabmap_args = LaunchConfiguration("rtabmap_args").perform(context)
    rtabmap_viz = LaunchConfiguration("rtabmap_viz").perform(context)
    viewer_port = LaunchConfiguration("viewer_port").perform(context)
    nav_backend = LaunchConfiguration("nav_backend").perform(context)
    model_path = LaunchConfiguration("model_path").perform(context)
    policy_variant = LaunchConfiguration("policy_variant").perform(context)

    # The sim bridge publishes /clock, so every node in this launch runs
    # on sim time. Pins the whole chain (TF, approx_sync, freshness
    # checks) to a single monotonic source.
    sim_time = "true"

    nodes = [
        # ── Robot description (URDF → TF tree) ─────────────────────────
        # TF from base_link → d555_link. Odom TF comes from the sim bridge;
        # map → odom comes from RTAB-Map.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(description_dir, "launch", "description.launch.py")
            ),
            launch_arguments={"use_sim_time": sim_time}.items(),
        ),

        # ── Timestamp fixer ─────────────────────────────────────────────
        # Re-stamps camera frames from the sim bridge so message sync
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
            # restamp=False — bridge already publishes with /clock-derived
            # sim time; restamping rewrites each message with the current
            # sim time at relay, which advances between depth + camera_info
            # callbacks and breaks depth_image_proc's exact synchronizer.
            parameters=[{"use_sim_time": True, "restamp": False}],
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

        # ── One-shot donut-coverage warmup ────────────────────────────
        # Publishes a slow 360 rotation on /cmd_vel once at bringup so
        # RTAB-Map gets ~20 depth frames at varied yaws at the spawn
        # pose. The node exits after one rotation; that's the
        # once-per-session guarantee. Operator should wait for the
        # warmup to complete before submitting the first nav goal —
        # the spin and the autonomy executor share /cmd_vel.
        #
        # Override angular_vel via:
        #   ros2 launch ... donut_warmup:=true \
        #     -p donut_warmup.angular_vel:=0.6
        # or disable entirely with donut_warmup:=false.
        Node(
            package="strafer_bringup",
            executable="donut_warmup",
            name="donut_warmup",
            output="screen",
            condition=IfCondition(LaunchConfiguration("donut_warmup")),
            parameters=[{"use_sim_time": True}],
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

    # ── RL inference backend (gated on nav_backend) ────────────────────
    # nav2/unset launches neither node; strafer_direct adds the inference
    # node; hybrid adds the subgoal generator too. Same STRAFER_NAV_BACKEND
    # the executor dispatch reads, so server and client stay in lockstep.
    if nav_backend in _POLICY_BACKENDS:
        if not model_path:
            logging.getLogger("launch").error(
                f"STRAFER_NAV_BACKEND={nav_backend} but "
                "STRAFER_INFERENCE_MODEL_PATH is empty — strafer_inference "
                "will NOT advertise navigate_to_pose; every mission will "
                "silently fall back to nav2. Set STRAFER_INFERENCE_MODEL_PATH."
            )
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(inference_dir, "launch", "inference.launch.py")
            ),
            launch_arguments={
                "model_path": model_path,
                "policy_variant": policy_variant,
                "use_sim_time": sim_time,
            }.items(),
        ))
    if nav_backend == "hybrid_nav2_strafer":
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(inference_dir, "launch", "subgoal_generator.launch.py")
            ),
            launch_arguments={"use_sim_time": sim_time}.items(),
        ))

    return nodes


def generate_launch_description():
    default_vlm = os.environ.get("VLM_URL", "")
    default_planner = os.environ.get("PLANNER_URL", "")
    default_backend = os.environ.get("STRAFER_NAV_BACKEND", "nav2")
    default_model = os.environ.get("STRAFER_INFERENCE_MODEL_PATH", "")
    default_variant = os.environ.get("STRAFER_POLICY_VARIANT", "DEPTH")

    return LaunchDescription([
        DeclareLaunchArgument(
            "vlm_url", default_value=default_vlm,
            description="VLM grounding service URL on the sim host (e.g. http://192.168.50.196:8100).",
        ),
        DeclareLaunchArgument(
            "nav_backend", default_value=default_backend,
            description=(
                "navigate_to_pose backend: nav2 (default; launches NEITHER "
                "inference node), strafer_direct (+ strafer_inference), "
                "hybrid_nav2_strafer (+ strafer_inference + "
                "strafer_subgoal_generator). Must match STRAFER_NAV_BACKEND "
                "on the autonomy executor."
            ),
        ),
        DeclareLaunchArgument(
            "model_path", default_value=default_model,
            description=(
                "Exported policy artifact for strafer_inference. Empty with a "
                "policy backend is a misconfiguration (no navigate_to_pose)."
            ),
        ),
        DeclareLaunchArgument(
            "policy_variant", default_value=default_variant,
            description="PolicyVariant name (DEPTH, NOCAM, NOCAM_SUBGOAL, DEPTH_SUBGOAL) for strafer_inference.",
        ),
        DeclareLaunchArgument(
            "planner_url", default_value=default_planner,
            description="LLM planner service URL on the sim host (e.g. http://192.168.50.196:8200).",
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
        DeclareLaunchArgument(
            "donut_warmup", default_value="true",
            description=(
                "Run a one-shot 360 rotation at bringup so RTAB-Map "
                "collects depth frames at varied yaws and the camera "
                "blind-spot donut around base_link is covered before "
                "the first nav goal. Set false to skip — useful when "
                "the operator already has a populated map or doesn't "
                "want the startup spin."
            ),
        ),
        OpaqueFunction(function=_launch_setup),
    ])
