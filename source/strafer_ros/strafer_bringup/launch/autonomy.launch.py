"""Strafer autonomy launch — navigation stack + executor.

Launches the full robot autonomy stack:
  - navigation.launch.py:     base + perception + SLAM + Nav2
  - goal_projection_node:     VLM bbox → map-frame goal pose service
  - strafer-executor:         autonomy command server (connects to DGX services)

Requires VLM_URL and PLANNER_URL to point at the DGX Spark services.

Usage:
  ros2 launch strafer_bringup autonomy.launch.py \
      vlm_url:=http://192.168.50.196:8100 \
      planner_url:=http://192.168.50.196:8200

  # Or with env vars already set:
  VLM_URL=http://192.168.50.196:8100 PLANNER_URL=http://192.168.50.196:8200 \
      ros2 launch strafer_bringup autonomy.launch.py
"""

import logging
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

# navigate_to_pose backends that need the RL inference node; the
# subgoal generator is needed only for the hybrid one.
_POLICY_BACKENDS = {"strafer_direct", "hybrid_nav2_strafer"}


def generate_launch_description():
    bringup_dir = get_package_share_directory("strafer_bringup")
    inference_dir = get_package_share_directory("strafer_inference")

    # Inherit from env if set, otherwise require launch argument.
    default_vlm = os.environ.get("VLM_URL", "")
    default_planner = os.environ.get("PLANNER_URL", "")

    # Backend gating is build-time (same os.environ precedent as the
    # VLM/planner URLs above). The real robot has no /clock, so the
    # inference nodes always run on wall time (use_sim_time="false").
    backend = os.environ.get("STRAFER_NAV_BACKEND", "nav2")
    model_path = os.environ.get("STRAFER_INFERENCE_MODEL_PATH", "")
    policy_variant = os.environ.get("STRAFER_POLICY_VARIANT", "DEPTH")

    vlm_url = LaunchConfiguration("vlm_url")
    planner_url = LaunchConfiguration("planner_url")

    # Forwarded navigation args
    front_port = LaunchConfiguration("front_port")
    rear_port = LaunchConfiguration("rear_port")
    localization = LaunchConfiguration("localization")
    database_path = LaunchConfiguration("database_path")
    rtabmap_args = LaunchConfiguration("rtabmap_args")
    rtabmap_viz = LaunchConfiguration("rtabmap_viz")
    nav_log_level = LaunchConfiguration("nav_log_level")

    entities = [
        # ── DGX service URLs ───────────────────────────────────────────
        DeclareLaunchArgument(
            "vlm_url", default_value=default_vlm,
            description="VLM grounding service URL (e.g. http://192.168.50.196:8100).",
        ),
        DeclareLaunchArgument(
            "planner_url", default_value=default_planner,
            description="LLM planner service URL (e.g. http://192.168.50.196:8200).",
        ),
        DeclareLaunchArgument(
            "nav_backend", default_value=backend,
            description=(
                "navigate_to_pose backend: nav2 (default; launches NEITHER "
                "inference node), strafer_direct (+ strafer_inference), "
                "hybrid_nav2_strafer (+ strafer_inference + "
                "strafer_subgoal_generator). Must match STRAFER_NAV_BACKEND "
                "on the autonomy executor."
            ),
        ),

        # ── Forwarded navigation arguments ─────────────────────────────
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

        # ── Navigation (base + perception + SLAM + Nav2) ──────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_dir, "launch", "navigation.launch.py")
            ),
            launch_arguments={
                "front_port": front_port,
                "rear_port": rear_port,
                "localization": localization,
                "database_path": database_path,
                "rtabmap_args": rtabmap_args,
                "rtabmap_viz": rtabmap_viz,
                "nav_log_level": nav_log_level,
            }.items(),
        ),
    ]

    # ── RL inference backend (gated on STRAFER_NAV_BACKEND) ────────────
    # nav2/unset launches neither node; strafer_direct adds the inference
    # node; hybrid adds the subgoal generator too. Same env var the
    # executor dispatch reads, so server and client stay in lockstep.
    if backend in _POLICY_BACKENDS:
        if not model_path:
            logging.getLogger("launch").error(
                f"STRAFER_NAV_BACKEND={backend} but "
                "STRAFER_INFERENCE_MODEL_PATH is empty — strafer_inference "
                "will NOT advertise navigate_to_pose; every mission will "
                "silently fall back to nav2. Set STRAFER_INFERENCE_MODEL_PATH."
            )
        entities.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(inference_dir, "launch", "inference.launch.py")
            ),
            launch_arguments={
                "model_path": model_path,
                "policy_variant": policy_variant,
                "use_sim_time": "false",
            }.items(),
        ))
    if backend == "hybrid_nav2_strafer":
        entities.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(inference_dir, "launch", "subgoal_generator.launch.py")
            ),
            launch_arguments={"use_sim_time": "false"}.items(),
        ))

    entities += [
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
            },
        ),
    ]
    return LaunchDescription(entities)
