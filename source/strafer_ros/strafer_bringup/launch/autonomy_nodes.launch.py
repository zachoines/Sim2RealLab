"""Decomposed autonomy layer: goal-projection service + autonomy executor.

This is the per-container counterpart to the top layer of ``autonomy.launch.py``.
In the containerized deployment Nav2, SLAM, perception, the driver, and the RL
inference node each run in their own container and are discovered over DDS, so
this launch brings up only the two nodes that are neither hardware nor a
navigation backend:

  - ``goal_projection`` (``strafer_perception``) — VLM bbox -> map-frame goal pose;
  - ``strafer-executor`` (``strafer_autonomy``) — the autonomy command server that
    dispatches missions against the remote VLM/planner services.

``autonomy.launch.py`` also brings up the whole nav stack in one process; this
launch does not (each layer is its own container). ``VLM_URL`` / ``PLANNER_URL``
come from the environment (compose ``autonomy.env``); ``STRAFER_USE_SIM_TIME``
pins sim time on BOTH nodes when a ``/clock`` publisher is upstream.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_vlm = os.environ.get("VLM_URL", "")
    default_planner = os.environ.get("PLANNER_URL", "")
    use_sim_time = os.environ.get("STRAFER_USE_SIM_TIME", "false").strip().lower() == "true"
    use_sim_time_str = "true" if use_sim_time else "false"

    return LaunchDescription([
        DeclareLaunchArgument(
            "vlm_url", default_value=default_vlm,
            description="VLM grounding service URL (e.g. http://192.168.50.196:8100).",
        ),
        DeclareLaunchArgument(
            "planner_url", default_value=default_planner,
            description="LLM planner service URL (e.g. http://192.168.50.196:8200).",
        ),

        # VLM bbox -> map-frame goal pose service (strafer_perception node)
        Node(
            package="strafer_perception",
            executable="goal_projection",
            name="goal_projection",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
        ),

        # Autonomy command server (connects to the remote VLM/planner services).
        # use_sim_time is passed through --ros-args (the executor is a plain
        # process, not a launch_ros Node), matching bringup_sim_in_the_loop.
        ExecuteProcess(
            cmd=["strafer-executor", "--ros-args", "-p", f"use_sim_time:={use_sim_time_str}"],
            name="strafer_executor",
            output="screen",
            additional_env={
                "VLM_URL": LaunchConfiguration("vlm_url"),
                "PLANNER_URL": LaunchConfiguration("planner_url"),
            },
        ),
    ])
