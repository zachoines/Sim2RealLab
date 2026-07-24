"""Standalone `foxglove_bridge` for inspecting the running graph.

Launches no robot nodes, so it attaches to whichever stack is already up.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _spawn_foxglove(context, *args, **kwargs):
    # foxglove_bridge's port and use_sim_time are typed; a raw substitution fails.
    port = int(LaunchConfiguration("viewer_port").perform(context))
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).strip().lower() in (
        "1", "true", "yes", "on",
    )
    return [
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            parameters=[{
                "port": port,
                "address": "0.0.0.0",
                "use_sim_time": use_sim_time,
            }],
        ),
    ]


def generate_launch_description():
    default_port = os.environ.get("STRAFER_VIEWER_PORT", "8765")
    default_sim_time = os.environ.get("STRAFER_USE_SIM_TIME", "false")

    return LaunchDescription([
        DeclareLaunchArgument(
            "viewer_port", default_value=default_port,
            description="Foxglove WebSocket TCP port, bound on 0.0.0.0.",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value=default_sim_time,
            description="Use the /clock topic for the node's own time.",
        ),
        OpaqueFunction(function=_spawn_foxglove),
    ])
