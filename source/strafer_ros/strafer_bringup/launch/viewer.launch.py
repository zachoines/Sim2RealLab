"""Standalone Foxglove viewer for the Strafer stack.

Spawns a single ``foxglove_bridge`` WebSocket server (default ``:8765``) so an
operator can inspect the live ROS 2 graph — TF, costmaps, depth, ``cmd_vel``,
subgoals, the policy's ``navigate_to_pose`` feedback — from Foxglove Studio.

It only *attaches* to the domain-42 graph and launches no robot nodes of its
own, so it is agnostic to which stack is up: use it for the full real-robot
deploy OR the DGX sim-bridge e2e. This is deliberately separate from
``bringup_sim_in_the_loop.launch.py`` (which spawns its own foxglove_bridge as
part of the sim-in-the-loop container) so the containerized deploy lane, which
composes nodes as independent services, gets a clean opt-in viewer service
instead of an implicit one.

Bare-metal:
    ros2 launch strafer_bringup viewer.launch.py
    ros2 launch strafer_bringup viewer.launch.py viewer_port:=8765 use_sim_time:=true

Containerized (deploy lane):
    docker compose --profile viewer up viewer

Connect Foxglove Studio to ``ws://<robot-ip>:8765`` (the bridge binds 0.0.0.0),
or tunnel it: ``ssh -L 8765:localhost:8765 <user>@<robot-ip>``.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _spawn_foxglove(context, *args, **kwargs):
    # Resolve here so we can hand foxglove_bridge a correctly-typed int port and
    # bool use_sim_time (its params are typed; passing raw substitutions fails).
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
                "address": "0.0.0.0",          # bind all interfaces so host-net / LAN can reach it
                "use_sim_time": use_sim_time,
            }],
        ),
    ]


def generate_launch_description():
    # Env defaults let the deploy compose service (or the sim-bridge overlay)
    # parameterize the viewer without overriding the launch command.
    default_port = os.environ.get("STRAFER_VIEWER_PORT", "8765")
    default_sim_time = os.environ.get("STRAFER_USE_SIM_TIME", "false")

    return LaunchDescription([
        DeclareLaunchArgument(
            "viewer_port", default_value=default_port,
            description="Foxglove WebSocket TCP port (bound on 0.0.0.0).",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value=default_sim_time,
            description=(
                "Use the /clock topic for the node's own time. true for the "
                "sim-bridge lane (Isaac drives /clock); false on the real robot."
            ),
        ),
        OpaqueFunction(function=_spawn_foxglove),
    ])
