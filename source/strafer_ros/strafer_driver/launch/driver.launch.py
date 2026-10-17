"""Launch file for the strafer_driver roboclaw_node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def end_launch_on_exit(event, context):
    """End the launch when the driver exits, and non-zero after a non-zero exit.

    A fence refusal exits 1. Launch returns non-zero only for an exception, so a
    bare Shutdown would end the launch with 0; compose restarts it either way.
    An exit during the launch's own shutdown (Ctrl-C) is not a refusal.
    """
    if event.returncode and not context.is_shutdown:
        raise RuntimeError(f"roboclaw_node exited {event.returncode}; ending the launch")
    return Shutdown()


def generate_launch_description():
    pkg_share = get_package_share_directory("strafer_driver")
    default_params = os.path.join(pkg_share, "config", "driver_params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Path to the driver parameters YAML file",
            ),
            DeclareLaunchArgument(
                "front_port",
                default_value="/dev/roboclaw0",
                description="Front RoboClaw (0x80) port; unused when auto_detect is true",
            ),
            DeclareLaunchArgument(
                "rear_port",
                default_value="/dev/roboclaw1",
                description="Rear RoboClaw (0x81) port; unused when auto_detect is true",
            ),
            Node(
                package="strafer_driver",
                executable="roboclaw_node",
                name="roboclaw_node",
                namespace="",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "front_port": LaunchConfiguration("front_port"),
                        "rear_port": LaunchConfiguration("rear_port"),
                    },
                ],
                remappings=[
                    ("/strafer/cmd_vel", "/cmd_vel"),
                ],
                # A refusal ends the launch, so compose's restart re-runs detection.
                on_exit=end_launch_on_exit,
            ),
        ]
    )
