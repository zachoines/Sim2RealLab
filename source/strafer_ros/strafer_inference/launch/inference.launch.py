"""Launch the strafer_inference node with config/inference.yaml.

The node lives in the ``strafer_inference`` namespace so its
``navigate_to_pose`` action server does not collide with Nav2's
when both backends are running side-by-side.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_dir = get_package_share_directory("strafer_inference")
    default_config = os.path.join(pkg_dir, "config", "inference.yaml")

    config_arg = DeclareLaunchArgument(
        "config_file", default_value=default_config,
        description="Path to the strafer_inference parameter YAML.",
    )
    log_level_arg = DeclareLaunchArgument(
        "log_level", default_value="info",
        description="Log level for the strafer_inference node.",
    )

    node = Node(
        package="strafer_inference",
        executable="inference_node",
        name="strafer_inference",
        namespace="strafer_inference",
        output="screen",
        parameters=[LaunchConfiguration("config_file")],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    return LaunchDescription([config_arg, log_level_arg, node])
