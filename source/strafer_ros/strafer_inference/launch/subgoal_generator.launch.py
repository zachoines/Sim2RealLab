"""Launch the rolling-subgoal generator node with config/subgoal_generator.yaml.

The node runs in the global namespace so its absolute ``/plan`` and
``/strafer/subgoal`` topics and the ``map -> base_link`` TF resolve without
remapping; the inference node's namespacing (to isolate its action server)
does not apply here.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_dir = get_package_share_directory("strafer_inference")
    default_config = os.path.join(pkg_dir, "config", "subgoal_generator.yaml")

    config_arg = DeclareLaunchArgument(
        "config_file", default_value=default_config,
        description="Path to the subgoal-generator parameter YAML.",
    )
    log_level_arg = DeclareLaunchArgument(
        "log_level", default_value="info",
        description="Log level for the subgoal-generator node.",
    )

    node = Node(
        package="strafer_inference",
        executable="subgoal_generator_node",
        name="strafer_subgoal_generator",
        output="screen",
        parameters=[LaunchConfiguration("config_file")],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    return LaunchDescription([config_arg, log_level_arg, node])
