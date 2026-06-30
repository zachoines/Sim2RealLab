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
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_dir = get_package_share_directory("strafer_inference")
    default_config = os.path.join(pkg_dir, "config", "subgoal_generator.yaml")

    default_use_sim = os.environ.get("STRAFER_USE_SIM_TIME", "false")

    config_arg = DeclareLaunchArgument(
        "config_file", default_value=default_config,
        description="Path to the subgoal-generator parameter YAML.",
    )
    log_level_arg = DeclareLaunchArgument(
        "log_level", default_value="info",
        description="Log level for the subgoal-generator node.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value=default_use_sim,
        description="true when a /clock publisher is upstream (sim-in-the-loop).",
    )

    node = Node(
        package="strafer_inference",
        executable="subgoal_generator_node",
        name="strafer_subgoal_generator",
        output="screen",
        parameters=[
            LaunchConfiguration("config_file"),
            {
                # use_sim_time is a bool param; a non-empty string is
                # truthy-by-presence, so coerce "false" to a real False.
                "use_sim_time": PythonExpression(
                    ["'", LaunchConfiguration("use_sim_time"), "' == 'true'"]
                ),
            },
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    return LaunchDescription([config_arg, log_level_arg, use_sim_time_arg, node])
