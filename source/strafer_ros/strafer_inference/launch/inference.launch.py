"""Launch the strafer_inference node with config/inference.yaml.

The node lives in the ``strafer_inference`` namespace so its
``navigate_to_pose`` action server does not collide with Nav2's
when both backends are running side-by-side.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_dir = get_package_share_directory("strafer_inference")
    default_config = os.path.join(pkg_dir, "config", "inference.yaml")

    default_model = os.environ.get("STRAFER_INFERENCE_MODEL_PATH", "")
    default_variant = os.environ.get("STRAFER_POLICY_VARIANT", "DEPTH")
    default_use_sim = os.environ.get("STRAFER_USE_SIM_TIME", "false")

    config_arg = DeclareLaunchArgument(
        "config_file", default_value=default_config,
        description="Path to the strafer_inference parameter YAML.",
    )
    log_level_arg = DeclareLaunchArgument(
        "log_level", default_value="info",
        description="Log level for the strafer_inference node.",
    )
    model_path_arg = DeclareLaunchArgument(
        "model_path", default_value=default_model,
        description=(
            "Exported policy artifact. Overrides the YAML; empty leaves the "
            "node refusing to advertise navigate_to_pose."
        ),
    )
    policy_variant_arg = DeclareLaunchArgument(
        "policy_variant", default_value=default_variant,
        description="PolicyVariant name (e.g. DEPTH, NOCAM_SUBGOAL). Overrides the YAML.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value=default_use_sim,
        description="true when a /clock publisher is upstream (sim-in-the-loop).",
    )

    node = Node(
        package="strafer_inference",
        executable="inference_node",
        name="strafer_inference",
        namespace="strafer_inference",
        output="screen",
        parameters=[
            # YAML first (topics, timeouts, ONNX providers); the override
            # dict below is applied last-wins for just these three knobs.
            LaunchConfiguration("config_file"),
            {
                "model_path": LaunchConfiguration("model_path"),
                "policy_variant": LaunchConfiguration("policy_variant"),
                # use_sim_time is a bool param; a non-empty string is
                # truthy-by-presence, so coerce "false" to a real False.
                "use_sim_time": PythonExpression(
                    ["'", LaunchConfiguration("use_sim_time"), "' == 'true'"]
                ),
            },
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        remappings=[
            # Consumers (RoboClaw driver, sim bridge) subscribe /cmd_vel;
            # the node's contract name is /strafer/cmd_vel. Same remap as
            # driver.launch.py — without it the policy commands nothing.
            ("/strafer/cmd_vel", "/cmd_vel"),
        ],
    )

    return LaunchDescription([
        config_arg, log_level_arg, model_path_arg,
        policy_variant_arg, use_sim_time_arg, node,
    ])
