"""Launch robot_state_publisher with the Strafer URDF.

Publishes TF transforms for:
  - base_link → chassis_link              (static)
  - base_link → wheel_{1..4}_link         (from /strafer/joint_states)
  - base_link → d555_link                 (static)
  - d555_link → d555_color_optical_frame  (static)

The odom → base_link transform comes from the driver node (not this package).

strafer_description.xacro_args sets every xacro argument from
strafer_shared.constants, so constants.py remains the single source of truth.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os

from strafer_description.xacro_args import robot_description


def generate_launch_description():
    pkg_dir = get_package_share_directory("strafer_description")
    xacro_path = os.path.join(pkg_dir, "urdf", "strafer.urdf.xacro")

    use_sim_time = LaunchConfiguration("use_sim_time")
    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time", default_value="false",
            description="Set to true when the /clock publisher is upstream (sim-in-the-loop).",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description(xacro_path),
                "use_sim_time": PythonExpression(["'", use_sim_time, "' == 'true'"]),
            }],
            remappings=[
                ("joint_states", "/strafer/joint_states"),
            ],
        ),
    ])
