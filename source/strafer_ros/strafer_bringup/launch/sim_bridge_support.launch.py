"""Support nodes the sim-bridge lane needs from `base` and `perception`.

Those two services carry hardware drivers that cannot run against simulated
sensors, so the lane does not start them — but two of their nodes are still
required: `robot_state_publisher` for the `base_link` -> `d555_link` TF, and
`timestamp_fixer` for the `/d555/*_sync` topics RTAB-Map subscribes to.
Sim-only; on the real robot those services provide these natively.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    description_dir = get_package_share_directory("strafer_description")

    use_sim_time = LaunchConfiguration("use_sim_time")
    sim_bool = PythonExpression(["'", use_sim_time, "' == 'true'"])

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="Use the /clock topic published by the simulated bridge.",
        ),
        DeclareLaunchArgument(
            "donut_warmup", default_value="true",
            description=(
                "Run a one-shot 360 spin on /cmd_vel at bringup so RTAB-Map seeds "
                "its map before the first nav goal."
            ),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(description_dir, "launch", "description.launch.py")
            ),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
        ),

        Node(
            package="strafer_perception",
            executable="timestamp_fixer",
            name="timestamp_fixer",
            output="screen",
            # Simulated depth is already registered to color, so the driver's
            # aligned topics are fed from the raw depth stream. Stamps already
            # come off /clock; restamping breaks the exact synchronizer.
            parameters=[{"use_sim_time": sim_bool, "restamp": False}],
            remappings=[
                ("/d555/aligned_depth_to_color/image_raw",
                 "/d555/depth/image_rect_raw"),
                ("/d555/aligned_depth_to_color/camera_info",
                 "/d555/depth/camera_info"),
            ],
        ),

        Node(
            package="strafer_bringup",
            executable="donut_warmup",
            name="donut_warmup",
            output="screen",
            condition=IfCondition(LaunchConfiguration("donut_warmup")),
            parameters=[{"use_sim_time": sim_bool}],
        ),
    ])
