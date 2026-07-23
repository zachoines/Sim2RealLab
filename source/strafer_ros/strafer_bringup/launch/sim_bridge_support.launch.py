"""Sim-bridge support layer for the containerized deploy lane.

The deploy compose splits the stack into per-driver services. In DGX sim-bridge
mode the hardware drivers (RoboClaw, D555) don't run, so `base` and `perception`
are dropped — but each ALSO carries a support node the sim-consuming pipeline
still needs, and dropping the service drops the support node with it:

  - robot_state_publisher (strafer_description)  -> base_link->d555_link TF
    (normally in the `base` service, alongside the RoboClaw driver)
  - timestamp_fixer (strafer_perception)         -> the /d555/*_sync topics that
    RTAB-Map consumes (normally in `perception`, alongside the D555 driver)

This launch wires exactly those, mirroring bringup_sim_in_the_loop.launch.py, plus
an optional one-shot donut_warmup spin to seed RTAB-Map at the spawn pose. With it,
the containerized `slam` + `navigation` + `inference` services become
self-sufficient against the bridge.

Opt-in and SIM-ONLY (compose `sim-bridge` profile). NEVER run on the real robot —
there `base`/`perception` provide these natively and the remaps below are wrong.

    docker compose --profile sim-bridge up sim-perception
    ros2 launch strafer_bringup sim_bridge_support.launch.py            # bare-metal
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
    # bool form for node params (use_sim_time as a LaunchConfiguration is a string)
    sim_bool = PythonExpression(["'", use_sim_time, "' == 'true'"])

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="Sim-bridge publishes /clock — keep true. Only meaningful in sim.",
        ),
        DeclareLaunchArgument(
            "donut_warmup", default_value="true",
            description=(
                "Run a one-shot 360 spin at bringup on /cmd_vel so RTAB-Map seeds "
                "its map at the spawn pose before the first nav goal. Wait for it "
                "to finish before injecting a goal (they share /cmd_vel)."
            ),
        ),

        # ── base_link -> d555_link / wheel TF (normally from the `base` service) ──
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(description_dir, "launch", "description.launch.py")
            ),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
        ),

        # ── /d555/*_sync for RTAB-Map (normally from the `perception` service) ────
        # restamp=False: the bridge already stamps camera frames on /clock (sim
        # time); restamping would rewrite each message with the current sim time at
        # relay, which advances between the depth and camera_info callbacks and
        # breaks the exact synchronizer. In sim the D555 is one co-registered RGB+D
        # sensor, so the bridge's raw /d555/depth/* IS the aligned depth — the
        # remaps hand it in under the aligned_depth_to_color/* contract rtabmap
        # subscribes to.
        Node(
            package="strafer_perception",
            executable="timestamp_fixer",
            name="timestamp_fixer",
            output="screen",
            parameters=[{"use_sim_time": sim_bool, "restamp": False}],
            remappings=[
                ("/d555/aligned_depth_to_color/image_raw",
                 "/d555/depth/image_rect_raw"),
                ("/d555/aligned_depth_to_color/camera_info",
                 "/d555/depth/camera_info"),
            ],
        ),

        # ── One-shot RTAB-Map warmup spin (shares /cmd_vel; exits after 360°) ─────
        Node(
            package="strafer_bringup",
            executable="donut_warmup",
            name="donut_warmup",
            output="screen",
            condition=IfCondition(LaunchConfiguration("donut_warmup")),
            parameters=[{"use_sim_time": sim_bool}],
        ),
    ])
