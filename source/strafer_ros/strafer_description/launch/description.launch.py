"""Launch robot_state_publisher with the Strafer URDF.

Publishes TF transforms for:
  - base_link → chassis_link       (static)
  - base_link → wheel_{1..4}_link  (from /strafer/joint_states)
  - base_link → d555_link          (static)

The odom → base_link transform comes from the driver node (not this package).

All numeric values are injected from strafer_shared.constants into the xacro
via mappings, so constants.py remains the single source of truth.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os
import xacro

from strafer_shared.constants import (
    WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_BASE, TRACK_WIDTH,
    CHASSIS_LENGTH, CHASSIS_WIDTH, CHASSIS_HEIGHT, CHASSIS_GROUND_CLEARANCE,
    CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z,
    CAMERA_LENGTH, CAMERA_WIDTH, CAMERA_HEIGHT,
    MASS_WHEEL_ASSEMBLY, MASS_CHASSIS, MASS_CAMERA,
)


def generate_launch_description():
    pkg_dir = get_package_share_directory("strafer_description")
    xacro_path = os.path.join(pkg_dir, "urdf", "strafer.urdf.xacro")

    # Pass ALL constants into xacro -- single source of truth
    mappings = {
        "wheel_radius": str(WHEEL_RADIUS),
        "wheel_width": str(WHEEL_WIDTH),
        "wheel_base": str(WHEEL_BASE),
        "track_width": str(TRACK_WIDTH),
        "chassis_length": str(CHASSIS_LENGTH),
        "chassis_width": str(CHASSIS_WIDTH),
        "chassis_height": str(CHASSIS_HEIGHT),
        "chassis_ground_clearance": str(CHASSIS_GROUND_CLEARANCE),
        "camera_x": str(CAMERA_OFFSET_X),
        "camera_y": str(CAMERA_OFFSET_Y),
        "camera_z": str(CAMERA_OFFSET_Z),
        "camera_length": str(CAMERA_LENGTH),
        "camera_width": str(CAMERA_WIDTH),
        "camera_height": str(CAMERA_HEIGHT),
        "mass_wheel_assembly": str(MASS_WHEEL_ASSEMBLY),
        "mass_chassis": str(MASS_CHASSIS),
        "mass_camera": str(MASS_CAMERA),
    }

    robot_description = xacro.process_file(xacro_path, mappings=mappings).toxml()

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
            }],
            remappings=[
                ("joint_states", "/strafer/joint_states"),
            ],
        ),
    ])
