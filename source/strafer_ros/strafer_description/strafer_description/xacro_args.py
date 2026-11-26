"""The Strafer URDF's xacro arguments, from strafer_shared.constants."""

import xacro

from strafer_shared.constants import (
    WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_BASE, TRACK_WIDTH,
    CHASSIS_LENGTH, CHASSIS_WIDTH, CHASSIS_HEIGHT, CHASSIS_GROUND_CLEARANCE,
    CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z,
    CAMERA_LENGTH, CAMERA_WIDTH, CAMERA_HEIGHT,
    MASS_WHEEL_ASSEMBLY, MASS_CHASSIS, MASS_CAMERA,
)


def xacro_mappings() -> dict[str, str]:
    """Return a fresh dict with one string per xacro:arg in strafer.urdf.xacro."""
    return {
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


def robot_description(xacro_path: str) -> str:
    """Return the URDF XML of the xacro at xacro_path, processed with xacro_mappings()."""
    return xacro.process_file(xacro_path, mappings=xacro_mappings()).toxml()
