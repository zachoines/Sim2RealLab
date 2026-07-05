"""Launch Nav2 navigation stack for the Strafer robot.

Requires (launched separately or via strafer_bringup):
  - strafer_driver:      /strafer/odom, /strafer/cmd_vel
  - strafer_perception:  /d555/* topics
  - strafer_description: robot_state_publisher (TF tree)
  - strafer_slam:        /rtabmap/map, map→odom TF, /scan

This launch file:
  1. Loads nav2_params.yaml
  2. Overrides robot-specific values from strafer_shared.constants
  3. Writes a patched YAML to /tmp and passes it to Nav2's navigation_launch.py

The Nav2 configuration is byte-identical on every lane by construction:
sim and real bringup both load nav2_params.yaml and receive the same
constants-derived velocity caps, footprint, resolution, and scan ranges.
Velocity caps come from NAV_*_VEL (indoor-safety fraction of the chassis
maximum) — one number for both lanes.

Usage:
  ros2 launch strafer_navigation navigation.launch.py
"""

import logging
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from strafer_shared.constants import (
    CHASSIS_LENGTH,
    DEPTH_MAX,
    DEPTH_MIN,
    MAP_RESOLUTION,
    NAV_ANGULAR_VEL,
    NAV_LINEAR_VEL,
    NAV_REVERSE_VEL,
    TRACK_WIDTH,
)


_logger = logging.getLogger(__name__)


def _build_footprint(length, width):
    """Build a Nav2 footprint string from chassis dimensions.

    Returns a rectangular polygon as ``[[x,y], ...]`` with the rectangle
    centred on base_link.  Half-length along X, half-width along Y.
    """
    hx = round(length / 2, 4)
    hy = round(width / 2, 4)
    return f"[[{hx}, {hy}], [{hx}, {-hy}], [{-hx}, {-hy}], [{-hx}, {hy}]]"


def _patch_params(params, footprint, nav_vel, nav_omega, nav_reverse,
                  resolution, depth_min, depth_max,
                  *, smoothing_bt_xml_path=None):
    """Inject robot constants into the nav2_params dict.

    Physical constants — velocity caps, costmap resolution, footprint,
    scan ranges — and the launch-resolved smoothing-BT path are the only
    values layered on top of nav2_params.yaml. Everything else, including
    the MPPI critic tuning, is a YAML default shared by every lane.

    ``smoothing_bt_xml_path``, when provided, wires the custom
    navigate-to-pose BT into ``bt_navigator`` (its path resolves from
    ament_index at launch time, so it can't be pinned in YAML).
    """
    ctrl = params["controller_server"]["ros__parameters"]["FollowPath"]

    # MPPI velocity caps, from NAV_*_VEL.
    ctrl["vx_max"] = nav_vel
    ctrl["vx_min"] = -nav_reverse
    ctrl["vy_max"] = nav_vel
    ctrl["wz_max"] = nav_omega

    # Behavior server.
    beh = params["behavior_server"]["ros__parameters"]
    beh["max_rotational_vel"] = nav_omega

    # Velocity smoother.
    vs = params["velocity_smoother"]["ros__parameters"]
    vs["max_velocity"] = [nav_vel, nav_vel, nav_omega]
    vs["min_velocity"] = [-nav_reverse, -nav_vel, -nav_omega]

    # Local costmap.
    lc = params["local_costmap"]["local_costmap"]["ros__parameters"]
    lc["resolution"] = resolution
    lc["footprint"] = footprint
    scan_lc = lc["obstacle_layer"]["scan"]
    scan_lc["raytrace_max_range"] = depth_max
    scan_lc["raytrace_min_range"] = depth_min
    scan_lc["obstacle_min_range"] = depth_min

    # Global costmap.
    gc = params["global_costmap"]["global_costmap"]["ros__parameters"]
    gc["resolution"] = resolution
    gc["footprint"] = footprint
    scan_gc = gc["obstacle_layer"]["scan"]
    scan_gc["raytrace_max_range"] = depth_max
    scan_gc["raytrace_min_range"] = depth_min
    scan_gc["obstacle_min_range"] = depth_min

    if smoothing_bt_xml_path:
        params["bt_navigator"]["ros__parameters"][
            "default_nav_to_pose_bt_xml"
        ] = smoothing_bt_xml_path


def _launch_setup(context, *args, **kwargs):
    """Resolve launch arguments and build the Nav2 include."""
    pkg_dir = get_package_share_directory("strafer_navigation")
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")

    log_level = LaunchConfiguration("log_level").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)

    # ── Load YAML ───────────────────────────────────────────────────────
    yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
    with open(yaml_path) as f:
        params = yaml.safe_load(f)

    # ── Derived values from constants ────────────────────────────────────
    footprint = _build_footprint(CHASSIS_LENGTH, TRACK_WIDTH)

    _logger.info(
        "Nav2 velocity caps (universal): linear=%.4f m/s, angular=%.4f rad/s, "
        "reverse=%.4f m/s",
        NAV_LINEAR_VEL, NAV_ANGULAR_VEL, NAV_REVERSE_VEL,
    )

    smoothing_bt_xml_path = os.path.join(
        pkg_dir, "config", "navigate_to_pose_w_smoothing_and_recovery.xml"
    )

    _patch_params(params, footprint, NAV_LINEAR_VEL, NAV_ANGULAR_VEL,
                  NAV_REVERSE_VEL,
                  MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX,
                  smoothing_bt_xml_path=smoothing_bt_xml_path)

    # ── Write patched YAML to temp file ─────────────────────────────────
    # Nav2's navigation_launch.py expects a file path.
    fd, patched_path = tempfile.mkstemp(
        prefix="strafer_nav2_", suffix=".yaml"
    )
    with os.fdopen(fd, "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    # ── Include Nav2 navigation stack ───────────────────────────────────
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, "launch", "navigation_launch.py")
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "params_file": patched_path,
                "autostart": "true",
                "use_composition": "False",
                "log_level": log_level,
            }.items(),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "log_level", default_value="warn",
            description="Log level for Nav2 nodes.",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="false",
            description="Set true when /clock is published upstream (sim-in-the-loop).",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
