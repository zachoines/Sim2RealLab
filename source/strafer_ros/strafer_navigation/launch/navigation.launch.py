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

Environment variables:
  STRAFER_NAV_VEL_SCALE : Override the constants.NAV_VEL_SCALE fraction
                         (default 0.5) used to derive Nav2's linear /
                         angular velocity caps. Set to 1.0 in
                         env_sim_in_the_loop.env to let MPPI ask for
                         full chassis dynamics in sim; leave unset on
                         real-robot bringup so the indoor safety cap
                         stays in force. Reverse cap stays scaled by
                         NAV_REVERSE_SCALE off the resolved forward cap.

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
    MAX_ANGULAR_VEL,
    MAX_LINEAR_VEL,
    NAV_ANGULAR_VEL,
    NAV_LINEAR_VEL,
    NAV_REVERSE_SCALE,
    NAV_REVERSE_VEL,
    NAV_VEL_SCALE,
    TRACK_WIDTH,
)


_logger = logging.getLogger(__name__)


def _resolved_nav_velocities():
    """Resolve Nav2 velocity caps, honoring ``STRAFER_NAV_VEL_SCALE``.

    Real-robot bringup leaves the env var unset and gets the
    constants-derived defaults (NAV_VEL_SCALE = 0.5, ~0.78 m/s linear cap)
    intended as an indoor safety bound. Sim-in-the-loop bringup sources
    ``env_sim_in_the_loop.env`` which exports STRAFER_NAV_VEL_SCALE=1.0,
    letting Nav2 in sim ask for hardware-max velocities — matching the
    envelope the trained policy and joystick teleop see, so MPPI is not
    artificially capped at half the chassis dynamics.

    Returns ``(linear, angular, reverse, envelope_factor)``. The factor
    is ``resolved_scale / NAV_VEL_SCALE`` — 1.0 on real-robot bringup,
    2.0 in sim — used by ``_patch_params`` to scale MPPI sampling noise
    and path-prune distance proportionally so the larger velocity cap is
    actually exploitable.
    """
    raw = os.environ.get("STRAFER_NAV_VEL_SCALE")
    if not raw:
        return NAV_LINEAR_VEL, NAV_ANGULAR_VEL, NAV_REVERSE_VEL, 1.0
    try:
        scale = float(raw)
    except ValueError:
        _logger.warning(
            "Ignoring non-numeric STRAFER_NAV_VEL_SCALE=%r; using default %.4f",
            raw, NAV_VEL_SCALE,
        )
        return NAV_LINEAR_VEL, NAV_ANGULAR_VEL, NAV_REVERSE_VEL, 1.0
    if scale <= 0.0:
        _logger.warning(
            "Ignoring non-positive STRAFER_NAV_VEL_SCALE=%s; using default %.4f",
            scale, NAV_VEL_SCALE,
        )
        return NAV_LINEAR_VEL, NAV_ANGULAR_VEL, NAV_REVERSE_VEL, 1.0
    linear = round(MAX_LINEAR_VEL * scale, 4)
    angular = round(MAX_ANGULAR_VEL * scale, 4)
    reverse = round(linear * NAV_REVERSE_SCALE, 4)
    envelope_factor = round(scale / NAV_VEL_SCALE, 4)
    _logger.info(
        "STRAFER_NAV_VEL_SCALE=%s overrides NAV_VEL_SCALE=%.4f "
        "(linear=%.4f m/s, angular=%.4f rad/s, reverse=%.4f m/s, "
        "MPPI envelope_factor=%.4f)",
        raw, NAV_VEL_SCALE, linear, angular, reverse, envelope_factor,
    )
    return linear, angular, reverse, envelope_factor


def _build_footprint(length, width):
    """Build a Nav2 footprint string from chassis dimensions.

    Returns a rectangular polygon as ``[[x,y], ...]`` with the rectangle
    centred on base_link.  Half-length along X, half-width along Y.
    """
    hx = round(length / 2, 4)
    hy = round(width / 2, 4)
    return f"[[{hx}, {hy}], [{hx}, {-hy}], [{-hx}, {-hy}], [{-hx}, {hy}]]"


def _patch_params(params, footprint, nav_vel, nav_omega, nav_reverse,
                  envelope_factor, resolution, depth_min, depth_max):
    """Inject constants into the loaded nav2_params dict."""

    # ── Controller (MPPI) ───────────────────────────────────────────────
    ctrl = params["controller_server"]["ros__parameters"]["FollowPath"]
    ctrl["vx_max"] = nav_vel
    ctrl["vx_min"] = -nav_reverse
    ctrl["vy_max"] = nav_vel
    ctrl["wz_max"] = nav_omega
    # MPPI samples per-step control noise from ``N(0, *_std)`` around the
    # current command. The YAML defaults (vx/vy_std=0.2, wz_std=0.4) are
    # tuned for the real-robot envelope (NAV_VEL_SCALE=0.5). When sim
    # raises STRAFER_NAV_VEL_SCALE=1.0 the cap doubles, but a fixed std
    # leaves the explored window narrower than the new cap and MPPI
    # plateaus mid-envelope (peak /cmd_vel.vx ≈ 0.55 m/s pre-fix against
    # vx_max=1.5683, the sim-velocity-attenuation symptom). Scale stds
    # and the path-prune distance by the envelope factor so MPPI explores
    # higher into the new envelope and sees more of the global path
    # within its rollout horizon. Identity at envelope_factor=1.0 leaves
    # real-robot bringup untouched. Empirically a super-linear (^1.5)
    # std scaling pushed exploration so wide that MPPI started preferring
    # strafe/spin over forward progress, so we keep this linear.
    for key in ("vx_std", "vy_std", "wz_std", "prune_distance"):
        if key in ctrl:
            ctrl[key] = round(float(ctrl[key]) * envelope_factor, 4)

    # ── Behavior server ─────────────────────────────────────────────────
    beh = params["behavior_server"]["ros__parameters"]
    beh["max_rotational_vel"] = nav_omega

    # ── Velocity smoother ───────────────────────────────────────────────
    vs = params["velocity_smoother"]["ros__parameters"]
    vs["max_velocity"] = [nav_vel, nav_vel, nav_omega]
    vs["min_velocity"] = [-nav_reverse, -nav_vel, -nav_omega]

    # ── Local costmap ───────────────────────────────────────────────────
    lc = params["local_costmap"]["local_costmap"]["ros__parameters"]
    lc["resolution"] = resolution
    lc["footprint"] = footprint
    scan_lc = lc["obstacle_layer"]["scan"]
    scan_lc["raytrace_max_range"] = depth_max
    scan_lc["raytrace_min_range"] = depth_min
    scan_lc["obstacle_min_range"] = depth_min

    # ── Global costmap ──────────────────────────────────────────────────
    gc = params["global_costmap"]["global_costmap"]["ros__parameters"]
    gc["resolution"] = resolution
    gc["footprint"] = footprint
    scan_gc = gc["obstacle_layer"]["scan"]
    scan_gc["raytrace_max_range"] = depth_max
    scan_gc["raytrace_min_range"] = depth_min
    scan_gc["obstacle_min_range"] = depth_min


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
    nav_linear_vel, nav_angular_vel, nav_reverse_vel, envelope_factor = (
        _resolved_nav_velocities()
    )

    _patch_params(params, footprint, nav_linear_vel, nav_angular_vel,
                  nav_reverse_vel, envelope_factor,
                  MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX)

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
