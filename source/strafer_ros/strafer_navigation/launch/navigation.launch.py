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
                  envelope_factor, resolution, depth_min, depth_max,
                  *, smoothing_bt_xml_path=None):
    """Inject constants and lane-specific overrides into the nav2_params dict.

    The body is split into the three buckets of the sim→real promotion
    contract (``docs/tasks/context/nav2-knob-promotion.md``):

      1. Universal defaults — physical constants and promoted behavioral
         defaults. Applied on every lane, identical at every
         ``envelope_factor``.
      2. Velocity-envelope coupling — MPPI sampling window and prune
         distance that MUST track the lifted velocity cap. Gated by
         construction, so they never graduate.
      3. Behavioral overrides under promotion — sim-only MPPI tuning gated
         on ``envelope_factor > 1.0``, each carrying the real-robot lap
         that has to land before it graduates to a universal default.

    ``smoothing_bt_xml_path``, when provided, wires the custom
    navigate-to-pose BT into ``bt_navigator`` (a universal default).
    """
    ctrl = params["controller_server"]["ros__parameters"]["FollowPath"]

    # ══ 1. Universal defaults — every lane, envelope_factor-independent ═════
    # MPPI velocity caps, resolved from NAV_*_VEL (× STRAFER_NAV_VEL_SCALE).
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

    # Smoothing / event-driven-replan BT: a universal behavioral default
    # (sim and real both load it). Injected here rather than pinned in YAML
    # because the path is resolved from ament_index at launch time.
    if smoothing_bt_xml_path:
        params["bt_navigator"]["ros__parameters"][
            "default_nav_to_pose_bt_xml"
        ] = smoothing_bt_xml_path

    # ══ 2. Velocity-envelope coupling — gated by construction ══════════════
    # MPPI samples per-step control noise from N(0, *_std) around the current
    # command. YAML stds fit the real-robot envelope (NAV_VEL_SCALE=0.5);
    # under STRAFER_NAV_VEL_SCALE=1.0 the cap doubles, so a fixed std leaves
    # exploration narrower than the cap and MPPI plateaus mid-envelope. Scale
    # linearly — super-linear (^1.5) tipped exploration wide enough that MPPI
    # preferred strafe/spin over forward progress. No-op at envelope_factor=1.0.
    for key in ("vx_std", "vy_std", "wz_std", "prune_distance"):
        if key in ctrl:
            ctrl[key] = round(float(ctrl[key]) * envelope_factor, 4)

    if envelope_factor > 1.0 and "vy_std" in ctrl:
        # Un-scale vy_std back to baseline: the lifted envelope is for forward
        # + rotation, not lateral. A wider vy_std hands MPPI more strafe
        # rollouts to weight against a noisy global path, letting the cost
        # minimum drift sideways off the planned forward line. vx/wz stay
        # scaled. This is the lateral half of the coupling — gated, but by
        # construction, not a promotion candidate.
        ctrl["vy_std"] = round(float(ctrl["vy_std"]) / envelope_factor, 4)

    # ══ 3. Behavioral overrides under promotion — envelope_factor > 1.0 ════
    # Sim-only MPPI critic + convergence tuning. Each is a candidate to
    # graduate to a universal default once a real-robot validation lap
    # confirms it holds at the indoor cap; until then it stays gated and
    # carries its promotion criterion inline.
    #
    # iteration_count > 1 is CPU-bound on the Jetson Orin Nano at
    # controller_frequency=20 Hz and makes controller_server miss its 50 ms
    # deadline — tune the cost landscape here, not the iteration depth.
    if envelope_factor > 1.0:
        if "PreferForwardCritic" in ctrl:
            # Graduates once a real-robot `translate forward 1 m` + cornering
            # smoke at the indoor cap shows no forward over-bias — no stall,
            # no reverse-along-path. Omni samples vy/reverse rollouts
            # diff-drive doesn't, so too low a weight drives tangent to or
            # backward along the path.
            ctrl["PreferForwardCritic"]["cost_weight"] = 10.0
        if "PathFollowCritic" in ctrl:
            # offset_from_furthest ~25 cm → ~1 m ahead at MAP_RESOLUTION=0.05,
            # so high-speed rollouts win on cost. Graduates once cornering +
            # a long-corridor lap at the ~0.78 m/s real cap (~1.3 s preview)
            # track the path without cutting corners.
            ctrl["PathFollowCritic"]["offset_from_furthest"] = 20
        # gamma is the sampled-vs-previous smoothing/lag trade-off; lower lets
        # the commanded mean reach the high-vx optimum instead of filtering
        # toward the prior step. Graduates once cornering + a `rotate 90°`
        # smoke at the indoor cap show no overshoot where chassis inertia
        # dominates command-following.
        ctrl["gamma"] = 0.008
        if "PathAlignCritic" in ctrl:
            # Tighter path tracking for the 2× envelope. Graduates only if at
            # the indoor cap MPPI tolerates small path-deviation noise without
            # spending lateral vy to chase it; otherwise stays gated with that
            # as its refreshed justification.
            ctrl["PathAlignCritic"]["cost_weight"] = 9.0


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

    smoothing_bt_xml_path = os.path.join(
        pkg_dir, "config", "navigate_to_pose_w_smoothing_and_recovery.xml"
    )

    _patch_params(params, footprint, nav_linear_vel, nav_angular_vel,
                  nav_reverse_vel, envelope_factor,
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
