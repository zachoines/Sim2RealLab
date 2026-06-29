"""Entry point for the Jetson autonomy executor.

Reads DGX service URLs from environment variables, constructs the HTTP
clients and ROS client, and spins the AutonomyCommandServer node.

Environment variables
---------------------
VLM_URL                : Base URL for the VLM grounding service  (required, e.g. http://192.168.50.196:8100)
PLANNER_URL            : Base URL for the LLM planner service    (required, e.g. http://192.168.50.196:8200)
OBSERVATION_MAX_AGE_S  : Override the freshness cap on cached camera frames (optional, seconds; default 0.5).
                         Raise this when the upstream publisher runs slower than the RealSense D555 — e.g.
                         an Isaac Sim bridge delivering at ~5 Hz needs 1.5-2.0 s of headroom to tolerate
                         jitter.
ROTATE_TIMEOUT_S       : Override the default rotate_in_place timeout (optional, seconds; default 15.0).
                         scan_for_target uses this default for its inter-heading rotations. Raise it when
                         sim real-time factor is sub-unity or the chassis rotates slowly under the
                         configured angular velocity.
STRAFER_CLOCK_STALL_BAIL_WALL_S : Override the wall-clock window after which a non-advancing ``/clock``
                         is treated as a stalled / crashed bridge and the executor's motion waits
                         (``rotate_in_place``, ``_wait_for_future``, ``_wait_for_nav_result``) bail out
                         (optional, seconds; default 15.0). This replaces the old absolute ``2 * timeout``
                         wall cap, which fired mid-motion at sub-unity RTF. The detector measures
                         *sim-time progress*, so a slow-but-live ``/clock`` is tolerated; only a frozen
                         clock trips it. Set to 0 to disable the stall detector entirely (the sim-clock
                         deadline then becomes the sole bound). On real hardware (``use_sim_time=False``)
                         the detector never fires regardless of this value.
STRAFER_NAVIGATION_TIMEOUT_S : Operator's per-mission navigation timeout ceiling (optional, seconds;
                         default 90.0). With ``STRAFER_NAV_PROGRESS_AWARE`` enabled (default), the
                         executor synthesizes per-step budgets from the requested displacement and
                         caps them at this value — short translates / rotations get tight budgets,
                         long traverses still respect this ceiling. With ``STRAFER_NAV_PROGRESS_AWARE=0``
                         this becomes the single deadline applied to every motion step, matching the
                         pre-progress-aware behavior. The executor's nav timeout ticks on the executor
                         node's clock, so under use_sim_time:=true this is a sim-time budget. Raise
                         it on sub-unity-RTF sim scenes where 90 s of sim time can correspond to
                         several minutes of wall time.
STRAFER_NAV_PROGRESS_AWARE : Toggle progress-aware motion timeouts (optional; default 1 / enabled).
                         When 1, per-step budgets for translate / rotate_by_degrees /
                         orient_to_direction / navigate_to_pose are derived from the requested
                         displacement (or robot→goal straight-line distance) divided by NAV_LINEAR_VEL
                         / NAV_ANGULAR_VEL, scaled by ``nav_budget_safety_factor`` (default 2.0)
                         and offset by ``nav_budget_setup_overhead_s`` (default 5.0), capped at
                         ``STRAFER_NAVIGATION_TIMEOUT_S``. Additionally, a stall watchdog on Nav2's
                         ``distance_remaining`` feedback aborts the active goal with
                         ``error_code=navigation_stalled`` if no progress is made for
                         ``nav_stall_window_s`` (default 20 s) of sim-time. Set to 0 to fall back
                         to the legacy single-deadline behavior for bisection.
STRAFER_NAV_BUDGET_SAFETY_FACTOR : Override the multiplier on the (distance / nominal_speed) term in
                         the progress-aware budget formula (optional, float; default 2.0). Raise on
                         cluttered sim scenes where Nav2 plans a longer path than the straight-line
                         estimate; lower on real-robot deployments with smoother planning.
                         No effect when STRAFER_NAV_PROGRESS_AWARE=0.
STRAFER_NAV_BUDGET_SETUP_OVERHEAD_S : Override the additive setup term in the progress-aware budget
                         formula (optional, seconds; default 5.0). Covers Nav2 plan + accel/decel
                         that don't scale with distance. Raise on slow-RTF sim where the planner
                         takes a noticeable fraction of a sim-second to converge.
                         No effect when STRAFER_NAV_PROGRESS_AWARE=0.
STRAFER_NAV_STALL_PROGRESS_M : Override the minimum distance_remaining decrease required within
                         STRAFER_NAV_STALL_WINDOW_S to consider Nav2 making progress (optional,
                         meters; default 0.10). Raise to be more permissive of slow approaches /
                         physics jitter; lower to fast-fail on hangs sooner.
                         No effect when STRAFER_NAV_PROGRESS_AWARE=0.
STRAFER_NAV_STALL_WINDOW_S : Override the rolling sim-time window over which the stall watchdog
                         measures progress (optional, seconds; default 20.0). The watchdog cannot
                         fire before the window is fully populated, so this is also the minimum
                         duration of any navigation attempt. Raise on noisy / oscillating planners;
                         lower for tighter dead-man behavior.
                         No effect when STRAFER_NAV_PROGRESS_AWARE=0.
STRAFER_NAV_STAGING_BUDGET : Override the maximum number of clamped intermediate Nav2 goals issued
                         by `_navigate_via_staging` when the projected target lands outside the global
                         costmap (optional, integer; default 4). One additional final Nav2 goal fires
                         once the projection lands inside the costmap, so the worst-case Nav2 goal
                         count per navigate-with-projection step is `budget + 1`.
STRAFER_NAV_BACKEND : Select the navigate-leg execution backend (optional). Unset/default uses
                         `nav2`; supported values are `nav2`, `strafer_direct`, and
                         `hybrid_nav2_strafer`. An unknown value is ignored with a warning,
                         keeping `nav2`. A per-step `execution_backend` in the plan still
                         overrides this default.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def _read_float_env(name: str, config_key: str, target: dict) -> None:
    """Copy ``$name`` into ``target[config_key]`` as a float, if set.

    Silently ignores unset env vars (preserving the dataclass default)
    and logs a warning on a non-numeric value rather than raising.
    """
    raw = os.environ.get(name)
    if not raw:
        return
    try:
        target[config_key] = float(raw)
        logger.info("%s overridden to %s via env", config_key, raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r", name, raw)


def _read_bool_env(name: str, config_key: str, target: dict) -> None:
    """Copy ``$name`` into ``target[config_key]`` as a bool, if set.

    Recognizes ``"0"`` / ``"false"`` / ``"no"`` / ``"off"`` (case-insensitive)
    as False; any other non-empty value as True. Unset preserves the
    dataclass default.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return
    target[config_key] = raw.strip().lower() not in {"0", "false", "no", "off"}
    logger.info("%s overridden to %s via env", config_key, target[config_key])


def _read_positive_int_env(name: str, config_key: str, target: dict) -> None:
    """Copy ``$name`` into ``target[config_key]`` as a positive int, if set.

    Mirrors ``_read_float_env`` semantics: unset preserves the default,
    non-numeric and non-positive values log a warning and fall through.
    """
    raw = os.environ.get(name)
    if not raw:
        return
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r", name, raw)
        return
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%d", name, value)
        return
    target[config_key] = value
    logger.info("%s overridden to %d via env", config_key, value)


def _read_str_env(
    name: str,
    config_key: str,
    target: dict,
    *,
    allowed: "tuple[str, ...] | None" = None,
) -> None:
    """Copy ``$name`` into ``target[config_key]`` as a string, if set.

    Unset preserves the dataclass default. When ``allowed`` is given and the
    value is not in it, warns and keeps the default rather than raising, so a
    typo cannot strand the operator on no backend.
    """
    raw = os.environ.get(name)
    if not raw:
        return
    if allowed is not None and raw not in allowed:
        logger.warning(
            "Ignoring unsupported %s=%r (expected one of %s)",
            name, raw, sorted(allowed),
        )
        return
    target[config_key] = raw
    logger.info("%s overridden to %s via env", config_key, raw)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] [%(name)s] %(message)s",
    )

    vlm_url = os.environ.get("VLM_URL")
    planner_url = os.environ.get("PLANNER_URL")

    if not vlm_url or not planner_url:
        logger.error(
            "Both VLM_URL and PLANNER_URL environment variables must be set.\n"
            "  Example: VLM_URL=http://192.168.50.196:8100 "
            "PLANNER_URL=http://192.168.50.196:8200 strafer-executor"
        )
        sys.exit(1)

    import rclpy

    from strafer_autonomy.clients.planner_client import (
        HttpPlannerClient,
        HttpPlannerClientConfig,
    )
    from strafer_autonomy.clients.ros_client import (
        JetsonRosClient,
        RosClientConfig,
        _SUPPORTED_BACKENDS,
    )
    from strafer_autonomy.clients.vlm_client import (
        HttpGroundingClient,
        HttpGroundingClientConfig,
    )
    from strafer_autonomy.executor.command_server import build_command_server
    from strafer_autonomy.executor.mission_runner import MissionRunnerConfig

    if not rclpy.ok():
        # Forward sys.argv so `--ros-args -p use_sim_time:=true` and
        # similar overrides reach the rclpy context and propagate to the
        # nodes this executor constructs.
        rclpy.init(args=sys.argv)

    planner_client = HttpPlannerClient(
        config=HttpPlannerClientConfig(base_url=planner_url),
    )
    grounding_client = HttpGroundingClient(
        config=HttpGroundingClientConfig(base_url=vlm_url),
    )

    ros_config_kwargs: dict = {}
    _read_float_env("OBSERVATION_MAX_AGE_S", "observation_max_age_s", ros_config_kwargs)
    _read_float_env("ROTATE_TIMEOUT_S", "default_rotate_timeout_s", ros_config_kwargs)
    _read_float_env(
        "STRAFER_CLOCK_STALL_BAIL_WALL_S",
        "clock_stall_bail_wall_s",
        ros_config_kwargs,
    )
    ros_client = JetsonRosClient(
        config=RosClientConfig(**ros_config_kwargs) if ros_config_kwargs else None,
    )

    mission_config_kwargs: dict = {}
    _read_float_env(
        "STRAFER_NAVIGATION_TIMEOUT_S",
        "default_navigation_timeout_s",
        mission_config_kwargs,
    )
    _read_positive_int_env(
        "STRAFER_NAV_STAGING_BUDGET",
        "staging_budget",
        mission_config_kwargs,
    )
    _read_bool_env(
        "STRAFER_NAV_PROGRESS_AWARE",
        "nav_progress_aware",
        mission_config_kwargs,
    )
    _read_float_env(
        "STRAFER_NAV_BUDGET_SAFETY_FACTOR",
        "nav_budget_safety_factor",
        mission_config_kwargs,
    )
    _read_float_env(
        "STRAFER_NAV_BUDGET_SETUP_OVERHEAD_S",
        "nav_budget_setup_overhead_s",
        mission_config_kwargs,
    )
    _read_float_env(
        "STRAFER_NAV_STALL_PROGRESS_M",
        "nav_stall_progress_m",
        mission_config_kwargs,
    )
    _read_float_env(
        "STRAFER_NAV_STALL_WINDOW_S",
        "nav_stall_window_s",
        mission_config_kwargs,
    )
    _read_str_env(
        "STRAFER_NAV_BACKEND",
        "default_navigation_backend",
        mission_config_kwargs,
        allowed=_SUPPORTED_BACKENDS,
    )
    runner_config = (
        MissionRunnerConfig(**mission_config_kwargs)
        if mission_config_kwargs
        else None
    )

    logger.info("Planner service: %s", planner_url)
    logger.info("VLM service:     %s", vlm_url)

    server, runner = build_command_server(
        planner_client=planner_client,
        grounding_client=grounding_client,
        ros_client=ros_client,
        runner_config=runner_config,
    )

    logger.info("Autonomy executor ready — spinning.")
    try:
        server.spin()
    except KeyboardInterrupt:
        logger.info("Shutting down autonomy executor.")
    finally:
        server.destroy()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
