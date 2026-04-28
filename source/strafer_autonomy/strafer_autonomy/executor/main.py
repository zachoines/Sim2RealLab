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
STRAFER_NAVIGATION_TIMEOUT_S : Override the default navigate_to_pose timeout (optional, seconds; default 90.0).
                         The executor's nav timeout ticks on the executor node's clock, so under
                         use_sim_time:=true this is a sim-time budget. Raise it on sub-unity-RTF sim
                         scenes where 90 s of sim time can correspond to several minutes of wall time.
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
    from strafer_autonomy.clients.ros_client import JetsonRosClient, RosClientConfig
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
    ros_client = JetsonRosClient(
        config=RosClientConfig(**ros_config_kwargs) if ros_config_kwargs else None,
    )

    mission_config_kwargs: dict = {}
    _read_float_env(
        "STRAFER_NAVIGATION_TIMEOUT_S",
        "default_navigation_timeout_s",
        mission_config_kwargs,
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
