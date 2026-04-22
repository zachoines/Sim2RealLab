"""Entry point for the Jetson autonomy executor.

Reads DGX service URLs from environment variables, constructs the HTTP
clients and ROS client, and spins the AutonomyCommandServer node.

Environment variables
---------------------
VLM_URL      : Base URL for the VLM grounding service  (required, e.g. http://192.168.50.196:8100)
PLANNER_URL  : Base URL for the LLM planner service    (required, e.g. http://192.168.50.196:8200)
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


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
    from strafer_autonomy.clients.ros_client import JetsonRosClient
    from strafer_autonomy.clients.vlm_client import (
        HttpGroundingClient,
        HttpGroundingClientConfig,
    )
    from strafer_autonomy.executor.command_server import build_command_server

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
    ros_client = JetsonRosClient()

    logger.info("Planner service: %s", planner_url)
    logger.info("VLM service:     %s", vlm_url)

    server, runner = build_command_server(
        planner_client=planner_client,
        grounding_client=grounding_client,
        ros_client=ros_client,
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
