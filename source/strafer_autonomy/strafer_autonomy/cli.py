"""Operator CLI for submitting, querying, and canceling autonomy missions."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence
import uuid

from strafer_autonomy.executor.command_server import (
    DEFAULT_EXECUTE_MISSION_ACTION,
    DEFAULT_STATUS_SERVICE,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="strafer-autonomy-cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit a natural-language mission.")
    submit_parser.add_argument("raw_command", help="Natural-language command to execute.")
    submit_parser.add_argument("--request-id", default="", help="Optional caller-supplied request id.")
    submit_parser.add_argument("--source", default="ssh_cli", help="Ingress source label.")
    submit_parser.add_argument(
        "--replace-active",
        action="store_true",
        help="Allow the new mission to replace an active mission.",
    )
    submit_parser.add_argument(
        "--action-name",
        default=DEFAULT_EXECUTE_MISSION_ACTION,
        help="ROS action name for ExecuteMission.",
    )
    submit_parser.add_argument(
        "--node-name",
        default="strafer_autonomy_cli_submit",
        help="ROS node name used by the submit client.",
    )
    submit_parser.add_argument(
        "--wait-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the action server to appear.",
    )
    submit_parser.add_argument(
        "--detach",
        action="store_true",
        help="Return after goal acceptance instead of following feedback until completion.",
    )

    status_parser = subparsers.add_parser("status", help="Query current mission status.")
    status_parser.add_argument(
        "--service-name",
        default=DEFAULT_STATUS_SERVICE,
        help="ROS service name for GetMissionStatus.",
    )
    status_parser.add_argument(
        "--node-name",
        default="strafer_autonomy_cli_status",
        help="ROS node name used by the status client.",
    )
    status_parser.add_argument(
        "--wait-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the status service to appear.",
    )

    cancel_parser = subparsers.add_parser("cancel", help="Cancel the active mission action goal.")
    cancel_parser.add_argument(
        "--action-name",
        default=DEFAULT_EXECUTE_MISSION_ACTION,
        help="ROS action name for ExecuteMission.",
    )
    cancel_parser.add_argument(
        "--node-name",
        default="strafer_autonomy_cli_cancel",
        help="ROS node name used by the cancel client.",
    )
    cancel_parser.add_argument(
        "--wait-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the cancel service to appear.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the operator CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "submit":
        return _run_submit(args)
    if args.command == "status":
        return _run_status(args)
    if args.command == "cancel":
        return _run_cancel(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _run_submit(args: argparse.Namespace) -> int:
    rclpy, action_client_cls, node_cls, execute_mission_action, _, _ = _require_ros_cli_types()
    request_id = args.request_id or uuid.uuid4().hex
    rclpy.init(args=None)
    node = node_cls(args.node_name)
    try:
        client = action_client_cls(node, execute_mission_action, args.action_name)
        if not client.wait_for_server(timeout_sec=args.wait_timeout):
            raise RuntimeError(f"Action server '{args.action_name}' is not available.")

        goal = execute_mission_action.Goal()
        goal.request_id = request_id
        goal.raw_command = args.raw_command
        goal.source = args.source
        goal.replace_active_mission = bool(args.replace_active)

        send_goal_future = client.send_goal_async(goal, feedback_callback=_feedback_callback)
        _spin_until_future_complete(rclpy, node, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            print(json.dumps({"accepted": False, "request_id": request_id}, indent=2))
            return 1

        print(json.dumps({"accepted": True, "request_id": request_id}, indent=2))
        if args.detach:
            return 0

        result_future = goal_handle.get_result_async()
        try:
            _spin_until_future_complete(rclpy, node, result_future)
        except KeyboardInterrupt:
            print("Detached from mission feedback; mission continues on the executor.")
            return 0

        result = result_future.result()
        if result is None:
            raise RuntimeError("No result was returned by the action server.")
        print(
            json.dumps(
                {
                    "accepted": result.result.accepted,
                    "mission_id": result.result.mission_id,
                    "final_state": result.result.final_state,
                    "error_code": result.result.error_code,
                    "message": result.result.message,
                },
                indent=2,
            )
        )
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _run_status(args: argparse.Namespace) -> int:
    rclpy, _, node_cls, _, status_service, _ = _require_ros_cli_types()
    rclpy.init(args=None)
    node = node_cls(args.node_name)
    try:
        client = node.create_client(status_service, args.service_name)
        if not client.wait_for_service(timeout_sec=args.wait_timeout):
            raise RuntimeError(f"Status service '{args.service_name}' is not available.")
        future = client.call_async(status_service.Request())
        _spin_until_future_complete(rclpy, node, future)
        response = future.result()
        if response is None:
            raise RuntimeError("No response was returned by the status service.")
        print(json.dumps(_status_response_to_dict(response), indent=2))
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _run_cancel(args: argparse.Namespace) -> int:
    rclpy, _, node_cls, _, _, cancel_goal_service = _require_ros_cli_types()
    rclpy.init(args=None)
    node = node_cls(args.node_name)
    try:
        service_name = _cancel_service_name(args.action_name)
        client = node.create_client(cancel_goal_service, service_name)
        if not client.wait_for_service(timeout_sec=args.wait_timeout):
            raise RuntimeError(f"Cancel service '{service_name}' is not available.")
        future = client.call_async(cancel_goal_service.Request())
        _spin_until_future_complete(rclpy, node, future)
        response = future.result()
        if response is None:
            raise RuntimeError("No response was returned by the cancel service.")
        print(json.dumps({"goals_canceling": len(response.goals_canceling)}, indent=2))
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _feedback_callback(feedback_msg) -> None:  # type: ignore[no-untyped-def]
    feedback = feedback_msg.feedback
    print(
        json.dumps(
            {
                "mission_id": feedback.mission_id,
                "state": feedback.state,
                "current_step_id": feedback.current_step_id,
                "current_skill": feedback.current_skill,
                "message": feedback.message,
                "elapsed_s": feedback.elapsed_s,
            },
            indent=2,
        )
    )


def _status_response_to_dict(response) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return {
        "active": bool(response.active),
        "mission_id": response.mission_id,
        "state": response.state,
        "raw_command": response.raw_command,
        "current_step_id": response.current_step_id,
        "current_skill": response.current_skill,
        "message": response.message,
        "elapsed_s": response.elapsed_s,
    }


def _spin_until_future_complete(rclpy, node, future) -> None:  # type: ignore[no-untyped-def]
    while not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)


def _cancel_service_name(action_name: str) -> str:
    normalized = action_name.rstrip("/")
    return f"{normalized}/_action/cancel_goal"


def _require_ros_cli_types():
    try:
        import rclpy
        from action_msgs.srv import CancelGoal
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from strafer_msgs.action import ExecuteMission
        from strafer_msgs.srv import GetMissionStatus
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python dependencies are not available. "
            "Run this CLI on the Jetson inside the ROS environment with strafer_msgs built."
        ) from exc
    return rclpy, ActionClient, Node, ExecuteMission, GetMissionStatus, CancelGoal


if __name__ == "__main__":
    raise SystemExit(main())
