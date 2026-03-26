"""Planner client abstractions and HTTP transport stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from strafer_autonomy.schemas import MissionPlan, PlannerRequest, SkillCall


@runtime_checkable
class PlannerClient(Protocol):
    """Executor-facing interface for requesting bounded mission plans."""

    def plan_mission(self, request: PlannerRequest) -> MissionPlan:
        """Return a mission plan for the given user command."""


@dataclass(frozen=True)
class HttpPlannerClientConfig:
    """Connection settings for the workstation-hosted planner service."""

    base_url: str
    timeout_s: float = 10.0
    plan_path: str = "/plan"
    health_path: str = "/health"
    headers: dict[str, str] | None = None


class HttpPlannerClient:
    """LAN HTTP planner client stub for the MVP workstation service."""

    def __init__(self, config: HttpPlannerClientConfig) -> None:
        self._config = config

    @property
    def config(self) -> HttpPlannerClientConfig:
        """Return the immutable planner client configuration."""

        return self._config

    def plan_mission(self, request: PlannerRequest) -> MissionPlan:
        """Send a planning request to the remote planner service."""

        raise NotImplementedError(
            "HTTP planner transport is not implemented yet. "
            "Use planner_request_to_payload() and mission_plan_from_payload() when wiring the first LAN client."
        )


def planner_request_to_payload(request: PlannerRequest) -> dict[str, Any]:
    """Convert a planner request schema into the JSON payload expected by POST /plan."""

    return {
        "request_id": request.request_id,
        "raw_command": request.raw_command,
        "robot_state": request.robot_state,
        "active_mission_summary": request.active_mission_summary,
        "available_skills": list(request.available_skills),
    }


def mission_plan_from_payload(payload: dict[str, Any]) -> MissionPlan:
    """Parse a planner JSON response into the shared mission plan schema."""

    steps = tuple(
        SkillCall(
            skill=str(step["skill"]),
            step_id=str(step["step_id"]),
            args=dict(step.get("args") or {}),
            timeout_s=float(step["timeout_s"]) if step.get("timeout_s") is not None else None,
            retry_limit=int(step.get("retry_limit", 0)),
        )
        for step in payload.get("steps", ())
    )
    return MissionPlan(
        mission_id=str(payload["mission_id"]),
        mission_type=str(payload["mission_type"]),
        raw_command=str(payload["raw_command"]),
        steps=steps,
        created_at=float(payload["created_at"]),
    )
