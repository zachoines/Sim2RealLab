"""Mission planning and execution schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlannerRequest:
    """Request sent from the executor to the planner service."""

    request_id: str
    raw_command: str
    robot_state: dict[str, Any] | None = None
    active_mission_summary: dict[str, Any] | None = None
    available_skills: tuple[str, ...] = ()


@dataclass(frozen=True)
class MissionIntent:
    """Bounded planner interpretation of a user command."""

    intent_type: str
    raw_command: str
    target_label: str | None = None
    orientation_mode: str | None = None
    wait_mode: str | None = None
    requires_grounding: bool = False


@dataclass(frozen=True)
class SkillCall:
    """One typed executable step inside a mission plan."""

    skill: str
    step_id: str
    args: dict[str, Any] = field(default_factory=dict)
    timeout_s: float | None = None
    retry_limit: int = 0


@dataclass(frozen=True)
class MissionPlan:
    """Ordered sequence of skill calls produced by the planner."""

    mission_id: str
    mission_type: str
    raw_command: str
    steps: tuple[SkillCall, ...]
    created_at: float


@dataclass(frozen=True)
class SkillResult:
    """Normalized executor-facing result for a skill call."""

    step_id: str
    skill: str
    status: str
    outputs: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    message: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
