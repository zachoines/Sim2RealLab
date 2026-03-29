"""Planner client abstractions and HTTP transport."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import requests

from strafer_autonomy.schemas import MissionPlan, PlannerRequest, SkillCall

logger = logging.getLogger(__name__)


class PlannerServiceUnavailable(Exception):
    """Raised when the planner service cannot be reached or returns a server error."""


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
    max_retries: int = 2
    retry_backoff_s: float = 0.5


class HttpPlannerClient:
    """LAN HTTP planner client for the MVP workstation service."""

    def __init__(self, config: HttpPlannerClientConfig) -> None:
        self._config = config
        self._session = requests.Session()
        if config.headers:
            self._session.headers.update(config.headers)

    @property
    def config(self) -> HttpPlannerClientConfig:
        """Return the immutable planner client configuration."""

        return self._config

    def plan_mission(self, request: PlannerRequest) -> MissionPlan:
        """Send a planning request to the remote planner service.

        Retries up to ``config.max_retries`` times on connection and server
        errors with exponential back-off.  Raises ``PlannerServiceUnavailable``
        if all attempts fail.
        """

        payload = planner_request_to_payload(request)
        url = self._config.base_url.rstrip("/") + self._config.plan_path

        last_exc: Exception | None = None
        for attempt in range(1 + self._config.max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=self._config.timeout_s)
                resp.raise_for_status()
                return mission_plan_from_payload(resp.json())
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                logger.warning(
                    "Planner request %s attempt %d/%d failed: %s",
                    request.request_id, attempt + 1, 1 + self._config.max_retries, exc,
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code < 500:
                    raise PlannerServiceUnavailable(
                        f"Planner service returned {exc.response.status_code}: {exc.response.text}"
                    ) from exc
                last_exc = exc
                logger.warning(
                    "Planner request %s attempt %d/%d server error: %s",
                    request.request_id, attempt + 1, 1 + self._config.max_retries, exc,
                )

            if attempt < self._config.max_retries:
                backoff = self._config.retry_backoff_s * (2 ** attempt)
                time.sleep(backoff)

        raise PlannerServiceUnavailable(
            f"Planner service unreachable after {1 + self._config.max_retries} attempts."
        ) from last_exc

    def health(self) -> dict[str, Any]:
        """Check the remote planner service health.

        Raises ``PlannerServiceUnavailable`` on connection or HTTP errors.
        """

        url = self._config.base_url.rstrip("/") + self._config.health_path
        try:
            resp = self._session.get(url, timeout=5.0)
            resp.raise_for_status()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            raise PlannerServiceUnavailable(
                f"Planner health check failed: {exc}"
            ) from exc
        return resp.json()


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
