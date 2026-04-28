"""Databricks Model Serving implementation of ``PlannerClient``.

This module mirrors the surface of :mod:`strafer_autonomy.clients.planner_client`
so the executor can transparently swap between LAN HTTP and Databricks Model
Serving by setting ``PLANNER_BACKEND=databricks`` and related env vars.

The request/response wire format matches the Databricks Model Serving
``inputs``/``predictions`` convention (:ref:`MLflow pyfunc`). The serving
endpoint is expected to be backed by an MLflow ``pyfunc`` model that wraps
:class:`strafer_autonomy.planner.llm_runtime.LLMRuntime` plus the
``build_messages → parse_intent → compile_plan`` pipeline, producing one
``MissionPlan`` payload per input row (see ``databricks/planner_model.py``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from strafer_autonomy.clients.planner_client import (
    PlannerServiceUnavailable,
    mission_plan_from_payload,
    planner_request_to_payload,
)
from strafer_autonomy.schemas import MissionPlan, PlannerRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabricksServingPlannerClientConfig:
    """Connection settings for a Databricks Model Serving planner endpoint."""

    endpoint_name: str
    workspace_url: str
    token: str
    timeout_s: float = 30.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def invocations_url(self) -> str:
        base = self.workspace_url.rstrip("/")
        return f"{base}/serving-endpoints/{self.endpoint_name}/invocations"

    def state_url(self) -> str:
        base = self.workspace_url.rstrip("/")
        return f"{base}/api/2.0/serving-endpoints/{self.endpoint_name}"


class DatabricksServingPlannerClient:
    """``PlannerClient`` implementation backed by Databricks Model Serving.

    The ``plan_mission`` call POSTs a single-row MLflow ``inputs`` payload
    to the serving endpoint and parses the first prediction back into a
    :class:`MissionPlan`. The underlying pyfunc model is responsible for
    running the full planner pipeline (LLM → intent → compile).
    """

    def __init__(self, config: DatabricksServingPlannerClientConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {config.token}",
                "Content-Type": "application/json",
            }
        )
        if config.extra_headers:
            self._session.headers.update(config.extra_headers)

    @property
    def config(self) -> DatabricksServingPlannerClientConfig:
        return self._config

    def plan_mission(self, request: PlannerRequest) -> MissionPlan:
        """Send a planning request to Databricks Model Serving.

        Raises ``PlannerServiceUnavailable`` after exhausting retries.
        """

        row = planner_request_to_payload(request)
        payload: dict[str, Any] = {"inputs": [row]}
        url = self._config.invocations_url()

        last_exc: Exception | None = None
        for attempt in range(1 + self._config.max_retries):
            try:
                resp = self._session.post(
                    url, json=payload, timeout=self._config.timeout_s,
                )
                resp.raise_for_status()
                body = resp.json()
                return _extract_mission_plan(body)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                logger.warning(
                    "Databricks planner request %s attempt %d/%d failed: %s",
                    request.request_id, attempt + 1, 1 + self._config.max_retries, exc,
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code < 500:
                    raise PlannerServiceUnavailable(
                        "Databricks planner endpoint returned "
                        f"{exc.response.status_code}: {exc.response.text}"
                    ) from exc
                last_exc = exc
                logger.warning(
                    "Databricks planner request %s attempt %d/%d server error: %s",
                    request.request_id, attempt + 1, 1 + self._config.max_retries, exc,
                )
            except ValueError as exc:
                raise PlannerServiceUnavailable(
                    f"Databricks planner response was not valid JSON: {exc}"
                ) from exc

            if attempt < self._config.max_retries:
                backoff = self._config.retry_backoff_s * (2 ** attempt)
                time.sleep(backoff)

        raise PlannerServiceUnavailable(
            "Databricks planner endpoint unreachable after "
            f"{1 + self._config.max_retries} attempts."
        ) from last_exc

    def health(self) -> dict[str, Any]:
        """Query the Databricks serving endpoint's readiness state.

        Uses the Databricks REST API's ``GET /api/2.0/serving-endpoints/{name}``
        response and maps the ``state.ready`` field into the same shape the
        executor expects from the LAN HTTP planner health check.
        """

        url = self._config.state_url()
        try:
            resp = self._session.get(url, timeout=self._config.timeout_s)
            resp.raise_for_status()
            body = resp.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            raise PlannerServiceUnavailable(
                f"Databricks planner health check failed: {exc}"
            ) from exc
        except ValueError as exc:
            raise PlannerServiceUnavailable(
                f"Databricks planner health response was not valid JSON: {exc}"
            ) from exc

        state = (body.get("state") or {}) if isinstance(body, dict) else {}
        ready_state = str(state.get("ready", "")).upper()
        config_update = str(state.get("config_update", "")).upper()
        is_ready = ready_state == "READY" and config_update in {"NOT_UPDATING", ""}
        return {
            "status": "ok" if is_ready else "loading",
            "model_loaded": is_ready,
            "model_name": self._config.endpoint_name,
            "backend": "databricks",
            "endpoint_state": ready_state or None,
            "config_update": config_update or None,
        }


def _extract_mission_plan(body: Any) -> MissionPlan:
    """Parse the first prediction in a Databricks serving response into a plan."""

    if not isinstance(body, dict):
        raise PlannerServiceUnavailable(
            f"Databricks planner response was not an object: {type(body).__name__}"
        )

    predictions = body.get("predictions")
    if predictions is None:
        # Some MLflow wrappers emit raw records under "outputs" or the top level.
        predictions = body.get("outputs") or body
    if isinstance(predictions, dict):
        predictions = [predictions]
    if not isinstance(predictions, list) or not predictions:
        raise PlannerServiceUnavailable(
            f"Databricks planner response missing predictions: {body!r}"
        )

    first = predictions[0]
    if not isinstance(first, dict):
        raise PlannerServiceUnavailable(
            f"Databricks planner prediction was not a dict: {first!r}"
        )
    return mission_plan_from_payload(first)
