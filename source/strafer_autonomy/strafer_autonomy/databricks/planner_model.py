"""MLflow ``pyfunc`` wrapper for the Strafer planner pipeline.

The wrapper replicates the planner service's full chain:
``build_messages → LLMRuntime.generate → parse_intent → compile_plan``
so the Databricks serving endpoint behaves identically to
``POST /plan`` on the LAN HTTP service.

Inputs (``model_input``) accept the same columns as ``PlanRequest``:
``request_id``, ``raw_command``, optional ``robot_state``,
``active_mission_summary``, and ``available_skills``. One plan is
returned per input row.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def _mission_plan_to_dict(plan: Any) -> dict[str, Any]:
    from strafer_autonomy.schemas import MissionPlan

    if not isinstance(plan, MissionPlan):
        raise TypeError(f"Expected MissionPlan, got {type(plan).__name__}")
    return {
        "mission_id": plan.mission_id,
        "mission_type": plan.mission_type,
        "raw_command": plan.raw_command,
        "steps": [asdict(step) for step in plan.steps],
        "created_at": plan.created_at,
    }


def _rows_from_model_input(model_input: Any) -> Iterable[dict[str, Any]]:
    """Yield rows from MLflow's accepted input types.

    MLflow passes ``pandas.DataFrame`` for DataFrame-split JSON, a list of
    dicts for ``inputs``/``dataframe_records``, or a single dict for
    degenerate cases. Handle all three without requiring pandas at import
    time.
    """

    if model_input is None:
        return []
    if isinstance(model_input, dict):
        return [model_input]
    if isinstance(model_input, list):
        return model_input
    # pandas.DataFrame path
    if hasattr(model_input, "to_dict"):
        return model_input.to_dict(orient="records")
    raise TypeError(
        f"Unsupported model_input type: {type(model_input).__name__}"
    )


class StraferPlannerModel:
    """``mlflow.pyfunc.PythonModel`` wrapper around the planner pipeline.

    We intentionally do NOT subclass ``mlflow.pyfunc.PythonModel`` at import
    time — ``mlflow`` is an optional dependency on the Jetson and we want
    this module to be importable without it. ``register.py`` mixes this
    class with ``mlflow.pyfunc.PythonModel`` at log time via a tiny
    subclass (see ``register.py``).
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        max_tokens: int = 256,
    ) -> None:
        self._model_path = model_path
        self._max_tokens = max_tokens
        self._runtime: Any = None

    def load_context(self, context: Any) -> None:  # noqa: D401 - MLflow signature
        """Load the LLM runtime from the MLflow artifact path."""
        from strafer_autonomy.planner.llm_runtime import LLMRuntime

        artifacts = getattr(context, "artifacts", None) or {}
        model_path = artifacts.get("model") or self._model_path
        if not model_path:
            raise RuntimeError(
                "StraferPlannerModel requires a 'model' artifact path or "
                "an explicit model_path in __init__."
            )

        runtime = LLMRuntime()
        runtime.load(model_name=model_path, max_tokens=self._max_tokens)
        self._runtime = runtime
        logger.info("StraferPlannerModel loaded runtime from %s", model_path)

    def predict(self, context: Any, model_input: Any, params: Any = None) -> list[dict[str, Any]]:
        """Run the planner pipeline for each input row.

        Returns a list of plan dicts (one per input row) in the same shape
        the LAN HTTP planner service returns from ``POST /plan``.
        """
        if self._runtime is None or not self._runtime.ready:
            raise RuntimeError(
                "StraferPlannerModel.predict called before load_context()."
            )

        from strafer_autonomy.planner.intent_parser import IntentParseError, parse_intent
        from strafer_autonomy.planner.plan_compiler import CompilationError, compile_plan
        from strafer_autonomy.planner.prompt_builder import build_messages
        from strafer_autonomy.schemas import PlannerRequest

        results: list[dict[str, Any]] = []
        for row in _rows_from_model_input(model_input):
            request_id = str(row.get("request_id") or f"mlflow_{uuid.uuid4().hex[:12]}")
            raw_command = str(row.get("raw_command") or "").strip()
            if not raw_command:
                raise ValueError(f"row {request_id} missing non-empty raw_command")

            planner_request = PlannerRequest(
                request_id=request_id,
                raw_command=raw_command,
                robot_state=row.get("robot_state"),
                active_mission_summary=row.get("active_mission_summary"),
                available_skills=tuple(row.get("available_skills") or ()),
            )
            messages = build_messages(planner_request)

            raw_output = self._runtime.generate(messages)
            try:
                intent = parse_intent(raw_output, raw_command)
            except IntentParseError as exc:
                raise ValueError(
                    f"row {request_id} produced unparseable planner output: {exc}"
                ) from exc
            try:
                plan = compile_plan(intent)
            except CompilationError as exc:
                raise ValueError(
                    f"row {request_id} failed plan compilation: {exc}"
                ) from exc

            results.append(_mission_plan_to_dict(plan))

        return results

    # Deterministic helpers so ``register.py`` can log a consistent model
    # signature without running the real runtime.
    @staticmethod
    def input_schema_columns() -> tuple[str, ...]:
        return (
            "request_id",
            "raw_command",
            "robot_state",
            "active_mission_summary",
            "available_skills",
        )

    @staticmethod
    def output_schema_columns() -> tuple[str, ...]:
        return (
            "mission_id",
            "mission_type",
            "raw_command",
            "steps",
            "created_at",
        )

    @staticmethod
    def epoch_seconds() -> float:
        return time.time()
