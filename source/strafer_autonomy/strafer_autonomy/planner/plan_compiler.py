"""Deterministic MissionIntent → MissionPlan compiler.

Expands a validated ``MissionIntent`` into a full ``MissionPlan`` with
correct step ordering, timeouts, retry limits, and skill arguments.
"""

from __future__ import annotations

import time
import uuid

from strafer_autonomy.schemas import MissionIntent, MissionPlan, SkillCall


class CompilationError(Exception):
    """Raised when a MissionIntent cannot be compiled into a valid plan."""


def compile_plan(intent: MissionIntent) -> MissionPlan:
    """Compile a ``MissionIntent`` into a ``MissionPlan``.

    Raises ``CompilationError`` for unsupported intent types.
    """
    compiler = _COMPILERS.get(intent.intent_type)
    if compiler is None:
        raise CompilationError(f"Unsupported intent_type: {intent.intent_type!r}")
    steps = compiler(intent)
    return MissionPlan(
        mission_id=f"mission_{uuid.uuid4().hex[:12]}",
        mission_type=intent.intent_type,
        raw_command=intent.raw_command,
        steps=tuple(steps),
        created_at=time.time(),
    )


def _compile_go_to_target(intent: MissionIntent) -> list[SkillCall]:
    return [
        SkillCall(
            step_id="step_01",
            skill="scan_for_target",
            args={"label": intent.target_label, "max_scan_steps": 6, "scan_arc_deg": 360},
            timeout_s=60.0,
            retry_limit=0,
        ),
        SkillCall(
            step_id="step_02",
            skill="project_detection_to_goal_pose",
            args={"standoff_m": 0.7},
            timeout_s=2.0,
            retry_limit=0,
        ),
        SkillCall(
            step_id="step_03",
            skill="navigate_to_pose",
            args={"goal_source": "projected_target", "execution_backend": "nav2"},
            timeout_s=90.0,
            retry_limit=0,
        ),
    ]


def _compile_wait_by_target(intent: MissionIntent) -> list[SkillCall]:
    steps = _compile_go_to_target(intent)
    steps.append(
        SkillCall(
            step_id=f"step_{len(steps) + 1:02d}",
            skill="wait",
            args={"mode": intent.wait_mode or "until_next_command"},
            timeout_s=None,
            retry_limit=0,
        ),
    )
    return steps


def _compile_cancel(intent: MissionIntent) -> list[SkillCall]:
    return [
        SkillCall(
            step_id="step_01",
            skill="cancel_mission",
            args={},
            timeout_s=5.0,
            retry_limit=0,
        ),
    ]


def _compile_status(intent: MissionIntent) -> list[SkillCall]:
    return [
        SkillCall(
            step_id="step_01",
            skill="report_status",
            args={},
            timeout_s=5.0,
            retry_limit=0,
        ),
    ]


_COMPILERS: dict[str, callable] = {
    "go_to_target": _compile_go_to_target,
    "wait_by_target": _compile_wait_by_target,
    "cancel": _compile_cancel,
    "status": _compile_status,
}
