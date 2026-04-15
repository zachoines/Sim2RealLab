"""Deterministic MissionIntent → MissionPlan compiler.

Expands a validated ``MissionIntent`` into a full ``MissionPlan`` with
correct step ordering, timeouts, retry limits, and skill arguments.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from strafer_autonomy.schemas import MissionIntent, MissionPlan, SkillCall

# Every compiler that ends in ``navigate_to_pose`` appends a
# ``verify_arrival`` step for a total of four steps per target:
# scan → project → navigate → verify. The verify step runs a
# CLIP top-k ranking against the semantic map at the arrival pose
# to confirm the robot reached the intended target region.
_STEPS_PER_TARGET = 4


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


def _make_verify_arrival_step(*, step_id: str, target_label: str) -> SkillCall:
    return SkillCall(
        step_id=step_id,
        skill="verify_arrival",
        args={
            "target_label": target_label,
            "goal_radius_m": 3.0,
            "top_k": 5,
            "majority": 3,
            "fallback_on_empty_map": "pass",
        },
        timeout_s=10.0,
        retry_limit=1,
    )


def _compile_single_target_steps(
    *, label: str, base: int, standoff_m: float = 0.7,
) -> list[SkillCall]:
    """Return the 4-step scan→project→navigate→verify sequence.

    ``base`` is the 1-indexed step number of the first step.
    """
    return [
        SkillCall(
            step_id=f"step_{base:02d}",
            skill="scan_for_target",
            args={"label": label, "max_scan_steps": 6, "scan_arc_deg": 360},
            timeout_s=60.0,
            retry_limit=0,
        ),
        SkillCall(
            step_id=f"step_{base + 1:02d}",
            skill="project_detection_to_goal_pose",
            args={"standoff_m": standoff_m},
            timeout_s=2.0,
            retry_limit=0,
        ),
        SkillCall(
            step_id=f"step_{base + 2:02d}",
            skill="navigate_to_pose",
            args={"goal_source": "projected_target", "execution_backend": "nav2"},
            timeout_s=90.0,
            retry_limit=0,
        ),
        _make_verify_arrival_step(
            step_id=f"step_{base + 3:02d}",
            target_label=label,
        ),
    ]


def _compile_go_to_target(intent: MissionIntent) -> list[SkillCall]:
    assert intent.target_label is not None
    return _compile_single_target_steps(label=intent.target_label, base=1)


def _compile_wait_by_target(intent: MissionIntent) -> list[SkillCall]:
    assert intent.target_label is not None
    steps = _compile_single_target_steps(label=intent.target_label, base=1)
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


def _compile_rotate(intent: MissionIntent) -> list[SkillCall]:
    """Compile a rotate intent into a single rotation step.

    The LLM sets ``orientation_mode`` to either a numeric degree value
    (interpreted as a relative rotation via ``rotate_by_degrees``) or a
    cardinal direction string ("north", "east", ...) dispatched to
    ``orient_to_direction``.
    """
    mode = (intent.orientation_mode or "").strip()
    try:
        degrees = float(mode)
        return [
            SkillCall(
                step_id="step_01",
                skill="rotate_by_degrees",
                args={"degrees": degrees},
                timeout_s=30.0,
                retry_limit=0,
            ),
        ]
    except ValueError:
        pass

    return [
        SkillCall(
            step_id="step_01",
            skill="orient_to_direction",
            args={"direction": mode or "north"},
            timeout_s=30.0,
            retry_limit=0,
        ),
    ]


def _compile_go_to_targets(intent: MissionIntent) -> list[SkillCall]:
    """Chain scan→project→navigate→verify per target in the ``targets`` list."""
    if not intent.targets:
        raise CompilationError("go_to_targets requires a non-empty targets list.")
    steps: list[SkillCall] = []
    for i, target in enumerate(intent.targets):
        label = str(target["label"])
        standoff = float(target.get("standoff_m", 0.7))
        base = i * _STEPS_PER_TARGET + 1
        steps.extend(
            _compile_single_target_steps(
                label=label, base=base, standoff_m=standoff,
            )
        )
    return steps


def _compile_describe(intent: MissionIntent) -> list[SkillCall]:
    return [
        SkillCall(
            step_id="step_01",
            skill="describe_scene",
            args={"prompt": intent.raw_command},
            timeout_s=30.0,
            retry_limit=0,
        ),
    ]


def _compile_query(intent: MissionIntent) -> list[SkillCall]:
    return [
        SkillCall(
            step_id="step_01",
            skill="query_environment",
            args={"query": intent.raw_command},
            timeout_s=5.0,
            retry_limit=0,
        ),
    ]


def _compile_patrol(intent: MissionIntent) -> list[SkillCall]:
    """Compile a patrol by chaining single-target visits per waypoint.

    Loop control is handled at the MissionRunner level — the compiler emits
    one pass through all targets.
    """
    if not intent.targets:
        raise CompilationError("patrol requires a non-empty targets list.")
    steps: list[SkillCall] = []
    for i, target in enumerate(intent.targets):
        label = str(target["label"])
        standoff = float(target.get("standoff_m", 0.7))
        base = i * _STEPS_PER_TARGET + 1
        steps.extend(
            _compile_single_target_steps(
                label=label, base=base, standoff_m=standoff,
            )
        )
    return steps


_COMPILERS: dict[str, Callable[[MissionIntent], list[SkillCall]]] = {
    "go_to_target": _compile_go_to_target,
    "wait_by_target": _compile_wait_by_target,
    "cancel": _compile_cancel,
    "status": _compile_status,
    "rotate": _compile_rotate,
    "go_to_targets": _compile_go_to_targets,
    "describe": _compile_describe,
    "query": _compile_query,
    "patrol": _compile_patrol,
}
