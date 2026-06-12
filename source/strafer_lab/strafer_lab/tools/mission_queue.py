"""Parser for ``mission_queue.yaml`` mission rows.

A mission queue is a YAML list of rows, each describing one mission for
the capture drivers. The canonical producer is the offline mission
generator (``build_mission_queue.py`` once it ships); hand-authored
queues with the same shape are equally valid. Row shape::

    - mission_id: 0123
      scene_name: scene_high_quality_dgx_000_seed0
      scene_seed: 0                       # optional
      start_pose: {x: 0.5, y: 0.5, yaw: 0.0}   # optional, advisory
      target_label: chair
      target_position_3d: [4.2, 1.8, 0.0]
      target_room: living_room            # optional
      start_room: kitchen                 # optional
      cross_room: true                    # optional
      mission_text: "Go to the chair by hugging the south wall."
      paraphrases: ["...", "..."]         # optional
      planned_path:                       # optional; waypoint list
        - {x: 0.5, y: 0.4}
      generator_metadata: {...}           # optional, free-form

Per-driver consumption: the bridge dispatches ``mission_text`` through
the autonomy stack and ignores ``planned_path`` (the Jetson planner
emits its own); the scripted driver consumes ``planned_path`` as its
waypoint sequence; teleop displays ``mission_text`` to the operator.

Pure Python (+PyYAML). No Isaac Sim, no ROS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from strafer_lab.sim_in_the_loop.mission import MissionSpec


class MissionQueueError(ValueError):
    """Raised when a queue file is missing or a row is malformed."""


@dataclass(frozen=True)
class QueueMissionRow:
    """One parsed mission row. Optional source fields default to empty."""

    mission_id: str
    mission_text: str
    target_label: str
    target_position_3d: tuple[float, float, float]
    scene_name: str | None = None
    scene_seed: int | None = None
    start_pose: tuple[float, float, float] | None = None  # (x, y, yaw)
    target_room: str | None = None
    start_room: str | None = None
    cross_room: bool | None = None
    paraphrases: tuple[str, ...] = ()
    planned_path: tuple[tuple[float, float], ...] = ()
    generator_metadata: dict[str, Any] = field(default_factory=dict)


def _parse_xyz(value: Any, *, row_id: str, key: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise MissionQueueError(
            f"mission {row_id!r}: {key} must be a 3-element list, got {value!r}",
        )
    return (float(value[0]), float(value[1]), float(value[2]))


def _parse_start_pose(value: Any, *, row_id: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        try:
            return (
                float(value["x"]),
                float(value["y"]),
                float(value.get("yaw", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MissionQueueError(
                f"mission {row_id!r}: start_pose mapping needs numeric x/y "
                f"(+optional yaw), got {value!r}",
            ) from exc
    raise MissionQueueError(
        f"mission {row_id!r}: start_pose must be a mapping with x/y/yaw, "
        f"got {value!r}",
    )


def _parse_planned_path(value: Any, *, row_id: str) -> tuple[tuple[float, float], ...]:
    if not value:
        return ()
    waypoints: list[tuple[float, float]] = []
    for i, wp in enumerate(value):
        if isinstance(wp, Mapping) and "x" in wp and "y" in wp:
            waypoints.append((float(wp["x"]), float(wp["y"])))
        elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
            waypoints.append((float(wp[0]), float(wp[1])))
        else:
            raise MissionQueueError(
                f"mission {row_id!r}: planned_path[{i}] must be "
                f"{{x, y}} or [x, y], got {wp!r}",
            )
    return tuple(waypoints)


def parse_mission_row(raw: Mapping[str, Any]) -> QueueMissionRow:
    """Validate + coerce one raw YAML mapping into a :class:`QueueMissionRow`."""
    if not isinstance(raw, Mapping):
        raise MissionQueueError(f"queue row must be a mapping, got {type(raw).__name__}")

    row_id = str(raw.get("mission_id", "?"))
    missing = [
        key for key in ("mission_id", "mission_text", "target_label", "target_position_3d")
        if raw.get(key) in (None, "")
    ]
    if missing:
        raise MissionQueueError(
            f"mission {row_id!r}: missing required field(s) {missing}",
        )

    return QueueMissionRow(
        mission_id=row_id,
        mission_text=str(raw["mission_text"]).strip(),
        target_label=str(raw["target_label"]).strip(),
        target_position_3d=_parse_xyz(
            raw["target_position_3d"], row_id=row_id, key="target_position_3d",
        ),
        scene_name=(str(raw["scene_name"]) if raw.get("scene_name") else None),
        scene_seed=(int(raw["scene_seed"]) if raw.get("scene_seed") is not None else None),
        start_pose=_parse_start_pose(raw.get("start_pose"), row_id=row_id),
        target_room=(str(raw["target_room"]) if raw.get("target_room") else None),
        start_room=(str(raw["start_room"]) if raw.get("start_room") else None),
        cross_room=(bool(raw["cross_room"]) if raw.get("cross_room") is not None else None),
        paraphrases=tuple(str(p) for p in (raw.get("paraphrases") or ())),
        planned_path=_parse_planned_path(raw.get("planned_path"), row_id=row_id),
        generator_metadata=dict(raw.get("generator_metadata") or {}),
    )


def load_mission_queue(path: Path | str) -> list[QueueMissionRow]:
    """Load + validate every row of a ``mission_queue.yaml``.

    Raises :class:`MissionQueueError` on a missing file, a non-list
    document, or any malformed row (fail-fast — a half-valid queue is
    a generator bug, not something to capture around).
    """
    import yaml

    queue_path = Path(path)
    if not queue_path.is_file():
        raise MissionQueueError(f"mission queue not found: {queue_path}")
    document = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    if document is None:
        return []
    if not isinstance(document, list):
        raise MissionQueueError(
            f"{queue_path}: top-level YAML must be a list of mission rows, "
            f"got {type(document).__name__}",
        )
    return [parse_mission_row(raw) for raw in document]


def queue_row_to_mission_spec(row: QueueMissionRow, *, scene_name: str) -> MissionSpec:
    """Adapt a queue row to the :class:`MissionSpec` the harness dispatches.

    ``raw_command`` is the row's ``mission_text`` verbatim — the queue
    author owns the phrasing. ``target_instance_id`` is unknown to the
    queue schema and lands as ``-1``; ``target_room_idx`` is likewise
    unresolvable from the room *name* alone and is left ``None`` (the
    injection planner re-derives it from scene objects when needed).
    """
    return MissionSpec(
        mission_id=f"{scene_name}__queue__{row.mission_id}",
        scene_name=scene_name,
        target_label=row.target_label,
        target_instance_id=-1,
        target_position_3d=row.target_position_3d,
        target_room_idx=None,
        raw_command=row.mission_text,
    )
