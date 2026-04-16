"""Mission specifications derived from Infinigen ``scene_metadata.json``.

Reads the per-scene metadata JSON written by
``scripts/extract_scene_metadata.py`` and produces a stream of
natural-language navigation missions targeting individual labeled
objects in the scene. The Jetson executor consumes the
``raw_command`` field unchanged via its ``execute_mission`` ROS2 action.

Pure Python — no Isaac Sim, no ROS, no Infinigen runtime imports.
Importable from ``.venv_vlm`` for unit tests.

scene_metadata.json schema (mirrors ``extract_scene_metadata`` dataclasses)::

    {
      "rooms": [
        {"room_type": "Kitchen", "footprint_xy": [[x,y], ...],
         "area_m2": 12.5, "story": 0},
        ...
      ],
      "objects": [
        {"instance_id": 42, "label": "Chair",
         "semantic_tags": ["seating", "wood"],
         "prim_path": "/World/Chair_42",
         "position_3d": [x, y, z],
         "bbox_3d_min": [...], "bbox_3d_max": [...],
         "room_idx": 0,
         "relations": [...], "materials": [...]},
        ...
      ],
      "room_adjacency": {...}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class MissionSpec:
    """One navigation mission targeting a single object in a scene."""

    mission_id: str
    scene_name: str
    target_label: str
    target_instance_id: int
    target_position_3d: tuple[float, float, float]
    target_room_idx: int | None
    raw_command: str
    # Free-form metadata copied from the source object record so the
    # harness can save it on every frame without re-reading the JSON.
    target_semantic_tags: tuple[str, ...] = ()
    target_prim_path: str | None = None


class MissionGenerator:
    """Iterates ``MissionSpec`` instances from a ``scene_metadata.json``.

    The generator does NOT pick goal poses — that is the harness's job
    once it can query Nav2's costmap. Mission generation only commits to
    *which* objects to navigate toward; the harness translates each
    spec into ``raw_command`` for the executor and the executor's planner
    + Nav2 stack figure out the actual goal pose from the natural-language
    prompt and the live SLAM map.

    Typical usage::

        gen = MissionGenerator.from_metadata_path(
            scene_metadata_path=Path("Assets/generated/scenes/kitchen_01/scene_metadata.json"),
            scene_name="kitchen_01",
            max_missions=20,
        )
        for spec in gen:
            ...  # submit to harness
    """

    def __init__(
        self,
        *,
        scene_name: str,
        objects: list[dict],
        rooms: list[dict] | None = None,
        max_missions: int | None = None,
        allowed_labels: Iterable[str] | None = None,
        blocked_labels: Iterable[str] = (),
        seed: int = 0,
    ) -> None:
        self._scene_name = scene_name
        self._objects = list(objects)
        self._rooms = list(rooms or [])
        self._max_missions = max_missions
        self._allowed_labels = (
            {self._normalize_label(l) for l in allowed_labels}
            if allowed_labels is not None
            else None
        )
        self._blocked_labels = {self._normalize_label(l) for l in blocked_labels}
        self._seed = int(seed)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_metadata_path(
        cls,
        *,
        scene_metadata_path: Path | str,
        scene_name: str | None = None,
        **kwargs,
    ) -> "MissionGenerator":
        """Load metadata from disk and return a generator for the scene.

        ``scene_name`` defaults to the parent directory name (matching the
        layout produced by ``extract_scene_metadata.py``, which writes
        ``Assets/generated/scenes/<scene_name>/scene_metadata.json``).
        """

        path = Path(scene_metadata_path)
        if not path.is_file():
            raise FileNotFoundError(f"scene_metadata.json not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))

        if scene_name is None:
            scene_name = path.parent.name

        return cls(
            scene_name=scene_name,
            objects=list(data.get("objects") or []),
            rooms=list(data.get("rooms") or []),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[MissionSpec]:
        # Deterministic ordering: sort by (label, instance_id) before
        # filtering. This keeps test fixtures stable and means a reader
        # of the dataset can predict mission order from the metadata.
        ordered = sorted(
            self._objects,
            key=lambda o: (
                self._normalize_label(o.get("label", "")),
                int(o.get("instance_id", 0)),
            ),
        )

        emitted = 0
        for obj in ordered:
            label = str(obj.get("label", "")).strip()
            if not label:
                continue
            normalized = self._normalize_label(label)
            if normalized in self._blocked_labels:
                continue
            if self._allowed_labels is not None and normalized not in self._allowed_labels:
                continue

            position = obj.get("position_3d") or [0.0, 0.0, 0.0]
            if len(position) < 3:
                continue
            target_position = (
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )

            instance_id = int(obj.get("instance_id", -1))
            mission_id = f"{self._scene_name}__{normalized}__{instance_id}"
            raw_command = self._build_raw_command(label)

            yield MissionSpec(
                mission_id=mission_id,
                scene_name=self._scene_name,
                target_label=label,
                target_instance_id=instance_id,
                target_position_3d=target_position,
                target_room_idx=(
                    int(obj["room_idx"])
                    if obj.get("room_idx") is not None
                    else None
                ),
                raw_command=raw_command,
                target_semantic_tags=tuple(obj.get("semantic_tags") or ()),
                target_prim_path=obj.get("prim_path"),
            )

            emitted += 1
            if self._max_missions is not None and emitted >= self._max_missions:
                return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Lowercase + strip so allow / block lists are case-insensitive."""

        return str(label).strip().lower()

    @staticmethod
    def _build_raw_command(label: str) -> str:
        """Produce the natural-language command the executor will plan over.

        The phrasing matches what real operators submit to the executor
        via ``strafer-autonomy-cli submit "..."``. Kept simple on purpose:
        a more elaborate command (``"go to the nearest red chair"``)
        would constrain the planner's interpretation in ways we can't
        validate without a richer label vocabulary.
        """

        return f"go to the {label.strip().lower()}"
