"""Hard-negative goal perturbation for capture drivers.

Implements the ``--inject-bad-grounding`` contract: with a per-mission
probability, swap the mission's physical goal to a deliberately-wrong
object while the recorded ``mission_text`` keeps naming the original
target. The resulting episode is a *grounding* hard negative — the
language says one thing, the trajectory does another.

Modes (all swap the physical goal; the recorded ``mission_text`` keeps
naming the original target):

- ``wrong_room``: the goal moves to a randomly-selected object in a
  different room.
- ``wrong_instance``: the goal moves to another same-label object in
  the same room; falls back to ``wrong_room`` when the target has no
  same-label sibling there.
- ``wrong_object``: the goal moves to a different-label object in the
  same room (category confusion, e.g. "go to the chair" → drive to the
  sofa); falls back to ``wrong_room`` when the room has no other-label
  object.

The plan distinguishes the *requested* mode from the *actual* mode
after fallback. When no candidate exists at all (single-room scene,
unique label, nothing in another room), the perturbation silently
drops: ``injection_mode`` stays set to the request while
``injection_mode_actual`` and ``original_target_position_3d`` are
``None`` — letting consumers audit the per-scene drop rate. Downstream
training filters MUST key off ``injection_mode_actual``.

Pure Python. Candidates come from the scene's ``objects[]`` records
(``label`` / ``instance_id`` / ``position_3d`` / ``room_idx``);
deterministic given the caller's ``random.Random``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


INJECTION_MODES = ("wrong_room", "wrong_instance", "wrong_object")

# Centroid radius (metres) within which a same-label object is treated as
# "the original target itself" and excluded from the candidate pool. Only
# consulted when the mission carries no instance id to exclude by (queue
# rows don't). This is NOT the same threshold as
# ``resolve_target_room_idx``'s ``match_radius_m`` (1.0 m): that one matches
# a labelled-and-positioned target to its nearest scene object to recover a
# room index, a looser association; this one is a tight self-identity guard.
_SAME_OBJECT_RADIUS_M = 0.25


@dataclass(frozen=True)
class InjectionPlan:
    """Resolved perturbation for one mission.

    ``target_*`` always describe the goal the driver should actually
    pursue: the perturbed object when the injection landed, the
    original target otherwise. ``injected`` is True only when a swap
    actually happened (``injection_mode_actual`` is set).
    """

    injection_mode: str | None
    injection_mode_actual: str | None
    target_label: str
    target_instance_id: int
    target_position_3d: tuple[float, float, float]
    original_target_position_3d: tuple[float, float, float] | None

    @property
    def injected(self) -> bool:
        return self.injection_mode_actual is not None


def _normalize(label: str) -> str:
    return str(label).strip().lower()


def _object_fields(obj: Mapping[str, Any]) -> tuple[str, int, tuple[float, float, float], int | None]:
    position = obj.get("position_3d") or (0.0, 0.0, 0.0)
    room_idx = obj.get("room_idx")
    return (
        str(obj.get("label", "")).strip(),
        int(obj.get("instance_id", -1)),
        (float(position[0]), float(position[1]), float(position[2])),
        int(room_idx) if room_idx is not None else None,
    )


def resolve_target_room_idx(
    *,
    target_label: str,
    target_position_3d: Sequence[float],
    objects: Sequence[Mapping[str, Any]],
    match_radius_m: float = 1.0,
) -> int | None:
    """Best-effort room index for a target known only by label + position.

    Mission-queue rows name rooms by string, not index; this matches the
    target against ``objects[]`` by label and XY proximity and returns
    the nearest match's ``room_idx``. ``None`` when nothing matches
    within ``match_radius_m``.
    """
    normalized = _normalize(target_label)
    tx, ty = float(target_position_3d[0]), float(target_position_3d[1])
    best: tuple[float, int | None] | None = None
    for obj in objects:
        label, _instance, position, room_idx = _object_fields(obj)
        if _normalize(label) != normalized:
            continue
        dist = math.hypot(position[0] - tx, position[1] - ty)
        if dist <= match_radius_m and (best is None or dist < best[0]):
            best = (dist, room_idx)
    return best[1] if best is not None else None


def plan_injection(
    *,
    mode: str,
    probability: float,
    rng: random.Random,
    target_label: str,
    target_instance_id: int,
    target_position_3d: Sequence[float],
    target_room_idx: int | None,
    objects: Sequence[Mapping[str, Any]],
) -> InjectionPlan:
    """Decide whether and how to perturb one mission's goal.

    ``mode`` is ``"off"`` or one of :data:`INJECTION_MODES`. The rng is
    consumed for the probability draw and the candidate pick, so a
    seeded ``random.Random`` reproduces a capture run's injection
    sequence exactly.
    """
    original = (
        float(target_position_3d[0]),
        float(target_position_3d[1]),
        float(target_position_3d[2]),
    )
    no_injection = InjectionPlan(
        injection_mode=None,
        injection_mode_actual=None,
        target_label=target_label,
        target_instance_id=int(target_instance_id),
        target_position_3d=original,
        original_target_position_3d=None,
    )
    if mode in (None, "off"):
        return no_injection
    if mode not in INJECTION_MODES:
        raise ValueError(f"unknown injection mode {mode!r}; valid: {INJECTION_MODES}")
    if rng.random() >= float(probability):
        return no_injection

    normalized_target = _normalize(target_label)
    same_room_siblings: list[tuple[str, int, tuple[float, float, float]]] = []
    same_room_diff_label: list[tuple[str, int, tuple[float, float, float]]] = []
    other_room: list[tuple[str, int, tuple[float, float, float]]] = []
    for obj in objects:
        label, instance_id, position, room_idx = _object_fields(obj)
        if not label:
            continue
        # The original target is excluded by instance id when known, and
        # by proximity otherwise (queue rows carry no instance id).
        if _normalize(label) == normalized_target and (
            instance_id == int(target_instance_id)
            or math.hypot(position[0] - original[0], position[1] - original[1])
            < _SAME_OBJECT_RADIUS_M
        ):
            continue
        same_room = target_room_idx is not None and room_idx == target_room_idx
        if same_room and _normalize(label) == normalized_target:
            same_room_siblings.append((label, instance_id, position))
        elif same_room and _normalize(label) != normalized_target:
            same_room_diff_label.append((label, instance_id, position))
        elif (
            room_idx is not None
            and target_room_idx is not None
            and room_idx != target_room_idx
        ):
            other_room.append((label, instance_id, position))

    # Per-mode candidate pool, each with a wrong_room fallback (a
    # different-room object is always a coherent grounding negative when the
    # requested in-room candidate doesn't exist).
    actual = mode
    if mode == "wrong_instance":
        candidates = same_room_siblings
    elif mode == "wrong_object":
        candidates = same_room_diff_label
    else:  # wrong_room
        candidates = other_room
    if mode in ("wrong_instance", "wrong_object") and not candidates:
        actual = "wrong_room"
        candidates = other_room
    if not candidates:
        # Perturbation silently drops; injection_mode stays set so the
        # per-scene drop rate is auditable downstream.
        return InjectionPlan(
            injection_mode=mode,
            injection_mode_actual=None,
            target_label=target_label,
            target_instance_id=int(target_instance_id),
            target_position_3d=original,
            original_target_position_3d=None,
        )

    label, instance_id, position = rng.choice(sorted(candidates, key=lambda c: (c[0], c[1])))
    return InjectionPlan(
        injection_mode=mode,
        injection_mode_actual=actual,
        target_label=label,
        target_instance_id=instance_id,
        target_position_3d=position,
        original_target_position_3d=original,
    )
