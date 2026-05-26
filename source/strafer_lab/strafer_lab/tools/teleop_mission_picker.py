"""Mission target picker for ``--driver teleop --mission-source scene-metadata``.

Reads ``scene_metadata.json``, enumerates the ``objects[]`` array, and
exposes both:

- A :class:`MissionCandidate` list for the console picker the driver
  presents to the operator (numeric prompt, prints labels and rooms).
- A pure-Python ``select_by_index`` so unit tests can exercise the
  filter / sort / lookup paths without a TTY.

UX choice (v1, documented in the harness brief): **numeric-prompt
console picker**. The operator types ``5<enter>`` on the terminal to
select index 5. Cycle-through-D-pad was the alternative; the brief
explicitly says "v1 design choice yours — document it" and notes that
free-form mission text is a follow-up (Tier 1.5). Numeric prompt is
strictly simpler to implement and to test, and the operator already has
a hand free between episodes — they're not driving while picking.

This module is pure-Python and importable from ``.venv_harness``. The
console-IO portion is isolated behind :func:`prompt_for_target` so the
unit tests can exercise the picker logic without faking stdin.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# Default labels skipped — match run_sim_in_the_loop.py's harness mode.
_DEFAULT_BLOCKED_LABELS: frozenset[str] = frozenset({"wall", "floor", "ceiling"})


@dataclass(frozen=True)
class MissionCandidate:
    """One pickable target from ``scene_metadata.json:objects[]``."""

    index: int                       # 0-based index in the candidate list
    instance_id: int
    label: str
    target_position_3d: tuple[float, float, float]
    target_room_idx: int | None
    target_room_type: str | None     # resolved from rooms[] when available
    prim_path: str | None
    mission_text: str                # canonical "go to the <label>" string

    def short_display(self) -> str:
        """One-line console rendering for the picker prompt."""
        room_suffix = (
            f"  ({self.target_room_type})" if self.target_room_type else ""
        )
        return (
            f"  [{self.index:3d}] {self.label:24s}  "
            f"id={self.instance_id:<5d}  "
            f"pos=({self.target_position_3d[0]:+.2f}, "
            f"{self.target_position_3d[1]:+.2f}, "
            f"{self.target_position_3d[2]:+.2f})"
            f"{room_suffix}"
        )


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()


def _build_room_lookup(rooms: Sequence[dict]) -> dict[int, str]:
    """Map ``room_idx`` → ``room_type`` for display only."""
    lookup: dict[int, str] = {}
    for idx, raw in enumerate(rooms or ()):
        if not isinstance(raw, dict):
            continue
        room_type = str(raw.get("room_type") or "").strip()
        lookup[idx] = room_type
    return lookup


def load_candidates(
    scene_metadata_path: Path | str,
    *,
    allowed_labels: Iterable[str] | None = None,
    blocked_labels: Iterable[str] | None = None,
) -> list[MissionCandidate]:
    """Load + filter + sort the ``objects[]`` array into candidates.

    Sorting matches :class:`strafer_lab.sim_in_the_loop.MissionGenerator`:
    ``(normalized_label, instance_id)``. This keeps mission ordering
    stable across captures of the same scene and lets the operator
    rebuild muscle memory ("the chair is always at index 12") between
    sessions.

    The default block list ``{"wall", "floor", "ceiling"}`` matches the
    bridge harness's defaults so a teleop and bridge run on the same
    scene see comparable target counts.
    """
    path = Path(scene_metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"scene_metadata.json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return load_candidates_from_data(
        data,
        allowed_labels=allowed_labels,
        blocked_labels=blocked_labels,
    )


def load_candidates_from_data(
    data: dict,
    *,
    allowed_labels: Iterable[str] | None = None,
    blocked_labels: Iterable[str] | None = None,
) -> list[MissionCandidate]:
    """Pure-data variant of :func:`load_candidates` — for tests."""
    rooms = list(data.get("rooms") or [])
    objects = list(data.get("objects") or [])
    room_types_by_idx = _build_room_lookup(rooms)

    allowed = (
        {_normalize_label(l) for l in allowed_labels}
        if allowed_labels is not None else None
    )
    blocked = {
        _normalize_label(l)
        for l in (blocked_labels if blocked_labels is not None else _DEFAULT_BLOCKED_LABELS)
    }

    ordered = sorted(
        objects,
        key=lambda o: (
            _normalize_label(o.get("label", "")),
            int(o.get("instance_id", 0)),
        ),
    )

    out: list[MissionCandidate] = []
    for obj in ordered:
        label = str(obj.get("label", "")).strip()
        if not label:
            continue
        normalized = _normalize_label(label)
        if normalized in blocked:
            continue
        if allowed is not None and normalized not in allowed:
            continue

        position = obj.get("position_3d") or [0.0, 0.0, 0.0]
        if len(position) < 3:
            continue
        pos_3d = (float(position[0]), float(position[1]), float(position[2]))

        instance_id = int(obj.get("instance_id", -1))
        room_idx = obj.get("room_idx")
        room_idx_int = int(room_idx) if room_idx is not None else None

        out.append(
            MissionCandidate(
                index=len(out),
                instance_id=instance_id,
                label=label,
                target_position_3d=pos_3d,
                target_room_idx=room_idx_int,
                target_room_type=(
                    room_types_by_idx.get(room_idx_int) or None
                    if room_idx_int is not None else None
                ),
                prim_path=obj.get("prim_path"),
                mission_text=f"go to the {normalized}",
            ),
        )
    return out


def select_by_index(
    candidates: Sequence[MissionCandidate],
    raw_input: str,
) -> MissionCandidate | None:
    """Parse a numeric console-prompt response into one candidate.

    Returns ``None`` for empty / invalid / out-of-range input so the
    caller can re-prompt. Accepts leading whitespace and trailing
    newlines from ``input()``.
    """
    if not candidates:
        return None
    trimmed = (raw_input or "").strip()
    if not trimmed:
        return None
    try:
        idx = int(trimmed)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(candidates):
        return None
    return candidates[idx]


def prompt_for_target(
    candidates: Sequence[MissionCandidate],
    *,
    stream_in=None,
    stream_out=None,
) -> MissionCandidate | None:
    """Print the candidates and ask the operator to pick one.

    Returns ``None`` when the operator hits Ctrl-D / Ctrl-C — the driver
    loop should treat that as "no more episodes, save and quit".

    The two stream kwargs allow tests to pump in canned input without
    touching real stdin/stdout.
    """
    sin = stream_in if stream_in is not None else sys.stdin
    sout = stream_out if stream_out is not None else sys.stdout

    if not candidates:
        print(
            "[mission-picker] no targets in scene_metadata.json after filtering.",
            file=sout,
        )
        return None

    print("", file=sout)
    print("Pick a mission target by index (Ctrl-D to quit):", file=sout)
    for cand in candidates:
        print(cand.short_display(), file=sout)
    print("", file=sout)

    while True:
        try:
            print("target index> ", end="", file=sout, flush=True)
            line = sin.readline()
        except (KeyboardInterrupt, EOFError):
            return None
        if line == "":  # EOF
            return None
        choice = select_by_index(candidates, line)
        if choice is not None:
            return choice
        print(
            "  (invalid index; enter 0 .. "
            f"{len(candidates) - 1}, or Ctrl-D to quit)",
            file=sout,
        )
