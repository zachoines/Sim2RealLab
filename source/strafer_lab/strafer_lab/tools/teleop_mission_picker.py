"""Mission target picker for ``--driver teleop --mission-source scene-metadata``.

Reads ``scene_metadata.json``, enumerates the ``objects[]`` array, and
exposes both a :class:`MissionCandidate` list (for the console picker)
and a pure-Python ``select_by_index`` (for unit-testing the filter,
sort, and lookup paths without a TTY).

The picker presents a numeric prompt: the operator types ``5<enter>`` on
the terminal to select index 5. Console I/O is isolated behind
:func:`prompt_for_target` so the rest of the module is importable +
testable without stdin.
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
    dedup_by_instance: bool = True,
) -> list[MissionCandidate]:
    """Pure-data variant of :func:`load_candidates` — for tests.

    ``dedup_by_instance`` (default True): Infinigen authors multi-part
    objects as several sub-prims, each picked up as its own row by
    ``extract_scene_metadata.py``. A "bed" might appear as 3 rows at
    slightly different XY positions (mattress, frame, footboard). The
    picker is much friendlier when one logical object = one entry. We
    collapse by ``(normalized_label, instance_id)``, keeping the row
    with the median Z (a heuristic that lands on the object's centroid
    rather than its top or floor-contact point). Pass False to see all
    sub-prims (debug only).
    """
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

    # First pass: filter to acceptable objects + parse into normalized form.
    raw_rows: list[dict] = []
    for obj in objects:
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

        raw_rows.append({
            "label": label,
            "normalized": normalized,
            "instance_id": instance_id,
            "pos_3d": pos_3d,
            "room_idx_int": room_idx_int,
            "prim_path": obj.get("prim_path"),
        })

    # Second pass: optional dedup by (normalized_label, instance_id).
    # Multiple rows with the same (label, instance) are sub-prims of one
    # logical object — keep the median-Z row as canonical.
    if dedup_by_instance:
        from collections import defaultdict
        groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for row in raw_rows:
            groups[(row["normalized"], row["instance_id"])].append(row)
        deduped: list[dict] = []
        for key, members in groups.items():
            if len(members) == 1:
                deduped.append(members[0])
                continue
            # Pick the row with median Z. With an odd count this is the
            # middle one; with even, the lower-middle one (stable sort).
            members_sorted = sorted(members, key=lambda r: r["pos_3d"][2])
            mid = len(members_sorted) // 2
            deduped.append(members_sorted[mid])
        raw_rows = deduped

    # Third pass: sort + assign dense indices.
    raw_rows.sort(key=lambda r: (r["normalized"], r["instance_id"]))

    out: list[MissionCandidate] = []
    for row in raw_rows:
        room_idx_int = row["room_idx_int"]
        out.append(
            MissionCandidate(
                index=len(out),
                instance_id=row["instance_id"],
                label=row["label"],
                target_position_3d=row["pos_3d"],
                target_room_idx=room_idx_int,
                target_room_type=(
                    room_types_by_idx.get(room_idx_int) or None
                    if room_idx_int is not None else None
                ),
                prim_path=row["prim_path"],
                mission_text=f"go to the {row['normalized']}",
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
