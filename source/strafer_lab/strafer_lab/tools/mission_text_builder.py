"""Scene-source-agnostic referring-expression builder for mission text.

Given a target object and the full object/room lists of a scene, produce a
single *anchor noun phrase* that picks the target out of its same-label
competitors — so a scene with 233 shelves does not emit the same
``the shelf`` string for every one of them.

The builder is a **pure function over plain dicts**: it never opens a USD,
never imports a renderer, and never inspects a source-specific name token. It
reads only the documented ``objects[]`` fields (``label``, ``position_3d``, the
flat ``bbox_3d_min`` / ``bbox_3d_max`` keys) and treats ``label`` as opaque
text. The one shared dependency is the structural-class vocabulary in
:mod:`strafer_lab.tools.scene_classes`, used to drop non-navigable surfaces
from the competitor pool the same way the mission generator does.

Disambiguation is a least-effort qualifier waterfall (Dale & Reiter style):
the cheapest qualifier that strictly shrinks the competitor set is accumulated,
and the search stops the instant exactly one candidate remains. Qualifiers
compose into a conjunction when no single one suffices, and a terminal
coordinate escape valve guarantees string-uniqueness when nothing groundable
remains.

Hard groundability constraint: a holonomic mecanum robot decouples heading from
travel and the anchor is authored before any trajectory exists, so the phrase
carries NO sidedness, cardinal, or surface-bearing words. Every emitted surface
form is allocentric (lowest/highest, largest/smallest, next-to a uniquely-named
neighbour) or the coordinate escape valve.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from strafer_lab.tools import scene_classes

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorResult:
    """One target's referring expression plus how it was resolved.

    ``text`` is the anchor noun phrase interpolated into mission text.
    ``groundable`` is True for every tier except the terminal coordinate
    escape valve — a camera-grounded model cannot read a raw coordinate off an
    image, so a coordinate anchor is flagged un-groundable. ``tier`` names the
    qualifier that resolved the target (see :data:`TIERS`).
    """

    text: str
    groundable: bool
    tier: str


# Resolution tiers, cheapest first. ``room_scope`` / ``region_scope`` are
# reserved no-ops (see the waterfall docstring) and never appear as a result
# tier in v1.
TIER_SINGLETON = "singleton"
TIER_Z_EXTREMUM = "z_extremum"
TIER_SIZE_EXTREMUM = "size_extremum"
TIER_NEAREST_NEIGHBOR = "nearest_neighbor"
TIER_CONJUNCTION = "conjunction"
TIER_COORDINATE = "coordinate_fallback"

TIERS = (
    TIER_SINGLETON,
    TIER_Z_EXTREMUM,
    TIER_SIZE_EXTREMUM,
    TIER_NEAREST_NEIGHBOR,
    TIER_CONJUNCTION,
    TIER_COORDINATE,
)

# A neighbour anchor only reads as "next to" when it is genuinely adjacent;
# beyond this centroid distance the phrase stops being groundable, so the tier
# is skipped. Tunable, deterministic.
NEIGHBOR_PROXIMITY_M = 2.0

# At most three qualifiers may be conjoined before the coordinate escape valve.
MAX_CONJUNCTS = 3

# The same ungroundable-word ban the mission generator enforces on its text.
# Applied as a guard on neighbour anchors so a stray bearing-ish label can
# never leak into an anchor.
_BANNED_WORDS = re.compile(r"\b(north|south|east|west|left|right|wall)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Field accessors (plain dicts; flat bbox keys; label treated as opaque)
# ---------------------------------------------------------------------------


def _normalize(label: Any) -> str:
    """Trivial label normalization shared by every mission-text consumer."""
    return str(label).strip().lower()


def _valid_position(obj: dict[str, Any]) -> tuple[float, float, float] | None:
    """Return ``(x, y, z)`` or ``None`` for a missing / origin-sentinel position."""
    pos = obj.get("position_3d")
    if not isinstance(pos, (list, tuple)) or len(pos) < 3:
        return None
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if abs(x) < 1e-3 and abs(y) < 1e-3 and abs(z) < 1e-3:
        return None  # origin sentinel == no valid position
    return (x, y, z)


def _bbox_volume(obj: dict[str, Any]) -> float | None:
    """Axis-aligned bbox volume from the flat ``bbox_3d_min`` / ``bbox_3d_max`` keys.

    Returns ``None`` when either key is absent or the box is degenerate (a
    non-positive extent on any axis), so the size tier skips the object cleanly.
    """
    mn = obj.get("bbox_3d_min")
    mx = obj.get("bbox_3d_max")
    if not (isinstance(mn, (list, tuple)) and isinstance(mx, (list, tuple))):
        return None
    if len(mn) < 3 or len(mx) < 3:
        return None
    dims = [float(mx[i]) - float(mn[i]) for i in range(3)]
    if any(d <= 0.0 for d in dims):
        return None
    return dims[0] * dims[1] * dims[2]


def _euclidean(a: dict[str, Any], b: dict[str, Any]) -> float:
    pa, pb = _valid_position(a), _valid_position(b)
    return math.dist(pa, pb)


def _target_pool(all_objects: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The post-filter target set: labelled, non-structural, validly-positioned.

    Mirrors the mission generator's target filter so the competitor set the
    anchor is disambiguated against is exactly the set that becomes missions.
    """
    pool: list[dict[str, Any]] = []
    for obj in all_objects:
        label = _normalize(obj.get("label", ""))
        if not label or label in scene_classes.STRUCTURAL_CLASSES:
            continue
        if _valid_position(obj) is None:
            continue
        pool.append(obj)
    return pool


# ---------------------------------------------------------------------------
# Qualifier tiers
# ---------------------------------------------------------------------------


def _strict_extreme(target: dict[str, Any], group: Sequence[dict[str, Any]], key) -> str | None:
    """``"low"`` / ``"high"`` if ``target`` is the lone min / max of ``group`` by ``key``.

    Requires every member of ``group`` to have a value (so an unknown-size
    competitor can never let the target falsely claim to be the largest), and a
    strict, untied extreme so the superlative names exactly one object.
    """
    values = [(key(o), o) for o in group]
    tval = key(target)
    if tval is None or any(v is None for v, _ in values):
        return None
    lo = min(v for v, _ in values)
    hi = max(v for v, _ in values)
    if sum(1 for v, _ in values if v == lo) == 1 and tval == lo:
        return "low"
    if sum(1 for v, _ in values if v == hi) == 1 and tval == hi:
        return "high"
    return None


def _unique_label_anchors(pool: Sequence[dict[str, Any]], exclude_label: str) -> list[dict[str, Any]]:
    """Objects whose label is globally unique in the pool — referenceable as ``the {label}``.

    Excludes the target's own label (anti-stutter) and any label carrying a
    banned bearing word, so ``next to the {neighbour}`` always grounds to one
    object and never smuggles in sidedness.
    """
    counts = Counter(_normalize(o.get("label", "")) for o in pool)
    anchors: list[dict[str, Any]] = []
    for obj in pool:
        label = _normalize(obj.get("label", ""))
        if counts[label] != 1 or label == exclude_label:
            continue
        if _BANNED_WORDS.search(label):
            continue
        anchors.append(obj)
    return anchors


def _nearest_anchor(
    obj: dict[str, Any], anchors: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    """The nearest unique-label anchor within :data:`NEIGHBOR_PROXIMITY_M`, or ``None``.

    Deterministic: ties in distance break on the anchor's position then label so
    the choice never depends on list order or any opaque id.
    """
    candidates: list[tuple[float, tuple[float, float, float], str, dict[str, Any]]] = []
    for anchor in anchors:
        dist = _euclidean(obj, anchor)
        if dist <= NEIGHBOR_PROXIMITY_M:
            candidates.append((dist, _valid_position(anchor), _normalize(anchor.get("label", "")), anchor))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    return candidates[0][3]


# ---------------------------------------------------------------------------
# Coordinate escape valve
# ---------------------------------------------------------------------------


def _coordinate_anchor(label: str, pos: tuple[float, float, float]) -> str:
    """Terminal escape-valve phrasing keyed on the full-precision position.

    Distinct float positions yield distinct strings (round-trippable ``repr``),
    so this is the sole guarantee that every produced anchor is unique when no
    groundable qualifier remains. Un-groundable by construction.
    """
    x, y, z = pos
    return f"the {label} approximately at ({float(x)!r}, {float(y)!r}, {float(z)!r})"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def disambiguate(
    target_obj: dict[str, Any],
    all_objects: Sequence[dict[str, Any]],
    rooms: Sequence[dict[str, Any]],
) -> AnchorResult:
    """Build the anchor noun phrase that uniquely names ``target_obj`` in its scene.

    Pure function. ``all_objects`` is the scene's raw ``objects[]`` list (the
    same post-filter target set the generator builds is derived internally);
    ``rooms`` is accepted for the reserved room/region tiers and is unused while
    those remain no-ops.

    Qualifier waterfall, cheapest first, stopping the instant one candidate
    remains:

    1. ``room_scope`` — reserved no-op. Object room membership is null in the
       embedded metadata, so it would have to be recomputed point-in-polygon
       against ``rooms[].footprint_xy``; and a room surface form is outside the
       v1 anchor grammar. Skipped, never an error.
    2. ``region_scope`` — reserved no-op. Any within-room region split could
       only be phrased with the banned bearing words, so it authors nothing.
    3. ``z_extremum`` — the lone lowest / highest competitor by vertical
       position (``the lowest {label}`` / ``the highest {label}``).
    4. ``size_extremum`` — the lone largest / smallest competitor by bbox volume
       (``the largest {label}`` / ``the smallest {label}``).
    5. ``nearest_neighbor`` — ``the {label} next to the {neighbour}`` where the
       neighbour is the nearest uniquely-named object and the phrase partitions
       the competitors.
    6. conjunction of the above (``the largest {label} next to the {neighbour}``).
    7. coordinate escape valve — terminal, un-groundable.

    A qualifier is composed only if it strictly reduces the competitor set.
    """
    label = _normalize(target_obj.get("label", ""))
    tpos = _valid_position(target_obj)

    # Every qualifier this builder authors is allocentric by construction, so a
    # banned bearing/surface word can only reach the anchor through the opaque
    # object label. The upstream label vocabulary yields bearing-free category
    # nouns and structural surfaces are filtered from the pool below, so this
    # never fires on a real scene; when it does, degrade to a label-free,
    # un-groundable coordinate anchor rather than emit a banned word, so the
    # no-bearing-word guarantee holds on the output however dirty the input is.
    if _BANNED_WORDS.search(label):
        text = _coordinate_anchor("object", tpos) if tpos is not None else "the object"
        return AnchorResult(text=text, groundable=False, tier=TIER_COORDINATE)

    pool = _target_pool(all_objects)
    competitors = [o for o in pool if _normalize(o.get("label", "")) == label and o is not target_obj]

    # No same-label competitor, or a target we cannot position: the bare phrase
    # already names it. Byte-identical to the historical ``the {label}`` seam.
    if not competitors or tpos is None:
        return AnchorResult(text=f"the {label}", groundable=True, tier=TIER_SINGLETON)

    anchors = _unique_label_anchors(pool, exclude_label=label)

    confusable = [target_obj, *competitors]
    adjectives: list[str] = []  # superlatives, rendered before the label
    prepositions: list[str] = []  # "next to the ..." clauses, rendered after
    used: list[str] = []

    while len(confusable) > 1 and len(used) < MAX_CONJUNCTS:
        progressed = False
        for tier in (TIER_Z_EXTREMUM, TIER_SIZE_EXTREMUM, TIER_NEAREST_NEIGHBOR):
            if tier in used:
                continue

            if tier == TIER_Z_EXTREMUM:
                extreme = _strict_extreme(target_obj, confusable, lambda o: _valid_position(o)[2])
                if extreme is not None:
                    adjectives.append("lowest" if extreme == "low" else "highest")
                    confusable = [target_obj]
                    used.append(tier)
                    progressed = True
                    break

            elif tier == TIER_SIZE_EXTREMUM:
                extreme = _strict_extreme(target_obj, confusable, _bbox_volume)
                if extreme is not None:
                    adjectives.append("largest" if extreme == "high" else "smallest")
                    confusable = [target_obj]
                    used.append(tier)
                    progressed = True
                    break

            else:  # TIER_NEAREST_NEIGHBOR
                neighbor = _nearest_anchor(target_obj, anchors)
                if neighbor is None:
                    continue
                kept = [o for o in confusable if _nearest_anchor(o, anchors) is neighbor]
                if len(kept) < len(confusable) and target_obj in kept:
                    prepositions.append(f"next to the {_normalize(neighbor.get('label', ''))}")
                    confusable = kept
                    used.append(tier)
                    progressed = True
                    break

        if not progressed:
            break

    if len(confusable) == 1:
        prefix = (" ".join(adjectives) + " ") if adjectives else ""
        text = f"the {prefix}{label}"
        for clause in prepositions:
            text += f" {clause}"
        tier = used[0] if len(used) == 1 else TIER_CONJUNCTION
        return AnchorResult(text=text, groundable=True, tier=tier)

    # Nothing groundable partitioned the competitors: terminal coordinate anchor.
    return AnchorResult(
        text=_coordinate_anchor(label, tpos), groundable=False, tier=TIER_COORDINATE
    )


# ---------------------------------------------------------------------------
# Measurement harness (repeatable; reads a captured scene dict, no USD needed)
# ---------------------------------------------------------------------------


def measure_scene(metadata: dict[str, Any]) -> dict[str, Any]:
    """Resolve every post-filter target and report how the anchors landed.

    Returns the per-tier histogram, the coordinate-fallthrough rate (fraction of
    targets whose anchor is un-groundable), the per-label tier breakdown, and
    whether every produced anchor string was distinct. This is the number the
    follow-up filter-vs-emit decision turns on.
    """
    all_objects = list(metadata.get("objects") or [])
    rooms = list(metadata.get("rooms") or [])
    pool = _target_pool(all_objects)

    hist: Counter[str] = Counter()
    by_label: dict[str, Counter[str]] = {}
    texts: list[str] = []
    for obj in pool:
        result = disambiguate(obj, all_objects, rooms)
        hist[result.tier] += 1
        by_label.setdefault(_normalize(obj.get("label", "")), Counter())[result.tier] += 1
        texts.append(result.text)

    total = len(pool)
    fallthrough = hist[TIER_COORDINATE]
    return {
        "total_targets": total,
        "tier_histogram": dict(hist),
        "coordinate_fallthrough": fallthrough,
        "coordinate_fallthrough_rate": (fallthrough / total) if total else 0.0,
        "by_label": {k: dict(v) for k, v in by_label.items()},
        "all_texts_distinct": len(set(texts)) == len(texts),
    }


def _format_report(name: str, summary: dict[str, Any], *, worst_n: int = 10) -> str:
    total = summary["total_targets"]
    lines = [f"=== {name}: {total} targets ==="]
    for tier in TIERS:
        count = summary["tier_histogram"].get(tier, 0)
        pct = (count / total) if total else 0.0
        lines.append(f"  {tier:22s} {count:5d}  ({pct:6.1%})")
    rate = summary["coordinate_fallthrough_rate"]
    lines.append(f"  >> coordinate-fallthrough rate = {summary['coordinate_fallthrough']}/{total} = {rate:.1%}")
    lines.append(f"  all anchor strings distinct: {summary['all_texts_distinct']}")
    worst = sorted(
        summary["by_label"].items(),
        key=lambda kv: -kv[1].get(TIER_COORDINATE, 0),
    )[:worst_n]
    lines.append(f"  worst offenders (label: total -> coordinate / groundable):")
    for label, tiers in worst:
        tot = sum(tiers.values())
        coord = tiers.get(TIER_COORDINATE, 0)
        lines.append(f"    {label:18s} {tot:5d} -> {coord:5d} coord, {tot - coord:4d} groundable")
    return "\n".join(lines)


def _load_scene_dict(path: str) -> dict[str, Any]:
    """Load a scene metadata dict from a captured JSON fixture or a scene USD."""
    import json
    from pathlib import Path

    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    from strafer_lab.tools.scene_metadata_reader import load as load_usd

    return load_usd(p)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m strafer_lab.tools.mission_text_builder <scene.json|scene.usdc> ...``"""
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print(__doc__)
        print("usage: python -m strafer_lab.tools.mission_text_builder <scene.json|scene.usdc> ...")
        return 0
    for path in args:
        metadata = _load_scene_dict(path)
        from pathlib import Path

        print(_format_report(Path(path).stem, measure_scene(metadata)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
