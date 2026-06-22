"""Tests for the scene-source-agnostic mission-text anchor builder.

Pure Python — no Isaac Sim, no Kit, no ``pxr``. The dense real-scene snapshot
reads a frozen captured projection of ``scene_high_quality_dgx_000_seed2``'s
embedded ``customData`` (``fixtures/``), so the central global-uniqueness bar
and the coordinate-fallthrough measurement run without opening a USD. Per-tier
behaviour is pinned on small synthetic scenes engineered so each live tier is
individually necessary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from strafer_lab.tools import mission_text_builder as mtb
from strafer_lab.tools.mission_text_builder import AnchorResult, disambiguate, measure_scene

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_SEED2 = _FIXTURES / "scene_high_quality_dgx_000_seed2.json"

# Frozen measurement snapshot for seed2 (646 post-filter targets). This is the
# deliverable the filter-vs-emit follow-up turns on; update deliberately if the
# fixture or the waterfall changes.
SEED2_TARGETS = 646
SEED2_HISTOGRAM = {
    "singleton": 9,
    "z_extremum": 76,
    "size_extremum": 41,
    "nearest_neighbor": 12,
    "conjunction": 18,
    "coordinate_fallback": 490,
}
SEED2_FALLTHROUGH = 490

# The complete v1 anchor grammar — the only surface forms a groundable anchor
# may take. The no-bearing ban is a stricter test layered on top of this.
_ALLOWED_GROUNDABLE = re.compile(
    r"^the (?:(?:lowest|highest|largest|smallest) )?[^()]+?(?: next to the [^()]+)?$"
)
_COORDINATE = re.compile(r"^the .+ approximately at \([^)]*\)$")
_BANNED = re.compile(r"\b(north|south|east|west|left|right|wall)\b", re.IGNORECASE)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _obj(label, pos, *, bmin=None, bmax=None, iid=0):
    """A minimal object dict in the shape the builder reads (flat bbox keys)."""
    o = {"label": label, "position_3d": list(pos), "instance_id": iid}
    if bmin is not None:
        o["bbox_3d_min"] = list(bmin)
        o["bbox_3d_max"] = list(bmax)
    return o


def _cube(label, pos, side, *, iid=0):
    """An object at ``pos`` whose bbox encodes a cube of edge ``side``.

    The bbox is placed at the origin on purpose: the builder reads bbox solely
    for volume, so anchoring it at the origin makes equal ``side`` values give
    bit-identical volumes (a real tie) and distinct values an unambiguous order
    — float noise from ``pos ± half`` would otherwise break intended ties.
    """
    return _obj(label, pos, bmin=(0.0, 0.0, 0.0), bmax=(side, side, side), iid=iid)


# ---------------------------------------------------------------------------
# Central bar: global string-uniqueness on a real dense scene (seed2)
# ---------------------------------------------------------------------------


class TestGlobalUniquenessSeed2:
    def test_every_anchor_is_distinct_and_deterministic(self):
        md = _load(_SEED2)
        objects, rooms = md["objects"], md["rooms"]
        pool = mtb._target_pool(objects)
        assert len(pool) == SEED2_TARGETS

        first = [disambiguate(o, objects, rooms) for o in pool]
        again = [disambiguate(o, objects, rooms) for o in pool]

        texts = [r.text for r in first]
        # Central correctness bar: no two targets share a string.
        assert len(set(texts)) == len(texts)
        # Determinism: a second pass reproduces text + groundable + tier exactly.
        assert again == first

    def test_coordinate_escape_valve_is_terminal_and_guarantees_uniqueness(self):
        md = _load(_SEED2)
        objects, rooms = md["objects"], md["rooms"]
        pool = mtb._target_pool(objects)
        results = [disambiguate(o, objects, rooms) for o in pool]

        # The only un-groundable tier is the coordinate fallback, and it is the
        # sole guarantee of string-uniqueness for everything it catches.
        ungroundable = [r for r in results if not r.groundable]
        assert ungroundable, "expected the dense scene to exercise the escape valve"
        assert all(r.tier == "coordinate_fallback" for r in ungroundable)
        coord_texts = [r.text for r in ungroundable]
        assert all(_COORDINATE.match(t) for t in coord_texts)
        assert len(set(coord_texts)) == len(coord_texts)  # full-precision keys never collide

    def test_no_target_position_collisions_underpin_the_guarantee(self):
        # The escape valve only guarantees uniqueness because no two targets
        # share an exact position; assert that precondition holds for the scene.
        pool = mtb._target_pool(_load(_SEED2)["objects"])
        positions = [mtb._valid_position(o) for o in pool]
        assert len(set(positions)) == len(positions)


# ---------------------------------------------------------------------------
# The measurement (per-tier histogram + coordinate-fallthrough rate)
# ---------------------------------------------------------------------------


class TestMeasurementSeed2:
    def test_tier_histogram_and_fallthrough_rate_are_pinned(self):
        summary = measure_scene(_load(_SEED2))
        assert summary["total_targets"] == SEED2_TARGETS
        assert summary["tier_histogram"] == SEED2_HISTOGRAM
        assert summary["coordinate_fallthrough"] == SEED2_FALLTHROUGH
        assert summary["coordinate_fallthrough_rate"] == pytest.approx(
            SEED2_FALLTHROUGH / SEED2_TARGETS
        )
        assert summary["all_texts_distinct"] is True

    def test_worst_offender_breakdown_is_dominated_by_high_cardinality_labels(self):
        summary = measure_scene(_load(_SEED2))
        shelf = summary["by_label"]["shelf"]
        assert sum(shelf.values()) == 233
        assert shelf["coordinate_fallback"] == 224  # the bulk fall to coordinates
        # Histogram sum equals the target count (every target is accounted for).
        assert sum(summary["tier_histogram"].values()) == SEED2_TARGETS

    def test_harness_report_renders(self):
        # The repeatable harness formats a table without raising.
        text = mtb._format_report("seed2", measure_scene(_load(_SEED2)))
        assert "coordinate-fallthrough rate" in text
        assert "shelf" in text


# ---------------------------------------------------------------------------
# No bearing / sidedness / surface words anywhere (incl. coordinate strings)
# ---------------------------------------------------------------------------


class TestNoBearingWords:
    def test_seed2_anchors_carry_no_banned_token(self):
        md = _load(_SEED2)
        objects, rooms = md["objects"], md["rooms"]
        for o in mtb._target_pool(objects):
            text = disambiguate(o, objects, rooms).text
            assert not _BANNED.search(text), f"ungroundable spatial word in: {text!r}"

    def test_seed2_groundable_anchors_match_the_allow_list(self):
        md = _load(_SEED2)
        objects, rooms = md["objects"], md["rooms"]
        for o in mtb._target_pool(objects):
            r = disambiguate(o, objects, rooms)
            if r.groundable:
                assert _ALLOWED_GROUNDABLE.match(r.text), f"off-grammar anchor: {r.text!r}"
            else:
                assert _COORDINATE.match(r.text)

    @pytest.mark.parametrize("dirty", ["north corridor", "left cabinet", "south wing"])
    def test_a_bearing_tainted_target_label_never_leaks_into_the_anchor(self, dirty):
        # A bearing word can only enter via the opaque label; such a label must
        # degrade to a label-free, un-groundable coordinate anchor — not emit the
        # banned word. (Cannot occur on real scenes; the upstream vocabulary is
        # bearing-free and structural surfaces are pre-filtered.)
        objs = [_obj(dirty, (1.0, 2.0, 0.5)), _obj(dirty, (3.0, 4.0, 0.5))]
        r = disambiguate(objs[0], objs, [])
        assert not _BANNED.search(r.text), f"banned word leaked: {r.text!r}"
        assert r.groundable is False
        assert r.tier == "coordinate_fallback"
        assert r.text.startswith("the object approximately at (")


# ---------------------------------------------------------------------------
# Byte-identical guarantee: a singleton-label target is unchanged
# ---------------------------------------------------------------------------


class TestByteIdentical:
    def test_no_same_label_competitor_emits_bare_the_label(self):
        objs = [_obj("chair", (1.0, 1.0, 0.3)), _obj("table", (2.0, 2.0, 0.3))]
        r = disambiguate(objs[0], objs, [])
        assert r == AnchorResult(text="the chair", groundable=True, tier="singleton")

    def test_label_normalization_is_applied(self):
        objs = [_obj("  Floor Lamp ", (1.0, 1.0, 0.3))]
        r = disambiguate(objs[0], objs, [])
        assert r.text == "the floor lamp"  # stripped + lowercased


# ---------------------------------------------------------------------------
# Per-qualifier unit scenes: each live tier individually necessary
# ---------------------------------------------------------------------------


class TestZExtremumTier:
    def test_lowest_and_highest_resolve_when_only_height_differs(self):
        # Two same-label boxes, identical bbox, differing only in z: the size
        # and neighbour tiers cannot fire, so z_extremum is necessary.
        low = _cube("box", (0.0, 0.0, 0.2), 0.1, iid=1)
        high = _cube("box", (0.0, 0.0, 1.0), 0.1, iid=2)
        objs = [low, high]
        assert disambiguate(low, objs, []) == AnchorResult("the lowest box", True, "z_extremum")
        assert disambiguate(high, objs, []) == AnchorResult("the highest box", True, "z_extremum")


class TestSizeExtremumTier:
    def test_largest_and_smallest_resolve_when_height_is_tied(self):
        # Same z (z tied -> z skips), differing bbox volume: size is necessary.
        big = _cube("crate", (0.0, 0.0, 0.5), 0.4, iid=1)
        small = _cube("crate", (1.0, 0.0, 0.5), 0.1, iid=2)
        objs = [big, small]
        assert disambiguate(big, objs, []) == AnchorResult("the largest crate", True, "size_extremum")
        assert disambiguate(small, objs, []) == AnchorResult("the smallest crate", True, "size_extremum")

    def test_degenerate_bbox_competitor_blocks_a_false_superlative(self):
        # A competitor with no usable bbox means we cannot claim "largest";
        # height is tied too, so the target falls through to coordinates.
        big = _cube("crate", (0.0, 0.0, 0.5), 0.4, iid=1)
        nobbox = _obj("crate", (1.0, 0.0, 0.5), iid=2)  # no bbox keys
        r = disambiguate(big, [big, nobbox], [])
        assert r.tier == "coordinate_fallback" and r.groundable is False


class TestNearestNeighborTier:
    def test_unique_anchor_within_band_resolves_when_extrema_are_tied(self):
        # Two identical boxes (same z, same bbox): only proximity to a uniquely
        # named anchor distinguishes them, so the neighbour tier is necessary.
        near = _cube("box", (0.0, 0.0, 0.5), 0.1, iid=1)
        far = _cube("box", (10.0, 10.0, 0.5), 0.1, iid=2)
        vase = _obj("vase", (0.4, 0.0, 0.5), iid=3)  # unique label, within band of `near`
        objs = [near, far, vase]
        assert disambiguate(near, objs, []) == AnchorResult(
            "the box next to the vase", True, "nearest_neighbor"
        )
        # The far box has no in-band unique anchor -> coordinate fallback.
        assert disambiguate(far, objs, []).tier == "coordinate_fallback"

    def test_non_unique_neighbor_label_is_not_used_as_an_anchor(self):
        # The would-be anchor label appears twice, so "next to the lamp" would
        # not ground; the box falls to coordinates instead.
        box_a = _cube("box", (0.0, 0.0, 0.5), 0.1, iid=1)
        box_b = _cube("box", (5.0, 0.0, 0.5), 0.1, iid=2)
        lamp1 = _obj("lamp", (0.3, 0.0, 0.5), iid=3)
        lamp2 = _obj("lamp", (5.3, 0.0, 0.5), iid=4)
        r = disambiguate(box_a, [box_a, box_b, lamp1, lamp2], [])
        assert r.tier == "coordinate_fallback"


class TestConjunctionTier:
    def test_neighbor_then_extremum_compose(self):
        # box1/box2 share a unique anchor (vase) so "next to the vase" reduces
        # but does not resolve; box3 sits at the other anchor (clock) and holds
        # the global z-extreme, so z only resolves AFTER the neighbour split.
        box1 = _cube("box", (0.3, 0.0, 0.30), 0.1, iid=1)  # target
        box2 = _cube("box", (0.3, 0.5, 0.90), 0.1, iid=2)  # also next to vase, higher
        box3 = _cube("box", (10.3, 0.0, 0.10), 0.1, iid=3)  # next to clock, global lowest
        vase = _obj("vase", (0.0, 0.0, 0.5), iid=4)
        clock = _obj("clock", (10.0, 0.0, 0.5), iid=5)
        objs = [box1, box2, box3, vase, clock]
        r = disambiguate(box1, objs, [])
        assert r == AnchorResult("the lowest box next to the vase", True, "conjunction")


class TestCoordinateFallbackTier:
    def test_indistinguishable_clones_fall_to_unique_coordinate_anchors(self):
        # Three boxes identical in z and bbox, with no unique anchor anywhere:
        # nothing groundable partitions them, so all three escape to coordinates
        # and the strings stay distinct via full-precision position.
        boxes = [
            _cube("box", (0.0, 0.0, 0.5), 0.1, iid=1),
            _cube("box", (1.0, 0.0, 0.5), 0.1, iid=2),
            _cube("box", (2.0, 0.0, 0.5), 0.1, iid=3),
        ]
        results = [disambiguate(b, boxes, []) for b in boxes]
        assert all(r.tier == "coordinate_fallback" and r.groundable is False for r in results)
        assert len({r.text for r in results}) == 3
        assert all("approximately at" in r.text for r in results)


# ---------------------------------------------------------------------------
# Second-scene robustness: an independent dense synthetic scene
# (seed1's USD currently embeds zero objects, so a synthetic dense scene
# stands in for the brief's "repeat on a second scene" robustness check.)
# ---------------------------------------------------------------------------


def _synthetic_dense_scene() -> list[dict]:
    objs: list[dict] = []
    # Three labels, several clusters, a few unique anchors, stacked heights.
    for i in range(12):
        objs.append(_cube("bottle", (float(i % 4), float(i // 4), 0.2 + 0.1 * i), 0.05, iid=100 + i))
    for i in range(8):
        objs.append(_cube("bowl", (10.0 + i * 0.5, 0.0, 0.5), 0.04 + 0.005 * i, iid=200 + i))
    objs.append(_obj("kettle", (0.3, 0.0, 0.5), iid=300))  # unique anchor
    objs.append(_obj("toaster", (10.2, 0.0, 0.5), iid=301))  # unique anchor
    objs.append(_obj("wall", (0.0, 0.0, 0.0), iid=999))  # structural -> excluded
    return objs


class TestSecondSceneRobustness:
    def test_synthetic_dense_scene_anchors_are_all_distinct(self):
        objs = _synthetic_dense_scene()
        pool = mtb._target_pool(objs)
        # The structural 'wall' is filtered out of the target pool.
        assert all(mtb._normalize(o["label"]) != "wall" for o in pool)
        results = [disambiguate(o, objs, []) for o in pool]
        texts = [r.text for r in results]
        assert len(set(texts)) == len(texts)
        # Escape valve is terminal: every un-groundable result is a coordinate.
        assert all(r.tier == "coordinate_fallback" for r in results if not r.groundable)
        # No bearing words anywhere.
        assert all(not _BANNED.search(t) for t in texts)
