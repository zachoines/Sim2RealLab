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

# Frozen measurement snapshot for seed2 (646 post-filter targets) with the LIVE
# room scope. This is the deliverable the downstream corpus run turns on; update
# deliberately if the fixture or the waterfall changes. Groundable yield (the
# corpus target-pool size after filtering the coordinate fallthrough) is the
# headline number.
SEED2_TARGETS = 646
SEED2_HISTOGRAM = {
    "singleton": 9,
    "room_scope": 12,
    "z_extremum": 54,
    "size_extremum": 27,
    "nearest_neighbor": 5,
    "conjunction": 82,
    "coordinate_fallback": 457,
}
SEED2_FALLTHROUGH = 457
SEED2_GROUNDABLE = 189  # 646 - 457; the corpus target-pool size

# The complete anchor grammar — the only surface forms a groundable anchor may
# take: an optional superlative, the label, an optional "next to the {neighbour}"
# clause, and an optional trailing "in the {room_type}" scope. The no-bearing ban
# is a stricter test layered on top of this.
_ALLOWED_GROUNDABLE = re.compile(
    r"^the (?:(?:lowest|highest|largest|smallest) )?[^()]+?(?: next to the [^()]+?)?(?: in the [^()]+)?$"
)
_COORDINATE = re.compile(r"^the .+ approximately at \([^)]*\)$")
# Mirror the builder's guard exactly (underscore counts as a separator, so an
# underscore-joined bearing token like ``north_storage`` is caught too).
_BANNED = re.compile(
    r"(?<![a-z0-9])(north|south|east|west|left|right|wall)(?![a-z0-9])", re.IGNORECASE
)


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

    def test_groundable_yield_sizes_the_corpus_pool(self):
        # The headline: how many targets survive the require_groundable filter.
        summary = measure_scene(_load(_SEED2))
        assert summary["groundable_targets"] == SEED2_GROUNDABLE
        assert summary["groundable_targets"] == SEED2_TARGETS - SEED2_FALLTHROUGH
        # The room scope is a live, groundable lever (not a no-op).
        assert summary["tier_histogram"]["room_scope"] == 12

    def test_worst_offender_breakdown_is_dominated_by_high_cardinality_labels(self):
        summary = measure_scene(_load(_SEED2))
        shelf = summary["by_label"]["shelf"]
        assert sum(shelf.values()) == 233
        assert shelf["coordinate_fallback"] == 218  # the bulk still fall to coordinates
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
                # The allow-list regex is permissive; also assert no dangling connective.
                assert not r.text.rstrip().endswith((" next to", " in", " the")), r.text
            else:
                assert _COORDINATE.match(r.text)

    @pytest.mark.parametrize(
        "dirty", ["north corridor", "left cabinet", "south wing", "north_corridor", "east_wall"]
    )
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


class TestRoomScope:
    # Two adjacent rooms along +x sharing the wall at x=4. Footprints are simple
    # squares; the test objects sit at interior points well clear of the seam.
    KITCHEN = {"room_type": "kitchen", "footprint_xy": [[0, 0], [4, 0], [4, 4], [0, 4]], "story": 0}
    HALL = {"room_type": "hall", "footprint_xy": [[4, 0], [8, 0], [8, 4], [4, 4]], "story": 0}
    BEDROOM_1 = {"room_type": "bedroom", "footprint_xy": [[0, 0], [4, 0], [4, 4], [0, 4]], "story": 0}
    BEDROOM_2 = {"room_type": "bedroom", "footprint_xy": [[4, 0], [8, 0], [8, 4], [4, 4]], "story": 0}

    def test_unique_room_scope_alone_resolves(self):
        # One box in the kitchen, one in the hall — identical otherwise, so only
        # the (globally unique) room name distinguishes them. Room scope alone
        # reduces the competitor set to empty and resolves each.
        in_kitchen = _cube("box", (1.0, 1.0, 0.5), 0.1, iid=1)
        in_hall = _cube("box", (5.0, 1.0, 0.5), 0.1, iid=2)
        rooms = [self.KITCHEN, self.HALL]
        objs = [in_kitchen, in_hall]
        assert disambiguate(in_kitchen, objs, rooms) == AnchorResult(
            "the box in the kitchen", True, "room_scope"
        )
        assert disambiguate(in_hall, objs, rooms) == AnchorResult(
            "the box in the hall", True, "room_scope"
        )

    def test_room_scope_composes_with_a_size_extremum(self):
        # Kitchen holds two boxes (big + small); the hall holds a third box the
        # same size as the kitchen's big one, so size-extremum cannot fire
        # globally (a tie). Room scope first narrows to the two kitchen boxes,
        # THEN size-extremum names the larger -> "the largest box in the kitchen".
        big = _cube("box", (1.0, 1.0, 0.5), 0.4, iid=1)
        small = _cube("box", (2.0, 1.0, 0.5), 0.1, iid=2)
        hall_same_size = _cube("box", (5.0, 1.0, 0.5), 0.4, iid=3)  # global size tie
        rooms = [self.KITCHEN, self.HALL]
        objs = [big, small, hall_same_size]
        assert disambiguate(big, objs, rooms) == AnchorResult(
            "the largest box in the kitchen", True, "conjunction"
        )

    def test_non_unique_room_type_does_not_fire(self):
        # Two bedrooms -> "the bedroom" is itself ungroundable, so room scope must
        # not fire and must never emit "in the bedroom"; the indistinguishable
        # boxes fall through to the coordinate anchor.
        a = _cube("box", (1.0, 1.0, 0.5), 0.1, iid=1)
        b = _cube("box", (5.0, 1.0, 0.5), 0.1, iid=2)
        rooms = [self.BEDROOM_1, self.BEDROOM_2]
        r = disambiguate(a, [a, b], rooms)
        assert "in the bedroom" not in r.text
        assert r.tier == "coordinate_fallback" and r.groundable is False

    def test_singleton_in_a_unique_room_stays_byte_identical(self):
        # A label with no same-label competitor needs no room scope: byte-identical
        # "the chair", not "the chair in the kitchen".
        chair = _cube("chair", (1.0, 1.0, 0.5), 0.1, iid=1)
        rooms = [self.KITCHEN, self.HALL]
        assert disambiguate(chair, [chair], rooms) == AnchorResult("the chair", True, "singleton")

    def test_room_scoped_anchors_obey_the_allow_list_and_no_bearing_ban(self):
        big = _cube("box", (1.0, 1.0, 0.5), 0.4, iid=1)
        small = _cube("box", (2.0, 1.0, 0.5), 0.1, iid=2)
        hall_same = _cube("box", (5.0, 1.0, 0.5), 0.4, iid=3)
        rooms = [self.KITCHEN, self.HALL]
        for o in (big, small, hall_same):
            r = disambiguate(o, [big, small, hall_same], rooms)
            if r.groundable:
                assert _ALLOWED_GROUNDABLE.match(r.text), f"off-grammar: {r.text!r}"
                # No dangling connective (the allow-list regex alone is permissive).
                assert not r.text.rstrip().endswith((" next to", " in", " the")), r.text
            assert not _BANNED.search(r.text), f"banned word: {r.text!r}"

    @pytest.mark.parametrize("dirty_room", ["north_storage", "south_wing", "east wall"])
    def test_bearing_tainted_room_type_does_not_fire_room_scope(self, dirty_room):
        # A bearing word in the (unique) room_type — including underscore-joined —
        # must NOT enter the anchor; room scope is skipped and the indistinguishable
        # boxes degrade to the coordinate anchor.
        kitchen = self.KITCHEN
        tainted = {"room_type": dirty_room, "footprint_xy": [[4, 0], [8, 0], [8, 4], [4, 4]], "story": 0}
        in_kitchen = _cube("box", (1.0, 1.0, 0.5), 0.1, iid=1)
        in_tainted = _cube("box", (5.0, 1.0, 0.5), 0.1, iid=2)
        r = disambiguate(in_tainted, [in_kitchen, in_tainted], [kitchen, tainted])
        assert not _BANNED.search(r.text), f"banned room_type leaked: {r.text!r}"
        assert "in the" not in r.text  # room scope did not fire
        assert r.tier == "coordinate_fallback" and r.groundable is False


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
