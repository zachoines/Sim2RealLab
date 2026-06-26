"""Pure-Python tests for the geometric start-frame grounding gate.

Covers the instance-segmentation parsing + per-instance target matching (in
``bbox_extractor``) and the provider's struct-shaping (in
``grounding_frame_provider``) — all without Kit, a model, or a rendered frame.

The load-bearing invariant: a KNOWN target is pinned by its USD ``prim_path``
through the ``instance_id_segmentation`` map (the prim path is the VALUE in
``idToLabels``, recovered by a reverse-scan), so a same-label SIBLING in frame
cannot satisfy the gate — exactly the same-room ambiguity the gate exists to
catch. ``bounding_box_2d_tight`` boxes per CLASS, so label alone would let the
sibling pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from strafer_lab.tools import build_mission_queue as bmq
from strafer_lab.tools.bbox_extractor import (
    INSTANCE_SEG_FIELDS,
    DetectedBbox,
    InstanceSegmentation,
    InstanceSegSchemaError,
    ReplicatorInstanceSegExtractor,
    bbox_row_for_segment,
    parse_bbox_data,
    parse_instance_seg_data,
    segment_ids_for_prim_path,
    segment_pixel_extent,
)
from strafer_lab.tools.grounding_frame_provider import (
    build_visibility_struct,
    edge_clip_fraction,
)

TARGET_PRIM = "/World/Room/chair_002"
SIBLING_PRIM = "/World/Room/chair_005"


def _mask_two_chairs():
    """6x6 id mask: target (id 7) top-left 2x2, same-label sibling (id 9) bottom-right 2x2."""
    mask = np.zeros((6, 6), dtype=np.uint32)
    mask[1:3, 1:3] = 7  # target
    mask[4:6, 4:6] = 9  # same-label sibling
    info = {"idToLabels": {"7": TARGET_PRIM, "9": SIBLING_PRIM}}
    return mask, info


def _class_bbox_rows(box, occ=0.1):
    """Parse a single per-CLASS ``bounding_box_2d_tight`` row (a list-of-dict fixture)."""
    raw = {
        "data": [
            {"semanticId": 1, "x_min": box[0], "y_min": box[1],
             "x_max": box[2], "y_max": box[3], "occlusionRatio": occ},
        ],
        "info": {"idToLabels": {"1": {"class": "chair"}}},
    }
    return parse_bbox_data(raw)


class TestInstanceSegParser:
    def test_parse_returns_typed_object(self):
        mask, info = _mask_two_chairs()
        seg = parse_instance_seg_data({"data": mask, "info": info})
        assert isinstance(seg, InstanceSegmentation)
        assert (seg.frame_w, seg.frame_h) == (6, 6)
        assert seg.info["idToLabels"]["7"] == TARGET_PRIM

    def test_none_or_no_data_returns_none(self):
        assert parse_instance_seg_data(None) is None
        assert parse_instance_seg_data({"data": None, "info": {"idToLabels": {}}}) is None

    def test_missing_required_info_key_raises(self):
        mask, _ = _mask_two_chairs()
        with pytest.raises(InstanceSegSchemaError):
            parse_instance_seg_data({"data": mask, "info": {"wrongKey": {}}})

    def test_fields_constant_is_frozen(self):
        # The frozen guard the operator confirms on the live schema freeze.
        assert INSTANCE_SEG_FIELDS == ("idToLabels",)


class TestSegmentIdsForPrimPath:
    def test_reverse_scan_finds_id_by_value(self):
        # prim_path is the VALUE in idToLabels, NOT the key — locks the reverse-scan.
        _, info = _mask_two_chairs()
        assert segment_ids_for_prim_path(info, TARGET_PRIM) == [7]
        assert segment_ids_for_prim_path(info, SIBLING_PRIM) == [9]

    def test_matches_descendant_mesh_subprim(self):
        # The renderer reports the drawn mesh prim, often a child of the labelled
        # object Xform (e.g. ".../chair_002/geom"); the object's prim_path must
        # still match it.
        info = {"idToLabels": {3: TARGET_PRIM + "/geom", 0: "INVALID"}}
        assert segment_ids_for_prim_path(info, TARGET_PRIM) == [3]

    def test_unions_multi_mesh_object(self):
        # A multi-mesh object owns several ids (one per drawn mesh).
        info = {"idToLabels": {1: TARGET_PRIM + "/seat", 2: TARGET_PRIM + "/legs", 3: SIBLING_PRIM}}
        assert sorted(segment_ids_for_prim_path(info, TARGET_PRIM)) == [1, 2]

    def test_matches_across_stage_composition_prefix(self):
        # Observed live: the env references the scene under /World/Room and the
        # mesh leaf repeats the object name, while the metadata recorded
        # /World/<name>. Match on the unique object-name segment.
        info = {"idToLabels": {
            "1": "INVALID",
            "2": "/World/Room/BookStackFactory_42__spawn_asset_7_/BookStackFactory_42__spawn_asset_7_",
            "3": "/World/Room/BottleFactory_9__spawn_asset_1_/BottleFactory_9__spawn_asset_1_",
        }}
        assert segment_ids_for_prim_path(info, "/World/BookStackFactory_42__spawn_asset_7_") == [2]
        # A different instance (sibling) must NOT match — names are per-instance.
        assert segment_ids_for_prim_path(info, "/World/BookStackFactory_99__spawn_asset_0_") == []

    def test_prefix_boundary_does_not_overmatch(self):
        # chair_002 must not match chair_0020.
        info = {"idToLabels": {1: "/World/Room/chair_0020", 2: TARGET_PRIM}}
        assert segment_ids_for_prim_path(info, TARGET_PRIM) == [2]

    def test_absent_or_invalid_returns_empty(self):
        _, info = _mask_two_chairs()
        assert segment_ids_for_prim_path(info, "/World/Room/nope") == []
        assert segment_ids_for_prim_path({"idToLabels": {0: "INVALID"}}, TARGET_PRIM) == []

    def test_none_inputs_return_empty(self):
        assert segment_ids_for_prim_path(None, TARGET_PRIM) == []
        assert segment_ids_for_prim_path({"idToLabels": {7: TARGET_PRIM}}, None) == []
        assert segment_ids_for_prim_path({"idToLabels": {}}, TARGET_PRIM) == []

    def test_mapping_value_probed_for_prim_path(self):
        info = {"idToLabels": {3: {"prim_path": TARGET_PRIM}}}
        assert segment_ids_for_prim_path(info, TARGET_PRIM) == [3]


class TestSegmentPixelExtent:
    def test_extent_is_per_instance(self):
        mask, _ = _mask_two_chairs()
        # id 7 occupies rows/cols 1-2 -> 4 px, bbox (1,1,3,3) with exclusive max.
        assert segment_pixel_extent(mask, 7) == (4, (1, 1, 3, 3))
        assert segment_pixel_extent(mask, 9) == (4, (4, 4, 6, 6))

    def test_unions_multiple_ids(self):
        mask, _ = _mask_two_chairs()
        # Both chairs as one object -> union spans both 2x2 blocks.
        assert segment_pixel_extent(mask, [7, 9]) == (8, (1, 1, 6, 6))

    def test_handles_trailing_channel_axis(self):
        # The real annotator data is (H, W, 1); the extent must reduce it to 2D.
        mask, _ = _mask_two_chairs()
        mask_3d = mask.reshape(mask.shape[0], mask.shape[1], 1)
        assert segment_pixel_extent(mask_3d, 7) == (4, (1, 1, 3, 3))

    def test_absent_segment_returns_none(self):
        mask, _ = _mask_two_chairs()
        assert segment_pixel_extent(mask, 999) is None
        assert segment_pixel_extent(mask, []) is None

    def test_pure_fallback_on_list_of_lists(self):
        # No numpy .shape -> the pure scan path (single pixel -> area 1).
        mask = [[0, 0], [0, 5]]
        assert segment_pixel_extent(mask, 5) == (1, (1, 1, 2, 2))


class TestBboxRowForSegment:
    def test_selects_overlapping_row(self):
        rows = [
            DetectedBbox(1, "chair", ("chair",), (1, 1, 3, 3), 0.1),
            DetectedBbox(2, "table", ("table",), (4, 4, 6, 6), 0.2),
        ]
        row = bbox_row_for_segment(rows, (1, 1, 3, 3))
        assert row is not None and row.label == "chair"

    def test_picks_largest_overlap(self):
        rows = [
            DetectedBbox(1, "a", ("a",), (0, 0, 2, 2), 0.1),  # overlap 1 px
            DetectedBbox(2, "b", ("b",), (1, 1, 5, 5), 0.2),  # overlap 4 px
        ]
        assert bbox_row_for_segment(rows, (1, 1, 3, 3)).label == "b"

    def test_no_overlap_returns_none(self):
        rows = [DetectedBbox(1, "chair", ("chair",), (1, 1, 3, 3), 0.1)]
        assert bbox_row_for_segment(rows, (10, 10, 12, 12)) is None


class TestSiblingRejection:
    """A same-label sibling in frame must NOT satisfy the gate for the target."""

    def test_target_visible_ships_with_per_instance_bbox(self):
        mask, info = _mask_two_chairs()
        seg = parse_instance_seg_data({"data": mask, "info": info})
        # One CLASS row spanning BOTH chairs — the merge the instance map defeats.
        bboxes = _class_bbox_rows((1, 1, 6, 6))
        s = build_visibility_struct(bboxes, seg, TARGET_PRIM)
        assert s["in_frame"] is True
        # The TARGET's per-instance box, NOT the merged class box (1,1,6,6).
        assert s["bbox"] == (1, 1, 3, 3)
        assert s["occlusion_ratio"] == 0.1

    def test_target_occluded_away_while_sibling_visible_is_not_in_frame(self):
        # Target id 7 renders ZERO pixels; the same-label sibling id 9 is visible.
        mask = np.zeros((6, 6), dtype=np.uint32)
        mask[4:6, 4:6] = 9
        info = {"idToLabels": {"7": TARGET_PRIM, "9": SIBLING_PRIM}}
        seg = parse_instance_seg_data({"data": mask, "info": info})
        # The class row covers the visible sibling — a label match would say "yes".
        bboxes = _class_bbox_rows((4, 4, 6, 6))
        s = build_visibility_struct(bboxes, seg, TARGET_PRIM)
        assert s["in_frame"] is False
        assert s["bbox"] is None
        # And the runner turns that into a "no".
        assert bmq.build_default_geometric_runner()(s, "go to the chair") == "no"

    def test_target_absent_from_id_map_is_not_in_frame(self):
        mask, _ = _mask_two_chairs()
        info = {"idToLabels": {"9": SIBLING_PRIM}}  # target never assigned an id
        seg = parse_instance_seg_data({"data": mask, "info": info})
        s = build_visibility_struct(_class_bbox_rows((1, 1, 6, 6)), seg, TARGET_PRIM)
        assert s["in_frame"] is False

    def test_matches_when_annotator_reports_mesh_subprim(self):
        # Real schema: the annotator labels the drawn mesh prim (.../geom), not the
        # object Xform; the object's prim_path must still resolve to its pixels.
        mask, _ = _mask_two_chairs()
        info = {"idToLabels": {7: TARGET_PRIM + "/geom", 9: SIBLING_PRIM + "/geom", 0: "INVALID"}}
        seg = parse_instance_seg_data({"data": mask, "info": info})
        s = build_visibility_struct(_class_bbox_rows((1, 1, 3, 3)), seg, TARGET_PRIM)
        assert s["in_frame"] is True
        assert s["bbox"] == (1, 1, 3, 3)

    def test_edge_clipped_sibling_does_not_reject_interior_target(self):
        # Target id 7 fully interior; a same-label sibling id 9 is truncated at the
        # right frame edge. bounding_box_2d_tight merges them into ONE per-class row
        # whose raw box overflows the frame — a class-row edge-clip would reject the
        # target. The per-instance edge-clip (target's own mask box) keeps it "yes".
        w, h = 640, 360
        mask = np.zeros((h, w), dtype=np.uint32)
        mask[100:160, 100:160] = 7        # interior target, 60x60
        mask[100:160, w - 20:w] = 9       # sibling hugging the right edge
        info = {"idToLabels": {"7": TARGET_PRIM, "9": SIBLING_PRIM}}
        seg = parse_instance_seg_data({"data": mask, "info": info})
        # One merged per-class row whose raw x_max runs far past the frame.
        merged = {
            "data": [{"semanticId": 1, "x_min": 100, "y_min": 100,
                      "x_max": w + 1000, "y_max": 160, "occlusionRatio": 0.05}],
            "info": {"idToLabels": {"1": {"class": "chair"}}},
        }
        s = build_visibility_struct(parse_bbox_data(merged), seg, TARGET_PRIM)
        assert s["in_frame"] is True
        assert s["bbox"] == (100, 100, 160, 160)  # the target's own interior box
        assert s["edge_clip_frac"] == 0.0          # unaffected by the clipped sibling
        assert bmq.build_default_geometric_runner()(s, "go to the chair") == "yes"


class TestBuildVisibilityStructSkips:
    def test_seg_none_is_skip(self):
        assert build_visibility_struct([], None, TARGET_PRIM) is None

    def test_no_prim_path_is_skip(self):
        mask, info = _mask_two_chairs()
        seg = parse_instance_seg_data({"data": mask, "info": info})
        assert build_visibility_struct([], seg, None) is None
        assert build_visibility_struct([], seg, "") is None

    def test_visible_target_with_no_bbox_row_has_none_occlusion(self):
        mask, info = _mask_two_chairs()
        seg = parse_instance_seg_data({"data": mask, "info": info})
        s = build_visibility_struct([], seg, TARGET_PRIM)  # empty bbox list
        assert s["in_frame"] is True
        assert s["occlusion_ratio"] is None
        # No occlusion evidence -> the verdict refuses to ship.
        assert bmq.geometric_visibility_verdict(s, "x") == "no"


class TestEdgeClipFraction:
    """Per-instance border-overlap fraction (sides of the box on a frame edge)."""

    def test_interior_box_is_zero(self):
        assert edge_clip_fraction((10, 10, 20, 20), 100, 100) == 0.0

    def test_one_border_is_a_quarter(self):
        assert edge_clip_fraction((0, 10, 20, 20), 100, 100) == pytest.approx(0.25)    # left
        assert edge_clip_fraction((10, 10, 100, 20), 100, 100) == pytest.approx(0.25)  # right (x2 >= W)

    def test_corner_two_borders_is_a_half(self):
        assert edge_clip_fraction((0, 0, 20, 20), 100, 100) == pytest.approx(0.5)

    def test_three_borders_exceeds_threshold(self):
        # Spans the full width against the top edge -> left + top + right.
        assert edge_clip_fraction((0, 0, 100, 20), 100, 100) == pytest.approx(0.75)

    def test_none_or_degenerate_is_one(self):
        assert edge_clip_fraction(None, 100, 100) == 1.0
        assert edge_clip_fraction((5, 5, 5, 5), 100, 100) == 1.0


class TestInstanceSegExtractorMockHook:
    def test_annotator_injection_returns_parsed(self):
        mask, info = _mask_two_chairs()

        class FakeAnnotator:
            def get_data(self):
                return {"data": mask, "info": info}

        ext = ReplicatorInstanceSegExtractor("rp/path", annotator=FakeAnnotator())
        seg = ext.extract()
        assert isinstance(seg, InstanceSegmentation)
        assert segment_ids_for_prim_path(seg.info, TARGET_PRIM) == [7]

    def test_no_frame_yet_returns_none(self):
        class EmptyAnnotator:
            def get_data(self):
                return None

        ext = ReplicatorInstanceSegExtractor("rp/path", annotator=EmptyAnnotator())
        assert ext.extract() is None


class TestStructToVerdictEndToEnd:
    """A realistic-resolution render flows struct -> runner -> verdict."""

    def test_large_unoccluded_target_ships(self):
        w, h = 640, 360
        mask = np.zeros((h, w), dtype=np.uint32)
        mask[100:160, 100:160] = 7  # 60x60 -> 3600 px, well over the floor
        info = {"idToLabels": {"7": TARGET_PRIM}}
        seg = parse_instance_seg_data({"data": mask, "info": info})
        bboxes = _class_bbox_rows((100, 100, 160, 160), occ=0.05)
        s = build_visibility_struct(bboxes, seg, TARGET_PRIM)
        assert bmq.build_default_geometric_runner()(s, "go to the chair") == "yes"

    def test_tiny_target_across_room_rejected(self):
        w, h = 640, 360
        mask = np.zeros((h, w), dtype=np.uint32)
        mask[100:103, 100:103] = 7  # 3x3 -> 9 px, below the ~115 floor
        info = {"idToLabels": {"7": TARGET_PRIM}}
        seg = parse_instance_seg_data({"data": mask, "info": info})
        bboxes = _class_bbox_rows((100, 100, 103, 103), occ=0.0)
        s = build_visibility_struct(bboxes, seg, TARGET_PRIM)
        assert bmq.build_default_geometric_runner()(s, "x") == "no"
