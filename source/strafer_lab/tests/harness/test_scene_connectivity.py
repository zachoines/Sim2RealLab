"""Unit tests for the scene connectivity tool.

Pure numpy — no Isaac Sim, no Kit, no pxr. The Kit-bound occupancy-map
generation is exercised separately on real scenes; here we verify the
reachability + connectivity-shape logic on synthetic occupancy grids and
synthetic room metadata. Grids follow the planner convention: row = x index,
col = y index, origin at the corner of cell (0, 0), cell size ``RES``.
"""

from __future__ import annotations

import numpy as np

from strafer_lab.tools.scene_connectivity import (
    aabb_to_footprint,
    compute_connectivity,
    connectivity_matrix,
    find_room_boundary_crossing,
    is_multi_room_incompatible,
    is_multi_story,
    occupancy_to_free_space,
    parse_room_floor_name,
    point_in_polygon,
    polygon_centroid,
    room_representative_xy,
)

RES = 0.1
ORIGIN = (0.0, 0.0)


# ---------------------------------------------------------------------------
# occupancy_to_free_space — invert + robot-radius disc inflation
# ---------------------------------------------------------------------------


class TestOccupancyToFreeSpace:
    def test_empty_occupancy_no_inflation_is_all_free(self):
        occ = np.zeros((10, 10), dtype=np.uint8)
        free = occupancy_to_free_space(occ, grid_res=RES, robot_radius_m=0.0)
        assert free.dtype == bool
        assert free.all()

    def test_inversion(self):
        occ = np.zeros((10, 10), dtype=np.uint8)
        occ[5, 5] = 1
        free = occupancy_to_free_space(occ, grid_res=RES, robot_radius_m=0.0)
        assert not free[5, 5]
        assert free[0, 0]

    def test_disc_inflation_radius(self):
        # One obstacle cell, 0.3 m radius at 0.1 m/cell -> 3-cell disc.
        occ = np.zeros((21, 21), dtype=np.uint8)
        occ[10, 10] = 1
        free = occupancy_to_free_space(occ, grid_res=RES, robot_radius_m=0.3)
        blocked = ~free
        # On-axis: cells within 3 of center blocked, the 4th free.
        assert blocked[10, 13] and blocked[13, 10]
        assert free[10, 14] and free[14, 10]
        # Disc, not square: the (3,3) corner (dist^2=18 > 9) stays free.
        assert free[13, 13]

    def test_nonzero_means_blocked_regardless_of_value(self):
        occ = np.full((6, 6), 7, dtype=np.uint8)  # all "occupied/unknown"
        free = occupancy_to_free_space(occ, grid_res=RES, robot_radius_m=0.0)
        assert not free.any()


# ---------------------------------------------------------------------------
# Room geometry helpers
# ---------------------------------------------------------------------------


class TestRoomGeometry:
    def test_parse_room_floor_name_underscored_types(self):
        assert parse_room_floor_name("living_room_0_0_floor") == ("living_room", 0, 0)
        assert parse_room_floor_name("dining_room_0_0_floor") == ("dining_room", 0, 0)
        assert parse_room_floor_name("bedroom_0_2_floor") == ("bedroom", 0, 2)
        assert parse_room_floor_name("bathroom_1_3_floor") == ("bathroom", 1, 3)

    def test_parse_room_floor_name_rejects_non_floor(self):
        assert parse_room_floor_name("PanelDoorFactory_99__spawn_asset_1_") is None
        assert parse_room_floor_name("kitchen") is None
        assert parse_room_floor_name("a_floor") is None  # too few parts
        assert parse_room_floor_name("room_x_y_floor") is None  # non-int story/idx

    def test_aabb_to_footprint_is_ccw_rectangle(self):
        fp = aabb_to_footprint((1.0, 2.0), (3.0, 5.0))
        assert fp == [[1.0, 2.0], [3.0, 2.0], [3.0, 5.0], [1.0, 5.0]]

    def test_polygon_centroid_of_rectangle(self):
        fp = aabb_to_footprint((0.0, 0.0), (4.0, 2.0))
        cx, cy = polygon_centroid(fp)
        assert abs(cx - 2.0) < 1e-9 and abs(cy - 1.0) < 1e-9

    def test_point_in_polygon(self):
        fp = aabb_to_footprint((0.0, 0.0), (2.0, 2.0))
        assert point_in_polygon(1.0, 1.0, fp)
        assert not point_in_polygon(3.0, 1.0, fp)

    def test_is_multi_story(self):
        assert not is_multi_story([{"story": 0}, {"story": 0}])
        assert is_multi_story([{"story": 0}, {"story": 1}])
        assert not is_multi_story([])


# ---------------------------------------------------------------------------
# Synthetic two-room worlds for compute_connectivity
# ---------------------------------------------------------------------------


def _two_room_metadata():
    """Room A: x[0.5,2.5] y[0.5,3.5]; Room B: x[3.5,5.5] y[0.5,3.5]."""
    return [
        {"room_type": "living_room", "story": 0,
         "footprint_xy": aabb_to_footprint((0.5, 0.5), (2.5, 3.5))},
        {"room_type": "bedroom", "story": 0,
         "footprint_xy": aabb_to_footprint((3.5, 0.5), (5.5, 3.5))},
    ]


def _free_grid(nx: int = 60, ny: int = 40) -> np.ndarray:
    return np.ones((nx, ny), dtype=bool)


def _wall_at_x3(free: np.ndarray, doorway: bool) -> np.ndarray:
    """Block the column at x≈3.0 (rows 28-32) across all y, optional doorway."""
    free = free.copy()
    free[28:33, :] = False
    if doorway:
        free[28:33, 15:26] = True  # ~1.1 m gap in y around y=2.0
    return free


class TestComputeConnectivity:
    def test_reachable_pair_through_doorway(self):
        rooms = _two_room_metadata()
        free = _wall_at_x3(_free_grid(), doorway=True)
        edges = compute_connectivity(
            rooms, free, grid_origin_xy=ORIGIN, grid_res=RES,
        )
        assert len(edges) == 1
        e = edges[0]
        assert (e["from_idx"], e["to_idx"]) == (0, 1)
        assert e["reachable"] is True
        assert e["path_length_m"] > 0
        dx, dy = e["via_doorway_xy"]
        # Crossing sits in the doorway gap region.
        assert 2.3 <= dx <= 3.7
        assert 1.3 <= dy <= 2.7

    def test_blocked_pair_no_doorway(self):
        rooms = _two_room_metadata()
        free = _wall_at_x3(_free_grid(), doorway=False)
        edges = compute_connectivity(
            rooms, free, grid_origin_xy=ORIGIN, grid_res=RES,
        )
        assert len(edges) == 1
        e = edges[0]
        assert e["reachable"] is False
        assert e["reason"] == "blocked"
        assert "via_doorway_xy" not in e

    def test_cross_story_pair_is_stairs_without_planning(self):
        rooms = _two_room_metadata()
        rooms[1]["story"] = 1
        # Fully open grid: if it planned anyway it'd be reachable. It must not.
        edges = compute_connectivity(
            rooms, _free_grid(), grid_origin_xy=ORIGIN, grid_res=RES,
        )
        assert edges[0]["reachable"] is False
        assert edges[0]["reason"] == "stairs"

    def test_room_adjacency_restricts_candidates(self):
        rooms = _two_room_metadata()
        rooms.append({"room_type": "closet", "story": 0,
                      "footprint_xy": aabb_to_footprint((0.5, 0.5), (1.5, 1.5))})
        free = _free_grid()
        # Only the (0,1) edge is a candidate; (0,2)/(1,2) are excluded.
        edges = compute_connectivity(
            rooms, free, grid_origin_xy=ORIGIN, grid_res=RES,
            room_adjacency=[[0, 1], [1, 0]],
        )
        assert [(e["from_idx"], e["to_idx"]) for e in edges] == [(0, 1)]

    def test_all_pairs_when_no_adjacency(self):
        rooms = _two_room_metadata()
        rooms.append({"room_type": "kitchen", "story": 0,
                      "footprint_xy": aabb_to_footprint((6.5, 0.5), (7.5, 3.5))})
        # 3 rooms, no adjacency -> 3 unordered candidate pairs.
        edges = compute_connectivity(
            rooms, _free_grid(80, 40), grid_origin_xy=ORIGIN, grid_res=RES,
        )
        assert sorted((e["from_idx"], e["to_idx"]) for e in edges) == [
            (0, 1), (0, 2), (1, 2),
        ]


# ---------------------------------------------------------------------------
# Scene-level flags + reporting helpers
# ---------------------------------------------------------------------------


class TestSceneFlags:
    def test_incompatible_only_when_no_cross_room_edge_reachable(self):
        rooms = _two_room_metadata()
        blocked = compute_connectivity(
            rooms, _wall_at_x3(_free_grid(), doorway=False),
            grid_origin_xy=ORIGIN, grid_res=RES,
        )
        assert is_multi_room_incompatible(blocked) is True

        reachable = compute_connectivity(
            rooms, _wall_at_x3(_free_grid(), doorway=True),
            grid_origin_xy=ORIGIN, grid_res=RES,
        )
        assert is_multi_room_incompatible(reachable) is False

    def test_single_room_scene_is_never_incompatible(self):
        # No cross-room candidate edges at all.
        assert is_multi_room_incompatible([]) is False

    def test_partial_blockage_is_non_fatal(self):
        # One reachable + one blocked cross-room edge -> still compatible.
        edges = [
            {"from_idx": 0, "to_idx": 1, "reachable": True},
            {"from_idx": 1, "to_idx": 2, "reachable": False, "reason": "blocked"},
        ]
        assert is_multi_room_incompatible(edges) is False

    def test_connectivity_matrix_is_symmetric(self):
        edges = [
            {"from_idx": 0, "to_idx": 1, "reachable": True},
            {"from_idx": 1, "to_idx": 2, "reachable": False, "reason": "blocked"},
        ]
        mat = connectivity_matrix(3, edges)
        assert mat[0, 1] and mat[1, 0]
        assert not mat[1, 2] and not mat[0, 2]


class TestDoorwayCrossing:
    def test_crossing_between_adjacent_rooms(self):
        a = {"footprint_xy": aabb_to_footprint((0.0, 0.0), (2.0, 2.0))}
        b = {"footprint_xy": aabb_to_footprint((3.0, 0.0), (5.0, 2.0))}
        path = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
        x, y = find_room_boundary_crossing(path, a, b)
        assert 2.0 <= x <= 3.0
        assert abs(y - 1.0) < 1e-6


class TestRepresentativePoint:
    def test_centroid_when_free(self):
        room = {"footprint_xy": aabb_to_footprint((0.5, 0.5), (2.5, 2.5))}
        free = _free_grid()
        x, y = room_representative_xy(room, free, origin_xy=ORIGIN, grid_res=RES)
        assert abs(x - 1.5) < 1e-6 and abs(y - 1.5) < 1e-6


# ---------------------------------------------------------------------------
# Door handling in the Kit script (pure helpers — door + leaf-detect regexes)
# ---------------------------------------------------------------------------


def _load_vsc_module():
    """Load validate_scene_connectivity.py — its top-level imports are Kit-free."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "validate_scene_connectivity.py"
    spec = importlib.util.spec_from_file_location("validate_scene_connectivity_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDoorHandling:
    def test_leaf_regex_matches_panel_not_frame(self):
        vsc = _load_vsc_module()
        assert vsc._DOOR_LEAF_RE.search("PanelDoorFactory_99__spawn_asset_2__001")
        assert not vsc._DOOR_LEAF_RE.search("PanelDoorFactory_99__spawn_asset_2_")
        # The door matcher still catches the GlassPanel variant + both prims.
        assert vsc._DOOR_PRIM_RE.search("GlassPanelDoorFactory_56__spawn_asset_4_")
        assert vsc._DOOR_PRIM_RE.search("GlassPanelDoorFactory_56__spawn_asset_4__001")

    def test_snaps_to_free_when_centroid_blocked(self):
        room = {"footprint_xy": aabb_to_footprint((0.5, 0.5), (2.5, 2.5))}
        free = _free_grid()
        # Block the centroid cell (1.5, 1.5) -> row 15, col 15.
        free[14:17, 14:17] = False
        x, y = room_representative_xy(room, free, origin_xy=ORIGIN, grid_res=RES)
        r = int((x - ORIGIN[0]) / RES)
        c = int((y - ORIGIN[1]) / RES)
        assert free[r, c]
