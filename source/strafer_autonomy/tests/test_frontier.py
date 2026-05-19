"""Tests for the wavefront frontier detector and ranking helpers."""

from __future__ import annotations

import numpy as np
import pytest

from strafer_autonomy.executor.frontier import (
    DEFAULT_OCCUPIED_THRESHOLD,
    FrontierCluster,
    detect_frontier_clusters,
    rank_frontiers,
)


def _make_grid(rows: list[str]) -> np.ndarray:
    """Build a synthetic occupancy grid from a list of row strings.

    Character mapping:
        ``"."`` free (0)
        ``"#"`` occupied (occupied_threshold)
        ``"?"`` unknown (-1)
    """
    height = len(rows)
    width = len(rows[0])
    data = np.zeros((height, width), dtype=np.int8)
    for r, row in enumerate(rows):
        assert len(row) == width, f"row {r} length mismatch"
        for c, ch in enumerate(row):
            if ch == "?":
                data[r, c] = -1
            elif ch == "#":
                data[r, c] = DEFAULT_OCCUPIED_THRESHOLD
            elif ch == ".":
                data[r, c] = 0
            else:
                raise ValueError(f"unknown char {ch!r}")
    return data


class TestDetectFrontierClusters:
    def test_no_frontiers_when_all_free(self):
        data = np.zeros((6, 6), dtype=np.int8)
        clusters = detect_frontier_clusters(
            data=data,
            origin_xy=(0.0, 0.0),
            resolution=0.1,
            min_cluster_cells=1,
        )
        assert clusters == []

    def test_no_frontiers_when_all_unknown(self):
        data = -np.ones((6, 6), dtype=np.int8)
        clusters = detect_frontier_clusters(
            data=data,
            origin_xy=(0.0, 0.0),
            resolution=0.1,
            min_cluster_cells=1,
        )
        # Free cells are the prerequisite — no free means no frontier.
        assert clusters == []

    def test_single_frontier_along_boundary(self):
        # 6x6 grid: top half known free, bottom half unknown.
        data = _make_grid([
            "......",
            "......",
            "......",
            "??????",
            "??????",
            "??????",
        ])
        clusters = detect_frontier_clusters(
            data=data,
            origin_xy=(0.0, 0.0),
            resolution=0.1,
            min_cluster_cells=1,
        )
        assert len(clusters) == 1
        cluster = clusters[0]
        # The frontier row is r=2 (free cells with unknown neighbors at r=3).
        assert cluster.cell_count == 6
        assert cluster.bbox_rows == (2, 2)
        assert cluster.bbox_cols == (0, 5)
        # Centroid in map frame: row 2.5 maps to y = origin + 0.25, col 2.5 -> x = 0.25.
        assert cluster.centroid_xy == pytest.approx((0.30, 0.25))

    def test_two_disconnected_frontiers(self):
        # Two unknown regions separated by a vertical occupied wall —
        # the wall fully prevents the two frontier strips from merging
        # through the gap.
        data = _make_grid([
            "...#...",
            "...#...",
            "...#...",
            "???#???",
        ])
        clusters = detect_frontier_clusters(
            data=data,
            origin_xy=(0.0, 0.0),
            resolution=1.0,
            min_cluster_cells=1,
        )
        assert len(clusters) == 2
        cols_a = clusters[0].centroid_xy[0]
        cols_b = clusters[1].centroid_xy[0]
        # One cluster on each side of the col-3 wall.
        assert (cols_a < 3.0 and cols_b > 4.0) or (cols_a > 4.0 and cols_b < 3.0)

    def test_min_cluster_cells_filters_noise(self):
        # One large frontier (left) + one one-cell speckle (right).
        data = _make_grid([
            "??????.....",
            "??????.....",
            "??????..?..",
            ".......????",
            ".......????",
        ])
        big = detect_frontier_clusters(
            data=data,
            origin_xy=(0.0, 0.0),
            resolution=1.0,
            min_cluster_cells=3,
        )
        # Only one cluster passes the size filter (the 4x1 frontier on
        # the left side, plus the bottom-right wedge — they may or may
        # not merge depending on geometry). Verify each surviving cluster
        # is at or above the floor.
        assert len(big) >= 1
        for cluster in big:
            assert cluster.cell_count >= 3

    def test_occupied_cells_block_frontier(self):
        # A wall of occupied cells between free and unknown means the
        # free cells are NOT frontier (their unknown-side neighbor is
        # occupied, not unknown).
        data = _make_grid([
            "......",
            "......",
            "######",
            "??????",
        ])
        clusters = detect_frontier_clusters(
            data=data,
            origin_xy=(0.0, 0.0),
            resolution=0.1,
            min_cluster_cells=1,
        )
        assert clusters == []

    def test_cluster_key_is_lex_min_cell(self):
        data = _make_grid([
            "...???",
            "...???",
            "...???",
        ])
        clusters = detect_frontier_clusters(
            data=data,
            origin_xy=(0.0, 0.0),
            resolution=0.1,
            min_cluster_cells=1,
        )
        assert len(clusters) == 1
        cluster = clusters[0]
        assert cluster.cluster_key == (0, 2)
        # Re-run on the same grid; key is identical (stable identity).
        again = detect_frontier_clusters(
            data=data,
            origin_xy=(0.0, 0.0),
            resolution=0.1,
            min_cluster_cells=1,
        )
        assert again[0].cluster_key == cluster.cluster_key

    def test_origin_offset_carries_through_to_centroid(self):
        data = _make_grid([
            "...???",
            "...???",
        ])
        clusters = detect_frontier_clusters(
            data=data,
            origin_xy=(10.0, -5.0),
            resolution=0.1,
            min_cluster_cells=1,
        )
        assert len(clusters) == 1
        cx, cy = clusters[0].centroid_xy
        # Column-2 cells, two rows → mean col 2, mean row 0.5.
        # cx = 10 + (2 + 0.5)*0.1 = 10.25; cy = -5 + (0.5 + 0.5)*0.1 = -4.9
        assert cx == pytest.approx(10.25)
        assert cy == pytest.approx(-4.9)


class TestRankFrontiers:
    def _make_cluster(
        self,
        *,
        key: tuple[int, int],
        cell_count: int,
        centroid_xy: tuple[float, float],
    ) -> FrontierCluster:
        return FrontierCluster(
            cluster_key=key,
            cell_count=cell_count,
            centroid_xy=centroid_xy,
            bbox_rows=(0, 0),
            bbox_cols=(0, 0),
        )

    def test_filters_by_max_distance(self):
        near = self._make_cluster(key=(0, 0), cell_count=10, centroid_xy=(1.0, 0.0))
        far = self._make_cluster(key=(1, 1), cell_count=10, centroid_xy=(20.0, 0.0))
        ranked = rank_frontiers(
            clusters=[near, far],
            robot_xy=(0.0, 0.0),
            max_distance_m=5.0,
        )
        assert len(ranked) == 1
        assert ranked[0].cluster.cluster_key == (0, 0)

    def test_skips_visited_keys(self):
        a = self._make_cluster(key=(0, 0), cell_count=10, centroid_xy=(1.0, 0.0))
        b = self._make_cluster(key=(1, 1), cell_count=10, centroid_xy=(2.0, 0.0))
        ranked = rank_frontiers(
            clusters=[a, b],
            robot_xy=(0.0, 0.0),
            max_distance_m=10.0,
            visited_keys={(0, 0)},
        )
        assert len(ranked) == 1
        assert ranked[0].cluster.cluster_key == (1, 1)

    def test_score_prefers_large_and_nearby(self):
        small_near = self._make_cluster(
            key=(0, 0), cell_count=10, centroid_xy=(1.0, 0.0)
        )
        big_far = self._make_cluster(
            key=(1, 1), cell_count=200, centroid_xy=(5.0, 0.0)
        )
        ranked = rank_frontiers(
            clusters=[small_near, big_far],
            robot_xy=(0.0, 0.0),
            max_distance_m=10.0,
        )
        # 200/5 = 40 > 10/1 = 10 → big_far wins despite being farther.
        assert ranked[0].cluster.cluster_key == (1, 1)
        assert ranked[1].cluster.cluster_key == (0, 0)

    def test_empty_input(self):
        ranked = rank_frontiers(
            clusters=[],
            robot_xy=(0.0, 0.0),
            max_distance_m=10.0,
        )
        assert ranked == []
