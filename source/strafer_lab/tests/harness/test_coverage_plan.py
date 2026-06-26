"""Unit tests for the geometric coverage planner.

Pure-Python: the planner is driven with a hand-built rooms fixture + an
all-free grid, never a USD read. The acceptance evidence is the coverage
metric — every room visited >= N from headings spread past the threshold —
plus seed determinism.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strafer_lab.tools import coverage_plan as cp

RES = 0.05
ORIGIN = (0.0, 0.0)


def _aabb(p0, p1):
    (x0, y0), (x1, y1) = p0, p1
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _rooms():
    """Three rooms laid left-to-right on the free grid."""
    return [
        {"room_type": "living_room", "story": 0, "area_m2": 6.0,
         "footprint_xy": _aabb((0.5, 0.5), (2.5, 3.5))},
        {"room_type": "kitchen", "story": 0, "area_m2": 6.0,
         "footprint_xy": _aabb((3.5, 0.5), (5.5, 3.5))},
        {"room_type": "bedroom", "story": 0, "area_m2": 6.0,
         "footprint_xy": _aabb((6.5, 0.5), (8.5, 3.5))},
    ]


def _free_grid(nx=200, ny=80):
    return np.ones((nx, ny), dtype=bool)


def _plan(visits_per_room=2, seed=0):
    return cp.build_coverage_plan(
        _rooms(), _free_grid(),
        grid_res=RES, grid_origin_xy=ORIGIN,
        visits_per_room=visits_per_room, seed=seed,
    )


class TestCoverageMetric:
    def test_every_room_visited_at_least_n(self):
        plan = _plan(visits_per_room=2)
        metric = cp.coverage_metric(plan)
        assert len(metric.rooms) == 3
        for room in metric.rooms:
            assert room.visit_count >= 2

    def test_per_room_heading_spread_exceeds_threshold(self):
        plan = _plan(visits_per_room=2)
        metric = cp.coverage_metric(plan)
        for room in metric.rooms:
            assert room.max_pairwise_heading_gap_rad > metric.heading_spread_threshold_rad

    def test_metric_satisfied(self):
        assert cp.coverage_metric(_plan(visits_per_room=2)).satisfied

    @pytest.mark.parametrize("visits", [2, 3])
    def test_metric_satisfied_across_seeds(self, visits):
        # The heading-spread guarantee must hold for every seed, not just the
        # default — the n=2 case is the tight one (single pair near pi).
        for seed in range(64):
            metric = cp.coverage_metric(_plan(visits_per_room=visits, seed=seed))
            assert metric.satisfied, f"seed={seed} visits={visits} not satisfied"

    def test_three_visits_also_spread(self):
        metric = cp.coverage_metric(_plan(visits_per_room=3))
        assert metric.satisfied
        for room in metric.rooms:
            assert room.visit_count == 3

    def test_single_visit_fails_spread(self):
        # One visit per room cannot demonstrate heading diversity.
        plan = _plan(visits_per_room=1)
        metric = cp.coverage_metric(plan)
        assert not metric.satisfied
        for room in metric.rooms:
            assert room.max_pairwise_heading_gap_rad == 0.0


class TestDeterminism:
    def test_same_seed_identical_plan(self):
        assert _plan(seed=7).waypoints == _plan(seed=7).waypoints

    def test_different_seed_changes_plan(self):
        assert _plan(seed=7).waypoints != _plan(seed=8).waypoints


class TestPlanShape:
    def test_waypoint_count_is_rooms_times_visits(self):
        plan = _plan(visits_per_room=2)
        assert len(plan.waypoints) == 3 * 2

    def test_revisits_share_target_but_differ_in_heading(self):
        plan = _plan(visits_per_room=2)
        by_room: dict[int, list] = {}
        for wp in plan.waypoints:
            by_room.setdefault(wp.room_index, []).append(wp)
        for visits in by_room.values():
            targets = {v.target_xy for v in visits}
            assert len(targets) == 1  # same place
            headings = [v.approach_heading_rad for v in visits]
            assert cp._circular_delta(headings[0], headings[1]) > math.pi / 2

    def test_targets_lie_on_a_free_grid_cell(self):
        plan = _plan()
        free = _free_grid()
        nrows, ncols = free.shape
        for wp in plan.waypoints:
            # planner maps x -> row, y -> col (path_planner._xy_to_cell).
            row = int((wp.target_xy[0] - ORIGIN[0]) / RES)
            col = int((wp.target_xy[1] - ORIGIN[1]) / RES)
            assert 0 <= row < nrows and 0 <= col < ncols
            assert free[row, col]  # representative point is a free interior cell

    def test_empty_rooms_yields_empty_plan(self):
        plan = cp.build_coverage_plan(
            [], _free_grid(), grid_res=RES, grid_origin_xy=ORIGIN,
        )
        assert plan.waypoints == ()

    def test_visits_per_room_must_be_positive(self):
        with pytest.raises(ValueError):
            _plan(visits_per_room=0)
