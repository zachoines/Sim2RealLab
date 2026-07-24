"""Unit tests for the planned-path curvature and clearance statistics.

Pure numpy on synthetic grids — no Isaac Sim, no Kit, no pxr, no scene
assets. Grids follow the planner convention: row = x index, col = y index,
origin at the corner of cell (0, 0).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from strafer_lab.tasks.navigation.path_planner import resample_polyline
from strafer_lab.tools import path_statistics as ps

RES = 0.1
ORIGIN = (0.0, 0.0)


def _open_grid(rows=60, cols=60):
    """Occupancy with no obstacles at all."""
    return np.zeros((rows, cols), dtype=np.uint8)


def _two_room_grid(door_cells, rows=60, cols=60, wall_row=30):
    """A wall across the grid with a centred gap of ``door_cells``."""
    occ = np.zeros((rows, cols), dtype=np.uint8)
    occ[wall_row, :] = 1
    lo = (cols - door_cells) // 2
    occ[wall_row, lo:lo + door_cells] = 0
    occ[0, :] = occ[-1, :] = 1
    occ[:, 0] = occ[:, -1] = 1
    return occ


def _inflate(occ, radius_cells):
    from strafer_lab.tools.scene_connectivity import _disc_dilate

    return ~_disc_dilate(occ != 0, radius_cells)


# ---------------------------------------------------------------------------
# turn_density
# ---------------------------------------------------------------------------


class TestTurnDensity:
    def test_straight_line_has_no_turning(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        density, tort, arc, straight = ps.turn_density(path)
        assert density == 0.0
        assert tort == pytest.approx(1.0)
        assert arc == pytest.approx(3.0)
        assert straight == pytest.approx(3.0)

    def test_right_angle_turn(self):
        path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
        density, tort, arc, _ = ps.turn_density(path)
        assert arc == pytest.approx(4.0)
        assert density == pytest.approx((math.pi / 2.0) / 4.0)
        assert tort == pytest.approx(4.0 / math.hypot(2.0, 2.0))

    def test_total_turning_survives_resampling(self):
        """The statistic must not depend on waypoint spacing — the planner
        resamples every path at a fixed arc length."""
        coarse = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [4.0, 2.0]])
        fine = resample_polyline(coarse, 0.05)
        d_coarse, t_coarse, arc_coarse, _ = ps.turn_density(coarse)
        d_fine, t_fine, arc_fine, _ = ps.turn_density(fine)
        assert arc_fine == pytest.approx(arc_coarse, rel=1e-6)
        assert d_fine == pytest.approx(d_coarse, rel=1e-6)
        assert t_fine == pytest.approx(t_coarse, rel=1e-6)

    def test_repeated_waypoint_injects_no_turn(self):
        """Along -x, a zero-length segment's arctan2(0, 0) reads 0 rad against
        the run's own pi, so an unfiltered repeat would fake a full reversal."""
        path = np.array([[2.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        density, _, arc, _ = ps.turn_density(path)
        assert density == 0.0
        assert arc == pytest.approx(2.0)

    def test_heading_wrap_across_the_branch_cut(self):
        """An out-and-back through -x crosses the atan2 discontinuity; without
        the wrap the turn reads as a full circle instead of a half one."""
        path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, -0.001], [0.0, -0.001]])
        density, _, arc, _ = ps.turn_density(path)
        assert density * arc == pytest.approx(math.pi, abs=0.01)

    def test_closed_loop_tortuosity_is_infinite_and_counted_out(self):
        """An infinite tortuosity in the percentiles would make the whole
        summary unserialisable and unreadable."""
        loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        density, tort, arc, straight = ps.turn_density(loop)
        assert math.isinf(tort)
        stub = _stub_stats(loop, np.ones(5))
        closed = ps.PathStats(**{**stub.__dict__, "tortuosity": tort,
                                 "arc_m": arc, "straight_line_m": straight})
        out = ps.summarize([closed])
        assert out["tortuosity"]["n_closed_loops"] == 1
        assert out["tortuosity"]["median"] is None
        json.dumps(out, allow_nan=False)

    def test_rejects_degenerate_input(self):
        with pytest.raises(ValueError):
            ps.turn_density(np.array([[0.0, 0.0]]))

    def test_float32_jitter_on_a_straight_path_reads_as_straight(self):
        """The planner emits float32, so a geometrically straight polyline
        carries per-vertex heading jitter that must not read as turning."""
        from strafer_lab.tasks.navigation.path_planner import plan_path

        path = plan_path(
            np.array([1.5, 2.0]), np.array([4.5, 4.0]),
            np.ones((60, 60), dtype=bool),
            grid_res=RES, grid_origin_xy=ORIGIN,
        )
        assert path.dtype == np.float32
        assert ps.turn_density(path)[0] == 0.0

    def test_a_real_corner_survives_the_deadband(self):
        """The probe is an absolute angle, not a multiple of the constant, so a
        deadband widened past what a corridor produces fails here."""
        turn = math.radians(2.0)
        path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, math.tan(turn)]])
        assert ps.turn_density(path)[0] > 0.0

        # A polyline of many small real turns must still read as turning.
        angles = np.cumsum(np.full(20, math.radians(2.0)))
        steps = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 0.2
        gentle = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
        assert ps.turn_density(gentle)[0] > 0.1


# ---------------------------------------------------------------------------
# waypoint_clearance — the resolution correction
# ---------------------------------------------------------------------------


class TestWaypointClearance:
    def test_measures_to_the_obstacle_face_not_its_centre(self):
        occ = np.zeros((20, 20), dtype=np.uint8)
        occ[10, 10] = 1  # cell centre at (1.05, 1.05), face at x = 1.0
        clearance = ps.waypoint_clearance(
            np.array([[0.5, 1.05]]), occ, grid_res=RES, grid_origin_xy=ORIGIN
        )
        assert clearance[0] == pytest.approx(0.5)

    def test_zero_inside_the_obstacle(self):
        occ = np.zeros((20, 20), dtype=np.uint8)
        occ[10, 10] = 1
        clearance = ps.waypoint_clearance(
            np.array([[1.05, 1.05]]), occ, grid_res=RES, grid_origin_xy=ORIGIN
        )
        assert clearance[0] == pytest.approx(0.0)

    def test_same_geometry_reads_the_same_at_two_resolutions(self):
        """The half-cell correction is what makes a 0.05 m grid and a 0.1 m
        grid comparable; without it they read a systematic half-cell apart."""
        probe = np.array([[1.0, 0.5]])
        readings = []
        for res in (0.05, 0.1):
            n = int(round(4.0 / res))
            occ = np.zeros((n, n), dtype=np.uint8)
            wall_from = int(round(2.0 / res))
            occ[wall_from:, :] = 1  # obstacle half-plane at x = 2.0
            readings.append(
                float(ps.waypoint_clearance(
                    probe, occ, grid_res=res, grid_origin_xy=ORIGIN
                )[0])
            )
        assert readings[0] == pytest.approx(1.0, abs=1e-9)
        assert readings[1] == pytest.approx(1.0, abs=1e-9)

    def test_obstacle_free_grid_is_rejected(self):
        """An infinity here would poison every quantile downstream rather than
        surfacing the degenerate input."""
        occ = _open_grid()
        with pytest.raises(ValueError):
            ps.waypoint_clearance(
                np.array([[1.0, 1.0]]), occ, grid_res=RES, grid_origin_xy=ORIGIN
            )

    def test_grid_origin_is_applied(self):
        """Both production callers pass a non-zero origin; dropping the term
        silently relocates every obstacle."""
        occ = np.zeros((80, 80), dtype=np.uint8)
        occ[40, 40] = 1  # env-local centre when the origin is (-4, -4)
        clearance = ps.waypoint_clearance(
            np.array([[0.05, 0.05]]), occ, grid_res=RES, grid_origin_xy=(-4.0, -4.0)
        )
        assert clearance[0] == pytest.approx(0.0)
        assert ps.waypoint_clearance(
            np.array([[0.05, 0.05]]), occ, grid_res=RES, grid_origin_xy=ORIGIN
        )[0] == pytest.approx(math.hypot(3.95, 3.95), abs=1e-6)

    def test_reports_a_profile_not_one_number_per_path(self):
        occ = np.zeros((80, 80), dtype=np.uint8)
        occ[40, 40] = 1
        path = np.array([[0.05, 0.05], [1.05, 0.05], [2.05, 0.05]])
        clearance = ps.waypoint_clearance(
            path, occ, grid_res=RES, grid_origin_xy=(-4.0, -4.0)
        )
        assert len(clearance) == 3
        assert clearance[0] < clearance[1] < clearance[2]


# ---------------------------------------------------------------------------
# arc_fraction_below
# ---------------------------------------------------------------------------


class TestArcFractionBelow:
    def test_weights_by_arc_length_not_waypoint_count(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        clearance = np.array([0.1, 0.1, 5.0])
        # Segment midpoints: 0.1 (length 1) and 2.55 (length 3).
        assert ps.arc_fraction_below(path, clearance, 1.0) == pytest.approx(0.25)

    def test_all_or_nothing(self):
        path = np.array([[0.0, 0.0], [2.0, 0.0]])
        assert ps.arc_fraction_below(path, np.array([0.1, 0.1]), 1.0) == 1.0
        assert ps.arc_fraction_below(path, np.array([2.0, 2.0]), 1.0) == 0.0

    def test_zero_length_path_is_zero(self):
        path = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert ps.arc_fraction_below(path, np.array([0.0, 0.0]), 1.0) == 0.0


# ---------------------------------------------------------------------------
# excess_clearance / threshold_from_body
# ---------------------------------------------------------------------------


def _stub_stats(path, clearance, inflation=0.2):
    return ps.PathStats(
        path=np.asarray(path, dtype=np.float64),
        arc_m=float(np.hypot(*np.diff(np.asarray(path), axis=0).T).sum()),
        straight_line_m=float(np.hypot(*(np.asarray(path)[-1] - np.asarray(path)[0]))),
        turn_density=0.0,
        tortuosity=1.0,
        clearance_m=np.asarray(clearance, dtype=np.float64),
        inflation_radius_m=inflation,
    )


class TestExcessClearance:
    def test_samples_the_segment_midpoint(self):
        """Taking the leading endpoint instead would shift the whole pooled
        distribution the threshold is read off."""
        s = _stub_stats([[0.0, 0.0], [1.0, 0.0]], [0.2, 0.8], inflation=0.0)
        values, _ = ps.excess_clearance([s])
        assert values == pytest.approx([0.5])

    def test_weighted_quantile_rejects_mismatched_inputs(self):
        with pytest.raises(ValueError):
            ps.weighted_quantile(np.array([1.0, 2.0, 3.0]), np.ones(4), 0.5)
        with pytest.raises(ValueError):
            ps.weighted_quantile(np.array([1.0, 2.0]), np.zeros(2), 0.5)

    def test_subtracts_the_source_inflation_radius(self):
        s = _stub_stats([[0.0, 0.0], [1.0, 0.0]], [0.5, 0.5], inflation=0.2)
        values, weights = ps.excess_clearance([s])
        assert values == pytest.approx([0.3])
        assert weights == pytest.approx([1.0])

    def test_long_path_outweighs_short_path(self):
        short = _stub_stats([[0.0, 0.0], [1.0, 0.0]], [1.0, 1.0])
        long = _stub_stats([[0.0, 0.0], [9.0, 0.0]], [0.0, 0.0])
        values, weights = ps.excess_clearance([short, long])
        assert weights == pytest.approx([1.0, 9.0])
        # 90% of the arc sits on the low-clearance path.
        assert ps.weighted_quantile(values, weights, 0.5) == pytest.approx(-0.2)
        assert ps.weighted_quantile(values, weights, 0.95) == pytest.approx(0.8)

    def test_weighted_quantile_never_interpolates(self):
        """An interpolated quantile invents a corridor width no waypoint has."""
        values = np.array([0.1, 0.9])
        weights = np.array([1.0, 1.0])
        for q in (0.25, 0.5, 0.75):
            assert ps.weighted_quantile(values, weights, q) in (0.1, 0.9)

    def test_threshold_comes_from_the_body_not_the_tail(self):
        """A threshold read off the tightest passage reads flat across every
        corridor wider than it, which cannot gate a widening lever."""
        wide = [_stub_stats([[0.0, 0.0], [10.0, 0.0]], [1.0, 1.0]) for _ in range(9)]
        pinch = [_stub_stats([[0.0, 0.0], [2.0, 0.0]], [0.25, 0.25])]
        reference = wide + pinch  # the pinch is 2 m of 92 m of arc
        body = ps.threshold_from_body(reference, 0.5)
        tail = ps.threshold_from_body(reference, 0.01)
        assert body == pytest.approx(0.8)
        assert tail == pytest.approx(0.05)
        # Only the body threshold registers a mid-clearance population.
        mid = [_stub_stats([[0.0, 0.0], [10.0, 0.0]], [0.6, 0.6])]
        assert ps.summarize(mid, excess_thresholds=(body,))["arc_below_excess"][
            ps._threshold_key(body)
        ] == pytest.approx(1.0)
        assert ps.summarize(mid, excess_thresholds=(tail,))["arc_below_excess"][
            ps._threshold_key(tail)
        ] == pytest.approx(0.0)

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError):
            ps.threshold_from_body([])


# ---------------------------------------------------------------------------
# endpoint sampling
# ---------------------------------------------------------------------------


class TestSampleEndpointPairs:
    def test_respects_minimum_separation(self):
        pts = np.stack(np.meshgrid(np.arange(6) * 0.5, np.arange(6) * 0.5), -1).reshape(-1, 2)
        pairs = ps.sample_endpoint_pairs(
            pts, np.random.default_rng(0), n_pairs=50, min_separation_m=1.5
        )
        assert pairs
        for a, b in pairs:
            assert float(np.hypot(*(b - a))) >= 1.5

    def test_deterministic_for_a_given_generator_seed(self):
        pts = np.arange(40, dtype=np.float64).reshape(-1, 2)
        a = ps.sample_endpoint_pairs(pts, np.random.default_rng(7), n_pairs=10,
                                     min_separation_m=0.0)
        b = ps.sample_endpoint_pairs(pts, np.random.default_rng(7), n_pairs=10,
                                     min_separation_m=0.0)
        assert [(p.tolist(), q.tolist()) for p, q in a] == [
            (p.tolist(), q.tolist()) for p, q in b
        ]

    def test_leaves_the_torch_global_stream_untouched(self):
        """The generator's RNG consumption is a frozen contract, so nothing
        here may draw from the stream it shares."""
        import torch

        torch.manual_seed(1234)
        before = torch.random.get_rng_state().clone()
        pts = np.arange(40, dtype=np.float64).reshape(-1, 2)
        ps.sample_endpoint_pairs(pts, np.random.default_rng(1), n_pairs=10,
                                 min_separation_m=0.0)
        assert torch.equal(torch.random.get_rng_state(), before)

    @pytest.mark.parametrize("n_points", [0, 1])
    def test_too_few_points_yields_nothing(self, n_points):
        """An empty pool must return, not raise: rng.integers(0, 0) errors."""
        assert ps.sample_endpoint_pairs(
            np.zeros((n_points, 2)), np.random.default_rng(0),
            n_pairs=5, min_separation_m=0.0,
        ) == []


# ---------------------------------------------------------------------------
# plan_over_pairs / summarize
# ---------------------------------------------------------------------------


class TestPlanOverPairs:
    def test_an_obstacle_free_grid_is_counted_not_raised(self):
        """Open-field difficulties really do generate these, and one of them
        must not abort the sweep it appears in."""
        occ = _open_grid()
        free = np.ones((60, 60), dtype=bool)
        pairs = [(np.array([1.0, 1.0]), np.array([4.0, 1.0]))] * 3
        stats, failures = ps.plan_over_pairs(
            pairs, free, occ, grid_res=RES, grid_origin_xy=ORIGIN,
            inflation_radius_m=0.3,
        )
        assert stats == []
        assert failures["no_obstacles"] == 3

    def test_the_group_label_reaches_the_measured_path(self):
        """The group is what the interval resamples; losing it here silently
        turns every clustered interval into a per-path one."""
        occ = _two_room_grid(door_cells=24)
        stats, _ = ps.plan_over_pairs(
            [(np.array([1.0, 3.0]), np.array([4.0, 3.0]))], _inflate(occ, 3), occ,
            grid_res=RES, grid_origin_xy=ORIGIN, inflation_radius_m=0.3,
            group="r0e7",
        )
        assert [s.group for s in stats] == ["r0e7"]

    def test_counts_unplannable_pairs_instead_of_dropping_them(self):
        occ = _two_room_grid(door_cells=0)
        free = _inflate(occ, 3)
        pairs = [(np.array([1.0, 3.0]), np.array([4.0, 3.0]))]
        stats, failures = ps.plan_over_pairs(
            pairs, free, occ, grid_res=RES, grid_origin_xy=ORIGIN,
            inflation_radius_m=0.3,
        )
        assert stats == []
        assert failures["no_path"] == 1

    def test_measures_a_plannable_pair(self):
        occ = _two_room_grid(door_cells=24)
        free = _inflate(occ, 3)
        stats, failures = ps.plan_over_pairs(
            [(np.array([1.0, 3.0]), np.array([4.0, 3.0]))], free, occ,
            grid_res=RES, grid_origin_xy=ORIGIN, inflation_radius_m=0.3,
        )
        assert failures == {"no_path": 0, "invalid_endpoint": 0, "no_obstacles": 0}
        assert len(stats) == 1
        assert stats[0].turn_density == pytest.approx(0.0, abs=1e-9)
        assert np.isfinite(stats[0].min_clearance_m)


class TestSummarize:
    def test_band_filters_by_straight_line_distance_not_arc(self):
        """A dog-leg has a long arc between near endpoints — exactly the
        corridor-threading case — so banding on arc would select it out."""
        dog_leg = _stub_stats(
            [[0.0, 0.0], [0.0, 3.0], [3.0, 3.0], [3.0, 0.0]], np.ones(4)
        )
        assert dog_leg.arc_m == pytest.approx(9.0)
        assert dog_leg.straight_line_m == pytest.approx(3.0)
        assert ps.summarize([dog_leg], straight_line_band=(1.5, 4.0))["n_paths"] == 1
        assert ps.summarize([dog_leg], straight_line_band=(8.0, 10.0))["n_paths"] == 0

    def test_band_reports_what_it_dropped(self):
        short = _stub_stats([[0.0, 0.0], [1.0, 0.0]], [1.0, 1.0])
        long = _stub_stats([[0.0, 0.0], [9.0, 0.0]], [1.0, 1.0])
        out = ps.summarize([short, long], straight_line_band=(2.0, 12.0))
        assert out["n_paths"] == 1
        assert out["n_paths_before_band"] == 2
        assert out["straight_line_band_m"] == [2.0, 12.0]

    def test_turning_fraction_reads_where_every_percentile_is_pinned(self):
        straight = [_stub_stats([[0.0, 0.0], [5.0, 0.0]], [1.0, 1.0]) for _ in range(19)]
        bent = _stub_stats([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], [1.0, 1.0, 1.0])
        bent = ps.PathStats(**{**bent.__dict__, "turn_density": 0.4, "tortuosity": 1.4})
        out = ps.summarize(straight + [bent])
        assert out["turn_density_rad_per_m"]["median"] == 0.0
        assert out["turn_density_rad_per_m"]["p90"] == 0.0
        assert out["frac_paths_turning"] == pytest.approx(0.05)
        assert out["frac_paths_bending"] == pytest.approx(0.05)

    def test_empty_population_reports_no_paths(self):
        out = ps.summarize([])
        assert out["n_paths"] == 0
        assert "turn_density_rad_per_m" not in out


# ---------------------------------------------------------------------------
# The statistics must move with the lever they are meant to gate
# ---------------------------------------------------------------------------


class TestSensitivityToTopology:
    def test_an_internal_wall_bends_paths_an_open_room_leaves_straight(self):
        # A perimeter-only grid stands in for the "open room": the clearance
        # statistic needs something to measure against.
        open_occ = _two_room_grid(door_cells=58)
        open_free = _inflate(open_occ, 3)
        walled_occ = _two_room_grid(door_cells=14)
        walled_free = _inflate(walled_occ, 3)
        # The straight line between these misses the gap, so the wall forces a
        # detour and the open grid does not.
        pair = [(np.array([1.5, 1.0]), np.array([4.5, 1.5]))]
        kw = dict(grid_res=RES, grid_origin_xy=ORIGIN, inflation_radius_m=0.3)

        open_stats, _ = ps.plan_over_pairs(pair, open_free, open_occ, **kw)
        walled_stats, _ = ps.plan_over_pairs(pair, walled_free, walled_occ, **kw)

        assert open_stats[0].turn_density == pytest.approx(0.0, abs=1e-9)
        assert walled_stats[0].turn_density > 0.05
        assert walled_stats[0].tortuosity > open_stats[0].tortuosity

    def test_a_wider_aperture_raises_clearance_through_it(self):
        """The threading statistic has to register ordinary gaps, not only
        gaps narrow enough to be doorways."""
        pair = [(np.array([1.5, 3.0]), np.array([4.5, 3.0]))]
        kw = dict(grid_res=RES, grid_origin_xy=ORIGIN, inflation_radius_m=0.3)
        narrow_occ = _two_room_grid(door_cells=10)
        wide_occ = _two_room_grid(door_cells=24)
        narrow, _ = ps.plan_over_pairs(pair, _inflate(narrow_occ, 3), narrow_occ, **kw)
        wide, _ = ps.plan_over_pairs(pair, _inflate(wide_occ, 3), wide_occ, **kw)
        assert narrow and wide
        assert wide[0].min_clearance_m > narrow[0].min_clearance_m

        tau = narrow[0].min_excess_clearance_m + 0.15
        narrow_frac = ps.summarize(narrow, excess_thresholds=(tau,))
        wide_frac = ps.summarize(wide, excess_thresholds=(tau,))
        key = ps._threshold_key(tau)
        assert (narrow_frac["arc_below_excess"][key]
                > wide_frac["arc_below_excess"][key])


class TestBootstrapGates:
    def test_interval_widens_when_paths_share_a_room(self):
        """Paths from one room are one observation, not six; resampling paths
        would report an interval several times too tight."""
        straight = [_stub_stats([[0.0, 0.0], [5.0, 0.0]], [1.0, 1.0]) for _ in range(60)]
        bent = [_stub_stats([[0.0, 0.0], [5.0, 0.0]], [1.0, 1.0]) for _ in range(60)]
        bent = [ps.PathStats(**{**b.__dict__, "tortuosity": 1.4}) for b in bent]
        population = straight + bent

        per_path = [ps.PathStats(**{**s.__dict__, "group": f"g{i}"})
                    for i, s in enumerate(population)]
        clustered = [ps.PathStats(**{**s.__dict__, "group": f"g{i // 30}"})
                     for i, s in enumerate(population)]

        wide = ps.bootstrap_gates(clustered, resamples=400, seed=1)
        tight = ps.bootstrap_gates(per_path, resamples=400, seed=1)
        assert wide["n_groups"] == 4
        assert tight["n_groups"] == 120
        assert wide["frac_paths_bending"]["value"] == pytest.approx(0.5)
        wide_width = (wide["frac_paths_bending"]["ci_hi"]
                      - wide["frac_paths_bending"]["ci_lo"])
        tight_width = (tight["frac_paths_bending"]["ci_hi"]
                       - tight["frac_paths_bending"]["ci_lo"])
        assert wide_width > 2.0 * tight_width

    def test_a_single_group_reports_no_interval(self):
        """One group resamples to itself, so a zero-width interval would be an
        artifact quoted as a 95% CI."""
        stats = [ps.PathStats(**{**_stub_stats([[0.0, 0.0], [5.0, 0.0]],
                                               [1.0, 1.0]).__dict__, "group": "one"})
                 for _ in range(10)]
        g = ps.bootstrap_gates(stats, resamples=50)
        assert g["n_groups"] == 1
        assert g["frac_paths_turning"]["ci_lo"] is None
        assert g["frac_paths_turning"]["ci_hi"] is None

    def test_the_requested_level_is_the_one_reported(self):
        stats = [ps.PathStats(**{**_stub_stats([[0.0, 0.0], [5.0, 0.0]],
                                               [float(i), float(i)]).__dict__,
                                 "group": f"g{i}"})
                 for i in range(60)]
        wide = ps.bootstrap_gates(stats, resamples=1500, level=0.95, seed=3)
        narrow = ps.bootstrap_gates(stats, resamples=1500, level=0.50, seed=3)
        assert wide["level"] == 0.95 and narrow["level"] == 0.50
        w = wide["min_clearance_median"]
        m = narrow["min_clearance_median"]
        assert (w["ci_hi"] - w["ci_lo"]) > 1.5 * (m["ci_hi"] - m["ci_lo"])

    def test_ungrouped_population_falls_back_to_one_group_per_path(self):
        stats = [_stub_stats([[0.0, 0.0], [5.0, 0.0]], [1.0, 1.0]) for _ in range(10)]
        assert ps.bootstrap_gates(stats, resamples=50)["n_groups"] == 10

    def test_empty_population_reports_nothing(self):
        assert ps.bootstrap_gates([])["n_paths"] == 0
