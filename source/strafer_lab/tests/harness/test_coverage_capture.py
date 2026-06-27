"""Pure-Python tests for the scripted coverage capture driver.

Exercise the driver's CLI surface, the policy-variant -> capture-env mapping,
and the per-leg path planner (``_leg_path``) without launching Isaac Sim — the
Isaac imports live inside ``main()``, so the module is importable plain. The
live env loop (the CaptureSubgoalCommand rolling the subgoal + the rsl_rl
runner stepping) is covered by the Kit smoke, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from strafer_lab.tools import scene_connectivity as sc
from strafer_lab.tools.coverage_plan import CoveragePlan, VisitWaypoint

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import coverage_capture as cc  # noqa: E402  (post-path-mutation import)


class _FakePlanError(Exception):
    pass


def _ok_plan_path(start, goal, free_space, *, grid_res, grid_origin_xy):
    """Stand-in planner that always connects start to goal."""
    return np.asarray([start, goal], dtype=np.float32)


def _plan(targets):
    """Build a CoveragePlan from a list of viewpoint XY (one visit each)."""
    waypoints = tuple(
        VisitWaypoint(
            room_index=i,
            visit_ordinal=0,
            target_xy=(float(x), float(y)),
            approach_heading_rad=0.0,
        )
        for i, (x, y) in enumerate(targets)
    )
    return CoveragePlan(
        waypoints=waypoints,
        visits_per_room=1,
        heading_spread_threshold_rad=1.5708,
        seed=0,
    )


def _occ(grid, origin_xy=(0.0, 0.0), res=0.1):
    return sc.CachedOccupancy(
        grid=grid, origin_xy=origin_xy, resolution_m=res, z_slice_m=0.0, meta={},
    )


class TestCli:
    def test_checkpoint_required(self):
        parser = cc._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_known_args(["--scene", "s", "--output", "/tmp/o"])

    def test_defaults(self):
        parser = cc._build_parser()
        args, _ = parser.parse_known_args(
            ["--scene", "s", "--output", "/tmp/o", "--checkpoint", "/m.pt"],
        )
        assert args.policy_variant == "nocam_subgoal"
        assert args.env is None
        assert args.num_envs == 1
        assert args.lookahead_m is None

    def test_unknown_flags_pass_through(self):
        # AppLauncher / pass-through flags must survive parse_known_args.
        parser = cc._build_parser()
        _, extra = parser.parse_known_args(
            ["--scene", "s", "--output", "/tmp/o", "--checkpoint", "/m.pt",
             "--headless", "--device", "cpu"],
        )
        assert "--headless" in extra and "--device" in extra


class TestVariantEnvMapping:
    def test_nocam_subgoal_maps_to_coverage_env(self):
        assert (
            cc._CAPTURE_ENV_BY_VARIANT["nocam_subgoal"]
            == "Isaac-Strafer-Nav-Capture-Coverage-v0"
        )


class TestSceneDirFor:
    def test_resolves_scene_directory(self):
        usd = Path("/x/Assets/generated/scenes/scene_foo_000/scene_foo_000.usdc")
        assert cc._scene_dir_for(usd).name == "scene_foo_000"


class TestDeriveSpawnXy:
    """The capture spawn is drawn from the occupancy free-space the driver
    already loads — a passable, plan-reachable cell in the occupancy frame —
    so it never depends on an external spawn list or a frame the grid
    disagrees with."""

    def _free_block(self, origin=(0.0, 0.0), res=0.1):
        """20x20 raw grid blocked except an interior block; returns (raw, free)."""
        raw = np.ones((20, 20), dtype=np.uint8)
        raw[6:14, 6:14] = 0
        free = sc.occupancy_to_free_space(raw, grid_res=res)
        return raw, free

    def test_raster_maps_grid_cell_to_local_xy(self):
        # No free viewpoint -> the deterministic free-cell raster runs, and its
        # cell->XY uses the occupancy origin/resolution (row=+X, col=+Y, the
        # +res/2 center offset).
        origin, res = (1.0, -2.0), 0.1
        raw, free = self._free_block(origin, res)
        plan = _plan([(0.0, 0.0)])  # maps outside the grid in this frame -> not free
        out = cc._derive_spawn_xy(
            free, plan, _occ(raw, origin, res),
            plan_path=_ok_plan_path, invalid_endpoint_errors=(_FakePlanError,),
        )
        r, c = (int(v) for v in np.argwhere(free)[0])
        assert out == [r * res + origin[0] + res / 2.0, c * res + origin[1] + res / 2.0]
        # and it round-trips back to that same free cell
        assert sc._xy_to_cell((out[0], out[1]), origin, res) == (r, c)
        assert free[r, c]

    def test_blocked_viewpoint_advances_to_free_cell(self):
        # The first viewpoint landed on a blocked cell (centroid under furniture);
        # the derived spawn is the next free viewpoint, free in the inflated grid
        # and never a raw-occupied cell.
        origin, res = (0.0, 0.0), 0.1
        raw, free = self._free_block(origin, res)
        blocked_xy = sc._cell_to_xy(0, 0, origin, res)
        fr, fc = (int(v) for v in np.argwhere(free)[0])
        free_xy = sc._cell_to_xy(fr, fc, origin, res)
        plan = _plan([blocked_xy, free_xy])
        out = cc._derive_spawn_xy(
            free, plan, _occ(raw, origin, res),
            plan_path=_ok_plan_path, invalid_endpoint_errors=(_FakePlanError,),
        )
        r, c = sc._xy_to_cell((out[0], out[1]), origin, res)
        assert free[r, c]
        assert raw[r, c] == 0
        assert out == [float(free_xy[0]), float(free_xy[1])]

    def test_disconnected_pocket_advances_to_reachable(self):
        # The first viewpoint is free but stranded in a disconnected pocket (its
        # first leg never plans); the spawn advances to the reachable viewpoint.
        origin, res = (0.0, 0.0), 0.1
        free = np.ones((40, 40), dtype=bool)
        occ = _occ(np.zeros((40, 40), dtype=np.uint8), origin, res)
        pocket = sc._cell_to_xy(2, 2, origin, res)
        good = sc._cell_to_xy(20, 20, origin, res)

        def fake_plan_path(start, goal, free_space, *, grid_res, grid_origin_xy):
            if np.allclose(start, pocket, atol=1e-6):
                raise _FakePlanError
            return np.asarray([start, goal], dtype=np.float32)

        out = cc._derive_spawn_xy(
            free, _plan([pocket, good]), occ,
            plan_path=fake_plan_path, invalid_endpoint_errors=(_FakePlanError,),
        )
        assert out == [float(good[0]), float(good[1])]

    def test_deterministic(self):
        origin, res = (0.0, 0.0), 0.1
        raw, free = self._free_block(origin, res)
        fr, fc = (int(v) for v in np.argwhere(free)[0])
        plan = _plan([sc._cell_to_xy(fr, fc, origin, res)])
        occ = _occ(raw, origin, res)
        kw = dict(plan_path=_ok_plan_path, invalid_endpoint_errors=(_FakePlanError,))
        assert cc._derive_spawn_xy(free, plan, occ, **kw) == cc._derive_spawn_xy(
            free, plan, occ, **kw
        )

    def test_degenerate_grid_raises(self):
        # No free cell anywhere -> a regenerate-occupancy blocker, surfaced loudly.
        occ = _occ(np.ones((10, 10), dtype=np.uint8), (0.0, 0.0), 0.1)
        with pytest.raises(RuntimeError):
            cc._derive_spawn_xy(
                np.zeros((10, 10), dtype=bool), _plan([(0.5, 0.5)]), occ,
                plan_path=_ok_plan_path, invalid_endpoint_errors=(_FakePlanError,),
            )


class TestLegPath:
    """``_leg_path`` stages one approach_distance behind the viewpoint along the
    approach heading, then a straight final segment to the viewpoint."""

    def _free(self):
        return np.ones((200, 200), dtype=bool)

    def test_staged_path_ends_at_target_facing_heading(self):
        target = np.array([5.0, 5.0], dtype=np.float32)
        calls: list[tuple] = []

        def fake_plan_path(start, goal, free_space, *, grid_res, grid_origin_xy):
            calls.append((tuple(start), tuple(goal)))
            return np.asarray([start, goal], dtype=np.float32)

        leg = cc._leg_path(
            np.array([0.0, 0.0], dtype=np.float32), target, 0.0, self._free(),
            grid_res=0.05, grid_origin_xy=(0.0, 0.0), approach_distance_m=0.6,
            plan_path=fake_plan_path, error_cls=_FakePlanError,
        )
        # Last waypoint is the viewpoint; the penultimate is the staging point
        # one approach_distance behind it along the heading (heading 0 -> -x).
        assert np.allclose(leg[-1], target)
        assert np.allclose(leg[-2], [target[0] - 0.6, target[1]], atol=1e-5)

    def test_falls_back_to_direct_when_staging_unplannable(self):
        target = np.array([5.0, 5.0], dtype=np.float32)
        state = {"n": 0}

        def fake_plan_path(start, goal, free_space, *, grid_res, grid_origin_xy):
            state["n"] += 1
            if state["n"] == 1:  # staged approach fails
                raise _FakePlanError
            return np.asarray([start, goal], dtype=np.float32)

        leg = cc._leg_path(
            np.array([0.0, 0.0], dtype=np.float32), target, 1.0, self._free(),
            grid_res=0.05, grid_origin_xy=(0.0, 0.0), approach_distance_m=0.6,
            plan_path=fake_plan_path, error_cls=_FakePlanError,
        )
        assert np.allclose(leg[-1], target)

    def test_returns_none_when_unplannable(self):
        def fake_plan_path(start, goal, free_space, *, grid_res, grid_origin_xy):
            raise _FakePlanError

        leg = cc._leg_path(
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([5.0, 5.0], dtype=np.float32), 0.0, self._free(),
            grid_res=0.05, grid_origin_xy=(0.0, 0.0), approach_distance_m=0.6,
            plan_path=fake_plan_path, error_cls=_FakePlanError,
        )
        assert leg is None
