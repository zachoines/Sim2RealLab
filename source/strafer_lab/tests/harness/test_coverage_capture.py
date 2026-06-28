"""Pure-Python tests for the scripted coverage capture driver.

Exercise the driver's CLI surface, the policy-variant -> capture-env mapping,
and the per-leg path planner (``_leg_path``) without launching Isaac Sim — the
Isaac imports live inside ``main()``, so the module is importable plain. The
live env loop (the CaptureSubgoalCommand rolling the subgoal + the rsl_rl
runner stepping) is covered by the Kit smoke, not here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from strafer_lab.tools import scene_connectivity as sc
from strafer_lab.tools import scene_metadata_reader as smr
from strafer_lab.tools.coverage_plan import CoveragePlan, VisitWaypoint
from strafer_lab.tools.lerobot_writer import hash_scene_metadata

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import coverage_capture as cc  # noqa: E402  (post-path-mutation import)


def _rooms_covering(shape, origin=(0.0, 0.0), res=0.1):
    """One room whose footprint covers the whole grid AABB.

    Containment is then always satisfied, so the spawn-derivation tests that
    predate the in-room guard exercise the same free/plannable logic unchanged.
    """
    rows, cols = shape
    lo = (origin[0] - res, origin[1] - res)
    hi = (origin[0] + rows * res + res, origin[1] + cols * res + res)
    return [{"footprint_xy": sc.aabb_to_footprint(lo, hi)}]


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
            rooms=_rooms_covering(free.shape, origin, res),
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
            rooms=_rooms_covering(free.shape, origin, res),
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
            rooms=_rooms_covering(free.shape, origin, res),
            plan_path=fake_plan_path, invalid_endpoint_errors=(_FakePlanError,),
        )
        assert out == [float(good[0]), float(good[1])]

    def test_deterministic(self):
        origin, res = (0.0, 0.0), 0.1
        raw, free = self._free_block(origin, res)
        fr, fc = (int(v) for v in np.argwhere(free)[0])
        plan = _plan([sc._cell_to_xy(fr, fc, origin, res)])
        occ = _occ(raw, origin, res)
        kw = dict(
            rooms=_rooms_covering(free.shape, origin, res),
            plan_path=_ok_plan_path, invalid_endpoint_errors=(_FakePlanError,),
        )
        assert cc._derive_spawn_xy(free, plan, occ, **kw) == cc._derive_spawn_xy(
            free, plan, occ, **kw
        )

    def test_degenerate_grid_raises(self):
        # No free cell anywhere -> a regenerate-occupancy blocker, surfaced loudly.
        occ = _occ(np.ones((10, 10), dtype=np.uint8), (0.0, 0.0), 0.1)
        with pytest.raises(RuntimeError):
            cc._derive_spawn_xy(
                np.zeros((10, 10), dtype=bool), _plan([(0.5, 0.5)]), occ,
                rooms=_rooms_covering((10, 10), (0.0, 0.0), 0.1),
                plan_path=_ok_plan_path, invalid_endpoint_errors=(_FakePlanError,),
            )


class TestSpawnContainment:
    """The in-room guard closes the porous-exterior hole: a free + reachable
    cell that is outside every room footprint (e.g. the exterior corner the
    row-major fallback scans first) must never be returned as a spawn."""

    def test_fallback_skips_exterior_corner_for_in_room_cell(self):
        # Whole grid free (incl. the exterior corner cell (0,0)); the only room
        # covers a central patch. The plan viewpoint is outside the room, so the
        # primary path rejects it and the row-major fallback runs — its first
        # free cell is the exterior corner, which the in-room guard skips.
        origin, res = (0.0, 0.0), 0.1
        free = np.ones((20, 20), dtype=bool)
        occ = _occ(np.zeros((20, 20), dtype=np.uint8), origin, res)
        room = {"footprint_xy": sc.aabb_to_footprint((0.8, 0.8), (1.2, 1.2))}
        out = cc._derive_spawn_xy(
            free, _plan([(1.8, 1.8)]), occ, rooms=[room],
            plan_path=_ok_plan_path, invalid_endpoint_errors=(_FakePlanError,),
        )
        assert sc.point_in_any_room(out[0], out[1], [room])
        corner = list(sc._cell_to_xy(0, 0, origin, res))
        assert out != corner

    def test_no_in_room_free_cell_raises(self):
        # The room footprint maps entirely off the grid, so no free cell is
        # in-room — a regenerate-occupancy blocker surfaced loudly.
        origin, res = (0.0, 0.0), 0.1
        free = np.ones((20, 20), dtype=bool)
        occ = _occ(np.zeros((20, 20), dtype=np.uint8), origin, res)
        room = {"footprint_xy": sc.aabb_to_footprint((100.0, 100.0), (101.0, 101.0))}
        with pytest.raises(RuntimeError):
            cc._derive_spawn_xy(
                free, _plan([(1.0, 1.0)]), occ, rooms=[room],
                plan_path=_ok_plan_path, invalid_endpoint_errors=(_FakePlanError,),
            )


class TestValidateSpawnReady:
    """Pre-capture grid gate: spawn must be in-room and the first leg must be
    in-room and plannable, else a clear SystemExit."""

    _occ_ = staticmethod(lambda: _occ(np.zeros((20, 20), np.uint8), (0.0, 0.0), 0.1))

    def _room(self):
        return [{"footprint_xy": sc.aabb_to_footprint((0.0, 0.0), (2.0, 2.0))}]

    def _kw(self):
        return dict(
            plan_path=_ok_plan_path,
            invalid_endpoint_errors=(_FakePlanError,),
            point_in_any_room=sc.point_in_any_room,
        )

    def test_passes_in_room_and_plannable(self):
        free = np.ones((20, 20), dtype=bool)
        cc._validate_spawn_ready(
            [1.0, 1.0], _plan([(1.0, 1.0), (0.5, 0.5)]), self._room(), free,
            self._occ_(), **self._kw(),
        )

    def test_spawn_out_of_room_raises(self):
        free = np.ones((20, 20), dtype=bool)
        with pytest.raises(SystemExit):
            cc._validate_spawn_ready(
                [5.0, 5.0], _plan([(1.0, 1.0), (0.5, 0.5)]), self._room(), free,
                self._occ_(), **self._kw(),
            )

    def test_leg_target_out_of_room_raises(self):
        free = np.ones((20, 20), dtype=bool)
        with pytest.raises(SystemExit):
            cc._validate_spawn_ready(
                [1.0, 1.0], _plan([(1.0, 1.0), (5.0, 5.0)]), self._room(), free,
                self._occ_(), **self._kw(),
            )

    def test_leg_unplannable_raises(self):
        free = np.ones((20, 20), dtype=bool)

        def _fail(start, goal, free_space, *, grid_res, grid_origin_xy):
            raise _FakePlanError

        kw = dict(self._kw())
        kw["plan_path"] = _fail
        with pytest.raises(SystemExit):
            cc._validate_spawn_ready(
                [1.0, 1.0], _plan([(1.0, 1.0), (0.5, 0.5)]), self._room(), free,
                self._occ_(), **kw,
            )


class _FakePrim:
    def __init__(self, custom_data=None, children=()):
        self._custom = custom_data or {}
        self._children = list(children)

    def IsValid(self):
        return True

    def GetCustomDataByKey(self, key):
        return self._custom.get(key)

    def GetChildren(self):
        return self._children


class _FakeStage:
    def __init__(self, prim):
        self._prim = prim

    def GetPrimAtPath(self, path):
        return self._prim


class TestSceneIdentity:
    """Pre-traversal runtime gate: the loaded geometry prim's embedded hash
    must match the requested scene; falls back to a cfg usd_path check only
    when the live prim exposes no metadata to hash."""

    _META = {"rooms": [{"footprint_xy": [[0, 0], [1, 0], [1, 1]]}],
             "objects": [], "room_adjacency": []}
    KEY = smr.CUSTOM_DATA_KEY

    def _prim(self, meta):
        return _FakePrim(custom_data={self.KEY: json.dumps(meta, sort_keys=True)})

    def _call(self, stage, cfg_usd="/x/seed7.usdc", expected_usd="/x/seed7.usdc"):
        cc._assert_loaded_scene_identity(
            stage,
            geometry_prim_path="/World/Room",
            cfg_usd_path=cfg_usd,
            expected_usd_path=expected_usd,
            expected_metadata=self._META,
            hash_fn=hash_scene_metadata,
            prim_metadata_reader=smr.metadata_from_prim,
        )

    def test_matching_hash_passes(self):
        self._call(_FakeStage(self._prim(self._META)))

    def test_mismatched_hash_raises(self):
        other = {"rooms": [], "objects": [], "room_adjacency": [["a"]]}
        with pytest.raises(SystemExit):
            self._call(_FakeStage(self._prim(other)))

    def test_metadata_on_child_prim_is_found(self):
        root = _FakePrim(children=[self._prim(self._META)])
        self._call(_FakeStage(root))

    def test_fallback_cfg_path_match_passes(self):
        # No customData on the prim -> fall back to cfg usd_path equality.
        bare = _FakePrim()
        self._call(_FakeStage(bare), cfg_usd="/x/seed7.usdc", expected_usd="/x/seed7.usdc")

    def test_fallback_cfg_path_mismatch_raises(self):
        bare = _FakePrim()
        with pytest.raises(SystemExit):
            self._call(_FakeStage(bare), cfg_usd="/x/seed1.usdc", expected_usd="/x/seed7.usdc")


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


class TestRenderCarbOverrides:
    """--render-carb parsing + the carb key-path transform (exposure probe)."""

    def test_coerces_value_types(self):
        assert cc._coerce_carb_value("true") is True
        assert cc._coerce_carb_value("False") is False
        assert cc._coerce_carb_value("6") == 6 and isinstance(cc._coerce_carb_value("6"), int)
        v = cc._coerce_carb_value("7.0")
        assert v == 7.0 and isinstance(v, float)
        assert cc._coerce_carb_value("aces") == "aces"

    def test_parses_repeated_key_value(self):
        overrides = cc._parse_render_carb_overrides([
            "rtx.post.histogram.enabled=false",
            "rtx.post.histogram.whiteScale=5.0",
        ])
        assert overrides == {
            "rtx.post.histogram.enabled": False,
            "rtx.post.histogram.whiteScale": 5.0,
        }

    def test_none_and_empty(self):
        assert cc._parse_render_carb_overrides(None) == {}
        assert cc._parse_render_carb_overrides([]) == {}

    def test_rejects_malformed(self):
        with pytest.raises(SystemExit):
            cc._parse_render_carb_overrides(["no_equals_sign"])
        with pytest.raises(SystemExit):
            cc._parse_render_carb_overrides(["=value"])

    def test_carb_path_matches_isaaclab_transform(self):
        # Mirror SimulationContext._apply_render_cfg_settings: dot/slash/underscore
        # all resolve to the same /slash/path the renderer reads.
        assert cc._carb_path("rtx.post.histogram.enabled") == "/rtx/post/histogram/enabled"
        assert cc._carb_path("/rtx/post/histogram/enabled") == "/rtx/post/histogram/enabled"
        assert cc._carb_path("rtx_post_histogram_enabled") == "/rtx/post/histogram/enabled"
