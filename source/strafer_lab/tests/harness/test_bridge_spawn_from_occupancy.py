"""The bridge/training robot spawn is derived per-loaded-scene from occupancy.

These tests pin the brief's invariants: the Infinigen-kind env cfg (and the
``run_sim_in_the_loop --scene-usd`` override) spawn the robot from the SINGLE
loaded scene's occupancy free-space — no cross-scene union, no pooled-max
spawn-z — using the one shared ``scene_connectivity.spawn_pool_from_occupancy``
core. The plan-free core itself is unit-tested with synthetic grids in
``test_scene_connectivity.py``; here we exercise the env-cfg wiring against the
shipped scene corpus (occupancy sidecars + embedded room footprints).

Runs in ``env_isaaclab3`` (needs isaaclab + ``pxr``); the corpus tests skip
cleanly where the scene assets or ``pxr`` are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# The whole module is env-cfg wiring — skip everywhere isaaclab is unavailable.
pytest.importorskip("isaaclab")

from strafer_lab.tools import scene_connectivity as sc  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_SCENES_ROOT = Path("Assets/generated/scenes")
_SCENE1 = "scene_high_quality_dgx_000_seed1"  # == _get_scene_usd_paths()[0]
_SCENE5 = "scene_high_quality_dgx_000_seed5"


def _scene_usd(stem: str) -> Path:
    return _SCENES_ROOT / f"{stem}.usdc"


def _corpus_has(stem: str) -> bool:
    return (_SCENES_ROOT / stem / "occupancy.npy").is_file() and _scene_usd(stem).exists()


# Corpus + pxr are required for the real-scene assertions.
_have_pxr = True
try:  # pragma: no cover - availability probe
    import pxr  # noqa: F401
except Exception:  # pragma: no cover
    _have_pxr = False

requires_corpus = pytest.mark.skipif(
    not (_have_pxr and _corpus_has(_SCENE1) and _corpus_has(_SCENE5)),
    reason="needs the shipped scene corpus (occupancy sidecars + USDs) and pxr",
)


def _scene_occ_rooms_free(stem: str):
    """Load a scene's occupancy + embedded rooms + sealed free grid (ground truth)."""
    from strafer_lab.tools import scene_metadata_reader as smr

    occ = sc.load_occupancy(_SCENES_ROOT / stem)
    rooms = smr.load(_scene_usd(stem)).get("rooms", [])
    free = sc.occupancy_to_free_space(occ.grid, grid_res=occ.resolution_m)
    free = sc.seal_free_space_to_rooms(
        free, rooms, origin_xy=occ.origin_xy, grid_res=occ.resolution_m
    )
    return occ, rooms, free


def _scene_floor_top_z(stem: str) -> float:
    import json

    meta = json.loads((_SCENES_ROOT / "scenes_metadata.json").read_text())
    return float(meta["scenes"][stem]["floor_top_z"])


def _assert_all_in_room_and_free(pool, occ, rooms, free):
    assert pool, "expected a non-empty spawn pool"
    for x, y in pool:
        assert sc.point_in_any_room(x, y, rooms), f"{(x, y)} outside every room"
        r, c = sc._xy_to_cell((x, y), occ.origin_xy, occ.resolution_m)
        assert 0 <= r < free.shape[0] and 0 <= c < free.shape[1], f"{(x, y)} off-grid"
        assert free[r, c], f"{(x, y)} not a free (inflated, sealed) cell"


def _fake_env_cfg():
    """A minimal stand-in for the bits the spawn wiring writes."""
    return SimpleNamespace(
        scene=SimpleNamespace(
            scene_geometry=SimpleNamespace(spawn=SimpleNamespace(usd_path="UNSET")),
        ),
        sim=SimpleNamespace(render=SimpleNamespace(carb_settings=None)),
        events=SimpleNamespace(
            reset_robot=SimpleNamespace(params={"spawn_points_xy": [], "spawn_z": 9.9}),
            lift_ground=SimpleNamespace(params={"target_z": 9.9}),
        ),
        commands=SimpleNamespace(goal_command=SimpleNamespace(spawn_points_xy=None)),
    )


# ---------------------------------------------------------------------------
# Graceful + fail-loud edges (no corpus required)
# ---------------------------------------------------------------------------


class TestDeriveEdges:
    def test_missing_occupancy_returns_empty(self, tmp_path):
        # A scene dir with no occupancy sidecar -> empty pool (the reset event
        # then spawns at the env origin), never a crash.
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            derive_infinigen_scene_spawn,
        )

        scenes = tmp_path / "scenes"
        (scenes / "scene_fresh_000").mkdir(parents=True)
        usd = scenes / "scene_fresh_000.usdc"
        usd.write_text("")  # present path, but no occupancy sidecar beside it
        assert derive_infinigen_scene_spawn(usd) == []

    def test_degenerate_grid_propagates_fail_loud(self, monkeypatch):
        # A present-but-degenerate occupancy (all blocked) must surface loudly
        # through the env-cfg helper, not be masked or fall back to a union read.
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            derive_infinigen_scene_spawn,
        )
        from strafer_lab.tools import scene_metadata_reader as smr

        occ = sc.CachedOccupancy(
            grid=np.ones((8, 8), dtype=np.uint8), origin_xy=(0.0, 0.0),
            resolution_m=0.1, z_slice_m=0.0, meta={},
        )
        room = {"footprint_xy": sc.aabb_to_footprint((-1.0, -1.0), (2.0, 2.0))}
        monkeypatch.setattr(sc, "scene_dir_for", lambda *_a, **_k: Path("/x"))
        monkeypatch.setattr(sc, "load_occupancy", lambda *_a, **_k: occ)
        monkeypatch.setattr(smr, "load", lambda *_a, **_k: {"rooms": [room]})
        with pytest.raises(RuntimeError):
            derive_infinigen_scene_spawn(Path("/x/scene_z_000.usdc"))


# ---------------------------------------------------------------------------
# Per-loaded-scene derivation against the shipped corpus
# ---------------------------------------------------------------------------


@requires_corpus
class TestPerSceneDerivation:
    def test_pool_is_in_room_free_and_bounded(self):
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            _INFINIGEN_SPAWN_POOL_SIZE,
            derive_infinigen_scene_spawn,
        )

        pool = derive_infinigen_scene_spawn(_scene_usd(_SCENE1))
        occ, rooms, free = _scene_occ_rooms_free(_SCENE1)
        _assert_all_in_room_and_free(pool, occ, rooms, free)
        # Bounded per-scene pool — NOT the cross-scene union (which pooled every
        # scene's floor samples into one ~hundreds-long list).
        assert len(pool) == _INFINIGEN_SPAWN_POOL_SIZE

    def test_derivation_is_scene_specific(self):
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            derive_infinigen_scene_spawn,
        )

        pool1 = derive_infinigen_scene_spawn(_scene_usd(_SCENE1))
        pool5 = derive_infinigen_scene_spawn(_scene_usd(_SCENE5))
        assert pool1 != pool5

        occ1, rooms1, free1 = _scene_occ_rooms_free(_SCENE1)
        occ5, rooms5, free5 = _scene_occ_rooms_free(_SCENE5)
        _assert_all_in_room_and_free(pool1, occ1, rooms1, free1)
        _assert_all_in_room_and_free(pool5, occ5, rooms5, free5)

        # Scene-specificity: a meaningful share of seed1's spawn cells are NOT
        # free in seed5's occupancy (the two floorplans differ), so the pool
        # genuinely tracks the loaded scene rather than a shared union.
        def free_in(pool, occ, free):
            n = 0
            for x, y in pool:
                r, c = sc._xy_to_cell((x, y), occ.origin_xy, occ.resolution_m)
                if 0 <= r < free.shape[0] and 0 <= c < free.shape[1] and free[r, c]:
                    n += 1
            return n

        assert free_in(pool1, occ5, free5) < len(pool1)


# ---------------------------------------------------------------------------
# Config-time wiring (_apply_infinigen_scene_setup) and constructed cfgs
# ---------------------------------------------------------------------------


@requires_corpus
class TestConfigTimeSetup:
    def test_apply_setup_writes_per_scene_spawn_and_floor(self):
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            _apply_infinigen_scene_setup,
        )

        cfg = _fake_env_cfg()
        _apply_infinigen_scene_setup(cfg)

        occ, rooms, free = _scene_occ_rooms_free(_SCENE1)  # scene[0]
        spawn = cfg.events.reset_robot.params["spawn_points_xy"]
        _assert_all_in_room_and_free(spawn, occ, rooms, free)
        # goal_command shares the per-scene pool (separate list object).
        assert cfg.commands.goal_command.spawn_points_xy == spawn
        assert cfg.commands.goal_command.spawn_points_xy is not spawn

        floor = _scene_floor_top_z(_SCENE1)
        assert cfg.events.reset_robot.params["spawn_z"] == pytest.approx(floor + 0.1)
        assert cfg.events.lift_ground.params["target_z"] == pytest.approx(floor - 0.002)
        # The scene USD bind points at scene[0], same scene the spawn came from.
        assert _SCENE1 in cfg.scene.scene_geometry.spawn.usd_path

    @pytest.mark.parametrize(
        "task",
        [
            "Isaac-Strafer-Nav-Capture-Bridge-v0",
            "Isaac-Strafer-Nav-Capture-Teleop-v0",
            "Isaac-Strafer-Nav-Capture-Coverage-v0",
        ],
    )
    def test_constructed_cfg_has_per_scene_in_room_spawn(self, task):
        import strafer_lab.tasks  # noqa: F401  (registers envs)
        from isaaclab_tasks.utils import parse_env_cfg

        cfg = parse_env_cfg(task, device="cpu", num_envs=1)
        occ, rooms, free = _scene_occ_rooms_free(_SCENE1)
        spawn = cfg.events.reset_robot.params["spawn_points_xy"]
        _assert_all_in_room_and_free(spawn, occ, rooms, free)

        floor = _scene_floor_top_z(_SCENE1)
        # spawn_z is the LOADED scene's floor + clearance, NOT the pooled MAX
        # across scenes (seed5's 0.61 would dominate the union).
        assert cfg.events.reset_robot.params["spawn_z"] == pytest.approx(floor + 0.1)
        assert cfg.events.lift_ground.params["target_z"] == pytest.approx(floor - 0.002)


# ---------------------------------------------------------------------------
# Runtime --scene-usd override coherence
# ---------------------------------------------------------------------------


@requires_corpus
class TestSceneUsdOverride:
    def test_override_re_derives_spawn_and_floor(self):
        import run_sim_in_the_loop as rsil

        cfg = _fake_env_cfg()
        rsil._apply_scene_usd_spawn_override(cfg, _scene_usd(_SCENE5))

        occ, rooms, free = _scene_occ_rooms_free(_SCENE5)
        spawn = cfg.events.reset_robot.params["spawn_points_xy"]
        _assert_all_in_room_and_free(spawn, occ, rooms, free)
        assert cfg.commands.goal_command.spawn_points_xy is not None
        assert len(cfg.commands.goal_command.spawn_points_xy) == len(spawn)

        floor = _scene_floor_top_z(_SCENE5)
        assert cfg.events.reset_robot.params["spawn_z"] == pytest.approx(floor + 0.1)
        assert cfg.events.lift_ground.params["target_z"] == pytest.approx(floor - 0.002)
        # The override bound seed5's resolved USD, not the default scene[0].
        assert cfg.scene.scene_geometry.spawn.usd_path == str(_scene_usd(_SCENE5).resolve())
