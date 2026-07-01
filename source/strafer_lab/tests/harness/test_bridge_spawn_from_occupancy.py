"""The bridge/training robot spawn is derived per-loaded-scene from occupancy.

These tests pin the brief's invariants: the env cfg (and the
``run_sim_in_the_loop --scene-usd`` override) spawn the robot from the SINGLE
loaded scene's occupancy free-space — no cross-scene union, no pooled-max
spawn-z — via the one shared ``scene_connectivity.spawn_pool_from_occupancy``
core. The spawn source is the scene-provider occupancy sidecar + embedded room
footprints; it is scene-generator agnostic.

Hermetic: each test authors a synthetic scene (an occupancy sidecar plus a
minimal USD carrying room footprints) in a tmp directory. Nothing reads the
on-disk scene corpus, which is a transient, regenerable artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from strafer_lab.tools import scene_connectivity as sc
from strafer_lab.tools import scene_metadata_reader as smr


def _author_scene(
    scenes_root: Path,
    stem: str,
    *,
    room_aabb: tuple[tuple[float, float], tuple[float, float]],
    grid: np.ndarray,
    origin: tuple[float, float] = (0.0, 0.0),
    res: float = 0.1,
    z: tuple[float, float] = (0.1, 0.3),
    write_occupancy: bool = True,
) -> tuple[Path, list[dict]]:
    """Author a synthetic scene: occupancy sidecar + a USD with room footprints.

    Lays it out like a real scene tree (``<scenes>/<stem>/<stem>.usdc`` beside
    ``occupancy.npy``) so ``scene_connectivity.scene_dir_for`` resolves the
    sidecar directory exactly as it does for a corpus scene.
    """
    sdir = scenes_root / stem
    sdir.mkdir(parents=True, exist_ok=True)
    if write_occupancy:
        sc.save_occupancy(
            sdir, np.asarray(grid, dtype=np.uint8),
            sc.occupancy_meta(origin_xy=origin, resolution_m=res, z_min_m=z[0], z_max_m=z[1]),
        )
    usd = sdir / f"{stem}.usdc"
    stage = Usd.Stage.CreateNew(str(usd))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    rooms = [
        {"footprint_xy": sc.aabb_to_footprint(*room_aabb), "room_type": "living_room", "story": 0}
    ]
    smr.write_custom_data(stage, {"rooms": rooms, "objects": [], "room_adjacency": []})
    stage.GetRootLayer().Save()
    return usd, rooms


def _free_block(n: int = 40, lo: int = 8, hi: int = 32) -> np.ndarray:
    """Raw occupancy: blocked everywhere except a free interior square."""
    grid = np.ones((n, n), dtype=np.uint8)
    grid[lo:hi, lo:hi] = 0
    return grid


def _sealed_free(usd: Path, rooms: list[dict]):
    occ = sc.load_occupancy(sc.scene_dir_for(usd))
    free = sc.occupancy_to_free_space(occ.grid, grid_res=occ.resolution_m)
    free = sc.seal_free_space_to_rooms(
        free, rooms, origin_xy=occ.origin_xy, grid_res=occ.resolution_m
    )
    return occ, free


def _assert_in_room_and_free(pool, occ, rooms, free):
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
# derive_infinigen_scene_spawn — the shared per-loaded-scene derivation
# ---------------------------------------------------------------------------


class TestDerive:
    def test_pool_is_in_room_free_and_bounded(self, tmp_path):
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            _INFINIGEN_SPAWN_POOL_SIZE,
            derive_infinigen_scene_spawn,
        )

        usd, rooms = _author_scene(
            tmp_path / "scenes", "scene_mock_000",
            room_aabb=((0.6, 0.6), (3.4, 3.4)), grid=_free_block(),
        )
        pool = derive_infinigen_scene_spawn(usd)
        occ, free = _sealed_free(usd, rooms)
        _assert_in_room_and_free(pool, occ, rooms, free)
        # Bounded per-scene pool — NOT a cross-scene union of every scene's points.
        assert len(pool) == _INFINIGEN_SPAWN_POOL_SIZE

    def test_derivation_is_scene_specific(self, tmp_path):
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            derive_infinigen_scene_spawn,
        )

        scenes = tmp_path / "scenes"
        # Two scenes at disjoint world coordinates.
        usd_a, rooms_a = _author_scene(
            scenes, "scene_mock_a", room_aabb=((0.6, 0.6), (3.4, 3.4)),
            grid=_free_block(), origin=(0.0, 0.0),
        )
        usd_b, rooms_b = _author_scene(
            scenes, "scene_mock_b", room_aabb=((10.6, 10.6), (13.4, 13.4)),
            grid=_free_block(), origin=(10.0, 10.0),
        )
        pool_a = derive_infinigen_scene_spawn(usd_a)
        pool_b = derive_infinigen_scene_spawn(usd_b)
        assert pool_a != pool_b

        occ_a, free_a = _sealed_free(usd_a, rooms_a)
        occ_b, free_b = _sealed_free(usd_b, rooms_b)
        _assert_in_room_and_free(pool_a, occ_a, rooms_a, free_a)
        _assert_in_room_and_free(pool_b, occ_b, rooms_b, free_b)
        # Every scene-A point lies in scene A's coordinate region, none in B's.
        for x, y in pool_a:
            assert not sc.point_in_any_room(x, y, rooms_b)

    def test_missing_occupancy_returns_empty(self, tmp_path):
        # A scene with a USD but no occupancy sidecar yet -> empty pool (the reset
        # event then spawns at the env origin), never a crash.
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            derive_infinigen_scene_spawn,
        )

        usd, _ = _author_scene(
            tmp_path / "scenes", "scene_mock_000",
            room_aabb=((0.6, 0.6), (3.4, 3.4)), grid=_free_block(),
            write_occupancy=False,
        )
        assert derive_infinigen_scene_spawn(usd) == []

    def test_degenerate_grid_fails_loud(self, tmp_path):
        # A present-but-degenerate occupancy (all blocked) surfaces loudly — it is
        # never masked or backed off to a union read.
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            derive_infinigen_scene_spawn,
        )

        usd, _ = _author_scene(
            tmp_path / "scenes", "scene_mock_000",
            room_aabb=((0.6, 0.6), (3.4, 3.4)), grid=np.ones((40, 40), dtype=np.uint8),
        )
        with pytest.raises(RuntimeError):
            derive_infinigen_scene_spawn(usd)


# ---------------------------------------------------------------------------
# Config-time wiring (_apply_infinigen_scene_setup) + constructed cfgs
# ---------------------------------------------------------------------------


def _point_cfg_at_scene(monkeypatch, usd: Path, floor_top_z: float):
    """Redirect the config-default scene bind + floor lookup at a mock scene."""
    from strafer_lab.tasks.navigation import strafer_env_cfg as cfg_mod

    monkeypatch.setattr(cfg_mod, "_get_scene_usd_paths", lambda: [str(usd)])
    monkeypatch.setattr(
        cfg_mod, "_get_infinigen_active_scene_floor_top_z",
        lambda stem: floor_top_z if stem == usd.stem else None,
    )


class TestConfigTimeSetup:
    def test_apply_setup_writes_per_scene_spawn_and_floor(self, tmp_path, monkeypatch):
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            _apply_infinigen_scene_setup,
        )

        usd, rooms = _author_scene(
            tmp_path / "scenes", "scene_mock_000",
            room_aabb=((0.6, 0.6), (3.4, 3.4)), grid=_free_block(),
        )
        floor = 0.321
        _point_cfg_at_scene(monkeypatch, usd, floor)

        cfg = _fake_env_cfg()
        _apply_infinigen_scene_setup(cfg)

        occ, free = _sealed_free(usd, rooms)
        spawn = cfg.events.reset_robot.params["spawn_points_xy"]
        _assert_in_room_and_free(spawn, occ, rooms, free)
        # goal_command shares the per-scene pool (a separate list object).
        assert cfg.commands.goal_command.spawn_points_xy == spawn
        assert cfg.commands.goal_command.spawn_points_xy is not spawn
        assert cfg.events.reset_robot.params["spawn_z"] == pytest.approx(floor + 0.1)
        assert cfg.events.lift_ground.params["target_z"] == pytest.approx(floor - 0.002)
        assert cfg.scene.scene_geometry.spawn.usd_path == str(usd.resolve())

    @pytest.mark.parametrize(
        "task",
        [
            "Isaac-Strafer-Nav-Capture-Bridge-v0",
            "Isaac-Strafer-Nav-Capture-Teleop-v0",
            "Isaac-Strafer-Nav-Capture-Coverage-v0",
        ],
    )
    def test_constructed_cfg_has_per_scene_in_room_spawn(self, task, tmp_path, monkeypatch):
        import strafer_lab.tasks  # noqa: F401  (registers envs)
        from isaaclab_tasks.utils import parse_env_cfg

        usd, rooms = _author_scene(
            tmp_path / "scenes", "scene_mock_000",
            room_aabb=((0.6, 0.6), (3.4, 3.4)), grid=_free_block(),
        )
        floor = 0.321
        _point_cfg_at_scene(monkeypatch, usd, floor)

        cfg = parse_env_cfg(task, device="cpu", num_envs=1)
        occ, free = _sealed_free(usd, rooms)
        spawn = cfg.events.reset_robot.params["spawn_points_xy"]
        _assert_in_room_and_free(spawn, occ, rooms, free)
        # spawn_z is the LOADED scene's floor + clearance, NOT a pooled max.
        assert cfg.events.reset_robot.params["spawn_z"] == pytest.approx(floor + 0.1)
        assert cfg.events.lift_ground.params["target_z"] == pytest.approx(floor - 0.002)


# ---------------------------------------------------------------------------
# Runtime --scene-usd override coherence
# ---------------------------------------------------------------------------


class TestSceneUsdOverride:
    def test_override_re_derives_spawn_and_floor(self, tmp_path, monkeypatch):
        _scripts = Path(__file__).resolve().parents[2] / "scripts"
        if str(_scripts) not in sys.path:
            sys.path.insert(0, str(_scripts))
        import run_sim_in_the_loop as rsil
        from strafer_lab.tasks.navigation import strafer_env_cfg as cfg_mod

        usd, rooms = _author_scene(
            tmp_path / "scenes", "scene_mock_000",
            room_aabb=((0.6, 0.6), (3.4, 3.4)), grid=_free_block(),
        )
        floor = 0.507
        monkeypatch.setattr(
            cfg_mod, "_get_infinigen_active_scene_floor_top_z",
            lambda stem: floor if stem == usd.stem else None,
        )

        cfg = _fake_env_cfg()
        rsil._apply_scene_usd_spawn_override(cfg, usd)

        occ, free = _sealed_free(usd, rooms)
        spawn = cfg.events.reset_robot.params["spawn_points_xy"]
        _assert_in_room_and_free(spawn, occ, rooms, free)
        assert cfg.commands.goal_command.spawn_points_xy is not None
        assert len(cfg.commands.goal_command.spawn_points_xy) == len(spawn)
        assert cfg.events.reset_robot.params["spawn_z"] == pytest.approx(floor + 0.1)
        assert cfg.events.lift_ground.params["target_z"] == pytest.approx(floor - 0.002)
        assert cfg.scene.scene_geometry.spawn.usd_path == str(usd.resolve())
