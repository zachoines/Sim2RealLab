# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the generator's placement sequence, park rank and column phase.

The sibling frozen-output guard proves the *default* path did not move. These
cover the ``placement`` surface: that an explicitly-vanilla ``PlacementCfg`` is
byte-for-byte the ``None`` path, that the group-atomic park rank reproduces and
generalizes the two-tier ladder, that the mid-room column phase is additive and
bounded away from the solvability seed, and that the enriched seed correction
removes blocked cells from the spawn pool.

No Kit / GPU. Run with the pure-Python lab tests.
"""
from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import pytest
import torch

from strafer_lab.tasks.navigation.mdp import proc_room as _proc_room
from strafer_lab.tasks.navigation.mdp.proc_room import (
    BFS_SEED_CLEARANCE,
    CLUTTER_SLOTS,
    CLUTTER_TALL_CYL_SLOTS,
    FURNITURE_SLOTS,
    GRID_RES,
    GRID_SIZE,
    OBJECT_SIZES,
    WALL_SLOTS,
    CompoundCfg,
    PlacementCfg,
    _relocate_blocked_seeds,
    _rotated_aabb_half_extents,
    _VANILLA_CLUTTER_SEQUENCE,
    _VANILLA_FURNITURE_SEQUENCE,
    _VANILLA_PARK_RANK,
    _xy_to_grid,
    column_protected_park_rank,
    generate_proc_room,
)


class _CaptureEntity:
    def __init__(self):
        self.body_poses = None
        self.root_pose = None

    def write_body_link_pose_to_sim_index(self, body_poses, env_ids, body_ids):
        self.body_poses = body_poses.clone()

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        self.root_pose = root_pose.clone()


class _StubScene:
    def __init__(self, entities, env_origins):
        self._entities = entities
        self.env_origins = env_origins

    def __getitem__(self, key):
        return self._entities[key]


def _make_env(num_envs, difficulty):
    entities = {"room_primitives": _CaptureEntity(), "ceiling": _CaptureEntity()}
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=_StubScene(entities, torch.zeros(num_envs, 3)),
    )
    env._proc_room_difficulty = torch.full((num_envs,), difficulty, dtype=torch.long)
    return env, entities


def _run(seed=20260101, num_envs=8, difficulty=7, **kwargs):
    torch.manual_seed(seed)
    env, entities = _make_env(num_envs, difficulty)
    generate_proc_room(env, torch.arange(num_envs), **kwargs)
    return env, entities


def _digest(env, entities):
    """Output plus stream position, the same quantities the frozen guard hashes."""
    h = hashlib.sha256()
    for t in (
        entities["room_primitives"].body_poses,
        env._proc_room_active_mask,
        env._proc_room_free_space,
        env._proc_room_spawn_pts,
        env._proc_room_spawn_count,
        torch.rand(4),
    ):
        c = t.detach().cpu().contiguous()
        h.update(str(tuple(c.shape)).encode())
        h.update(str(c.dtype).encode())
        h.update(c.numpy().tobytes())
    return h.hexdigest()


_VANILLA_PLACEMENT = PlacementCfg(
    furniture_sequence=_VANILLA_FURNITURE_SEQUENCE,
    clutter_sequence=_VANILLA_CLUTTER_SEQUENCE,
    park_rank=_VANILLA_PARK_RANK,
)


# ---------------------------------------------------------------------------
# The refactor's equivalence, stated as data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", (20260101, 20275939))
@pytest.mark.parametrize("difficulty", range(8))
@pytest.mark.parametrize("num_envs", (1, 8))
def test_explicit_vanilla_placement_matches_the_none_path(seed, difficulty, num_envs):
    """A ``PlacementCfg`` spelling out the vanilla order is byte-for-byte ``None``.

    The frozen guard pins the ``None`` path against history; this pins the two
    against each other over a seed × difficulty × batch sweep, so the sequence
    and rank are the data the loops hold hardcoded."""
    implicit = _digest(*_run(seed=seed, num_envs=num_envs, difficulty=difficulty))
    explicit = _digest(
        *_run(seed=seed, num_envs=num_envs, difficulty=difficulty,
              placement=_VANILLA_PLACEMENT)
    )
    assert implicit == explicit


def test_health_sink_does_not_perturb_generation():
    """The instrument observes; it must not move a pose or a draw."""
    without = _digest(*_run())
    sink = {}
    with_sink = _digest(*_run(health_sink=sink))
    assert without == with_sink
    assert sink["envs"] == 8


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------


def _active_xy(env, entities, slots):
    mask = env._proc_room_active_mask[:, slots]
    xy = entities["room_primitives"].body_poses[:, slots, :2]
    return sorted(tuple(round(v, 6) for v in p) for p in xy[mask].tolist())


def test_clutter_permutation_relabels_an_identical_geometry():
    """The clutter sampler never reads the slot's size and its rejection tests
    compare XY only, so permuting the clutter sequence attaches the *same*
    geometric sequence to different slots. This is what lets the mid-room-column
    presence lever reorder the clutter sequence free of side effects.

    Stated for the placement phase alone: the retry ladder reads slots, so on a
    room it strips the relabeling does move geometry. Sparse rooms keep the
    ladder out of it, and the sink proves they did."""
    promoted = tuple(CLUTTER_TALL_CYL_SLOTS) + tuple(
        s for s in CLUTTER_SLOTS if s not in set(CLUTTER_TALL_CYL_SLOTS)
    )
    sink_base, sink_perm = {}, {}
    base = _run(difficulty=4, health_sink=sink_base, placement=_VANILLA_PLACEMENT)
    perm = _run(difficulty=4, health_sink=sink_perm,
                placement=PlacementCfg(clutter_sequence=promoted))
    assert sink_base["ladder_passes"] == sink_perm["ladder_passes"] == 0
    assert _active_xy(*base, CLUTTER_SLOTS) == _active_xy(*perm, CLUTTER_SLOTS)
    # Non-vacuous: the slots those positions landed on did change.
    assert not torch.equal(
        base[0]._proc_room_active_mask[:, CLUTTER_SLOTS],
        perm[0]._proc_room_active_mask[:, CLUTTER_SLOTS],
    )


def test_furniture_permutation_changes_the_geometry():
    """The furniture sampler *does* read ``OBJECT_SIZES[slot, :2]`` for its wall
    inset, so the two categories cannot be assumed symmetric — a furniture
    permutation of unlike footprints is a real geometry change, not a
    relabeling."""
    perm = (22, 21, 20) + tuple(FURNITURE_SLOTS[3:])  # 1.2x0.3 shelf for a 0.8x0.6 table
    base = _run(difficulty=4, placement=_VANILLA_PLACEMENT)
    swapped = _run(difficulty=4, placement=PlacementCfg(furniture_sequence=perm))
    assert _active_xy(*base, FURNITURE_SLOTS) != _active_xy(*swapped, FURNITURE_SLOTS)


def test_a_short_sequence_bounds_the_level_budget():
    """Level 7 asks for 8 furniture; a 3-slot sequence caps it at 3."""
    env, _ = _run(placement=PlacementCfg(furniture_sequence=(20, 21, 22)))
    active = env._proc_room_active_mask[:, FURNITURE_SLOTS]
    assert active.sum(dim=1).max() <= 3
    assert not active[:, 3:].any()


# ---------------------------------------------------------------------------
# Park rank
# ---------------------------------------------------------------------------


def _park_ladder(active_row, rank, passes):
    """Reference walk of a rank over one env's active mask."""
    active = dict(enumerate(active_row))
    for _ in range(passes):
        for group in rank:
            if any(active[s] for s in group):
                for s in group:
                    active[s] = False
                break
    return active


def test_column_protected_rank_reorders_within_clutter_only():
    rank = column_protected_park_rank()
    assert sorted(s for g in rank for s in g) == sorted(CLUTTER_SLOTS + FURNITURE_SLOTS)
    flat = [s for g in rank for s in g]
    columns = set(CLUTTER_TALL_CYL_SLOTS)
    ordinary = [s for s in CLUTTER_SLOTS if s not in columns]
    # Every ordinary clutter slot is parked before either column, and every
    # furniture slot after both — protection bounded by its own category.
    assert max(flat.index(s) for s in ordinary) < min(flat.index(s) for s in columns)
    assert max(flat.index(s) for s in columns) < min(
        flat.index(s) for s in FURNITURE_SLOTS
    )


def test_park_rank_parks_groups_atomically():
    """A compound must leave as a unit — the property groups exist for."""
    group = (28, 29, 30)
    rest = [s for s in reversed(CLUTTER_SLOTS) if s not in group]
    rank = (group,) + tuple((s,) for s in rest + list(reversed(FURNITURE_SLOTS)))
    active = _park_ladder([True] * 44, rank, passes=1)
    assert not any(active[s] for s in group)
    assert all(active[s] for s in rest)


def test_park_rank_terminates_in_one_pass_per_group():
    """Groups are disjoint and nothing re-activates, so the index of the first
    active group strictly increases; ``len(rank)`` passes strip everything."""
    rank = column_protected_park_rank()
    active = _park_ladder([True] * 44, rank, passes=len(rank))
    assert not any(active[s] for g in rank for s in g)
    # The terminal geometry is the bare perimeter box — walls are not ranked.
    assert all(active[s] for s in WALL_SLOTS)


def test_unsolvable_rooms_strip_to_the_perimeter_and_stay_solvable():
    """Drive the ladder to exhaustion under the protected rank and confirm both
    the termination bound and that walls survive it."""
    env, _ = _run(
        num_envs=4,
        difficulty=7,
        span_max=4.0,
        placement=PlacementCfg(park_rank=column_protected_park_rank()),
    )
    assert env._proc_room_spawn_count.min() > 0
    assert env._proc_room_active_mask[:, WALL_SLOTS].any(dim=1).all()


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    (
        {"furniture_sequence": (20, 20)},
        {"furniture_sequence": (20, 30)},
        {"clutter_sequence": (28, 28)},
        {"clutter_sequence": (28, 20)},
        {"park_rank": ()},
        {"park_rank": ((),)},
        {"park_rank": ((28,), (28,))},
        {"park_rank": tuple((s,) for s in list(CLUTTER_SLOTS) + FURNITURE_SLOTS + [0])},
        {"park_rank": tuple((s,) for s in CLUTTER_SLOTS)},
        {"column_prob": 1.5},
        {"column_prob": -0.1},
        {"column_count": 3},
        {"column_count": -1},
        {"column_slots": (28, 28)},
        {"column_slots": (20,)},
        {"column_seed_clearance": BFS_SEED_CLEARANCE - 0.01},
        {"column_radius_max": 0.4},
    ),
)
def test_placement_cfg_rejects_an_unsound_lever(kwargs):
    with pytest.raises(ValueError):
        PlacementCfg(**kwargs)


def test_placement_cfg_accepts_the_shipped_shapes():
    PlacementCfg()
    PlacementCfg(park_rank=column_protected_park_rank())
    PlacementCfg(column_prob=0.5, column_count=2)


# ---------------------------------------------------------------------------
# Mid-room columns
# ---------------------------------------------------------------------------

_COLUMNS = PlacementCfg(
    park_rank=column_protected_park_rank(),
    column_prob=1.0,
    column_count=2,
    relocate_blocked_bfs_seed=True,
)


def _column_presence(env):
    return env._proc_room_active_mask[:, list(CLUTTER_TALL_CYL_SLOTS)].any(dim=1)


@pytest.mark.parametrize("difficulty", (4, 5, 6, 7))
def test_column_phase_raises_mid_room_presence(difficulty):
    """The two tall cylinders sit at the end of the clutter sequence, so today
    only level 7 reaches them at all. The phase makes them a per-episode draw."""
    off, _ = _run(num_envs=32, difficulty=difficulty)
    on, _ = _run(num_envs=32, difficulty=difficulty, placement=_COLUMNS)
    assert _column_presence(on).float().mean() > _column_presence(off).float().mean()
    assert _column_presence(on).all()


def test_column_phase_is_a_probability_not_a_reorder():
    sink = {}
    _run(num_envs=256, difficulty=6, health_sink=sink,
         placement=PlacementCfg(column_prob=0.5, column_count=2))
    # 256 Bernoulli(0.5) draws: a five-sigma band is ~40 wide.
    assert 88 < sink["column_phase_fired"] < 168


def test_columns_clear_the_bfs_seed_and_stay_inside_the_room():
    """The radial law is bounded below by the seed clearance and above by the
    room inset, so no draw can enclose the solvability seed or leave the room."""
    env, entities = _run(num_envs=64, difficulty=7, placement=_COLUMNS)
    slots = list(CLUTTER_TALL_CYL_SLOTS)
    mask = env._proc_room_active_mask[:, slots]
    xy = entities["room_primitives"].body_poses[:, slots, :2]
    assert mask.any()
    half_diag = max(
        (OBJECT_SIZES[s, 0].item() ** 2 + OBJECT_SIZES[s, 1].item() ** 2) ** 0.5 / 2.0
        for s in slots
    )
    radius = xy[mask].norm(dim=-1)
    assert (radius - half_diag >= BFS_SEED_CLEARANCE - 1e-6).all()
    # Inside the room: the walls the generator built bound every column.
    wall_xy = entities["room_primitives"].body_poses[:, WALL_SLOTS, :2].abs()
    wall_mask = env._proc_room_active_mask[:, WALL_SLOTS]
    half_span = torch.where(wall_mask.unsqueeze(-1), wall_xy, torch.zeros_like(wall_xy))
    half_span = half_span.amax(dim=1)  # (B, 2)
    inside = (xy.abs() <= half_span.unsqueeze(1) + 1e-6) | ~mask.unsqueeze(-1)
    assert inside.all()


def test_column_phase_is_additive_not_a_displacement():
    """The phase gives the ordinary scatter its own budget back: at level 4 the
    same four leading clutter slots are still the ones attempted, and the
    columns arrive on top of them rather than in place of two."""
    env, _ = _run(num_envs=32, difficulty=4, placement=_COLUMNS)
    active = env._proc_room_active_mask
    ordinary = [s for s in CLUTTER_SLOTS if s not in set(CLUTTER_TALL_CYL_SLOTS)]
    assert active[:, list(CLUTTER_TALL_CYL_SLOTS)].all()
    assert not active[:, ordinary[4:]].any()
    assert active[:, ordinary[:4]].sum(dim=1).max() == 4


def test_column_phase_off_leaves_the_enriched_path_alone():
    """``column_prob`` at zero draws nothing, so an enriched arm carrying only
    the park rank is byte-identical to one carrying no placement at all."""
    rank_only = _digest(*_run(placement=PlacementCfg(park_rank=column_protected_park_rank())))
    knob_off = _digest(
        *_run(placement=PlacementCfg(park_rank=column_protected_park_rank(),
                                     column_prob=0.0, column_count=2))
    )
    assert rank_only == knob_off


def test_column_phase_off_places_no_mid_room_column():
    """The off arm (`prob=0.0`) never runs the phase, so it is the clean
    no-mid-room-column baseline the D4 A/B differences against. The two tall
    cylinders may still reach the room by ordinary scatter at level 7, but the
    phase's own promotion counter must read zero."""
    sink = {}
    _run(num_envs=32, difficulty=7, health_sink=sink,
         placement=PlacementCfg(column_prob=0.0, column_count=2))
    assert sink["column_phase_fired"] == 0


# ---------------------------------------------------------------------------
# BFS seed correction
# ---------------------------------------------------------------------------


def test_relocate_blocked_seeds_finds_the_nearest_free_cell():
    free = torch.zeros(1, 9, 9, dtype=torch.bool)
    free[0, 4, 7] = True   # distance 3
    free[0, 6, 4] = True   # distance 2 — the nearest
    cells, moved = _relocate_blocked_seeds(free, torch.tensor([[4, 4]]))
    assert moved.tolist() == [True]
    assert cells.tolist() == [[6, 4]]


def test_relocate_blocked_seeds_leaves_a_free_seed_alone():
    free = torch.ones(2, 9, 9, dtype=torch.bool)
    start = torch.tensor([[4, 4], [1, 2]])
    cells, moved = _relocate_blocked_seeds(free, start)
    assert not moved.any()
    assert torch.equal(cells, start)


def test_relocate_blocked_seeds_keeps_a_sealed_env_seeded():
    """With no free cell anywhere the seed must stay put — the unconditional
    seed write is what keeps the reachable count non-zero."""
    free = torch.zeros(1, 9, 9, dtype=torch.bool)
    cells, moved = _relocate_blocked_seeds(free, torch.tensor([[4, 4]]))
    assert not moved.any()
    assert cells.tolist() == [[4, 4]]


def test_relocate_blocked_seeds_breaks_ties_row_major():
    """Equidistant free cells resolve on flattened cell order (argmin), so the
    correction is deterministic and draw-free. (4,6) and (6,4) are both
    distance 2 from (4,4); the lower flat index 4·9+6 < 6·9+4 wins."""
    free = torch.zeros(1, 9, 9, dtype=torch.bool)
    free[0, 4, 6] = True
    free[0, 6, 4] = True
    cells, moved = _relocate_blocked_seeds(free, torch.tensor([[4, 4]]))
    assert moved.tolist() == [True]
    assert cells.tolist() == [[4, 6]]


def _blocked_spawn_cells(env):
    """Spawn points whose own grid cell is not free — the hazard the seed
    correction closes."""
    pts = env._proc_room_spawn_pts
    counts = env._proc_room_spawn_count
    origin = -GRID_SIZE * GRID_RES / 2.0
    idx = ((pts - origin) / GRID_RES).long().clamp(0, GRID_SIZE - 1)
    valid = torch.arange(pts.shape[1]).unsqueeze(0) < counts.unsqueeze(1)
    free = env._proc_room_free_space
    b = torch.arange(pts.shape[0]).unsqueeze(1).expand_as(idx[..., 0])
    return int((~free[b, idx[..., 0], idx[..., 1]] & valid).sum())


def test_seed_correction_clears_blocked_cells_from_the_spawn_pool():
    """``_gpu_bfs`` marks its seed reachable free or not, so a grid-blocked room
    centre puts a cell inside an obstacle into the pool the robot, the goal and
    the planner endpoints all draw from."""
    enriched = dict(
        wall_height=2.7, door_width_max=2.0, span_max=7.5, max_span_sum=13.4,
        clutter_wall_bias_prob=0.3, robot_spawn_inflation_cells=1,
    )
    sink = {}
    off, _ = _run(num_envs=64, difficulty=7, **enriched)
    on, _ = _run(num_envs=64, difficulty=7, health_sink=sink, **enriched,
                 placement=PlacementCfg(relocate_blocked_bfs_seed=True))
    assert _blocked_spawn_cells(off) > 0, "no blocked seed in this batch — test is vacuous"
    assert _blocked_spawn_cells(on) == 0
    assert sink["bfs_seed_relocated"] > 0


# ---------------------------------------------------------------------------
# Health sink
# ---------------------------------------------------------------------------


def test_health_sink_reports_placement_and_park_counts():
    sink = {}
    env, _ = _run(num_envs=32, difficulty=7, span_max=4.0, health_sink=sink,
                  placement=_COLUMNS)
    active = env._proc_room_active_mask
    ordinary = [s for s in CLUTTER_SLOTS if s not in set(CLUTTER_TALL_CYL_SLOTS)]
    assert sink["envs"] == 32
    assert sink["column_phase_fired"] == 32
    assert sink["placed_columns"] == 64
    # Parks are what the ladder took back off the pre-ladder placement.
    assert sink["parked_clutter"] == sink["placed_clutter"] - int(active[:, ordinary].sum())
    assert sink["parked_furniture"] == (
        sink["placed_furniture"] - int(active[:, FURNITURE_SLOTS].sum())
    )
    assert sink["ladder_passes"] > 0, "tight rooms did not exercise the ladder"
    assert sink["spawn_count_min"] > 0


# ---------------------------------------------------------------------------
# Compounds (H1 primitive clusters)
# ---------------------------------------------------------------------------

# A wide "wardrobe": the two 1.2x0.3 shelves side by side against a wall,
# deliberately overlapping in x (dx = +-0.4 -> 0.4 m within-compound overlap),
# flush to the wall (dy = 0.15 = half the 0.3 m depth).
_WARDROBE = CompoundCfg(members=(22, 23), offsets=((-0.4, 0.15), (0.4, 0.15)))
_COMPOUND_RANK = tuple(
    [(s,) for s in list(reversed(CLUTTER_SLOTS))]
    + [(f,) for f in reversed(FURNITURE_SLOTS) if f not in (22, 23)]
    + [(22, 23)]  # the compound members park as one group, last
)
_COMPOUNDS = PlacementCfg(
    park_rank=_COMPOUND_RANK, compounds=(_WARDROBE,), compound_prob=1.0,
    furniture_aabb_reject=True,
)
_ENRICHED_KW = dict(
    wall_height=2.7, door_width_max=2.0, span_max=7.5, max_span_sum=13.4,
    clutter_wall_bias_prob=0.3, robot_spawn_inflation_cells=1,
    tall_object_heights={"shelf": 2.0, "cabinet": 2.1, "tall_cyl": 1.8},
)


# A second shape for the menu: the two 0.5x0.5 cabinets side by side (dy=0.25
# flush to their 0.5 m depth). Disjoint members from the wardrobe.
_CABINETS = CompoundCfg(members=(24, 25), offsets=((-0.35, 0.25), (0.35, 0.25)))
_MENU_RANK = tuple(
    [(s,) for s in list(reversed(CLUTTER_SLOTS))]
    + [(f,) for f in reversed(FURNITURE_SLOTS) if f not in (22, 23, 24, 25)]
    + [(22, 23), (24, 25)]
)


def _yaw_of(poses, b, slot):
    return 2.0 * math.atan2(poses[b, slot, 5].item(), poses[b, slot, 6].item())


def test_compound_catalogue_places_one_shape_per_episode():
    """A multi-shape catalogue is a menu, not a set: each firing episode seats
    exactly one shape. With the catalogue slots held out of the ordinary sequence
    they can only arrive via a compound, so this reads the menu draw directly."""
    menu = PlacementCfg(
        furniture_sequence=(20, 21, 26, 27), park_rank=_MENU_RANK,
        compounds=(_WARDROBE, _CABINETS), compound_prob=1.0, furniture_aabb_reject=True)
    env, _ = _run(num_envs=128, difficulty=7, placement=menu, **_ENRICHED_KW)
    active = env._proc_room_active_mask
    wardrobe = active[:, 22] & active[:, 23]
    cabinets = active[:, 24] & active[:, 25]
    assert not (wardrobe & cabinets).any(), "an episode seated two shapes"
    assert not (active[:, 22] ^ active[:, 23]).any(), "a half wardrobe leaked in"
    assert not (active[:, 24] ^ active[:, 25]).any(), "a half cabinet-pair leaked in"
    assert wardrobe.any() and cabinets.any(), "the menu never exercised both shapes"


def test_compound_phase_off_leaves_the_enriched_path_alone():
    """``compound_prob`` at zero draws nothing, so an arm carrying a compound
    catalogue at zero probability digests identically to one carrying none."""
    none = _digest(*_run(placement=PlacementCfg(park_rank=_COMPOUND_RANK)))
    knob_off = _digest(*_run(placement=PlacementCfg(
        park_rank=_COMPOUND_RANK, compounds=(_WARDROBE,), compound_prob=0.0)))
    assert none == knob_off


def test_compound_places_and_parks_as_a_unit():
    """A compound is all-or-nothing per env and the retry ladder never splits it,
    which is the property the group-atomic park rank exists to guarantee."""
    env, _ = _run(num_envs=64, difficulty=7, placement=_COMPOUNDS, **_ENRICHED_KW)
    active = env._proc_room_active_mask
    assert (active[:, 22] & active[:, 23]).any(), "the wardrobe never seated"
    assert not (active[:, 22] ^ active[:, 23]).any(), "a member placed without its pair"
    # Drive the ladder to exhaustion on tight rooms — still never a half-compound.
    tight, _ = _run(num_envs=16, difficulty=7, placement=_COMPOUNDS,
                    **{**_ENRICHED_KW, "span_max": 4.0, "max_span_sum": 8.0})
    tight_active = tight._proc_room_active_mask
    assert not (tight_active[:, 22] ^ tight_active[:, 23]).any()


def test_compound_keeps_within_overlap_but_rejects_between_overlap():
    """Within-compound overlap is intentional (the members interpenetrate by
    design); between-unit overlap is not — the true-AABB test keeps ordinary
    furniture out of a placed compound member."""
    env, ents = _run(num_envs=64, difficulty=7, placement=_COMPOUNDS, **_ENRICHED_KW)
    active = env._proc_room_active_mask
    poses = ents["room_primitives"].body_poses
    placed = torch.where(active[:, 22] & active[:, 23])[0].tolist()
    assert placed

    for b in placed:
        hx, hy = _rotated_aabb_half_extents(1.2, 0.3, _yaw_of(poses, b, 22))
        ox = 2 * hx - abs(poses[b, 22, 0].item() - poses[b, 23, 0].item())
        oy = 2 * hy - abs(poses[b, 22, 1].item() - poses[b, 23, 1].item())
        assert ox > 0.001 and oy > 0.001, "within-compound overlap was suppressed"

    worst = 0.0
    for b in placed:
        for f in FURNITURE_SLOTS:
            if f in (22, 23) or not active[b, f]:
                continue
            hxf, hyf = _rotated_aabb_half_extents(
                OBJECT_SIZES[f, 0].item(), OBJECT_SIZES[f, 1].item(), _yaw_of(poses, b, f))
            for m in (22, 23):
                hxm, hym = _rotated_aabb_half_extents(1.2, 0.3, _yaw_of(poses, b, m))
                ox = hxf + hxm - abs(poses[b, f, 0].item() - poses[b, m, 0].item())
                oy = hyf + hym - abs(poses[b, f, 1].item() - poses[b, m, 1].item())
                if ox > 0.001 and oy > 0.001:
                    worst = max(worst, min(ox, oy))
    assert worst < 0.01, f"ordinary furniture interpenetrates a compound by {worst:.3f} m"


def test_compound_footprint_reaches_the_occupancy_grid():
    """Each member rasterizes its own AABB into occupancy, so BFS sees the whole
    composite footprint — a member's centre cell is never passable."""
    env, ents = _run(num_envs=64, difficulty=7, placement=_COMPOUNDS, **_ENRICHED_KW)
    active = env._proc_room_active_mask
    poses = ents["room_primitives"].body_poses
    free = env._proc_room_free_space
    b = torch.where(active[:, 22] & active[:, 23])[0][0].item()
    for m in (22, 23):
        r, c = _xy_to_grid(poses[b, m, 0], poses[b, m, 1])
        assert not free[b, r, c], "a compound member's centre is passable in BFS"


def test_compound_consumes_the_furniture_budget():
    """Members count against the level's furniture budget rather than adding to
    it: at level 4 (budget 4) a placed 2-member compound leaves 4 furniture, not
    4 + 2 — held by construction so density does not silently inflate."""
    sink = {}
    env, _ = _run(num_envs=64, difficulty=4, health_sink=sink,
                  placement=_COMPOUNDS, **_ENRICHED_KW)
    active = env._proc_room_active_mask
    placed = active[:, 22] & active[:, 23]
    assert placed.any()
    per_env = active[:, FURNITURE_SLOTS].sum(dim=1)
    assert (per_env[placed] <= 4).all(), "compound added to the budget instead of consuming it"
    assert sink["compound_phase_fired"] == 64
    assert sink["placed_compound_members"] == 2 * int(placed.sum())


@pytest.mark.parametrize(
    "kwargs",
    (
        {"members": (22,), "offsets": ((0.0, 0.15),)},                    # < 2 members
        {"members": (22, 23), "offsets": ((0.0, 0.15),)},                 # length mismatch
        {"members": (22, 22), "offsets": ((0.0, 0.15), (0.0, 0.15))},     # repeated member
        {"members": (22, 30), "offsets": ((0.0, 0.15), (0.0, 0.15))},     # clutter member
        {"members": (22, 23), "offsets": ((0.0, 0.0), (0.0, 0.15))},      # dy pierces the wall
    ),
)
def test_compound_cfg_rejects_an_unsound_shape(kwargs):
    with pytest.raises(ValueError):
        CompoundCfg(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"compound_prob": 1.5},
        {"compound_prob": -0.1},
        {"compounds": (_WARDROBE,)},  # no park_rank holds the members in one group
    ),
)
def test_placement_cfg_rejects_an_unsound_compound_lever(kwargs):
    with pytest.raises(ValueError):
        PlacementCfg(**kwargs)


def test_placement_cfg_rejects_compounds_sharing_a_member():
    rank = tuple([(22, 23, 26)] + [(s,) for s in CLUTTER_SLOTS] + [(20,), (21,), (24,), (25,), (27,)])
    with pytest.raises(ValueError):
        PlacementCfg(park_rank=rank, compounds=(
            CompoundCfg((22, 23), ((-0.4, 0.15), (0.4, 0.15))),
            CompoundCfg((23, 26), ((-0.4, 0.15), (0.4, 0.30))),
        ))


def test_compound_cfg_accepts_the_shipped_shapes():
    CompoundCfg(members=(22, 23), offsets=((-0.4, 0.15), (0.4, 0.15)))
    PlacementCfg(park_rank=_COMPOUND_RANK, compounds=(_WARDROBE,), compound_prob=1.0)


@pytest.fixture(autouse=True)
def _object_sizes_immutable():
    before = hashlib.sha256(_proc_room.OBJECT_SIZES.numpy().tobytes()).hexdigest()
    yield
    after = hashlib.sha256(_proc_room.OBJECT_SIZES.numpy().tobytes()).hexdigest()
    assert after == before, "OBJECT_SIZES was mutated during the test"
