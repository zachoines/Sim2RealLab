"""Unit tests for the rolling-subgoal command machinery.

Two layers, neither launches Isaac Sim:

1. :class:`PathCursor` — the pure-torch projection / cursor / lookahead math,
   exercised directly against synthetic paths and robot trajectories.
2. :class:`SubgoalCommand` — the Isaac Lab command term, exercised against a
   stub env graph (the same SimpleNamespace pattern the observation-parity
   tests use) so reset-time planning and per-tick updates run end to end.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from strafer_lab.tasks.navigation.mdp.commands import (
    SubgoalCommand,
    SubgoalCommandCfg,
    dwell_step,
)
from strafer_lab.tasks.navigation.path_planner import PathCursor

LOOKAHEAD = 1.0


# -----------------------------------------------------------------------------
# PathCursor
# -----------------------------------------------------------------------------


def _straight_path(length: float = 5.0, spacing: float = 0.05) -> torch.Tensor:
    n = int(length / spacing) + 1
    xs = torch.linspace(0.0, length, n)
    return torch.stack([xs, torch.zeros_like(xs)], dim=1)


def _make_cursor(path: torch.Tensor, num_envs: int = 1) -> PathCursor:
    cursor = PathCursor(num_envs=num_envs, max_points=256, device="cpu")
    cursor.set_paths(torch.arange(num_envs), [path] * num_envs)
    return cursor


def test_cursor_advances_monotonically_along_straight_path():
    cursor = _make_cursor(_straight_path())
    prev_arc = -1.0
    total_progress = 0.0
    for x in torch.arange(0.0, 4.0, 0.25):
        state = cursor.update(torch.tensor([[x, 0.0]]), LOOKAHEAD)
        arc = float(state.cursor_arc[0])
        assert arc >= prev_arc, "cursor moved backward"
        assert float(state.along_track_progress[0]) >= 0.0
        total_progress += float(state.along_track_progress[0])
        prev_arc = arc
    # Cursor tracked the robot's 3.75 m advance (within projection slack).
    assert total_progress == pytest.approx(3.75, abs=0.05)


def test_cursor_does_not_retreat_when_robot_backs_up():
    cursor = _make_cursor(_straight_path())
    cursor.update(torch.tensor([[2.0, 0.0]]), LOOKAHEAD)
    state = cursor.update(torch.tensor([[1.0, 0.0]]), LOOKAHEAD)
    assert float(state.cursor_arc[0]) == pytest.approx(2.0, abs=1e-4)
    assert float(state.along_track_progress[0]) == 0.0


def test_subgoal_holds_lookahead_distance_mid_path():
    cursor = _make_cursor(_straight_path())
    for x in (0.5, 1.0, 2.5, 3.5):
        state = cursor.update(torch.tensor([[x, 0.0]]), LOOKAHEAD)
        assert float(state.subgoal_xy[0, 0]) == pytest.approx(x + LOOKAHEAD, abs=1e-3)
        assert float(state.subgoal_xy[0, 1]) == pytest.approx(0.0, abs=1e-6)
        assert float(state.subgoal_heading[0]) == pytest.approx(0.0, abs=1e-5)


def test_lateral_offset_reports_cross_track_and_keeps_subgoal_on_path():
    cursor = _make_cursor(_straight_path())
    state = cursor.update(torch.tensor([[2.0, 0.08]]), LOOKAHEAD)
    assert float(state.cross_track[0]) == pytest.approx(0.08, abs=1e-4)
    # Subgoal stays on the path: lookahead ahead of the *projection*.
    assert float(state.subgoal_xy[0, 0]) == pytest.approx(2.0 + LOOKAHEAD, abs=1e-3)
    assert float(state.subgoal_xy[0, 1]) == pytest.approx(0.0, abs=1e-6)


def test_subgoal_clamps_to_final_point_near_path_end():
    cursor = _make_cursor(_straight_path(length=5.0))
    state = cursor.update(torch.tensor([[4.6, 0.0]]), LOOKAHEAD)
    assert float(state.subgoal_xy[0, 0]) == pytest.approx(5.0, abs=1e-4)
    assert float(state.end_distance[0]) == pytest.approx(0.4, abs=1e-4)


def test_per_env_lookahead_tensor_applies_distinct_distances():
    """A (num_envs,) lookahead tensor places each env's subgoal at its own
    distance; the cursor indexes it by env_ids."""
    cursor = _make_cursor(_straight_path(), num_envs=2)
    robot = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    lookahead = torch.tensor([0.5, 1.5])
    state = cursor.update(robot, lookahead)
    assert float(state.subgoal_xy[0, 0]) == pytest.approx(1.5, abs=1e-3)  # 1.0 + 0.5
    assert float(state.subgoal_xy[1, 0]) == pytest.approx(2.5, abs=1e-3)  # 1.0 + 1.5


def test_set_paths_rewinds_cursor():
    cursor = _make_cursor(_straight_path())
    cursor.update(torch.tensor([[3.0, 0.0]]), LOOKAHEAD)
    cursor.set_paths(torch.tensor([0]), [_straight_path(length=2.0)])
    state = cursor.update(torch.tensor([[0.0, 0.0]]), LOOKAHEAD)
    assert float(state.cursor_arc[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(state.total_arc[0]) == pytest.approx(2.0, abs=1e-4)


def test_cursor_follows_a_corner_path_tangent():
    # L-shaped path: +x for 2 m, then +y for 2 m.
    leg1 = torch.stack([torch.linspace(0, 2, 41), torch.zeros(41)], dim=1)
    leg2 = torch.stack([torch.full((40,), 2.0), torch.linspace(0.05, 2, 40)], dim=1)
    path = torch.cat([leg1, leg2])
    cursor = _make_cursor(path)
    # Robot halfway up the second leg: subgoal ahead on the +y segment.
    state = cursor.update(torch.tensor([[2.0, 0.6]]), LOOKAHEAD)
    assert float(state.subgoal_xy[0, 0]) == pytest.approx(2.0, abs=1e-3)
    assert float(state.subgoal_xy[0, 1]) == pytest.approx(1.6, abs=1e-2)
    assert float(state.subgoal_heading[0]) == pytest.approx(math.pi / 2, abs=1e-3)


def test_restricted_update_leaves_other_envs_untouched():
    cursor = _make_cursor(_straight_path(), num_envs=2)
    robot = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    cursor.update(robot, LOOKAHEAD)
    before = float(cursor._cursor[1])
    state = cursor.update(
        torch.tensor([[2.0, 0.0], [99.0, 99.0]]), LOOKAHEAD,
        env_ids=torch.tensor([0]),
    )
    assert state.subgoal_xy.shape[0] == 1
    assert float(cursor._cursor[1]) == before


# -----------------------------------------------------------------------------
# SubgoalCommand against a stub env
# -----------------------------------------------------------------------------


class _StubScene(dict):
    """Mapping for scene entities that also exposes ``env_origins``."""

    def __init__(self, entities: dict, env_origins: torch.Tensor):
        super().__init__(entities)
        self.env_origins = env_origins


def _make_stub_env(num_envs: int = 2):
    """Stub env graph exposing exactly what SubgoalCommand reads: robot root
    pose, env origins, and the proc-room free-space / spawn-point state."""
    device = "cpu"
    root_pos = torch.zeros(num_envs, 3)
    robot = SimpleNamespace(
        is_initialized=True,
        data=SimpleNamespace(
            root_pos_w=wp.from_torch(root_pos),
            # Body-frame linear velocity read by the dwell speed gate. Defaults
            # to zero (parked); tests override it via ``_set_body_speed``.
            root_lin_vel_b=wp.from_torch(torch.zeros(num_envs, 3)),
        ),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device=device,
        scene=_StubScene({"robot": robot}, torch.zeros(num_envs, 3)),
    )
    env._proc_room_free_space = torch.ones(num_envs, 80, 80, dtype=torch.bool)
    # Reachable goal candidates: one too close (0.5 m), several valid.
    pts = torch.tensor([[0.5, 0.0], [2.5, 0.0], [0.0, 2.5], [-2.5, 0.0], [0.0, -2.5]])
    env._proc_room_spawn_pts = pts.unsqueeze(0).repeat(num_envs, 1, 1)
    env._proc_room_spawn_count = torch.full((num_envs,), len(pts), dtype=torch.long)
    return env, root_pos


def _make_term(env) -> SubgoalCommand:
    cfg = SubgoalCommandCfg(asset_name="robot", debug_vis=False)
    return SubgoalCommand(cfg, env)


def _set_body_speed(env, speed: float) -> None:
    """Set every stub robot's body-frame speed to ``speed`` m/s (forward vx)."""
    vel = torch.zeros(env.num_envs, 3)
    vel[:, 0] = float(speed)
    env.scene["robot"].data.root_lin_vel_b = wp.from_torch(vel)


def test_reset_plans_path_and_emits_initial_subgoal():
    env, root_pos = _make_stub_env()
    term = _make_term(env)
    term.reset(env_ids=[0, 1])

    assert term.command.shape == (2, 3)
    for b in range(2):
        # Initial subgoal sits ~lookahead ahead of the robot along the path.
        dist = float(torch.norm(term.command[b, :2] - root_pos[b, :2]))
        assert dist == pytest.approx(term.cfg.lookahead_m, abs=0.1)
        # The sampled path endpoint honors the min goal distance.
        last = term.path_cursor.path_len[b] - 1
        end = term.path_cursor.paths[b, last]
        assert float(torch.norm(end - root_pos[b, :2])) >= term.cfg.min_goal_distance - 1e-3


def test_tracking_advances_subgoal_and_requires_parking_to_complete():
    env, root_pos = _make_stub_env(num_envs=1)
    term = _make_term(env)
    term.reset(env_ids=[0])

    end = term.path_cursor.paths[0, term.path_cursor.path_len[0] - 1].clone()
    total_progress = 0.0
    # Walk the robot straight to the path end in small steps, moving above the
    # dwell speed cap the whole way so completion cannot fire mid-traverse.
    for alpha in torch.linspace(0.05, 1.0, 20):
        root_pos[0, :2] = alpha * end
        _set_body_speed(env, 0.5)
        term.compute(dt=1.0 / 30.0)
        total_progress += float(term.along_track_progress[0])
        assert float(term.cross_track_error[0]) < 0.2

    assert total_progress == pytest.approx(float(end.norm()), abs=0.15)
    # Arriving at speed is NOT completion — parking is.
    assert not bool(term.path_complete[0])
    assert int(term.dwell_counter[0]) == 0
    # At the end the subgoal converges on the final path point.
    assert float(torch.norm(term.command[0, :2] - end)) < 0.05

    # Park at the end: hold still inside the radius. Completion fires exactly on
    # the dwell_steps-th consecutive holding step, not before.
    root_pos[0, :2] = end
    _set_body_speed(env, 0.0)
    for step in range(1, term.cfg.dwell_steps + 1):
        term.compute(dt=1.0 / 30.0)
        assert bool(term.path_complete[0]) is (step >= term.cfg.dwell_steps)
    assert int(term.dwell_counter[0]) == term.cfg.dwell_steps


def test_resample_after_completion_rewinds_tracking_state():
    env, root_pos = _make_stub_env(num_envs=1)
    term = _make_term(env)
    term.reset(env_ids=[0])
    end = term.path_cursor.paths[0, term.path_cursor.path_len[0] - 1].clone()
    root_pos[0, :2] = end
    _set_body_speed(env, 0.0)
    for _ in range(term.cfg.dwell_steps):
        term.compute(dt=1.0 / 30.0)
    assert bool(term.path_complete[0])
    assert int(term.dwell_counter[0]) == term.cfg.dwell_steps

    term.reset(env_ids=[0])
    assert not bool(term.path_complete[0])
    assert int(term.dwell_counter[0]) == 0
    assert float(term.along_track_progress[0]) == 0.0
    assert float(term.path_cursor._cursor[0]) == pytest.approx(0.0, abs=1e-4)


def test_waypoint_noise_keeps_path_in_free_space():
    env, _ = _make_stub_env(num_envs=1)
    # Add an obstacle slab so noise clamping has something to respect.
    env._proc_room_free_space[0, 50:54, :] = False
    term = _make_term(env)
    term.reset(env_ids=[0])
    n = int(term.path_cursor.path_len[0])
    path = term.path_cursor.paths[0, :n]
    origin = -80 * 0.1 / 2.0
    rows = ((path[:, 0] - origin) / 0.1).long().clamp(0, 79)
    cols = ((path[:, 1] - origin) / 0.1).long().clamp(0, 79)
    assert env._proc_room_free_space[0][rows, cols].all()


def test_fixed_lookahead_is_constant_across_envs():
    """With no randomization band the per-env lookahead stays at the cfg
    default for every env."""
    env, _ = _make_stub_env(num_envs=4)
    term = _make_term(env)
    term.reset(env_ids=[0, 1, 2, 3])
    assert torch.allclose(
        term._lookahead, torch.full((4,), term.cfg.lookahead_m)
    )


def test_lookahead_randomization_samples_per_env_within_band():
    """With a band set, each env draws its own lookahead inside the band, and
    its initial subgoal sits at that env's distance."""
    env, root_pos = _make_stub_env(num_envs=4)
    cfg = SubgoalCommandCfg(
        asset_name="robot", debug_vis=False,
        lookahead_randomization_m=(0.7, 1.3),
    )
    term = SubgoalCommand(cfg, env)
    term.reset(env_ids=[0, 1, 2, 3])
    la = term._lookahead
    assert (la >= 0.7).all() and (la <= 1.3).all()
    assert la.unique().numel() > 1, "expected distinct per-env lookahead draws"
    for b in range(4):
        dist = float(torch.norm(term.command[b, :2] - root_pos[b, :2]))
        assert dist == pytest.approx(float(la[b]), abs=0.1)


# -----------------------------------------------------------------------------
# dwell_step — the pure per-step parking predicate (Kit-free truth table)
# -----------------------------------------------------------------------------


def test_dwell_step_accumulates_and_fires_on_nth_consecutive_step():
    counter = torch.zeros(1, dtype=torch.int32)
    inside = torch.ones(1, dtype=torch.bool)
    slow = torch.ones(1, dtype=torch.bool)
    for step in range(1, 4):
        counter, parked = dwell_step(counter, inside, slow, dwell_steps=3)
        assert int(counter[0]) == step
        assert bool(parked[0]) is (step >= 3)


def test_dwell_step_resets_on_fast_or_outside():
    # Four envs: inside+slow, inside+fast, outside+slow, outside+fast.
    counter = torch.full((4,), 5, dtype=torch.int32)
    inside = torch.tensor([True, True, False, False])
    slow = torch.tensor([True, False, True, False])
    counter, parked = dwell_step(counter, inside, slow, dwell_steps=3)
    # Only the inside-and-slow env keeps accumulating; the rest reset to zero.
    torch.testing.assert_close(counter, torch.tensor([6, 0, 0, 0], dtype=torch.int32))
    torch.testing.assert_close(parked, torch.tensor([True, False, False, False]))


def test_dwell_step_reset_clears_then_reaccumulates_from_zero():
    counter = torch.full((1,), 9, dtype=torch.int32)
    # A broken hold (fast) zeroes an almost-complete counter.
    counter, parked = dwell_step(
        counter, torch.tensor([True]), torch.tensor([False]), dwell_steps=10
    )
    assert int(counter[0]) == 0 and not bool(parked[0])
    # The next hold starts the count over from one, not from where it left off.
    counter, parked = dwell_step(
        counter, torch.tensor([True]), torch.tensor([True]), dwell_steps=10
    )
    assert int(counter[0]) == 1 and not bool(parked[0])


# -----------------------------------------------------------------------------
# Dwell integration through the command term
# -----------------------------------------------------------------------------


def _reset_at_end(num_envs: int = 1):
    """Reset a subgoal term with the robot parked exactly at its path end."""
    env, root_pos = _make_stub_env(num_envs=num_envs)
    term = _make_term(env)
    term.reset(env_ids=list(range(num_envs)))
    end = term.path_cursor.paths[0, term.path_cursor.path_len[0] - 1].clone()
    root_pos[0, :2] = end
    _set_body_speed(env, 0.0)
    return env, root_pos, term, end


def test_dwell_counter_resets_when_speed_exceeds_cap():
    env, root_pos, term, _ = _reset_at_end()
    for _ in range(term.cfg.dwell_steps - 1):
        term.compute(dt=1.0 / 30.0)
    assert int(term.dwell_counter[0]) == term.cfg.dwell_steps - 1
    assert not bool(term.path_complete[0])
    # One step above the speed cap (still inside the radius) resets the dwell.
    _set_body_speed(env, term.cfg.dwell_speed_max_m_s + 1.0)
    term.compute(dt=1.0 / 30.0)
    assert int(term.dwell_counter[0]) == 0
    assert not bool(term.path_complete[0])


def test_dwell_counter_resets_when_leaving_radius():
    env, root_pos, term, _ = _reset_at_end()
    for _ in range(term.cfg.dwell_steps - 1):
        term.compute(dt=1.0 / 30.0)
    assert int(term.dwell_counter[0]) == term.cfg.dwell_steps - 1
    # Step back to the path start — well outside the completion radius.
    root_pos[0, :2] = term.path_cursor.paths[0, 0]
    term.compute(dt=1.0 / 30.0)
    assert int(term.dwell_counter[0]) == 0
    assert not bool(term.path_complete[0])


def test_capture_command_completes_on_touch_without_parking():
    """The capture cfg exempts itself from dwell: the coverage driver advances
    legs on touch, so a fast single in-radius step must still complete."""
    from strafer_lab.tasks.navigation.mdp.capture_commands import (
        CaptureSubgoalCommand,
        CaptureSubgoalCommandCfg,
    )

    env, root_pos = _make_stub_env(num_envs=1)
    term = CaptureSubgoalCommand(
        CaptureSubgoalCommandCfg(asset_name="robot", debug_vis=False), env
    )
    term.reset(env_ids=[0])
    leg = _straight_path(length=2.0)
    term.set_leg(leg)
    root_pos[0, :2] = leg[-1]
    _set_body_speed(env, 1.0)  # moving fast, never parking
    term.compute(dt=1.0 / 30.0)
    assert bool(term.path_complete[0])


# -----------------------------------------------------------------------------
# Config-level: the dwell params are present and wired to the subgoal task
# -----------------------------------------------------------------------------


def test_subgoal_cfg_defaults_enable_dwell_parking():
    cfg = SubgoalCommandCfg(asset_name="robot")
    assert cfg.dwell_radius_m == pytest.approx(0.3)
    assert cfg.dwell_speed_max_m_s == pytest.approx(0.1)
    assert cfg.dwell_steps == 10


def test_capture_cfg_keeps_instant_touch_completion():
    from strafer_lab.tasks.navigation.mdp.capture_commands import CaptureSubgoalCommandCfg

    cfg = CaptureSubgoalCommandCfg(asset_name="robot")
    assert cfg.dwell_steps == 1
    assert math.isinf(cfg.dwell_speed_max_m_s)
    # Radius is unchanged from the shared default — only the dwell is disabled.
    assert cfg.dwell_radius_m == pytest.approx(0.3)


def test_subgoal_task_command_cfg_carries_dwell_params():
    from strafer_lab.tasks.navigation.strafer_env_cfg import (
        CommandsCfg_ProcRoom_Subgoal,
    )

    goal = CommandsCfg_ProcRoom_Subgoal().goal_command
    assert isinstance(goal, SubgoalCommandCfg)
    assert goal.dwell_steps == 10
    assert goal.dwell_speed_max_m_s == pytest.approx(0.1)
    assert goal.dwell_radius_m == pytest.approx(0.3)
