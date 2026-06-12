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

from strafer_lab.tasks.navigation.mdp.commands import SubgoalCommand, SubgoalCommandCfg
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
        data=SimpleNamespace(root_pos_w=wp.from_torch(root_pos)),
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


def test_tracking_a_path_advances_subgoal_and_completes():
    env, root_pos = _make_stub_env(num_envs=1)
    term = _make_term(env)
    term.reset(env_ids=[0])

    end = term.path_cursor.paths[0, term.path_cursor.path_len[0] - 1].clone()
    total_progress = 0.0
    # Walk the robot straight to the path end in small steps.
    for alpha in torch.linspace(0.05, 1.0, 20):
        root_pos[0, :2] = alpha * end
        term.compute(dt=1.0 / 30.0)
        total_progress += float(term.along_track_progress[0])
        assert float(term.cross_track_error[0]) < 0.2

    assert total_progress == pytest.approx(float(end.norm()), abs=0.15)
    assert bool(term.path_complete[0])
    # At the end the subgoal converges on the final path point.
    assert float(torch.norm(term.command[0, :2] - end)) < 0.05


def test_resample_after_completion_rewinds_tracking_state():
    env, root_pos = _make_stub_env(num_envs=1)
    term = _make_term(env)
    term.reset(env_ids=[0])
    end = term.path_cursor.paths[0, term.path_cursor.path_len[0] - 1].clone()
    root_pos[0, :2] = end
    term.compute(dt=1.0 / 30.0)
    assert bool(term.path_complete[0])

    term.reset(env_ids=[0])
    assert not bool(term.path_complete[0])
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
