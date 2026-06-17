"""Unit tests for the path-tracking reward and termination functions.

These functions are thin readers of the per-tick state the subgoal command
term maintains, so the tests pin the read contract (which attribute, which
dtype, threshold semantics) against a stub command term.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from strafer_lab.tasks.navigation.mdp.rewards import (
    off_path_divergence_penalty,
    path_along_track_progress,
    path_complete_reward,
    path_cross_track_error,
)
from strafer_lab.tasks.navigation.mdp.terminations import (
    off_path_divergence,
    path_complete,
)


def _make_env(cross_track, progress, complete):
    term = SimpleNamespace(
        cross_track_error=torch.as_tensor(cross_track, dtype=torch.float32),
        along_track_progress=torch.as_tensor(progress, dtype=torch.float32),
        path_complete=torch.as_tensor(complete, dtype=torch.bool),
    )
    return SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: term)
    )


def test_along_track_progress_passthrough():
    env = _make_env([0.1, 0.5], [0.03, 0.0], [False, False])
    out = path_along_track_progress(env, "goal_command")
    torch.testing.assert_close(out, torch.tensor([0.03, 0.0]))


def test_cross_track_error_passthrough():
    env = _make_env([0.12, 0.4], [0.0, 0.0], [False, False])
    out = path_cross_track_error(env, "goal_command")
    torch.testing.assert_close(out, torch.tensor([0.12, 0.4]))


def test_path_complete_reward_is_binary_float():
    env = _make_env([0.0, 0.0], [0.0, 0.0], [True, False])
    out = path_complete_reward(env, "goal_command")
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.tensor([1.0, 0.0]))


def test_off_path_threshold_is_strictly_greater():
    # Exactly at the bound does NOT fire; just above does.
    env = _make_env([0.5, 0.5001, 0.1], [0.0] * 3, [False] * 3)
    pen = off_path_divergence_penalty(env, "goal_command", max_off_path_m=0.5)
    torch.testing.assert_close(pen, torch.tensor([0.0, 1.0, 0.0]))
    done = off_path_divergence(env, "goal_command", max_off_path_m=0.5)
    assert done.tolist() == [False, True, False]


def test_path_complete_termination_passthrough():
    env = _make_env([0.0, 0.0], [0.0, 0.0], [False, True])
    done = path_complete(env, "goal_command")
    assert done.dtype == torch.bool
    assert done.tolist() == [False, True]


def test_reward_and_termination_share_off_path_semantics():
    env = _make_env([0.7], [0.0], [False])
    pen = off_path_divergence_penalty(env, "goal_command", max_off_path_m=0.5)
    done = off_path_divergence(env, "goal_command", max_off_path_m=0.5)
    assert bool(done[0]) and float(pen[0]) == 1.0
