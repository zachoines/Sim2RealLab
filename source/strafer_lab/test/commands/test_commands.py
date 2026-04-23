# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the GoalCommand command term.

Class under test: ``strafer_lab.tasks.navigation.mdp.commands.GoalCommand``

Tests cover:
* Command tensor shape and content.
* Goal resampling at episode reset and mid-episode.
* Goal distance metrics tracking.
* ``goal_resampled`` flag behavior.
* Minimum distance constraint enforcement.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/commands/test_commands.py -v
"""

import math
import torch
import pytest
import warp as wp


# =====================================================================
# Fixture alias
# =====================================================================

@pytest.fixture(scope="module")
def env(cmd_env):
    """Reuse the shared commands conftest environment."""
    return cmd_env


# =====================================================================
# Helpers
# =====================================================================

def _warm_up_env(env, n_steps: int = 10):
    """Run the env for a few steps so physics state is stable."""
    action = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(n_steps):
        env.step(action)


# =====================================================================
# GoalCommand — Shape and Content Tests
# =====================================================================


def test_command_tensor_shape(env):
    """Goal command must have shape (num_envs, 3) = (x, y, heading)."""
    env.reset()
    _warm_up_env(env, 5)

    command = env.command_manager.get_command("goal_command")

    assert command.shape == (env.num_envs, 3), (
        f"Expected shape ({env.num_envs}, 3), got {command.shape}"
    )


def test_command_contains_valid_positions(env):
    """Goal x and y must be finite real numbers."""
    env.reset()
    _warm_up_env(env, 5)

    command = env.command_manager.get_command("goal_command")
    xy = command[:, :2]

    assert torch.isfinite(xy).all(), (
        f"Goal positions contain non-finite values: "
        f"NaN count = {torch.isnan(xy).sum().item()}, "
        f"Inf count = {torch.isinf(xy).sum().item()}"
    )


def test_command_heading_in_range(env):
    """Goal heading must be in [-pi, pi]."""
    env.reset()
    _warm_up_env(env, 5)

    command = env.command_manager.get_command("goal_command")
    heading = command[:, 2]

    print(f"\n  Goal heading range:")
    print(f"    [{heading.min().item():.4f}, {heading.max().item():.4f}]")

    assert (heading >= -math.pi - 0.01).all() and (heading <= math.pi + 0.01).all(), (
        f"Heading out of [-pi, pi] range: "
        f"[{heading.min().item():.4f}, {heading.max().item():.4f}]"
    )


# =====================================================================
# GoalCommand — Resampling Tests
# =====================================================================


def test_goal_changes_on_reset(env):
    """After a full env reset, goals should be resampled."""
    env.reset()
    _warm_up_env(env, 5)

    goal_before = env.command_manager.get_command("goal_command")[:, :2].clone()

    env.reset()

    goal_after = env.command_manager.get_command("goal_command")[:, :2]

    diff = (goal_after - goal_before).abs()
    any_changed = (diff > 0.01).any().item()

    print(f"\n  Goal change on reset:")
    print(f"    Max position diff: {diff.max().item():.4f}")

    assert any_changed, (
        "Goals did not change after env reset — resampling may be broken"
    )


def test_goal_distance_metric_exists(env):
    """GoalCommand must track distance_to_goal in its metrics dict."""
    env.reset()
    _warm_up_env(env, 5)

    command_term = env.command_manager.get_term("goal_command")

    assert "distance_to_goal" in command_term.metrics, (
        f"Missing 'distance_to_goal' metric. "
        f"Available metrics: {list(command_term.metrics.keys())}"
    )

    distance = command_term.metrics["distance_to_goal"]
    assert distance.shape == (env.num_envs,), (
        f"Expected shape ({env.num_envs},), got {distance.shape}"
    )
    assert (distance >= 0).all(), (
        f"Negative distance_to_goal: min={distance.min().item():.4f}"
    )


def test_goal_resampled_flag_exists(env):
    """GoalCommand must expose the goal_resampled boolean flag."""
    env.reset()

    command_term = env.command_manager.get_term("goal_command")

    assert hasattr(command_term, "goal_resampled"), (
        "GoalCommand missing 'goal_resampled' attribute"
    )
    assert command_term.goal_resampled.dtype == torch.bool, (
        f"goal_resampled should be bool, got {command_term.goal_resampled.dtype}"
    )
    assert command_term.goal_resampled.shape == (env.num_envs,), (
        f"Expected shape ({env.num_envs},), got {command_term.goal_resampled.shape}"
    )


def test_goal_minimum_distance_from_robot(env):
    """Sampled goals must be at least min_goal_distance from the robot."""
    env.reset()
    _warm_up_env(env, 5)

    command_term = env.command_manager.get_term("goal_command")
    min_dist = command_term.cfg.min_goal_distance

    robot_pos = wp.to_torch(env.scene["robot"].data.root_pos_w)[:, :2]
    goal_pos = command_term.command[:, :2]

    distances = torch.norm(goal_pos - robot_pos, dim=-1)

    print(f"\n  Goal minimum distance:")
    print(f"    cfg.min_goal_distance: {min_dist}")
    print(f"    Actual min distance: {distances.min().item():.4f}")

    # Some tolerance since the robot may have moved slightly
    assert (distances >= min_dist - 0.5).all(), (
        f"Goal placed too close to robot: "
        f"min_dist={distances.min().item():.4f}, required={min_dist}"
    )


def test_goal_reach_threshold_config(env):
    """GoalCommandCfg must expose goal_reach_threshold."""
    command_term = env.command_manager.get_term("goal_command")

    assert hasattr(command_term.cfg, "goal_reach_threshold"), (
        "GoalCommandCfg missing 'goal_reach_threshold' attribute"
    )
    assert command_term.cfg.goal_reach_threshold > 0, (
        f"goal_reach_threshold should be positive, "
        f"got {command_term.cfg.goal_reach_threshold}"
    )
