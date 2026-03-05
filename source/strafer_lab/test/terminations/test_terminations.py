# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for termination functions.

Functions under test (``strafer_lab.tasks.navigation.mdp.terminations``):

* ``robot_flipped``  — terminate when robot tips past threshold angle.
* ``goal_reached``   — terminate when robot reaches the goal.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/terminations/test_terminations.py -v
"""

import torch
import pytest

from strafer_lab.tasks.navigation.mdp.terminations import (
    robot_flipped,
    goal_reached,
)


# =====================================================================
# Fixture alias
# =====================================================================

@pytest.fixture(scope="module")
def env(term_env):
    """Reuse the shared terminations conftest environment."""
    return term_env


# =====================================================================
# Helpers
# =====================================================================

def _warm_up_env(env, n_steps: int = 10):
    """Run the env for a few steps so physics state is stable."""
    action = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(n_steps):
        env.step(action)


# =====================================================================
# robot_flipped — Tests
# =====================================================================


def test_robot_flipped_false_when_upright(env):
    """An upright robot should not be flagged as flipped.

    Projected gravity for an upright robot is approximately (0, 0, -1).
    The z-component (-1.0) is well below any positive threshold.
    """
    env.reset()
    _warm_up_env(env, 10)

    flipped = robot_flipped(env, threshold=0.5)

    print(f"\n  robot_flipped (upright):")
    print(f"    Flipped count: {flipped.sum().item()}/{env.num_envs}")

    gravity_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    print(f"    gravity_z mean: {gravity_z.mean().item():.4f}")

    assert not flipped.any(), (
        f"Upright robot flagged as flipped: "
        f"{flipped.sum().item()}/{env.num_envs} environments"
    )


def test_robot_flipped_returns_bool_tensor(env):
    """Return type must be a boolean tensor with shape (num_envs,)."""
    env.reset()
    _warm_up_env(env, 5)

    flipped = robot_flipped(env, threshold=0.5)

    assert flipped.dtype == torch.bool, (
        f"Expected bool dtype, got {flipped.dtype}"
    )
    assert flipped.shape == (env.num_envs,), (
        f"Expected shape ({env.num_envs},), got {flipped.shape}"
    )


def test_robot_flipped_threshold_zero_still_upright(env):
    """Even at threshold=0, an upright robot (gravity_z ~ -1.0) is not flipped.

    projected_gravity_b[:, 2] ≈ -1.0 (upright), so the condition
    ``gravity_z > 0.0`` is not met.
    """
    env.reset()
    _warm_up_env(env, 10)

    flipped = robot_flipped(env, threshold=0.0)

    gravity_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    print(f"\n  robot_flipped (threshold=0):")
    print(f"    gravity_z range: [{gravity_z.min().item():.4f}, {gravity_z.max().item():.4f}]")
    print(f"    Flipped: {flipped.sum().item()}/{env.num_envs}")

    # Upright robot has gravity_z ≈ -1.0, so even threshold=0 should not trigger
    assert not flipped.any(), (
        f"Threshold=0 incorrectly flags upright robot as flipped. "
        f"gravity_z range: [{gravity_z.min().item():.4f}, {gravity_z.max().item():.4f}]"
    )


# =====================================================================
# goal_reached — Tests
# =====================================================================


def test_goal_reached_returns_bool(env):
    """Return type must be a boolean tensor with shape (num_envs,)."""
    env.reset()
    _warm_up_env(env, 5)

    reached = goal_reached(env, command_name="goal_command", threshold=0.3)

    assert reached.dtype == torch.bool, (
        f"Expected bool dtype, got {reached.dtype}"
    )
    assert reached.shape == (env.num_envs,), (
        f"Expected shape ({env.num_envs},), got {reached.shape}"
    )


def test_goal_reached_none_at_tiny_threshold(env):
    """With a 1 mm threshold, no env should reach the goal after reset."""
    env.reset()
    _warm_up_env(env, 5)

    reached = goal_reached(env, command_name="goal_command", threshold=0.001)

    n_reached = reached.sum().item()
    print(f"\n  goal_reached (tiny threshold):")
    print(f"    Reached: {n_reached}/{env.num_envs}")

    assert n_reached == 0, (
        f"Expected no envs within 1 mm of goal, but {n_reached} triggered"
    )


def test_goal_reached_all_at_huge_threshold(env):
    """With a 1 km threshold, all envs must be within range."""
    env.reset()
    _warm_up_env(env, 5)

    reached = goal_reached(env, command_name="goal_command", threshold=1000.0)

    n_reached = reached.sum().item()
    print(f"\n  goal_reached (huge threshold):")
    print(f"    Reached: {n_reached}/{env.num_envs}")

    assert n_reached == env.num_envs, (
        f"Expected all {env.num_envs} envs within 1km threshold, "
        f"but only {n_reached} triggered"
    )
