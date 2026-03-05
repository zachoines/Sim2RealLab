# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for curriculum learning functions.

Classes under test (``strafer_lab.tasks.navigation.mdp.curriculums``):

* ``GoalDistanceCurriculum``  — expands goal range based on success rate.
* ``ObstacleCurriculum``      — activates more obstacles as agent improves.

Tests access curriculum terms through ``env.curriculum_manager`` internals.
The class-based term instances live at ``_term_cfgs[idx].func``.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/curriculums/test_curriculums.py -v
"""

import torch
import pytest

from strafer_lab.tasks.navigation.mdp.curriculums import (
    GoalDistanceCurriculum,
    ObstacleCurriculum,
)


# =====================================================================
# Fixture alias
# =====================================================================

@pytest.fixture(scope="module")
def env(curr_env):
    """Reuse the shared curriculums conftest environment."""
    return curr_env


# =====================================================================
# Helpers
# =====================================================================

def _get_curriculum_term(env, term_name: str):
    """Retrieve a curriculum term instance from the environment.

    Isaac Lab's CurriculumManager stores class-based term instances at
    ``_term_cfgs[idx].func``, indexed parallel to ``_term_names``.
    """
    cm = env.curriculum_manager
    idx = cm._term_names.index(term_name)
    return cm._term_cfgs[idx].func


def _warm_up_env(env, n_steps: int = 10):
    """Run the env for a few steps so physics state is stable."""
    action = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(n_steps):
        env.step(action)


# =====================================================================
# GoalDistanceCurriculum — Tests
# =====================================================================


def test_goal_distance_curriculum_exists(env):
    """The environment must have a goal_distance curriculum term."""
    term = _get_curriculum_term(env, "goal_distance")

    assert isinstance(term, GoalDistanceCurriculum), (
        f"Expected GoalDistanceCurriculum, got {type(term).__name__}"
    )


def test_goal_distance_curriculum_has_zero_initial_difficulty(env):
    """After reset, difficulty should start at zero for all envs."""
    env.reset()

    term = _get_curriculum_term(env, "goal_distance")

    assert term._difficulty.shape == (env.num_envs,), (
        f"Expected shape ({env.num_envs},), got {term._difficulty.shape}"
    )

    print(f"\n  GoalDistanceCurriculum initial state:")
    print(f"    Difficulty range: [{term._difficulty.min().item():.1f}, "
          f"{term._difficulty.max().item():.1f}]")


def test_goal_distance_curriculum_range_clamped(env):
    """Manually forcing high difficulty: range must not exceed max_range."""
    env.reset()

    term = _get_curriculum_term(env, "goal_distance")
    original_difficulty = term._difficulty.clone()

    # Force high difficulty
    term._difficulty[:] = 100.0
    current_range = term._current_range()

    max_range = term.cfg.params.get("max_range", 5.0)

    print(f"\n  GoalDistanceCurriculum range clamp:")
    print(f"    Max range config: {max_range}")
    print(f"    Current range at difficulty=100: {current_range.max().item():.2f}")

    assert (current_range <= max_range + 1e-6).all(), (
        f"Range exceeds max_range: {current_range.max().item():.2f} > {max_range}"
    )

    # Restore
    term._difficulty[:] = original_difficulty


def test_goal_distance_curriculum_returns_float(env):
    """Calling the curriculum via the manager should produce a float result."""
    env.reset()
    _warm_up_env(env, 5)

    term = _get_curriculum_term(env, "goal_distance")
    env_ids = torch.arange(env.num_envs, device=env.device)

    result = term(env, env_ids)

    assert isinstance(result, float), (
        f"Expected float, got {type(result)}"
    )

    print(f"\n  GoalDistanceCurriculum return:")
    print(f"    Mean range: {result:.2f}")


# =====================================================================
# ObstacleCurriculum — Tests
# =====================================================================


def test_obstacle_curriculum_exists(env):
    """The environment must have an obstacle_difficulty curriculum term."""
    term = _get_curriculum_term(env, "obstacle_difficulty")

    assert isinstance(term, ObstacleCurriculum), (
        f"Expected ObstacleCurriculum, got {type(term).__name__}"
    )


def test_obstacle_curriculum_active_count_property(env):
    """active_obstacle_count property must expose per-env counts."""
    env.reset()

    term = _get_curriculum_term(env, "obstacle_difficulty")
    counts = term.active_obstacle_count

    assert counts.shape == (env.num_envs,), (
        f"Expected shape ({env.num_envs},), got {counts.shape}"
    )

    initial_count = term.cfg.params.get("initial_count", 2)
    print(f"\n  ObstacleCurriculum active counts:")
    print(f"    Config initial_count: {initial_count}")
    print(f"    Unique counts: {counts.unique().tolist()}")

    assert (counts >= 0).all(), (
        f"Negative obstacle counts: min={counts.min().item()}"
    )
