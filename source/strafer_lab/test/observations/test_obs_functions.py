# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for individual observation functions.

Functions under test (``strafer_lab.tasks.navigation.mdp.observations``):

* ``goal_position_relative``  — goal (x, y) in robot's local frame.
* ``goal_distance``           — scalar Euclidean distance to goal.
* ``goal_heading_relative``   — angular error to desired heading.
* ``body_velocity_xy``        — body-frame linear velocity (vx, vy).
* ``last_action``             — previous action from action manager.
* ``privileged_ground_truth`` — critic-only ground truth observation.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/observations/test_obs_functions.py -v
"""

import math
import numpy as np
import torch
import pytest

from test.common import CONFIDENCE_LEVEL, N_SETTLE_STEPS
from test.common.stats import one_sample_t_test

import warp as wp
from strafer_lab.tasks.navigation.mdp.observations import (
    goal_position_relative,
    goal_distance,
    goal_heading_relative,
    body_velocity_xy,
    last_action,
    privileged_ground_truth,
)


# =====================================================================
# Fixture alias
# =====================================================================

@pytest.fixture(scope="module")
def env(obs_env):
    """Reuse the shared observations conftest environment."""
    return obs_env


# =====================================================================
# Helpers
# =====================================================================

def _warm_up_env(env, n_steps: int = 10):
    """Run the env for a few steps so physics state is stable."""
    action = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(n_steps):
        env.step(action)


# =====================================================================
# goal_position_relative — Tests
# =====================================================================


def test_goal_position_relative_shape(env):
    """Output shape must be (num_envs, 2)."""
    env.reset()
    _warm_up_env(env, 5)

    rel = goal_position_relative(env, command_name="goal_command")

    assert rel.shape == (env.num_envs, 2), (
        f"Expected shape ({env.num_envs}, 2), got {rel.shape}"
    )


def test_goal_position_relative_finite(env):
    """All values must be finite."""
    env.reset()
    _warm_up_env(env, 5)

    rel = goal_position_relative(env, command_name="goal_command")

    assert torch.isfinite(rel).all(), (
        f"goal_position_relative contains non-finite values"
    )


# =====================================================================
# goal_distance — Tests
# =====================================================================


def test_goal_distance_shape(env):
    """Output shape must be (num_envs, 1)."""
    env.reset()
    _warm_up_env(env, 5)

    dist = goal_distance(env, command_name="goal_command")

    assert dist.shape == (env.num_envs, 1), (
        f"Expected shape ({env.num_envs}, 1), got {dist.shape}"
    )


def test_goal_distance_nonnegative(env):
    """Distance must be non-negative."""
    env.reset()
    _warm_up_env(env, 5)

    dist = goal_distance(env, command_name="goal_command")

    assert (dist >= 0).all(), (
        f"Negative distance: min={dist.min().item():.4f}"
    )


def test_goal_distance_matches_manual_computation(env):
    """Distance should match manual Euclidean computation."""
    env.reset()
    _warm_up_env(env, 5)

    dist = goal_distance(env, command_name="goal_command")

    # Manual computation
    robot_pos = wp.to_torch(env.scene["robot"].data.root_pos_w)[:, :2]
    goal_pos = env.command_manager.get_command("goal_command")[:, :2]
    expected = torch.norm(goal_pos - robot_pos, dim=-1, keepdim=True)

    err = (dist - expected).abs().max().item()

    print(f"\n  goal_distance vs manual:")
    print(f"    Max error: {err:.8f}")

    assert err < 1e-4, (
        f"goal_distance deviates from manual: max err = {err:.6f}"
    )


# =====================================================================
# goal_heading_relative — Tests
# =====================================================================


def test_goal_heading_relative_shape(env):
    """Output shape must be (num_envs, 1)."""
    env.reset()
    _warm_up_env(env, 5)

    heading = goal_heading_relative(env, command_name="goal_command")

    assert heading.shape == (env.num_envs, 1), (
        f"Expected shape ({env.num_envs}, 1), got {heading.shape}"
    )


def test_goal_heading_relative_bounded(env):
    """Heading error must be in [-pi, pi]."""
    env.reset()
    _warm_up_env(env, 5)

    heading = goal_heading_relative(env, command_name="goal_command")

    print(f"\n  goal_heading_relative bounds:")
    print(f"    Range: [{heading.min().item():.4f}, {heading.max().item():.4f}]")

    assert (heading >= -math.pi - 0.01).all() and (heading <= math.pi + 0.01).all(), (
        f"Heading error out of [-pi, pi]: "
        f"[{heading.min().item():.4f}, {heading.max().item():.4f}]"
    )


# =====================================================================
# body_velocity_xy — Tests
# =====================================================================


def test_body_velocity_xy_shape(env):
    """Output shape must be (num_envs, 2)."""
    env.reset()
    _warm_up_env(env, 5)

    vel = body_velocity_xy(env)

    assert vel.shape == (env.num_envs, 2), (
        f"Expected shape ({env.num_envs}, 2), got {vel.shape}"
    )


def test_body_velocity_near_zero_when_stationary(env):
    """Body velocity magnitude should be statistically near zero under zero action.

    Uses a one-sample t-test across all envs to check that mean speed is
    below a physical threshold (0.5 m/s).  Collecting over several timesteps
    gives N = num_envs * n_sample_steps independent-ish samples.
    """
    env.reset()

    zero_action = torch.zeros(env.num_envs, 3, device=env.device)

    # Let physics settle (roller contacts, initial transients)
    for _ in range(N_SETTLE_STEPS):
        env.step(zero_action)

    # Collect speed samples over multiple steps for statistical power
    n_sample_steps = 10
    speed_samples = []
    for _ in range(n_sample_steps):
        env.step(zero_action)
        vel = body_velocity_xy(env)  # (num_envs, 2)
        speed = vel.norm(dim=-1)     # (num_envs,)
        speed_samples.append(speed.cpu().numpy())

    # Flatten to 1-D array: (n_sample_steps * num_envs,)
    all_speeds = np.concatenate(speed_samples)

    # H0: mean speed >= 0.5 m/s   (robot is NOT stationary)
    # H1: mean speed <  0.5 m/s   (robot IS stationary)
    threshold = 0.5  # m/s — generous for a robot at rest
    result = one_sample_t_test(
        all_speeds,
        null_value=threshold,
        alternative="less",
        confidence_level=CONFIDENCE_LEVEL,
    )

    print(f"\n  body_velocity_xy stationary CI test:")
    print(f"    Mean speed:  {result.mean:.4f} m/s")
    print(f"    95% CI:      [{result.ci_low:.4f}, {result.ci_high:.4f}]")
    print(f"    p-value:     {result.p_value:.6f}")
    print(f"    N samples:   {result.n_samples}")

    assert result.reject_null, (
        f"Cannot reject that mean speed >= {threshold} m/s at "
        f"{CONFIDENCE_LEVEL:.0%} confidence.  "
        f"Mean={result.mean:.4f}, CI=[{result.ci_low:.4f}, {result.ci_high:.4f}], "
        f"p={result.p_value:.6f}"
    )


# =====================================================================
# last_action — Tests
# =====================================================================


def test_last_action_shape(env):
    """Output shape must be (num_envs, 3) for vx, vy, omega."""
    env.reset()
    _warm_up_env(env, 5)

    action = last_action(env)

    assert action.shape == (env.num_envs, 3), (
        f"Expected shape ({env.num_envs}, 3), got {action.shape}"
    )


# =====================================================================
# privileged_ground_truth — Tests
# =====================================================================


def test_privileged_ground_truth_shape(env):
    """Output shape must be (num_envs, 4) for body_vel(2) + dist(1) + heading(1)."""
    env.reset()
    _warm_up_env(env, 5)

    priv = privileged_ground_truth(env, command_name="goal_command")

    assert priv.shape == (env.num_envs, 4), (
        f"Expected shape ({env.num_envs}, 4), got {priv.shape}"
    )


def test_privileged_ground_truth_finite(env):
    """All values must be finite."""
    env.reset()
    _warm_up_env(env, 5)

    priv = privileged_ground_truth(env, command_name="goal_command")

    assert torch.isfinite(priv).all(), (
        "privileged_ground_truth contains non-finite values"
    )


def test_privileged_ground_truth_distance_nonnegative(env):
    """The distance component (index 2) must be non-negative."""
    env.reset()
    _warm_up_env(env, 5)

    priv = privileged_ground_truth(env, command_name="goal_command")
    distance = priv[:, 2]

    assert (distance >= 0).all(), (
        f"Privileged distance is negative: min={distance.min().item():.4f}"
    )
