# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reward function behaviour.

Reward functions covered:

* ``goal_progress_reward``  — dense reward for reducing distance to goal.
* ``action_smoothness_penalty`` — penalises large step-to-step action changes.

The original file focused on **reset-spike regression** (reward/penalty spike
on episode reset caused by stale ``_prev_goal_distance`` / ``_prev_action``).
Those regression tests are retained and new functional-coverage tests will be
added in Phase 2 of the test-update plan.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/rewards/test_rewards.py -v

NOTE: Uses the ``reward_env`` fixture from ``test/rewards/conftest.py``.
"""

import torch
import pytest

from strafer_lab.tasks.navigation.mdp.rewards import (
    goal_progress_reward,
    action_smoothness_penalty,
)


# =====================================================================
# Fixture alias — use the shared conftest env
# =====================================================================

@pytest.fixture(scope="module")
def env(reward_env):
    """Reuse the shared rewards conftest environment."""
    return reward_env


# =====================================================================
# Helpers
# =====================================================================

def _warm_up_env(env, n_steps: int = 20):
    """Run the env for a few steps so that state buffers are populated."""
    action = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(n_steps):
        env.step(action)


# =====================================================================
# Goal Progress Reward Reset Tests
# =====================================================================


def test_goal_progress_no_spike_after_reset(env):
    """Verify no reward spike on the first step after env.reset().

    The bug: if ``_prev_goal_distance`` was set to the *old* episode's
    final distance, the first step's "progress" = old_distance - new_distance
    could be enormous.

    After the fix, the reset mask sets ``_prev_goal_distance`` to the
    current distance on step 0, yielding progress ≈ 0.
    """
    # Warm up so _prev_goal_distance exists from a previous episode
    _warm_up_env(env, 30)

    # Full reset — triggers episode_length_buf = 0 on next step
    env.reset()

    # Take one step with zero action
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)
    env.step(zero_action)

    # Compute the reward
    reward = goal_progress_reward(env, command_name="goal_command")

    # On the first step after reset, progress should be near-zero
    # because the reset mask sets _prev_goal_distance = current_distance.
    # Allow some tolerance for physics stepping moving the robot slightly.
    max_abs_reward = reward.abs().max().item()

    print(f"\n  Goal progress after reset:")
    print(f"    Max |reward|: {max_abs_reward:.6f}")
    print(f"    Mean reward:  {reward.mean().item():.6f}")

    # A spike would be > 1.0 (corresponds to >1 m jump).
    # Normal step-to-step should be << 0.1
    assert max_abs_reward < 0.5, (
        f"Reward spike detected after reset: max |reward| = {max_abs_reward:.4f}. "
        "This suggests _prev_goal_distance was not reset properly."
    )


def test_goal_progress_reset_mask_applied(env):
    """Verify _prev_goal_distance is updated for envs where episode_length_buf == 0."""
    _warm_up_env(env, 20)
    env.reset()

    zero_action = torch.zeros(env.num_envs, 3, device=env.device)
    env.step(zero_action)

    # After stepping, call reward to ensure _prev_goal_distance is set
    _ = goal_progress_reward(env, command_name="goal_command")

    # The _prev_goal_distance should now exist and match the current distance
    assert hasattr(env, "_prev_goal_distance"), (
        "env._prev_goal_distance not initialised after calling goal_progress_reward"
    )

    # Compute current distance for comparison
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    command = env.command_manager.get_command("goal_command")
    goal_pos = command[:, :2]
    current_distance = torch.norm(goal_pos - robot_pos, dim=-1)

    diff = (env._prev_goal_distance - current_distance).abs()
    max_diff = diff.max().item()

    print(f"\n  _prev_goal_distance consistency:")
    print(f"    Max |prev - current|: {max_diff:.6f}")

    # After just one step with zero action, the robot barely moves,
    # so prev and current should be very close.
    assert max_diff < 0.1, (
        f"_prev_goal_distance deviates from current by {max_diff:.4f}. "
        "Reset mask may not be applied correctly."
    )


# =====================================================================
# Action Smoothness Penalty Reset Tests
# =====================================================================


def test_action_smoothness_no_spike_after_reset(env):
    """Verify no penalty spike on the first step after env.reset().

    The bug: if ``_prev_action`` retained the *old* episode's last action,
    the new episode's first action might differ greatly, yielding a huge
    ``(current - prev)²`` penalty.

    After the fix, the reset mask sets ``_prev_action`` = ``current_action``
    on step 0, so the penalty is ≈ 0.
    """
    # Warm up with non-zero actions to set _prev_action to something non-zero
    action = torch.ones(env.num_envs, 3, device=env.device) * 0.5
    for _ in range(20):
        env.step(action)

    # Reset — triggers episode_length_buf = 0
    env.reset()

    # Step with zero action (maximally different from prev 0.5 if not reset!)
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)
    env.step(zero_action)

    penalty = action_smoothness_penalty(env)

    max_penalty = penalty.max().item()

    print(f"\n  Action smoothness after reset:")
    print(f"    Max penalty: {max_penalty:.6f}")
    print(f"    Mean penalty: {penalty.mean().item():.6f}")

    # Without the fix, penalty ≈ 3 * 0.5² = 0.75 (3 dims × 0.25).
    # With the fix, penalty should be near zero because reset mask
    # sets _prev_action = current_action on step 0.
    assert max_penalty < 0.1, (
        f"Penalty spike detected after reset: max penalty = {max_penalty:.4f}. "
        "This suggests _prev_action was not reset properly."
    )


def test_action_smoothness_steady_state_zero(env):
    """Verify zero penalty when action is constant (no change)."""
    env.reset()

    constant_action = torch.ones(env.num_envs, 3, device=env.device) * 0.3

    # A few warm-up steps with the same action
    for _ in range(5):
        env.step(constant_action)

    # Now the penalty should be near zero since action hasn't changed
    penalty = action_smoothness_penalty(env)
    max_penalty = penalty.max().item()

    print(f"\n  Action smoothness steady state:")
    print(f"    Max penalty: {max_penalty:.6f}")

    assert max_penalty < 1e-6, (
        f"Non-zero penalty with constant action: {max_penalty:.6f}"
    )
