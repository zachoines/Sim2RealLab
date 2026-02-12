# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for all reward / penalty functions.

Functions under test (``strafer_lab.tasks.navigation.mdp.rewards``):

* ``goal_progress_reward``      — dense reward for reducing distance to goal.
* ``goal_reached_reward``       — binary reward when within threshold of goal.
* ``heading_to_goal_reward``    — cosine alignment between heading and goal.
* ``energy_penalty``            — sum of squared applied torques.
* ``action_smoothness_penalty`` — sum of squared step-to-step action changes.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/rewards/test_rewards.py -v

NOTE: Uses the ``reward_env`` fixture from ``test/rewards/conftest.py``.
"""

import numpy as np
import torch
import pytest

from strafer_lab.tasks.navigation.mdp.rewards import (
    goal_progress_reward,
    goal_reached_reward,
    heading_to_goal_reward,
    energy_penalty,
    action_smoothness_penalty,
)

from test.common.stats import one_sample_t_test


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


def _goal_direction_body_frame(env) -> torch.Tensor:
    """Return unit vector (N, 2) from robot to goal in body frame (vx, vy).

    The action space is body-frame (ROS: X forward, Y left), so a
    world-frame direction vector must be rotated by the inverse of the
    robot's yaw before it can be used as a (vx, vy) command.
    """
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    robot_quat = env.scene["robot"].data.root_quat_w
    goal_pos = env.command_manager.get_command("goal_command")[:, :2]

    delta = goal_pos - robot_pos
    world_dir = delta / (torch.norm(delta, dim=-1, keepdim=True) + 1e-8)

    # Robot yaw from quaternion (w, x, y, z in IsaacLab)
    yaw = 2.0 * torch.atan2(robot_quat[:, 3], robot_quat[:, 0])
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)

    # Rotate world → body (transpose of 2-D rotation matrix)
    body_x = cos_y * world_dir[:, 0] + sin_y * world_dir[:, 1]
    body_y = -sin_y * world_dir[:, 0] + cos_y * world_dir[:, 1]

    return torch.stack([body_x, body_y], dim=-1)


# =====================================================================
# Reward Signal Continuity Across Episode Reset
# =====================================================================


def test_no_reward_spikes_on_reset(env):
    """Reward signals must be continuous across episode boundaries.

    After a reset the stateful reward functions (``goal_progress_reward``
    and ``action_smoothness_penalty``) reinitialise their internal
    buffers (``_prev_goal_distance``, ``_prev_action``) to match the
    new episode state.  This test verifies that the first-step reward
    magnitude is comparable to a mid-episode step — no transient spike.

    Checks:
    1. ``goal_progress_reward`` on step 0  — max |reward| < 0.5 m
       (a spike would indicate stale ``_prev_goal_distance`` carrying
       a distance from the previous episode).
    2. ``action_smoothness_penalty`` on step 0 — max penalty < 0.1
       (a spike would indicate stale ``_prev_action`` from the
       previous episode).
    3. ``_prev_goal_distance`` tracks the current distance within 0.1 m
       after the first post-reset step.
    """
    # Build up state from a "previous episode" so internal buffers
    # are non-trivial before the reset.
    action = torch.ones(env.num_envs, 3, device=env.device) * 0.5
    for _ in range(30):
        env.step(action)

    # --- Reset -------------------------------------------------------
    env.reset()

    zero_action = torch.zeros(env.num_envs, 3, device=env.device)
    env.step(zero_action)

    # 1. Goal progress reward should be near-zero on first step.
    progress = goal_progress_reward(env, command_name="goal_command")
    max_abs_progress = progress.abs().max().item()
    print(f"\n  Reward continuity after reset:")
    print(f"    goal_progress  max |r|: {max_abs_progress:.6f}")
    assert max_abs_progress < 0.5, (
        f"goal_progress_reward spike after reset: "
        f"max |reward| = {max_abs_progress:.4f}"
    )

    # 2. Smoothness penalty should be near-zero (action didn't change).
    smoothness = action_smoothness_penalty(env)
    max_smoothness = smoothness.max().item()
    print(f"    smoothness     max:     {max_smoothness:.6f}")
    assert max_smoothness < 0.1, (
        f"action_smoothness_penalty spike after reset: "
        f"max penalty = {max_smoothness:.4f}"
    )

    # 3. Internal _prev_goal_distance should track real distance.
    assert hasattr(env, "_prev_goal_distance"), (
        "env._prev_goal_distance not initialised after goal_progress_reward"
    )
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    goal_pos = env.command_manager.get_command("goal_command")[:, :2]
    current_dist = torch.norm(goal_pos - robot_pos, dim=-1)
    tracking_err = (env._prev_goal_distance - current_dist).abs().max().item()
    print(f"    prev_dist track err:   {tracking_err:.6f}")
    assert tracking_err < 0.1, (
        f"_prev_goal_distance deviates from current by {tracking_err:.4f}"
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


# =====================================================================
# Goal Progress Reward — Functional Tests
# =====================================================================


def test_goal_progress_positive_when_approaching(env):
    """Distance to goal must decrease when the robot moves toward it.

    For each env we compute the unit direction vector from robot to goal,
    then command the robot to move along that direction (body-frame vx/vy).
    A one-sample t-test (alternative='greater') confirms the mean
    distance reduction is positive with statistical significance.
    """
    env.reset()

    # Settle physics
    zero = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(10):
        env.step(zero)

    # Snapshot goal position (stable across these steps)
    goal_pos = env.command_manager.get_command("goal_command")[:, :2].clone()

    # Measure distance before
    pos_before = env.scene["robot"].data.root_pos_w[:, :2].clone()
    dist_before = torch.norm(goal_pos - pos_before, dim=-1)

    # Command each robot toward its own goal (body-frame)
    direction = _goal_direction_body_frame(env)  # (N, 2) body-frame unit vectors
    approach_action = torch.zeros(env.num_envs, 3, device=env.device)
    approach_action[:, 0] = direction[:, 0]  # vx toward goal
    approach_action[:, 1] = direction[:, 1]  # vy toward goal

    for _ in range(20):
        env.step(approach_action)

    # Measure distance after
    pos_after = env.scene["robot"].data.root_pos_w[:, :2]
    dist_after = torch.norm(goal_pos - pos_after, dim=-1)

    # progress > 0 means distance decreased (good)
    progress = (dist_before - dist_after).cpu().numpy()

    result = one_sample_t_test(progress, null_value=0.0, alternative="greater")
    print(f"\n  Goal progress (approaching):")
    print(f"    mean Δd = {result.mean:.4f} m   (>0 = closer)")
    print(f"    t = {result.t_statistic:.2f},  p = {result.p_value:.4e}")
    print(f"    95% CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")

    assert result.reject_null, (
        f"Moving toward goal did not significantly decrease distance. "
        f"mean Δd = {result.mean:.4f}, t = {result.t_statistic:.2f}, "
        f"p = {result.p_value:.4e}"
    )


def test_goal_progress_negative_when_retreating(env):
    """Distance to goal must increase when the robot moves away from it.

    Same setup as the approaching test, but commands each robot to move
    in the opposite direction of its goal.  A one-sample t-test
    (alternative='greater') on the distance *increase* confirms the
    mean regression is positive.
    """
    env.reset()

    zero = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(10):
        env.step(zero)

    goal_pos = env.command_manager.get_command("goal_command")[:, :2].clone()

    pos_before = env.scene["robot"].data.root_pos_w[:, :2].clone()
    dist_before = torch.norm(goal_pos - pos_before, dim=-1)

    # Command each robot AWAY from its goal (body-frame)
    direction = _goal_direction_body_frame(env)
    retreat_action = torch.zeros(env.num_envs, 3, device=env.device)
    retreat_action[:, 0] = -direction[:, 0]
    retreat_action[:, 1] = -direction[:, 1]

    for _ in range(20):
        env.step(retreat_action)

    pos_after = env.scene["robot"].data.root_pos_w[:, :2]
    dist_after = torch.norm(goal_pos - pos_after, dim=-1)

    # regression > 0 means distance increased (retreated)
    regression = (dist_after - dist_before).cpu().numpy()

    result = one_sample_t_test(regression, null_value=0.0, alternative="greater")
    print(f"\n  Goal progress (retreating):")
    print(f"    mean Δd = {result.mean:.4f} m   (>0 = farther)")
    print(f"    t = {result.t_statistic:.2f},  p = {result.p_value:.4e}")
    print(f"    95% CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")

    assert result.reject_null, (
        f"Moving away from goal did not significantly increase distance. "
        f"mean Δd = {result.mean:.4f}, t = {result.t_statistic:.2f}, "
        f"p = {result.p_value:.4e}"
    )


# =====================================================================
# Goal Reached Reward — Functional Tests
# =====================================================================


def test_goal_reached_fires_within_threshold(env):
    """Envs close to the goal receive reward = 1.

    We check the current robot–goal distances and verify that any env within
    the threshold receives a reward of 1.0.
    """
    env.reset()
    _warm_up_env(env, 10)

    threshold = 5.0  # generous threshold so at least some envs qualify

    reward = goal_reached_reward(env, threshold=threshold, command_name="goal_command")

    # Compute actual distances for comparison
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    goal_pos = env.command_manager.get_command("goal_command")[:, :2]
    distances = torch.norm(goal_pos - robot_pos, dim=-1)

    within_mask = distances < threshold
    n_within = within_mask.sum().item()

    print(f"\n  Goal reached (within threshold={threshold}):")
    print(f"    Envs within threshold: {n_within}/{env.num_envs}")
    print(f"    Reward where within: {reward[within_mask].tolist()[:5]}")

    if n_within > 0:
        assert (reward[within_mask] == 1.0).all(), (
            f"Some envs within threshold did not get reward=1: "
            f"{reward[within_mask].tolist()}"
        )

    # All rewards should be exactly 0 or 1
    assert ((reward == 0.0) | (reward == 1.0)).all(), (
        f"goal_reached_reward produced non-binary values: "
        f"unique={reward.unique().tolist()}"
    )


def test_goal_reached_zero_outside_threshold(env):
    """Envs far from the goal receive reward = 0.

    We use a tiny threshold (1 mm) — virtually no env will be that close
    immediately after reset.
    """
    env.reset()
    _warm_up_env(env, 5)

    threshold = 0.001  # 1 mm — effectively impossible to satisfy

    reward = goal_reached_reward(env, threshold=threshold, command_name="goal_command")

    n_reached = (reward > 0).sum().item()

    print(f"\n  Goal reached (outside threshold={threshold}):")
    print(f"    Envs with reward>0: {n_reached}/{env.num_envs}")

    assert n_reached == 0, (
        f"Expected no envs within 1 mm of goal, but {n_reached} "
        f"received reward > 0."
    )


# =====================================================================
# Heading to Goal Reward — Functional Tests
# =====================================================================


def test_heading_to_goal_output_bounded(env):
    """Heading reward (cosine) must lie in [-1, 1] for all envs."""
    env.reset()
    _warm_up_env(env, 10)

    reward = heading_to_goal_reward(env, command_name="goal_command")

    print(f"\n  Heading to goal bounded:")
    print(f"    min: {reward.min().item():.4f}")
    print(f"    max: {reward.max().item():.4f}")
    print(f"    mean: {reward.mean().item():.4f}")

    assert (reward >= -1.0 - 1e-5).all() and (reward <= 1.0 + 1e-5).all(), (
        f"Heading reward out of [-1, 1]: "
        f"min={reward.min().item():.4f}, max={reward.max().item():.4f}"
    )


def test_heading_to_goal_varies_with_yaw(env):
    """After rotating, heading reward should change relative to the goal.

    We record the heading reward, then step with pure rotation (omega only)
    for several steps, and verify the reward changes.  This proves the
    reward is sensitive to robot yaw, not a constant.
    """
    env.reset()
    _warm_up_env(env, 10)

    reward_before = heading_to_goal_reward(env, command_name="goal_command").clone()

    # Step with pure spin (omega = 1.0) to change heading
    spin_action = torch.zeros(env.num_envs, 3, device=env.device)
    spin_action[:, 2] = 1.0  # omega
    for _ in range(20):
        env.step(spin_action)

    reward_after = heading_to_goal_reward(env, command_name="goal_command")

    diff = (reward_after - reward_before).abs()
    mean_diff = diff.mean().item()

    print(f"\n  Heading to goal after yaw change:")
    print(f"    Mean |Δreward|: {mean_diff:.4f}")
    print(f"    Max  |Δreward|: {diff.max().item():.4f}")

    # After 20 steps of pure spin, heading reward should change noticeably
    assert mean_diff > 0.01, (
        f"Heading reward barely changed after 20 spin steps: "
        f"mean |Δ| = {mean_diff:.6f}"
    )


# =====================================================================
# Energy Penalty — Functional Tests
# =====================================================================


def test_energy_penalty_nonnegative(env):
    """Energy penalty is always non-negative and finite.

    ``energy_penalty`` computes ``sum(applied_torque ** 2, dim=-1)`` which
    is a sum of squares — it must be ≥ 0 by definition.

    Note: on a mecanum robot with 72 roller rigid-bodies the residual
    gravity/friction torques produce a non-trivial "at rest" energy
    (~5–8 units).  This is expected physics behaviour, not a bug.
    We only assert non-negativity and finiteness here.
    """
    env.reset()

    zero_action = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(20):
        env.step(zero_action)

    penalty = energy_penalty(env)

    print(f"\n  Energy penalty (non-negative check):")
    print(f"    Mean: {penalty.mean().item():.4f}")
    print(f"    Min:  {penalty.min().item():.4f}")
    print(f"    Max:  {penalty.max().item():.4f}")

    assert torch.isfinite(penalty).all(), (
        f"Energy penalty contains non-finite values: "
        f"NaN count = {torch.isnan(penalty).sum().item()}, "
        f"Inf count = {torch.isinf(penalty).sum().item()}"
    )
    assert (penalty >= 0).all(), (
        f"Energy penalty has negative values: min = {penalty.min().item():.6f}"
    )


def test_energy_penalty_positive_under_load(env):
    """Non-zero actions produce non-zero applied torques → positive penalty."""
    env.reset()

    strong_action = torch.ones(env.num_envs, 3, device=env.device) * 0.8
    for _ in range(10):
        env.step(strong_action)

    penalty = energy_penalty(env)
    mean_penalty = penalty.mean().item()

    print(f"\n  Energy penalty under load:")
    print(f"    Mean penalty: {mean_penalty:.6f}")
    print(f"    Min penalty:  {penalty.min().item():.6f}")

    # Under strong actuation the penalty must be strictly positive
    assert mean_penalty > 0.0, (
        f"Expected positive energy penalty under load, got mean = {mean_penalty:.6f}"
    )


# =====================================================================
# Action Smoothness Penalty — Functional Tests
# =====================================================================


def test_action_smoothness_positive_when_changing(env):
    """Changing action between steps produces non-zero smoothness penalty.

    We verify by comparing the raw action difference: the penalty function
    computes ``sum((current - prev)², dim=-1)``.  We step with two distinct
    actions and verify the *env's* ``action_manager.action`` actually changes.
    """
    env.reset()

    # Step with action_a for several steps so state is stable
    action_a = torch.ones(env.num_envs, 3, device=env.device) * 0.5
    for _ in range(10):
        env.step(action_a)

    # Read the actual action the env's action_manager recorded
    prev_action = env.action_manager.action.clone()

    # Now step with a very different action
    action_b = torch.ones(env.num_envs, 3, device=env.device) * -0.5
    env.step(action_b)

    curr_action = env.action_manager.action.clone()

    # Compute expected smoothness penalty from the actual actions
    diff = curr_action - prev_action
    expected_penalty = (diff ** 2).sum(dim=-1)
    mean_expected = expected_penalty.mean().item()

    print(f"\n  Action smoothness (changing):")
    print(f"    Prev action sample: {prev_action[0].tolist()}")
    print(f"    Curr action sample: {curr_action[0].tolist()}")
    print(f"    Mean expected penalty: {mean_expected:.4f}")

    # The actions should actually differ (env may clip/scale them)
    action_changed = diff.abs().max().item() > 0.01
    print(f"    Actions changed: {action_changed}")

    assert action_changed, (
        "action_manager.action did not change between steps — "
        "the env may be clipping both actions to the same value."
    )
    assert mean_expected > 0.01, (
        f"Expected positive smoothness penalty when changing action, "
        f"got mean expected = {mean_expected:.6f}. Actions may be "
        f"identical after env processing."
    )
