# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for combined motor dynamics (all components active).

These tests verify the combined system behaves reasonably when all
dynamics components (motor filter, command delay, slew rate) are active
simultaneously. Uses qualitative tests since the exact response depends
on component interactions.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/actions/test_combined_dynamics.py -v
"""

import torch
import numpy as np

from test.actions.conftest import (
    reset_env_and_action_term,
    N_RESPONSE_STEPS,
)


def test_all_dynamics_enabled(realistic_env):
    """Verify realistic config has all dynamics enabled."""
    action_term = realistic_env.action_manager._terms["wheel_velocities"]

    print(f"\n  Realistic config - dynamics status:")
    print(f"    Motor dynamics: {action_term._enable_motor_dynamics}")
    print(f"    Command delay: {action_term._enable_command_delay}")
    print(f"    Slew rate: {action_term._enable_slew_rate}")

    assert action_term._enable_motor_dynamics, "Motor dynamics should be enabled"
    assert action_term._enable_command_delay, "Command delay should be enabled"
    assert action_term._enable_slew_rate, "Slew rate should be enabled"


def test_combined_response_is_gradual(realistic_env):
    """Verify combined system produces gradual (not instant) response."""
    action_term = reset_env_and_action_term(realistic_env, ideal_mode=False)

    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=realistic_env.device)
    step_action = step_action.repeat(realistic_env.num_envs, 1)

    responses = []
    for _ in range(N_RESPONSE_STEPS):
        action_term.process_actions(step_action)
        responses.append(action_term.processed_actions[0, 0].cpu().item())

    responses = np.abs(np.array(responses))
    initial = responses[0]
    final = responses[-1]

    print(f"\n  Combined dynamics - gradual response:")
    print(f"    Initial: {initial:.4f}")
    print(f"    Final: {final:.4f}")
    print(f"    Ratio: {initial / (final + 1e-9):.4f}")

    # Initial should be much less than final
    assert initial < final * 0.3, \
        f"Combined dynamics should produce gradual response (initial={initial:.4f}, final={final:.4f})"


def test_combined_no_overshoot(realistic_env):
    """Verify combined system doesn't overshoot (monotonic increase)."""
    action_term = reset_env_and_action_term(realistic_env, ideal_mode=False)

    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=realistic_env.device)
    step_action = step_action.repeat(realistic_env.num_envs, 1)

    responses = []
    for _ in range(N_RESPONSE_STEPS):
        action_term.process_actions(step_action)
        responses.append(action_term.processed_actions[0, 0].cpu().item())

    responses = np.abs(np.array(responses))
    v_final = responses[-1]
    noise_floor = max(1e-7 * v_final, 1e-6)
    diffs = np.diff(responses)
    significant_decreases = np.sum(diffs < -3 * noise_floor)

    print(f"\n  Combined dynamics - no overshoot:")
    print(f"    Significant decreases: {significant_decreases}")

    # Allow small number due to numerical noise
    assert significant_decreases <= 3, \
        f"Combined system should not overshoot, found {significant_decreases} significant decreases"


def test_combined_converges_to_steady_state(realistic_env):
    """Verify combined system converges to steady state."""
    action_term = reset_env_and_action_term(realistic_env, ideal_mode=False)

    step_action = torch.tensor([[0.5, 0.0, 0.0]], device=realistic_env.device)
    step_action = step_action.repeat(realistic_env.num_envs, 1)

    responses = []
    for _ in range(N_RESPONSE_STEPS * 2):
        action_term.process_actions(step_action)
        responses.append(action_term.processed_actions[0, 0].cpu().item())

    responses = np.abs(np.array(responses))
    final_samples = responses[-20:]
    mean_val = np.mean(final_samples)
    cv = np.std(final_samples) / (mean_val + 1e-9)

    print(f"\n  Combined dynamics - steady state:")
    print(f"    Mean final value: {mean_val:.4f}")
    print(f"    Coefficient of variation: {cv*100:.4f}%")

    assert cv < 0.05, f"Should converge to steady state (CV={cv*100:.2f}%)"


def test_reset_clears_all_state(realistic_env):
    """Verify reset clears all dynamics state."""
    action_term = reset_env_and_action_term(realistic_env, ideal_mode=False)

    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=realistic_env.device)
    step_action = step_action.repeat(realistic_env.num_envs, 1)

    # Build up velocity
    for _ in range(50):
        action_term.process_actions(step_action)

    pre_reset_vel = torch.abs(action_term.processed_actions).mean().item()

    # Reset
    action_term.reset(env_ids=torch.arange(realistic_env.num_envs, device=realistic_env.device))

    # Check state is cleared
    post_reset_vel = 0.0
    if hasattr(action_term, '_smoothed_wheel_vels'):
        post_reset_vel = max(post_reset_vel, torch.abs(action_term._smoothed_wheel_vels).mean().item())
    if hasattr(action_term, '_prev_wheel_vels'):
        post_reset_vel = max(post_reset_vel, torch.abs(action_term._prev_wheel_vels).mean().item())

    print(f"\n  Reset clears state:")
    print(f"    Pre-reset velocity: {pre_reset_vel:.4f}")
    print(f"    Post-reset state: {post_reset_vel:.6f}")

    assert pre_reset_vel > 0.1, "Should have nonzero velocity before reset"
    assert post_reset_vel < 0.01, "State should be cleared after reset"
