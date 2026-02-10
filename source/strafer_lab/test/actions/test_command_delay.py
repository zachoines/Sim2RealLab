# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for command delay buffer in action processing.

These tests verify command delay in ISOLATION by disabling motor dynamics
and slew rate limiting. This allows exact verification that delays are
precisely N timesteps without signal distortion.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/actions/test_command_delay.py -v
"""

import torch
import numpy as np

from test.actions.conftest import (
    configure_action_term_dynamics,
    CORRELATION_SIGNAL_PRESERVE,
)


def test_exact_step_shift(action_env):
    """Verify output is input shifted by exactly delay_steps.

    With delay only (no smoothing), a step input at t=0 should appear
    at output at t = delay_steps * dt.
    """
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=False, enable_delay=True, enable_slew=False
    )
    cfg = action_env.cfg.actions.wheel_velocities

    # Get the actual delay for env 0
    if hasattr(action_term, '_action_delay_buffer'):
        actual_delay = action_term._action_delay_buffer._time_lags[0].item()
    else:
        actual_delay = 0

    # Apply zero, then step to 1.0
    zero_action = torch.zeros(action_env.num_envs, 3, device=action_env.device)
    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=action_env.device).repeat(action_env.num_envs, 1)

    # Fill buffer with zeros
    for _ in range(cfg.max_delay_steps + 5):
        action_term.process_actions(zero_action)

    baseline = action_term.processed_actions[0, 0].cpu().item()

    # Now apply step and track when output changes
    responses = []
    for i in range(cfg.max_delay_steps + 10):
        action_term.process_actions(step_action)
        responses.append(action_term.processed_actions[0, 0].cpu().item())

    responses = np.array(responses)

    # Find first step where output deviates from baseline significantly
    threshold = 0.5 * (np.max(np.abs(responses)) - abs(baseline))
    first_change_idx = np.argmax(np.abs(responses - baseline) > threshold)

    print(f"\n  Command delay - exact step shift:")
    print(f"    Configured max delay: {cfg.max_delay_steps} steps")
    print(f"    Actual delay (env 0): {actual_delay} steps")
    print(f"    First output change at step: {first_change_idx}")
    print(f"    Expected: {actual_delay} steps")

    # Output should change exactly at step = actual_delay
    assert first_change_idx == actual_delay, \
        f"Delay should be exactly {actual_delay} steps, but output changed at step {first_change_idx}"


def test_signal_shape_preservation(action_env):
    """Verify delay buffer preserves signal shape (no distortion).

    A ramp input should produce a ramp output, just shifted in time.
    """
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=False, enable_delay=True, enable_slew=False
    )
    cfg = action_env.cfg.actions.wheel_velocities

    # Get actual delay for env 0
    if hasattr(action_term, '_action_delay_buffer'):
        actual_delay = action_term._action_delay_buffer._time_lags[0].item()
    else:
        actual_delay = 0

    # Create ramp input
    n_steps = 50 + cfg.max_delay_steps
    ramp_values = np.linspace(0, 1, n_steps)

    responses = []
    for i in range(n_steps):
        action = torch.tensor([[float(ramp_values[i]), 0.0, 0.0]], device=action_env.device)
        action = action.repeat(action_env.num_envs, 1)
        action_term.process_actions(action)
        responses.append(action_term.processed_actions[0, 0].cpu().item())

    responses = np.array(responses)

    # Output should equal input shifted by actual_delay steps
    if actual_delay > 0 and actual_delay < n_steps:
        input_segment = ramp_values[:-actual_delay] if actual_delay > 0 else ramp_values
        output_segment = np.abs(responses[actual_delay:])

        # Trim to same length
        min_len = min(len(input_segment), len(output_segment))
        input_segment = input_segment[:min_len]
        output_segment = output_segment[:min_len]

        # Correlation should be very high for identical signals
        correlation = np.corrcoef(input_segment, output_segment)[0, 1]

        print(f"\n  Command delay - signal preservation:")
        print(f"    Delay: {actual_delay} steps")
        print(f"    Correlation(input, shifted output): {correlation:.6f}")

        assert correlation > CORRELATION_SIGNAL_PRESERVE, \
            f"Delay buffer should preserve signal shape (correlation={correlation:.4f} < {CORRELATION_SIGNAL_PRESERVE})"
