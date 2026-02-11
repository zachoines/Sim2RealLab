# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests comparing ideal vs realistic motor dynamics.

These tests use Welch's t-test to statistically compare response
characteristics between ideal (no dynamics) and realistic (motor filter +
delay + slew rate) modes. Welch's t-test is preferred over Student's
t-test because it doesn't assume equal variances between groups.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/actions/test_ideal_vs_realistic.py -v
"""

import torch
import numpy as np

from test.actions.conftest import (
    reset_env_and_action_term,
    collect_step_response,
    N_RESPONSE_STEPS,
)
from test.common import (
    CONFIDENCE_LEVEL,
    welch_t_test,
)


def test_ideal_has_instant_response(action_env):
    """Verify ideal config produces instant velocity response."""
    action_term = reset_env_and_action_term(action_env, ideal_mode=True)

    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=action_env.device)
    step_action = step_action.repeat(action_env.num_envs, 1)

    action_term.process_actions(step_action)
    first_response = np.abs(action_term.processed_actions[0, 0].cpu().item())

    for _ in range(10):
        action_term.process_actions(step_action)
    final_response = np.abs(action_term.processed_actions[0, 0].cpu().item())

    print(f"\n  Ideal (instant) response:")
    print(f"    First step: {first_response:.4f}")
    print(f"    After 10 steps: {final_response:.4f}")
    print(f"    Ratio: {first_response / (final_response + 1e-9):.2f}")

    assert first_response > final_response * 0.9, \
        f"Ideal config should have instant response (first={first_response:.4f}, final={final_response:.4f})"


def test_realistic_has_gradual_response(action_env):
    """Verify realistic config produces gradual velocity response."""
    action_term = reset_env_and_action_term(action_env, ideal_mode=False)

    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=action_env.device)
    step_action = step_action.repeat(action_env.num_envs, 1)

    action_term.process_actions(step_action)
    first_response = np.abs(action_term.processed_actions[0, 0].cpu().item())

    for _ in range(50):
        action_term.process_actions(step_action)
    final_response = np.abs(action_term.processed_actions[0, 0].cpu().item())

    print(f"\n  Realistic (gradual) response:")
    print(f"    First step: {first_response:.4f}")
    print(f"    After 50 steps: {final_response:.4f}")
    print(f"    Ratio: {first_response / (final_response + 1e-9):.2f}")

    assert first_response < final_response * 0.5, \
        f"Realistic config should have gradual response (first={first_response:.4f}, final={final_response:.4f})"


def test_ideal_vs_realistic_rise_time_welch(action_env):
    """Compare rise times using Welch's t-test (unequal variance t-test).

    Rise time is defined as time to reach 63.2% of final value (1 time constant).
    For ideal mode, this should be ~0 (instant).
    For realistic mode, this should be approximately τ_motor.

    Welch's t-test is used because:
    1. Variances may differ between ideal (near-zero variance) and realistic
    2. It's more robust than Student's t-test for small/unequal samples
    3. It doesn't require homogeneity of variance assumption
    """
    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=action_env.device)
    step_action = step_action.repeat(action_env.num_envs, 1)
    physics_dt = action_env.physics_dt

    # Collect rise times for ideal mode (across all envs and wheels)
    reset_env_and_action_term(action_env, ideal_mode=True)
    ideal_responses = collect_step_response(action_env, step_action, N_RESPONSE_STEPS, ideal_mode=True)
    ideal_rise_times = []

    for env_idx in range(action_env.num_envs):
        for wheel_idx in range(4):
            response = np.abs(ideal_responses[:, env_idx, wheel_idx])
            v_final = response[-1]
            if v_final > 0.01:  # Only count active wheels
                threshold = 0.632 * v_final
                rise_idx = np.argmax(response >= threshold)
                rise_time = rise_idx * physics_dt
                ideal_rise_times.append(rise_time)

    # Collect rise times for realistic mode
    reset_env_and_action_term(action_env, ideal_mode=False)
    realistic_responses = collect_step_response(action_env, step_action, N_RESPONSE_STEPS, ideal_mode=False)
    realistic_rise_times = []

    for env_idx in range(action_env.num_envs):
        for wheel_idx in range(4):
            response = np.abs(realistic_responses[:, env_idx, wheel_idx])
            v_final = response[-1]
            if v_final > 0.01:
                threshold = 0.632 * v_final
                rise_idx = np.argmax(response >= threshold)
                rise_time = rise_idx * physics_dt
                realistic_rise_times.append(rise_time)

    ideal_rise_times = np.array(ideal_rise_times)
    realistic_rise_times = np.array(realistic_rise_times)

    # Welch's t-test: H1: realistic rise time > ideal rise time (one-sided)
    welch_result = welch_t_test(
        realistic_rise_times,
        ideal_rise_times,
        alternative="greater"
    )

    alpha = 1 - CONFIDENCE_LEVEL

    print(f"\n  Ideal vs Realistic rise time comparison (Welch's t-test):")
    print(f"    Ideal rise times: mean={welch_result['mean_b']*1000:.2f}ms, std={welch_result['std_b']*1000:.2f}ms, n={welch_result['n_b']}")
    print(f"    Realistic rise times: mean={welch_result['mean_a']*1000:.2f}ms, std={welch_result['std_a']*1000:.2f}ms, n={welch_result['n_a']}")
    print(f"    Welch's t-statistic: {welch_result['t_statistic']:.2f}")
    print(f"    One-sided p-value (realistic > ideal): {welch_result['p_value']:.4f}")
    print(f"    Cohen's d effect size: {welch_result['cohens_d']:.2f}")
    print(f"    Interpretation: {'Large' if abs(welch_result['cohens_d']) > 0.8 else 'Medium' if abs(welch_result['cohens_d']) > 0.5 else 'Small'} effect")

    # Test: Realistic rise time significantly greater than ideal
    assert welch_result['p_value'] < alpha, \
        f"Realistic rise time should be significantly greater than ideal (p={welch_result['p_value']:.4f} >= {alpha})"

    # Test: Effect size should be large (d > 0.8)
    assert welch_result['cohens_d'] > 0.8, \
        f"Effect size should be large (Cohen's d={welch_result['cohens_d']:.2f} <= 0.8)"
