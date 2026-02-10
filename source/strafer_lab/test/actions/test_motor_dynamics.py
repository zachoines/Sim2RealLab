# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for first-order motor dynamics in action processing.

These tests verify motor dynamics (first-order low-pass filter) in ISOLATION
by disabling delay and slew rate limiting. This allows exact analytical
verification of first-order system properties.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/actions/test_motor_dynamics.py -v
"""

import torch
import numpy as np

from test.actions.conftest import (
    configure_action_term_dynamics,
    fit_first_order_response,
    N_RESPONSE_STEPS,
    R_SQUARED_EXPONENTIAL_FIT,
)


def test_time_to_63_percent(action_env):
    """Verify time to reach 63.21% matches configured tau exactly.

    For first-order system: v(tau) = v_final * (1 - e^-1) = 0.6321 * v_final

    This is the DEFINITION of time constant, so it should match exactly
    (within discretization error of +/-1 timestep).
    """
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=True, enable_delay=False, enable_slew=False
    )
    cfg = action_env.cfg.actions.wheel_velocities
    configured_tau = cfg.motor_time_constant
    physics_dt = action_env.physics_dt

    # Apply step input
    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=action_env.device)
    step_action = step_action.repeat(action_env.num_envs, 1)

    responses = []
    for _ in range(N_RESPONSE_STEPS):
        action_term.process_actions(step_action)
        responses.append(action_term.processed_actions.cpu().numpy().copy())
    responses = np.abs(np.array(responses))

    # Analyze first wheel of first env
    wheel_response = responses[:, 0, 0]
    v_final = wheel_response[-1]
    times = np.arange(N_RESPONSE_STEPS) * physics_dt

    # Find time to reach 63.21% of final value
    target_63 = 0.6321 * v_final
    idx_63 = np.argmax(wheel_response >= target_63)
    measured_t63 = times[idx_63]

    # Discretization error: +/-1 timestep
    max_error = physics_dt

    print(f"\n  Motor dynamics - time to 63.21%:")
    print(f"    Configured tau: {configured_tau*1000:.2f}ms")
    print(f"    Measured t_63: {measured_t63*1000:.2f}ms")
    print(f"    Difference: {(measured_t63 - configured_tau)*1000:.2f}ms")
    print(f"    Max allowed error (+/-1 dt): +/-{max_error*1000:.2f}ms")

    assert abs(measured_t63 - configured_tau) <= max_error, \
        f"t_63 ({measured_t63*1000:.2f}ms) should equal tau ({configured_tau*1000:.2f}ms) +/-{max_error*1000:.2f}ms"


def test_10_90_rise_time(action_env):
    """Verify 10-90% rise time equals tau * ln(9) = 2.197tau exactly.

    For first-order system:
        t_10 = tau * ln(10/9) = 0.1054tau
        t_90 = tau * ln(10)   = 2.3026tau
        T_rise = t_90 - t_10 = tau * ln(9) = 2.1972tau
    """
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=True, enable_delay=False, enable_slew=False
    )
    cfg = action_env.cfg.actions.wheel_velocities
    configured_tau = cfg.motor_time_constant
    physics_dt = action_env.physics_dt

    # Exact analytical values
    LN_9 = np.log(9)  # ≈ 2.197
    expected_rise_time = configured_tau * LN_9

    # Apply step input
    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=action_env.device)
    step_action = step_action.repeat(action_env.num_envs, 1)

    responses = []
    for _ in range(N_RESPONSE_STEPS):
        action_term.process_actions(step_action)
        responses.append(action_term.processed_actions.cpu().numpy().copy())
    responses = np.abs(np.array(responses))

    wheel_response = responses[:, 0, 0]
    v_final = wheel_response[-1]
    times = np.arange(N_RESPONSE_STEPS) * physics_dt

    # Find times to reach 10% and 90%
    target_10 = 0.10 * v_final
    target_90 = 0.90 * v_final

    idx_10 = np.argmax(wheel_response >= target_10)
    idx_90 = np.argmax(wheel_response >= target_90)

    measured_t10 = times[idx_10]
    measured_t90 = times[idx_90]
    measured_rise_time = measured_t90 - measured_t10

    # Discretization error: +/-2 timesteps (one for each threshold crossing)
    max_error = 2 * physics_dt

    print(f"\n  Motor dynamics - 10-90% rise time:")
    print(f"    Configured tau: {configured_tau*1000:.2f}ms")
    print(f"    Expected rise time (tau*ln(9)): {expected_rise_time*1000:.2f}ms")
    print(f"    Measured t_10: {measured_t10*1000:.2f}ms")
    print(f"    Measured t_90: {measured_t90*1000:.2f}ms")
    print(f"    Measured rise time: {measured_rise_time*1000:.2f}ms")
    print(f"    Difference: {(measured_rise_time - expected_rise_time)*1000:.2f}ms")
    print(f"    Max allowed error (+/-2 dt): +/-{max_error*1000:.2f}ms")

    assert abs(measured_rise_time - expected_rise_time) <= max_error, \
        f"Rise time ({measured_rise_time*1000:.2f}ms) should equal 2.197*tau ({expected_rise_time*1000:.2f}ms) +/-{max_error*1000:.2f}ms"


def test_exponential_fit_quality(action_env):
    """Verify response fits first-order exponential with R² > 0.99.

    With only motor dynamics (no slew/delay), the response should be
    a near-perfect exponential, giving R² very close to 1.0.
    """
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=True, enable_delay=False, enable_slew=False
    )
    physics_dt = action_env.physics_dt

    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=action_env.device)
    step_action = step_action.repeat(action_env.num_envs, 1)

    responses = []
    for _ in range(N_RESPONSE_STEPS):
        action_term.process_actions(step_action)
        responses.append(action_term.processed_actions.cpu().numpy().copy())
    responses = np.abs(np.array(responses))

    wheel_response = responses[:, 0, 0]
    v_final = wheel_response[-1]
    times = np.arange(N_RESPONSE_STEPS) * physics_dt

    fit_result = fit_first_order_response(times, wheel_response, v_final)
    r_squared = fit_result['r_squared']

    print(f"\n  Motor dynamics - exponential fit quality:")
    print(f"    R²: {r_squared:.6f}")
    print(f"    Expected: > {R_SQUARED_EXPONENTIAL_FIT} (pure first-order response)")

    assert r_squared > R_SQUARED_EXPONENTIAL_FIT, \
        f"Isolated motor dynamics should give R² > {R_SQUARED_EXPONENTIAL_FIT}, got {r_squared:.4f}"


def test_no_overshoot(action_env):
    """Verify first-order response has no overshoot (strictly monotonic).

    A pure first-order system NEVER overshoots. With motor dynamics
    isolated, we should see strictly monotonic increase.
    """
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=True, enable_delay=False, enable_slew=False
    )

    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=action_env.device)
    step_action = step_action.repeat(action_env.num_envs, 1)

    responses = []
    for _ in range(N_RESPONSE_STEPS):
        action_term.process_actions(step_action)
        responses.append(action_term.processed_actions.cpu().numpy().copy())
    responses = np.abs(np.array(responses))

    wheel_response = responses[:, 0, 0]
    v_final = wheel_response[-1]

    # Noise threshold based on numerical precision
    relative_precision = 1e-7
    absolute_minimum = 1e-9
    noise_floor = max(relative_precision * v_final, absolute_minimum)
    diffs = np.diff(wheel_response)
    significant_decreases = np.sum(diffs < -3 * noise_floor)

    print(f"\n  Motor dynamics - monotonicity check:")
    print(f"    Significant decreases: {significant_decreases}")
    print(f"    Min diff: {np.min(diffs):.2e}")
    print(f"    Noise threshold (3σ): {-3*noise_floor:.2e}")

    assert significant_decreases == 0, \
        f"Pure first-order system should never decrease, found {significant_decreases} decreases"
