# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for slew rate limiting in action processing.

These tests verify slew rate limiting (acceleration limiting) in ISOLATION
by disabling motor dynamics and command delay. This allows exact analytical
verification of acceleration bounds and ramp characteristics.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/actions/test_slew_rate.py -v
"""

import torch
import numpy as np
from scipy import stats

from test.actions.conftest import (
    configure_action_term_dynamics,
    FLOAT_PRECISION_FACTOR,
    R_SQUARED_LINEAR_FIT,
)


def test_acceleration_bounded(action_env):
    """Verify acceleration never exceeds max_acceleration.

    For a large step input, the output should ramp linearly at exactly
    max_acceleration until reaching the target.
    """
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=False, enable_delay=False, enable_slew=True
    )
    cfg = action_env.cfg.actions.wheel_velocities
    max_accel = cfg.max_acceleration_rad_s2
    physics_dt = action_env.physics_dt

    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=action_env.device)
    step_action = step_action.repeat(action_env.num_envs, 1)

    responses = []
    for _ in range(100):
        action_term.process_actions(step_action)
        responses.append(action_term.processed_actions[0, 0].cpu().item())

    responses = np.abs(np.array(responses))
    diffs = np.diff(responses)
    accelerations = diffs / physics_dt
    max_measured_accel = np.max(np.abs(accelerations))

    print(f"\n  Slew rate - acceleration bound:")
    print(f"    Configured max accel: {max_accel:.1f} rad/s^2")
    print(f"    Max measured accel: {max_measured_accel:.1f} rad/s^2")
    print(f"    Difference: {max_measured_accel - max_accel:.4f} rad/s^2")

    # Should be bounded exactly (allow small float accumulation error)
    assert max_measured_accel <= max_accel * FLOAT_PRECISION_FACTOR, \
        f"Acceleration ({max_measured_accel:.1f}) should not exceed {max_accel:.1f} rad/s^2"


def test_ramp_time_matches_physics(action_env):
    """Verify ramp time equals v_final / max_acceleration exactly.

    For a step from 0 to v_final with slew rate limiting:
        T_ramp = v_final / max_accel
    """
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=False, enable_delay=False, enable_slew=True
    )
    cfg = action_env.cfg.actions.wheel_velocities
    max_accel = cfg.max_acceleration_rad_s2
    physics_dt = action_env.physics_dt

    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=action_env.device)
    step_action = step_action.repeat(action_env.num_envs, 1)

    responses = []
    for _ in range(200):
        action_term.process_actions(step_action)
        responses.append(action_term.processed_actions[0, 0].cpu().item())

    responses = np.abs(np.array(responses))
    v_final = responses[-1]
    times = np.arange(len(responses)) * physics_dt

    # Expected ramp time
    expected_ramp_time = v_final / max_accel

    # Find when we reach 99% of final (end of ramp)
    idx_99 = np.argmax(responses >= 0.99 * v_final)
    measured_ramp_time = times[idx_99]

    # Error allowance: +/-2 timesteps
    max_error = 2 * physics_dt

    print(f"\n  Slew rate - ramp time:")
    print(f"    v_final: {v_final:.2f} rad/s")
    print(f"    max_accel: {max_accel:.1f} rad/s^2")
    print(f"    Expected ramp time: {expected_ramp_time*1000:.2f}ms")
    print(f"    Measured ramp time (to 99%): {measured_ramp_time*1000:.2f}ms")
    print(f"    Difference: {(measured_ramp_time - expected_ramp_time)*1000:.2f}ms")

    assert abs(measured_ramp_time - expected_ramp_time) <= max_error, \
        f"Ramp time ({measured_ramp_time*1000:.2f}ms) should equal v/a ({expected_ramp_time*1000:.2f}ms) +/-{max_error*1000:.2f}ms"


def test_linear_ramp_shape(action_env):
    """Verify ramp is linear (constant acceleration) during slew.

    During the ramp phase, velocity should increase linearly with time.
    """
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=False, enable_delay=False, enable_slew=True
    )
    physics_dt = action_env.physics_dt

    step_action = torch.tensor([[1.0, 0.0, 0.0]], device=action_env.device)
    step_action = step_action.repeat(action_env.num_envs, 1)

    responses = []
    for _ in range(100):
        action_term.process_actions(step_action)
        responses.append(action_term.processed_actions[0, 0].cpu().item())

    responses = np.abs(np.array(responses))
    v_final = responses[-1]

    # Find ramp region (10% to 90% of final)
    ramp_mask = (responses > 0.1 * v_final) & (responses < 0.9 * v_final)
    ramp_indices = np.where(ramp_mask)[0]

    if len(ramp_indices) > 5:
        ramp_times = ramp_indices * physics_dt
        ramp_values = responses[ramp_indices]

        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(ramp_times, ramp_values)
        r_squared = r_value ** 2

        print(f"\n  Slew rate - linear ramp shape:")
        print(f"    Ramp region: {len(ramp_indices)} samples")
        print(f"    Linear fit R²: {r_squared:.6f}")
        print(f"    Slope (acceleration): {slope:.1f} rad/s^2")

        # Should be very linear - constant acceleration produces v(t) = a*t
        assert r_squared > R_SQUARED_LINEAR_FIT, \
            f"Ramp should be linear (R²={r_squared:.4f} < {R_SQUARED_LINEAR_FIT}), indicates non-constant acceleration"
