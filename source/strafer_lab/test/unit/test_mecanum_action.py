# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for MecanumWheelAction dynamics components.

These tests verify the individual dynamics components (motor filter, command delay,
slew rate limiter) in ISOLATION by calling process_actions() directly without
stepping the physics simulation.

Unit test characteristics:
- Test single component behavior with other components disabled
- Call action_term.process_actions() directly (not env.step())
- Verify exact analytical properties without tolerance factors
- No physics simulation involved

For integration tests that verify the full pipeline (action → physics → robot motion),
see test/integration/test_motor_dynamics.py.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/unit/test_mecanum_action.py -v
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import numpy as np
import pytest
from scipy import stats

from isaaclab.envs import ManagerBasedRLEnv

from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg_NoCam,
    ActionsCfg_Realistic,
)

# Import shared constants from common module
from test.common import DEVICE


# =============================================================================
# Test Configuration
# =============================================================================

NUM_ENVS = 4                 # Minimal envs for unit tests
N_RESPONSE_STEPS = 200       # Steps to measure step response


# =============================================================================
# Module-scoped Fixtures
# =============================================================================

_module_env = None


def _get_or_create_env():
    """Get or create environment for testing action term."""
    global _module_env

    if _module_env is not None:
        return _module_env

    cfg = StraferNavEnvCfg_NoCam()
    cfg.scene.num_envs = NUM_ENVS
    cfg.actions = ActionsCfg_Realistic()

    _module_env = ManagerBasedRLEnv(cfg)
    _module_env.reset()

    return _module_env


@pytest.fixture(scope="module")
def env():
    """Provide environment for action term testing."""
    env = _get_or_create_env()
    yield env


def pytest_sessionfinish(session, exitstatus):
    """Clean up environments after all tests complete."""
    global _module_env

    if _module_env is not None:
        _module_env.close()
        _module_env = None

    simulation_app.close()


# =============================================================================
# Helper Functions
# =============================================================================

def configure_action_term_dynamics(
    env,
    enable_motor: bool = False,
    enable_delay: bool = False,
    enable_slew: bool = False
):
    """Configure action term with specific dynamics enabled/disabled.

    This allows testing each dynamic component in isolation.

    Args:
        env: The environment
        enable_motor: Enable first-order motor dynamics filter
        enable_delay: Enable command delay buffer
        enable_slew: Enable slew rate (acceleration) limiting

    Returns:
        The configured action term
    """
    env.reset()
    action_term = env.action_manager._terms["wheel_velocities"]

    # Configure specific dynamics
    action_term._enable_motor_dynamics = enable_motor
    action_term._enable_command_delay = enable_delay
    action_term._enable_slew_rate = enable_slew

    # Reset and clear all state buffers
    action_term.reset(env_ids=torch.arange(env.num_envs, device=env.device))

    if hasattr(action_term, '_smoothed_wheel_vels'):
        action_term._smoothed_wheel_vels.zero_()
    if hasattr(action_term, '_prev_wheel_vels'):
        action_term._prev_wheel_vels.zero_()
    if hasattr(action_term, '_action_delay_buffer'):
        action_term._action_delay_buffer.reset(None)

    return action_term


def fit_first_order_response(times: np.ndarray, response: np.ndarray, v_final: float) -> dict:
    """Fit first-order exponential response and return parameters.

    Model: v(t) = v_final * (1 - exp(-t/tau))

    Args:
        times: Time array
        response: Response array (absolute values)
        v_final: Final steady-state value

    Returns:
        Dict with 'tau', 'tau_ci_lower', 'tau_ci_upper', 'r_squared'
    """
    # Normalize response
    v_norm = response / (v_final + 1e-9)

    # Transform to linearize: log(1 - v_norm) = -t/tau
    # Only use points where 1 - v_norm > 0
    valid_mask = v_norm < 0.99
    if not np.any(valid_mask):
        return {'tau': np.nan, 'tau_ci_lower': np.nan, 'tau_ci_upper': np.nan, 'r_squared': 0}

    t_valid = times[valid_mask]
    y_valid = np.log(1 - v_norm[valid_mask] + 1e-9)

    # Linear regression: y = -t/tau  →  slope = -1/tau
    slope, intercept, r_value, p_value, std_err = stats.linregress(t_valid, y_valid)

    if slope >= 0:
        return {'tau': np.nan, 'tau_ci_lower': np.nan, 'tau_ci_upper': np.nan, 'r_squared': 0}

    tau = -1.0 / slope
    r_squared = r_value ** 2

    # Confidence interval for tau using delta method
    tau_std = std_err / (slope ** 2)
    t_crit = stats.t.ppf(0.975, len(t_valid) - 2)
    tau_ci_lower = tau - t_crit * tau_std
    tau_ci_upper = tau + t_crit * tau_std

    return {
        'tau': tau,
        'tau_ci_lower': tau_ci_lower,
        'tau_ci_upper': tau_ci_upper,
        'r_squared': r_squared,
    }


# =============================================================================
# Tests: Isolated Motor Dynamics (First-Order Response Only)
# =============================================================================

class TestMotorDynamicsIsolated:
    """Test first-order motor dynamics in ISOLATION (no delay, no slew rate).

    By testing with only motor dynamics enabled, we can verify exact analytical
    properties of a first-order system without tolerance factors.

    For a first-order system with time constant tau:
        v(t) = v_final * (1 - e^(-t/tau))

    Exact analytical values:
        - At t = tau:        v = 63.21% of v_final  (1 - e^-1)
        - At t = 2.197tau:   v = 90% of v_final    (1 - e^-ln(10))
        - At t = 0.105tau:   v = 10% of v_final    (1 - e^-ln(10/9))
        - 10-90% rise time = tau * ln(9) = 2.197tau
    """

    def test_time_to_63_percent(self, env):
        """Verify time to reach 63.21% matches configured tau exactly.

        For first-order system: v(tau) = v_final * (1 - e^-1) = 0.6321 * v_final

        This is the DEFINITION of time constant, so it should match exactly
        (within discretization error of +/-1 timestep).
        """
        action_term = configure_action_term_dynamics(
            env, enable_motor=True, enable_delay=False, enable_slew=False
        )
        cfg = env.cfg.actions.wheel_velocities
        configured_tau = cfg.motor_time_constant
        physics_dt = env.physics_dt

        # Apply step input
        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
        step_action = step_action.repeat(env.num_envs, 1)

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

        print(f"\n  Motor dynamics isolated - time to 63.21%:")
        print(f"    Configured tau: {configured_tau*1000:.2f}ms")
        print(f"    Measured t_63: {measured_t63*1000:.2f}ms")
        print(f"    Difference: {(measured_t63 - configured_tau)*1000:.2f}ms")
        print(f"    Max allowed error (+/-1 dt): +/-{max_error*1000:.2f}ms")

        assert abs(measured_t63 - configured_tau) <= max_error, \
            f"t_63 ({measured_t63*1000:.2f}ms) should equal tau ({configured_tau*1000:.2f}ms) +/-{max_error*1000:.2f}ms"

    def test_10_90_rise_time(self, env):
        """Verify 10-90% rise time equals tau * ln(9) = 2.197tau exactly.

        For first-order system:
            t_10 = tau * ln(10/9) = 0.1054tau
            t_90 = tau * ln(10)   = 2.3026tau
            T_rise = t_90 - t_10 = tau * ln(9) = 2.1972tau
        """
        action_term = configure_action_term_dynamics(
            env, enable_motor=True, enable_delay=False, enable_slew=False
        )
        cfg = env.cfg.actions.wheel_velocities
        configured_tau = cfg.motor_time_constant
        physics_dt = env.physics_dt

        # Exact analytical values
        LN_9 = np.log(9)  # ≈ 2.197
        expected_rise_time = configured_tau * LN_9

        # Apply step input
        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
        step_action = step_action.repeat(env.num_envs, 1)

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

        print(f"\n  Motor dynamics isolated - 10-90% rise time:")
        print(f"    Configured tau: {configured_tau*1000:.2f}ms")
        print(f"    Expected rise time (tau*ln(9)): {expected_rise_time*1000:.2f}ms")
        print(f"    Measured t_10: {measured_t10*1000:.2f}ms")
        print(f"    Measured t_90: {measured_t90*1000:.2f}ms")
        print(f"    Measured rise time: {measured_rise_time*1000:.2f}ms")
        print(f"    Difference: {(measured_rise_time - expected_rise_time)*1000:.2f}ms")
        print(f"    Max allowed error (+/-2 dt): +/-{max_error*1000:.2f}ms")

        assert abs(measured_rise_time - expected_rise_time) <= max_error, \
            f"Rise time ({measured_rise_time*1000:.2f}ms) should equal 2.197*tau ({expected_rise_time*1000:.2f}ms) +/-{max_error*1000:.2f}ms"

    def test_exponential_fit_r_squared(self, env):
        """Verify response fits first-order exponential with R² > 0.99.

        With only motor dynamics (no slew/delay), the response should be
        a near-perfect exponential, giving R² very close to 1.0.
        """
        action_term = configure_action_term_dynamics(
            env, enable_motor=True, enable_delay=False, enable_slew=False
        )
        physics_dt = env.physics_dt

        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
        step_action = step_action.repeat(env.num_envs, 1)

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

        print(f"\n  Motor dynamics isolated - exponential fit quality:")
        print(f"    R²: {r_squared:.6f}")
        print(f"    Expected: > 0.99 (pure first-order response)")

        # With isolated motor dynamics, should get near-perfect fit
        assert r_squared > 0.99, \
            f"Isolated motor dynamics should give R² > 0.99, got {r_squared:.4f}"

    def test_no_overshoot(self, env):
        """Verify first-order response has no overshoot (strictly monotonic).

        A pure first-order system NEVER overshoots. With motor dynamics
        isolated, we should see strictly monotonic increase.
        """
        action_term = configure_action_term_dynamics(
            env, enable_motor=True, enable_delay=False, enable_slew=False
        )

        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
        step_action = step_action.repeat(env.num_envs, 1)

        responses = []
        for _ in range(N_RESPONSE_STEPS):
            action_term.process_actions(step_action)
            responses.append(action_term.processed_actions.cpu().numpy().copy())
        responses = np.abs(np.array(responses))

        wheel_response = responses[:, 0, 0]
        v_final = wheel_response[-1]

        # Noise threshold based on numerical precision
        noise_floor = max(1e-7 * v_final, 1e-6)
        diffs = np.diff(wheel_response)
        significant_decreases = np.sum(diffs < -3 * noise_floor)

        print(f"\n  Motor dynamics isolated - monotonicity check:")
        print(f"    Significant decreases: {significant_decreases}")
        print(f"    Min diff: {np.min(diffs):.2e}")
        print(f"    Noise threshold: {-3*noise_floor:.2e}")

        # Should be strictly monotonic (0 decreases, allowing for float noise)
        assert significant_decreases == 0, \
            f"Pure first-order system should never decrease, found {significant_decreases} decreases"


# =============================================================================
# Tests: Isolated Command Delay
# =============================================================================

class TestCommandDelayIsolated:
    """Test command delay buffer in ISOLATION (no motor dynamics, no slew rate).

    With only delay enabled, the output should be an exact copy of the input
    shifted by exactly D timesteps, where D is the configured delay.
    """

    def test_delay_exact_step_shift(self, env):
        """Verify output is input shifted by exactly delay_steps.

        With delay only (no smoothing), a step input at t=0 should appear
        at output at t = delay_steps * dt.
        """
        action_term = configure_action_term_dynamics(
            env, enable_motor=False, enable_delay=True, enable_slew=False
        )
        cfg = env.cfg.actions.wheel_velocities

        # Get the actual delay for env 0
        if hasattr(action_term, '_action_delay_buffer'):
            actual_delay = action_term._action_delay_buffer._time_lags[0].item()
        else:
            actual_delay = 0

        # Apply zero, then step to 1.0
        zero_action = torch.zeros(env.num_envs, 3, device=env.device)
        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)

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

        print(f"\n  Command delay isolated - exact step shift:")
        print(f"    Configured max delay: {cfg.max_delay_steps} steps")
        print(f"    Actual delay (env 0): {actual_delay} steps")
        print(f"    First output change at step: {first_change_idx}")
        print(f"    Expected: {actual_delay} steps")
        print(f"    Responses[0:10]: {responses[:10]}")

        # Output should change exactly at step = actual_delay
        assert first_change_idx == actual_delay, \
            f"Delay should be exactly {actual_delay} steps, but output changed at step {first_change_idx}"

    def test_delay_preserves_signal_shape(self, env):
        """Verify delay buffer preserves signal shape (no distortion).

        A ramp input should produce a ramp output, just shifted in time.
        """
        action_term = configure_action_term_dynamics(
            env, enable_motor=False, enable_delay=True, enable_slew=False
        )
        cfg = env.cfg.actions.wheel_velocities

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
            action = torch.tensor([[float(ramp_values[i]), 0.0, 0.0]], device=env.device)
            action = action.repeat(env.num_envs, 1)
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

            # Correlation should be very high (>0.999) for identical signals
            correlation = np.corrcoef(input_segment, output_segment)[0, 1]

            print(f"\n  Command delay isolated - signal preservation:")
            print(f"    Delay: {actual_delay} steps")
            print(f"    Correlation(input, shifted output): {correlation:.6f}")

            assert correlation > 0.999, \
                f"Delay buffer should preserve signal shape (correlation={correlation:.4f})"


# =============================================================================
# Tests: Isolated Slew Rate Limiting
# =============================================================================

class TestSlewRateIsolated:
    """Test slew rate limiting in ISOLATION (no motor dynamics, no delay).

    With only slew rate enabled, the output should be a rate-limited version
    of the input: dv/dt ≤ max_acceleration at all times.
    """

    def test_acceleration_exactly_bounded(self, env):
        """Verify acceleration never exceeds max_acceleration.

        For a large step input, the output should ramp linearly at exactly
        max_acceleration until reaching the target.
        """
        action_term = configure_action_term_dynamics(
            env, enable_motor=False, enable_delay=False, enable_slew=True
        )
        cfg = env.cfg.actions.wheel_velocities
        max_accel = cfg.max_acceleration_rad_s2
        physics_dt = env.physics_dt

        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
        step_action = step_action.repeat(env.num_envs, 1)

        responses = []
        for _ in range(100):
            action_term.process_actions(step_action)
            responses.append(action_term.processed_actions[0, 0].cpu().item())

        responses = np.abs(np.array(responses))
        diffs = np.diff(responses)
        accelerations = diffs / physics_dt
        max_measured_accel = np.max(np.abs(accelerations))

        print(f"\n  Slew rate isolated - acceleration bound:")
        print(f"    Configured max accel: {max_accel:.1f} rad/s^2")
        print(f"    Max measured accel: {max_measured_accel:.1f} rad/s^2")
        print(f"    Difference: {max_measured_accel - max_accel:.4f} rad/s^2")

        # Should be bounded exactly (within float precision)
        assert max_measured_accel <= max_accel * 1.001, \
            f"Acceleration ({max_measured_accel:.1f}) should not exceed {max_accel:.1f} rad/s^2"

    def test_ramp_time_matches_physics(self, env):
        """Verify ramp time equals v_final / max_acceleration exactly.

        For a step from 0 to v_final with slew rate limiting:
            T_ramp = v_final / max_accel
        """
        action_term = configure_action_term_dynamics(
            env, enable_motor=False, enable_delay=False, enable_slew=True
        )
        cfg = env.cfg.actions.wheel_velocities
        max_accel = cfg.max_acceleration_rad_s2
        physics_dt = env.physics_dt

        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
        step_action = step_action.repeat(env.num_envs, 1)

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
        # - 1 timestep for discretization at ramp end
        # - 1 timestep for 99% threshold approximation (vs true 100%)
        max_error = 2 * physics_dt

        print(f"\n  Slew rate isolated - ramp time:")
        print(f"    v_final: {v_final:.2f} rad/s")
        print(f"    max_accel: {max_accel:.1f} rad/s^2")
        print(f"    Expected ramp time: {expected_ramp_time*1000:.2f}ms")
        print(f"    Measured ramp time (to 99%): {measured_ramp_time*1000:.2f}ms")
        print(f"    Difference: {(measured_ramp_time - expected_ramp_time)*1000:.2f}ms")

        assert abs(measured_ramp_time - expected_ramp_time) <= max_error, \
            f"Ramp time ({measured_ramp_time*1000:.2f}ms) should equal v/a ({expected_ramp_time*1000:.2f}ms) +/-{max_error*1000:.2f}ms"

    def test_linear_ramp_shape(self, env):
        """Verify ramp is linear (constant acceleration) during slew.

        During the ramp phase, velocity should increase linearly with time.
        """
        action_term = configure_action_term_dynamics(
            env, enable_motor=False, enable_delay=False, enable_slew=True
        )
        physics_dt = env.physics_dt

        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
        step_action = step_action.repeat(env.num_envs, 1)

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

            print(f"\n  Slew rate isolated - linear ramp shape:")
            print(f"    Ramp region: {len(ramp_indices)} samples")
            print(f"    Linear fit R²: {r_squared:.6f}")
            print(f"    Slope (acceleration): {slope:.1f} rad/s^2")

            # Should be very linear (R² > 0.999)
            assert r_squared > 0.999, \
                f"Ramp should be linear (R²={r_squared:.4f}), indicates non-constant acceleration"


# =============================================================================
# Run pytest
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
