# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures and utilities for action dynamics tests.

This conftest.py provides:
- Environment fixtures for action term testing
- Helper functions for configuring action dynamics components
- Response analysis utilities (exponential fit, linear fit)

NOTE: AppLauncher is initialized in test/conftest.py (root).
This file must NOT launch Isaac Sim - only provide fixtures and helpers.
"""

import torch
import numpy as np
from scipy import stats
import pytest

from isaaclab.envs import ManagerBasedRLEnv

from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg_NoCam,
    ActionsCfg_Realistic,
)

from test.common import DEVICE


# =============================================================================
# Test Configuration Constants
# =============================================================================

NUM_ENVS = 4                 # Minimal envs for unit tests
N_RESPONSE_STEPS = 200       # Steps to measure step response

# Thresholds for deterministic analytical tests (not statistical tolerances)
R_SQUARED_EXPONENTIAL_FIT = 0.99    # First-order systems fit exponential near-perfectly
R_SQUARED_LINEAR_FIT = 0.999        # Constant acceleration gives perfect linear ramp
CORRELATION_SIGNAL_PRESERVE = 0.999 # Delay buffer preserves signal exactly
FLOAT_PRECISION_FACTOR = 1.001      # Allow 0.1% for floating-point accumulation


# =============================================================================
# Module-scoped Environment
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
def action_env():
    """Provide environment for action term testing."""
    env = _get_or_create_env()
    yield env


def pytest_sessionfinish(session, exitstatus):
    """Clean up the environment after all tests complete."""
    global _module_env
    if _module_env is not None:
        _module_env.close()
        _module_env = None
    # Note: simulation_app.close() is handled by root test/conftest.py


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
