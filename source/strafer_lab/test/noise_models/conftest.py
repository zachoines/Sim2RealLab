# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest fixtures for noise model tests.

This module provides common environment setup and observation collection
for noise model validation tests.

Note: Isaac Sim is launched by test/conftest.py (root conftest).
"""

# --- Imports (Isaac Sim launched by root conftest.py) ---

import torch
import pytest

from isaaclab.envs import ManagerBasedRLEnv

# Import shared utilities from common module
from test.common import (
    NUM_ENVS,
    N_SETTLE_STEPS,
    N_SAMPLES_STEPS,
    DEVICE,
    IMU_ACCEL_MAX,
    IMU_GYRO_MAX,
    ENCODER_VEL_MAX,
)

# Import robot utilities from common module
from test.common.robot import (
    reset_robot_pose,
    freeze_robot_in_place,
    clear_frozen_state,
)

from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg_NoCam,
    ActionsCfg_Ideal,
    ObsCfg_NoCam_Ideal,
    ObsCfg_NoCam_Realistic,
)
from strafer_lab.tasks.navigation.sim_real_cfg import (
    REAL_ROBOT_CONTRACT,
    get_imu_accel_noise,
    get_imu_gyro_noise,
    get_encoder_noise,
)

# =============================================================================
# Observation Term Indices (for NoCam config: 15 total dims)
# =============================================================================
# imu_accel(3) + imu_gyro(3) + encoders(4) + goal(2) + action(3)
IMU_ACCEL_SLICE = slice(0, 3)
IMU_GYRO_SLICE = slice(3, 6)
ENCODER_SLICE = slice(6, 10)
GOAL_SLICE = slice(10, 12)
ACTION_SLICE = slice(12, 15)


# =============================================================================
# Module-scoped Environment Management
# =============================================================================

_module_env = None
_env_config_type = None  # Track which config is loaded


def _get_or_create_env(use_noise: bool = True):
    """Get or create the shared test environment.

    Args:
        use_noise: If True, use Realistic config with noise enabled.
                   If False, use Ideal config without noise.
    """
    global _module_env, _env_config_type

    config_type = "realistic" if use_noise else "ideal"

    # If env exists with different config, we can't change it (single context)
    if _module_env is not None:
        if _env_config_type != config_type:
            raise RuntimeError(
                f"Cannot switch env config from '{_env_config_type}' to '{config_type}'. "
                "Isaac Sim only allows one SimulationContext per process."
            )
        return _module_env

    cfg = StraferNavEnvCfg_NoCam()
    cfg.scene.num_envs = NUM_ENVS
    cfg.actions = ActionsCfg_Ideal()  # Always ideal actions for predictability

    if use_noise:
        cfg.observations = ObsCfg_NoCam_Realistic()
    else:
        cfg.observations = ObsCfg_NoCam_Ideal()

    _module_env = ManagerBasedRLEnv(cfg)
    _env_config_type = config_type
    _module_env.reset()

    return _module_env


@pytest.fixture(scope="module")
def noisy_env():
    """Provide Strafer environment with realistic noise enabled (no camera)."""
    env = _get_or_create_env(use_noise=True)
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


def reset_observation_noise_models(env, debug: bool = False) -> int:
    """Reset all noise models in the observation manager.

    This clears accumulated bias drift state in IMU noise models,
    ensuring tests start from a clean noise state.

    Args:
        env: The Isaac Lab environment
        debug: If True, print debug information about found noise models

    Returns:
        Number of noise models reset
    """
    reset_count = 0
    try:
        obs_manager = env.observation_manager
        # Create tensor of all env_ids for reset
        all_env_ids = torch.arange(env.num_envs, device=env.device)

        # Iterate over all observation term groups
        for group_name, terms in obs_manager._group_obs_term_names.items():
            for term_name in terms:
                term = obs_manager._terms[group_name][term_name]
                # Check if the term has a noise model with a reset method
                if hasattr(term, '_noise_model') and term._noise_model is not None:
                    noise_model = term._noise_model
                    if hasattr(noise_model, 'reset'):
                        # IMU/Encoder noise models require env_ids argument
                        try:
                            noise_model.reset(all_env_ids)
                            reset_count += 1
                            if debug:
                                print(f"    Reset noise model for {group_name}/{term_name}: {type(noise_model).__name__}")
                        except TypeError:
                            # Some noise models may not require env_ids
                            noise_model.reset()
                            reset_count += 1
                            if debug:
                                print(f"    Reset noise model (no args) for {group_name}/{term_name}: {type(noise_model).__name__}")
                elif debug:
                    has_noise = hasattr(term, '_noise_model')
                    print(f"    {group_name}/{term_name}: has _noise_model={has_noise}, value={getattr(term, '_noise_model', 'N/A')}")

        if debug:
            print(f"  Total noise models reset: {reset_count}")

    except Exception as e:
        if debug:
            print(f"  Error resetting noise models: {e}")
    
    return reset_count


def collect_stationary_observations(env, n_steps: int, freeze_robot: bool = True) -> torch.Tensor:
    """Collect observations from a stationary robot over multiple steps.

    With zero actions and settled physics, any observation variance
    comes from sensor noise.

    Args:
        env: The Isaac Lab environment
        n_steps: Number of observation steps to collect
        freeze_robot: If True (default), continuously zero out robot velocities
                      to eliminate physics settling noise. This ensures measured
                      variance comes purely from sensor noise.

    Returns:
        Tensor of shape (n_steps, num_envs, obs_dim)
    """
    # Full environment reset to ensure clean state
    env.reset()

    # Reset noise model internal state (bias drift, delay buffers)
    # This prevents state leakage between tests sharing the same environment
    reset_observation_noise_models(env)

    # Reset robots to fixed positions at grid origins (avoids randomization overlap)
    reset_robot_pose(env)

    # Clear frozen state so it captures fresh pose on next freeze call
    clear_frozen_state()

    # Zero action (stationary)
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)

    # Let physics settle
    for _ in range(N_SETTLE_STEPS):
        env.step(zero_action)
        if freeze_robot:
            freeze_robot_in_place(env)

    # Collect observations
    observations = []
    for _ in range(n_steps):
        obs_dict, _, _, _, _ = env.step(zero_action)
        observations.append(obs_dict["policy"].clone())
        if freeze_robot:
            freeze_robot_in_place(env)

    return torch.stack(observations, dim=0)


def extract_noise_samples(observations: torch.Tensor, term_slice: slice):
    """Extract samples for a specific observation term and flatten for analysis.

    For a stationary robot, the "signal" should be constant and any
    deviation is noise. We compute deviation from per-environment mean.

    Args:
        observations: (n_steps, num_envs, obs_dim)
        term_slice: Slice for the observation term

    Returns:
        Flattened noise samples as numpy array
    """
    import numpy as np

    term_obs = observations[:, :, term_slice]  # (n_steps, num_envs, term_dim)

    # Compute mean per env (the "true" signal)
    mean_per_env = term_obs.mean(dim=0, keepdim=True)  # (1, num_envs, term_dim)

    # Deviation from mean is the noise
    noise = term_obs - mean_per_env

    return noise.cpu().numpy().flatten()


def extract_first_differences(observations: torch.Tensor, term_slice: slice):
    """Extract first differences for a specific observation term.

    First differences eliminate the need to estimate the "true" signal
    and give us a clean theoretical prediction for variance:

        Δy_t = y_t - y_{t-1}
        Var(Δy) = drift_rate² * dt + 2 * white_noise_std²

    Args:
        observations: (n_steps, num_envs, obs_dim)
        term_slice: Slice for the observation term

    Returns:
        Flattened first-difference samples as numpy array
        Shape: ((n_steps-1) * num_envs * term_dim,)
    """
    term_obs = observations[:, :, term_slice]  # (n_steps, num_envs, term_dim)

    # Compute first differences: y[t] - y[t-1]
    first_diffs = term_obs[1:] - term_obs[:-1]  # (n_steps-1, num_envs, term_dim)

    return first_diffs.cpu().numpy().flatten()


# =============================================================================
# Noise Configuration Constants
# =============================================================================
# Get expected noise parameters from contract configuration

_ACCEL_NOISE_CFG = get_imu_accel_noise(REAL_ROBOT_CONTRACT)
_GYRO_NOISE_CFG = get_imu_gyro_noise(REAL_ROBOT_CONTRACT)
_ENCODER_NOISE_CFG = get_encoder_noise(REAL_ROBOT_CONTRACT)

# RAW noise std (in physical units)
RAW_ACCEL_STD = _ACCEL_NOISE_CFG.accel_noise_std if _ACCEL_NOISE_CFG else 0.0  # m/s²
RAW_GYRO_STD = _GYRO_NOISE_CFG.gyro_noise_std if _GYRO_NOISE_CFG else 0.0  # rad/s
RAW_ENCODER_STD = (
    _ENCODER_NOISE_CFG.velocity_noise_std * _ENCODER_NOISE_CFG.max_velocity
    if _ENCODER_NOISE_CFG
    else 0.0
)  # ticks/s

# NORMALIZED noise std (what we measure in observation output)
# normalized_std = raw_std / max_value = raw_std * scale
EXPECTED_ACCEL_STD = RAW_ACCEL_STD / IMU_ACCEL_MAX if RAW_ACCEL_STD else 0.0
EXPECTED_GYRO_STD = RAW_GYRO_STD / IMU_GYRO_MAX if RAW_GYRO_STD else 0.0
EXPECTED_ENCODER_STD = RAW_ENCODER_STD / ENCODER_VEL_MAX if RAW_ENCODER_STD else 0.0
