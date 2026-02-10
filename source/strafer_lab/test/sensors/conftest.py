# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest fixtures for sensor and observation tests.

This module provides environment setup fixtures for observation validation tests.
It reuses core fixtures from the noise_models conftest for consistency.

Note: Isaac Sim is launched by test/conftest.py (root conftest).
"""

# --- Imports (Isaac Sim launched by root conftest.py) ---

import torch
import pytest

from isaaclab.envs import ManagerBasedRLEnv

from test.common import NUM_ENVS, N_SETTLE_STEPS

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
# Module-Scoped Environment Management
# =============================================================================

_module_env = None
_env_config_type = None


def _get_or_create_env(use_noise: bool = True):
    """Get or create the shared test environment."""
    global _module_env, _env_config_type

    config_type = "realistic" if use_noise else "ideal"

    if _module_env is not None:
        if _env_config_type != config_type:
            raise RuntimeError(
                f"Cannot switch env config from '{_env_config_type}' to '{config_type}'. "
                "Isaac Sim only allows one SimulationContext per process."
            )
        return _module_env

    cfg = StraferNavEnvCfg_NoCam()
    cfg.scene.num_envs = NUM_ENVS
    cfg.actions = ActionsCfg_Ideal()

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


def collect_stationary_observations(env, n_steps: int, freeze_robot: bool = True) -> torch.Tensor:
    """Collect observations from a stationary robot over multiple steps.

    Args:
        env: The Isaac Lab environment
        n_steps: Number of observation steps to collect
        freeze_robot: If True, continuously zero out robot velocities

    Returns:
        Tensor of shape (n_steps, num_envs, obs_dim)
    """
    env.reset()
    reset_robot_pose(env)
    clear_frozen_state()

    zero_action = torch.zeros(env.num_envs, 3, device=env.device)

    for _ in range(N_SETTLE_STEPS):
        env.step(zero_action)
        if freeze_robot:
            freeze_robot_in_place(env)

    observations = []
    for _ in range(n_steps):
        obs_dict, _, _, _, _ = env.step(zero_action)
        observations.append(obs_dict["policy"].clone())
        if freeze_robot:
            freeze_robot_in_place(env)

    return torch.stack(observations, dim=0)


def extract_noise_samples(observations: torch.Tensor, term_slice: slice):
    """Extract noise samples for a specific observation term."""
    import numpy as np

    term_obs = observations[:, :, term_slice]
    mean_per_env = term_obs.mean(dim=0, keepdim=True)
    noise = term_obs - mean_per_env
    return noise.cpu().numpy().flatten()
