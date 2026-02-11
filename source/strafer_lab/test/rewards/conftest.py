# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest fixtures for reward function tests.

Provides a NoCam Realistic environment with ``command_manager`` and
``action_manager`` — the two managers that reward functions rely on.

Note: Isaac Sim is launched by test/conftest.py (root conftest).
"""

# --- Imports (Isaac Sim launched by root conftest.py) ---

import torch
import pytest

from isaaclab.envs import ManagerBasedRLEnv

from test.common import NUM_ENVS

from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg_NoCam,
    ActionsCfg_Ideal,
    ObsCfg_NoCam_Realistic,
)


# =============================================================================
# Module-Scoped Environment Management
# =============================================================================

_module_env = None


def _get_or_create_env():
    """Get or create the shared reward-test environment.

    Uses NoCam Realistic config so that ``command_manager`` and
    ``action_manager`` are both available.  Actions are ideal for
    predictability (no motor dynamics interfering with reward tests).
    """
    global _module_env

    if _module_env is not None:
        return _module_env

    cfg = StraferNavEnvCfg_NoCam()
    cfg.scene.num_envs = NUM_ENVS
    cfg.actions = ActionsCfg_Ideal()
    cfg.observations = ObsCfg_NoCam_Realistic()

    _module_env = ManagerBasedRLEnv(cfg)
    _module_env.reset()

    return _module_env


@pytest.fixture(scope="module")
def reward_env():
    """Provide Strafer environment configured for reward testing."""
    env = _get_or_create_env()
    yield env


def pytest_sessionfinish(session, exitstatus):
    """Clean up the environment after all tests complete."""
    global _module_env
    if _module_env is not None:
        _module_env.close()
        _module_env = None
    # Note: simulation_app.close() is handled by root test/conftest.py
