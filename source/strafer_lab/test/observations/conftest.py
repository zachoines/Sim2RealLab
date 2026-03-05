# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest fixtures for observation function tests.

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
    ObsCfg_NoCam_Ideal,
)


# =============================================================================
# Module-Scoped Environment Management
# =============================================================================

_module_env = None


def _get_or_create_env():
    """Get or create the shared observation-test environment."""
    global _module_env

    if _module_env is not None:
        return _module_env

    cfg = StraferNavEnvCfg_NoCam()
    cfg.scene.num_envs = NUM_ENVS
    cfg.actions = ActionsCfg_Ideal()
    cfg.observations = ObsCfg_NoCam_Ideal()
    cfg.commands.goal_command.debug_vis = False

    _module_env = ManagerBasedRLEnv(cfg)
    _module_env.reset()

    return _module_env


@pytest.fixture(scope="module")
def obs_env():
    """Provide Strafer environment for observation testing."""
    env = _get_or_create_env()
    yield env


def pytest_sessionfinish(session, exitstatus):
    """Clean up the environment after all tests complete."""
    global _module_env
    if _module_env is not None:
        _module_env.close()
        _module_env = None
