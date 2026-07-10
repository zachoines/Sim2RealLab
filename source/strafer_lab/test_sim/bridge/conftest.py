# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for the bridge obs-dump term-evaluation tests.

Brings up a minimal depth env (plane scene, ideal profile) that carries exactly
the handles the dumper evaluates against: the ``d555_imu`` sensor, the
``d555_camera`` policy camera, and the ``robot`` articulation. The ideal profile
means the env's own observation manager assembles the same terms with no
corruption noise, so it is a clean oracle for the dump's computable dims.

Note: Isaac Sim is launched by test_sim/conftest.py (root conftest).
"""

import pytest

from isaaclab.envs import ManagerBasedRLEnv

from strafer_lab.tasks.navigation.composed_env_cfg import StraferNavCfg_Depth_Ideal
from strafer_lab.tasks.navigation.strafer_env_cfg import (
    ActionsCfg_Ideal,
    ObsCfg_Depth_Ideal,
)

# Two envs exercises the batch axis the dumper indexes off (row 0) while keeping
# camera rendering cheap.
_NUM_ENVS = 2

_module_env = None


def _get_or_create_env():
    """Get or create the shared depth env (plane, ideal, camera + IMU)."""
    global _module_env

    if _module_env is not None:
        return _module_env

    cfg = StraferNavCfg_Depth_Ideal()
    cfg.scene.num_envs = _NUM_ENVS
    cfg.actions = ActionsCfg_Ideal()
    cfg.observations = ObsCfg_Depth_Ideal()
    cfg.commands.goal_command.debug_vis = False

    _module_env = ManagerBasedRLEnv(cfg)
    _module_env.reset()

    return _module_env


@pytest.fixture(scope="module")
def depth_env():
    """Provide the shared depth env for obs-dump term-evaluation tests."""
    yield _get_or_create_env()


def pytest_sessionfinish(session, exitstatus):
    """Close the env after all tests complete (Isaac Sim is closed by root)."""
    global _module_env
    if _module_env is not None:
        _module_env.close()
        _module_env = None
