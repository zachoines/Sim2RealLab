# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Gymnasium environment registration (composed-variant scheme).

Strafer Lab registers a small set of composed navigation variants in
``strafer_lab.tasks.navigation.__init__``. These tests verify:

1. Every expected (new-scheme) environment ID is in the registry.
2. The entry-point + env_cfg + runner cfg are wired correctly.
3. **No legacy gym ID resolves** — the clean-break proof at the test layer.

Lightweight (no simulation needed) — only inspects registry metadata.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/env/test_env_registration.py -v
"""

import pytest
import gymnasium as gym

# Importing the package triggers the gym.register() calls
import strafer_lab.tasks.navigation  # noqa: F401
from strafer_lab.tasks.navigation.composed_env_cfg import (
    StraferNavCfg_RLDepth_Real_PLAY,
    StraferNavCfg_RLNoCam_PLAY,
    StraferNavCfg_RLDepth_Real,
    StraferNavCfg_TeleopCapture,
)


# =====================================================================
# Expected environment IDs (the composed-variant scheme)
# =====================================================================

EXPECTED_ENVS = [
    # RL training (fixed stack)
    "Isaac-Strafer-Nav-RLDepth-Real-v0",
    "Isaac-Strafer-Nav-RLDepth-Real-Play-v0",
    "Isaac-Strafer-Nav-RLDepth-Robust-v0",
    "Isaac-Strafer-Nav-RLDepth-Robust-Play-v0",
    "Isaac-Strafer-Nav-RLNoCam-v0",
    "Isaac-Strafer-Nav-RLNoCam-Play-v0",
    # Capture (operator-selectable stack)
    "Isaac-Strafer-Nav-Capture-Teleop-v0",
    "Isaac-Strafer-Nav-Capture-Bridge-v0",
    "Isaac-Strafer-Nav-Capture-Coverage-v0",
]


# =====================================================================
# Tests
# =====================================================================


@pytest.mark.parametrize("env_id", EXPECTED_ENVS)
def test_env_registered(env_id: str):
    """Each expected (new-scheme) environment ID is in the registry."""
    assert env_id in gym.envs.registry, (
        f"Environment '{env_id}' not found in gym.envs.registry."
    )


@pytest.mark.parametrize("env_id", EXPECTED_ENVS)
def test_env_entry_point(env_id: str):
    """Every registered env uses the IsaacLab ManagerBasedRLEnv entry-point."""
    spec = gym.envs.registry[env_id]
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"


def test_only_composed_scheme_ids_registered():
    """The clean-break proof: the registered Strafer set is EXACTLY the new
    scheme — so no legacy gym ID resolves (a caller still on an old ID would
    fail to launch rather than silently work)."""
    strafer_envs = [
        eid for eid in gym.envs.registry if eid.startswith("Isaac-Strafer-Nav")
    ]
    assert sorted(strafer_envs) == sorted(EXPECTED_ENVS), (
        f"Registered Strafer envs {sorted(strafer_envs)} != "
        f"expected {sorted(EXPECTED_ENVS)}"
    )


@pytest.mark.parametrize("env_id", EXPECTED_ENVS)
def test_env_spec_has_env_cfg(env_id: str):
    """Every registered env has an env_cfg_entry_point in its kwargs."""
    spec = gym.envs.registry[env_id]
    assert "env_cfg_entry_point" in (spec.kwargs or {})


@pytest.mark.parametrize("env_id", EXPECTED_ENVS)
def test_env_spec_runner_matches_obs_profile(env_id: str):
    """Proprioceptive (no-image obs) variants use the MLP runner; depth-image
    variants use the CNN depth runner."""
    spec = gym.envs.registry[env_id]
    runner = (spec.kwargs or {}).get("rsl_rl_cfg_entry_point", "")
    proprio = any(tag in env_id for tag in ("NoCam", "Teleop", "Coverage"))
    expected = "STRAFER_PPO_RUNNER_CFG" if proprio else "STRAFER_PPO_DEPTH_RUNNER_CFG"
    assert runner.endswith(expected), (
        f"{env_id} runner = '{runner}', expected suffix '{expected}'"
    )


@pytest.mark.parametrize(
    ("cfg_cls", "expected_num_envs"),
    [
        (StraferNavCfg_RLDepth_Real_PLAY, 8),
        (StraferNavCfg_RLNoCam_PLAY, 50),
    ],
)
def test_play_cfgs_override_only_scene_env_count(cfg_cls, expected_num_envs: int):
    """Play configs reduce the scene env count for evaluation."""
    assert cfg_cls().scene.num_envs == expected_num_envs


@pytest.mark.parametrize(
    "cfg_cls",
    [StraferNavCfg_RLDepth_Real, StraferNavCfg_TeleopCapture],
)
def test_nav_cfgs_share_runtime_defaults(cfg_cls):
    """Composed variants keep the shared runtime contract identical."""
    cfg = cfg_cls()
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert cfg.sim.render_interval == 4
    assert cfg.decimation == 4
    assert cfg.episode_length_s == 20.0
