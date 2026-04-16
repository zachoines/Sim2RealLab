# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Gymnasium environment registration.

Strafer Lab registers 30 environment variants in
``strafer_lab.tasks.navigation.__init__``.  These tests verify:

1. Every expected environment ID appears in ``gymnasium.envs.registry``.
2. The correct entry-point & env_cfg class are wired in the spec.

These tests are lightweight (no simulation needed) since they only
inspect the registry metadata.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/env/test_env_registration.py -v
"""

import pytest
import gymnasium as gym

# Importing the package triggers the gym.register() calls
import strafer_lab.tasks.navigation  # noqa: F401
from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg,
    StraferNavEnvCfg_PLAY,
    StraferNavEnvCfg_Real_ProcRoom_Depth,
    StraferNavEnvCfg_Real_ProcRoom_Depth_PLAY,
    StraferNavEnvCfg_Real_ProcRoom_NoCam,
    StraferNavEnvCfg_Real_ProcRoom_NoCam_PLAY,
)


# =====================================================================
# Expected environment IDs
# =====================================================================

EXPECTED_ENVS = [
    # Ideal (no noise, no motor dynamics)
    "Isaac-Strafer-Nav-v0",
    "Isaac-Strafer-Nav-Play-v0",
    "Isaac-Strafer-Nav-Depth-v0",
    "Isaac-Strafer-Nav-Depth-Play-v0",
    "Isaac-Strafer-Nav-NoCam-v0",
    "Isaac-Strafer-Nav-NoCam-Play-v0",
    # Realistic (motor dynamics + noise)
    "Isaac-Strafer-Nav-Real-v0",
    "Isaac-Strafer-Nav-Real-Play-v0",
    "Isaac-Strafer-Nav-Real-Depth-v0",
    "Isaac-Strafer-Nav-Real-Depth-Play-v0",
    "Isaac-Strafer-Nav-Real-NoCam-v0",
    "Isaac-Strafer-Nav-Real-NoCam-Play-v0",
    # Robust (aggressive noise + dynamics)
    "Isaac-Strafer-Nav-Robust-v0",
    "Isaac-Strafer-Nav-Robust-Play-v0",
    "Isaac-Strafer-Nav-Robust-Depth-v0",
    "Isaac-Strafer-Nav-Robust-Depth-Play-v0",
    "Isaac-Strafer-Nav-Robust-NoCam-v0",
    "Isaac-Strafer-Nav-Robust-NoCam-Play-v0",
    # Infinigen (Infinigen scene geometry)
    "Isaac-Strafer-Nav-Real-InfinigenDepth-v0",
    "Isaac-Strafer-Nav-Real-InfinigenDepth-Play-v0",
    "Isaac-Strafer-Nav-Robust-InfinigenDepth-v0",
    "Isaac-Strafer-Nav-Robust-InfinigenDepth-Play-v0",
    # Infinigen perception (640x360 camera, Play-only)
    "Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0",
    # ProcRoom (procedural primitive rooms)
    "Isaac-Strafer-Nav-Real-ProcRoom-NoCam-v0",
    "Isaac-Strafer-Nav-Real-ProcRoom-NoCam-Play-v0",
    "Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0",
    "Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0",
    "Isaac-Strafer-Nav-Robust-ProcRoom-NoCam-v0",
    "Isaac-Strafer-Nav-Robust-ProcRoom-NoCam-Play-v0",
    "Isaac-Strafer-Nav-Robust-ProcRoom-Depth-v0",
    "Isaac-Strafer-Nav-Robust-ProcRoom-Depth-Play-v0",
]


# =====================================================================
# Tests
# =====================================================================


@pytest.mark.parametrize("env_id", EXPECTED_ENVS)
def test_env_registered(env_id: str):
    """Verify each expected environment ID is in the Gymnasium registry."""
    assert env_id in gym.envs.registry, (
        f"Environment '{env_id}' not found in gym.envs.registry. "
        "Check that strafer_lab.tasks.navigation.__init__ registers it."
    )


@pytest.mark.parametrize("env_id", EXPECTED_ENVS)
def test_env_entry_point(env_id: str):
    """Verify every registered env uses the IsaacLab ManagerBasedRLEnv entry-point."""
    spec = gym.envs.registry[env_id]
    expected_ep = "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.entry_point == expected_ep, (
        f"{env_id} entry_point = '{spec.entry_point}', "
        f"expected '{expected_ep}'"
    )


def test_expected_env_count():
    """Verify the registered Strafer env count matches ``EXPECTED_ENVS``."""
    strafer_envs = [
        eid for eid in gym.envs.registry
        if eid.startswith("Isaac-Strafer-Nav")
    ]
    print(f"\n  Registered Strafer envs ({len(strafer_envs)}):")
    for eid in sorted(strafer_envs):
        print(f"    {eid}")

    assert len(strafer_envs) == len(EXPECTED_ENVS), (
        f"Expected {len(EXPECTED_ENVS)} Strafer env registrations, "
        f"found {len(strafer_envs)}"
    )


# =====================================================================
# Realism tier grouping sanity check
# =====================================================================

_IDEAL_ENVS = [e for e in EXPECTED_ENVS if "Real" not in e and "Robust" not in e]
_REAL_ENVS = [e for e in EXPECTED_ENVS if "Real" in e]
_ROBUST_ENVS = [e for e in EXPECTED_ENVS if "Robust" in e]
_INFINIGEN_ENVS = [e for e in EXPECTED_ENVS if "InfinigenDepth" in e]
_PROCROOM_ENVS = [e for e in EXPECTED_ENVS if "ProcRoom" in e]


def test_ideal_tier_count():
    """6 Ideal-tier environments (3 sensor configs x Train+Play)."""
    assert len(_IDEAL_ENVS) == 6, f"Expected 6 Ideal envs, got {len(_IDEAL_ENVS)}"


def test_realistic_tier_count():
    """12 Realistic-tier environments (3 sensor + 1 InfinigenDepth + 2 ProcRoom x Train+Play)."""
    assert len(_REAL_ENVS) == 12, f"Expected 12 Realistic envs, got {len(_REAL_ENVS)}"


def test_robust_tier_count():
    """12 Robust-tier environments (3 sensor + 1 InfinigenDepth + 2 ProcRoom x Train+Play)."""
    assert len(_ROBUST_ENVS) == 12, f"Expected 12 Robust envs, got {len(_ROBUST_ENVS)}"


def test_infinigen_tier_count():
    """4 Infinigen environments (2 realism x Train+Play)."""
    assert len(_INFINIGEN_ENVS) == 4, f"Expected 4 InfinigenDepth envs, got {len(_INFINIGEN_ENVS)}"


def test_procroom_tier_count():
    """8 ProcRoom environments (2 realism x 2 sensor configs x Train+Play)."""
    assert len(_PROCROOM_ENVS) == 8, f"Expected 8 ProcRoom envs, got {len(_PROCROOM_ENVS)}"


@pytest.mark.parametrize("env_id", EXPECTED_ENVS)
def test_env_spec_has_env_cfg(env_id: str):
    """Verify every registered env has an env_cfg_entry_point in its kwargs."""
    spec = gym.envs.registry[env_id]
    kwargs = spec.kwargs or {}
    assert "env_cfg_entry_point" in kwargs, (
        f"{env_id} is missing 'env_cfg_entry_point' in kwargs"
    )


@pytest.mark.parametrize("env_id", EXPECTED_ENVS)
def test_env_spec_uses_expected_runner_cfg(env_id: str):
    """NoCam variants use the proprio runner; all others use the depth runner."""
    spec = gym.envs.registry[env_id]
    kwargs = spec.kwargs or {}
    runner_cfg_entry = kwargs.get("rsl_rl_cfg_entry_point", "")
    expected_runner = "STRAFER_PPO_RUNNER_CFG" if "NoCam" in env_id else "STRAFER_PPO_DEPTH_RUNNER_CFG"
    assert runner_cfg_entry.endswith(expected_runner), (
        f"{env_id} runner cfg = '{runner_cfg_entry}', expected suffix '{expected_runner}'"
    )


@pytest.mark.parametrize(
    ("cfg_cls", "expected_num_envs"),
    [
        (StraferNavEnvCfg_PLAY, 50),
        (StraferNavEnvCfg_Real_ProcRoom_NoCam_PLAY, 50),
        (StraferNavEnvCfg_Real_ProcRoom_Depth_PLAY, 8),
    ],
)
def test_play_cfgs_override_only_scene_env_count(cfg_cls, expected_num_envs: int):
    """Play configs should just reduce the scene env count for evaluation."""
    cfg = cfg_cls()
    assert cfg.scene.num_envs == expected_num_envs


@pytest.mark.parametrize(
    "cfg_cls",
    [
        StraferNavEnvCfg,
        StraferNavEnvCfg_Real_ProcRoom_NoCam,
        StraferNavEnvCfg_Real_ProcRoom_Depth,
    ],
)
def test_nav_cfgs_share_runtime_defaults(cfg_cls):
    """Refactored env bases should keep the shared runtime contract identical."""
    cfg = cfg_cls()
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert cfg.sim.render_interval == 4
    assert cfg.decimation == 4
    assert cfg.episode_length_s == 20.0
