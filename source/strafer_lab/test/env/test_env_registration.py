# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Gymnasium environment registration.

Strafer Lab registers 18 environment variants in
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


# =====================================================================
# Expected environment IDs (9 configs × Train + Play = 18)
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
    """Verify exactly 18 Strafer environments are registered."""
    strafer_envs = [
        eid for eid in gym.envs.registry
        if eid.startswith("Isaac-Strafer-Nav")
    ]
    print(f"\n  Registered Strafer envs ({len(strafer_envs)}):")
    for eid in sorted(strafer_envs):
        print(f"    {eid}")

    assert len(strafer_envs) == 18, (
        f"Expected 18 Strafer env registrations, found {len(strafer_envs)}"
    )


# =====================================================================
# Realism tier grouping sanity check
# =====================================================================

_IDEAL_ENVS = [e for e in EXPECTED_ENVS if "Real" not in e and "Robust" not in e]
_REAL_ENVS = [e for e in EXPECTED_ENVS if "Real" in e]
_ROBUST_ENVS = [e for e in EXPECTED_ENVS if "Robust" in e]


def test_ideal_tier_count():
    """6 Ideal-tier environments (3 sensor configs × Train+Play)."""
    assert len(_IDEAL_ENVS) == 6, f"Expected 6 Ideal envs, got {len(_IDEAL_ENVS)}"


def test_realistic_tier_count():
    """6 Realistic-tier environments (3 sensor configs × Train+Play)."""
    assert len(_REAL_ENVS) == 6, f"Expected 6 Realistic envs, got {len(_REAL_ENVS)}"


def test_robust_tier_count():
    """6 Robust-tier environments (3 sensor configs × Train+Play)."""
    assert len(_ROBUST_ENVS) == 6, f"Expected 6 Robust envs, got {len(_ROBUST_ENVS)}"


@pytest.mark.parametrize("env_id", EXPECTED_ENVS)
def test_env_spec_has_env_cfg(env_id: str):
    """Verify every registered env has an env_cfg_entry_point in its kwargs."""
    spec = gym.envs.registry[env_id]
    kwargs = spec.kwargs or {}
    assert "env_cfg_entry_point" in kwargs, (
        f"{env_id} is missing 'env_cfg_entry_point' in kwargs"
    )
