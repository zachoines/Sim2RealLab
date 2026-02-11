# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for domain-randomization events (friction randomization).

``randomize_friction`` modifies wheel contact-surface material properties
at episode reset to simulate varying floor conditions (carpet, tile, etc.).

These tests verify:
1. Sampled friction values fall within the configured range.
2. Different environments receive different friction values.
3. The function does not crash and updates PhysX material properties.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/actions/test_events.py -v

NOTE: Uses the shared ``action_env`` fixture from ``test/actions/conftest.py``.
"""

import torch
import pytest
import numpy as np

from strafer_lab.tasks.navigation.mdp.events import randomize_friction


# =====================================================================
# Fixture alias — use the shared conftest env
# =====================================================================

@pytest.fixture(scope="module")
def env(action_env):
    """Reuse the shared actions conftest environment."""
    return action_env


# =====================================================================
# Constants
# =====================================================================

FRICTION_RANGE = (0.3, 1.2)

# =====================================================================
# Tests
# =====================================================================


def test_friction_within_range(env):
    """Verify static friction values fall within the configured range after randomization."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    randomize_friction(env, env_ids, friction_range=FRICTION_RANGE)

    robot = env.scene["robot"]
    materials = robot.root_physx_view.get_material_properties()
    # materials shape: (num_envs, num_shapes, 3) → [0]=static, [1]=dynamic, [2]=restitution

    # Check STATIC friction (index 0) — should be within FRICTION_RANGE
    static_friction = materials[:, :, 0]  # (num_envs, num_shapes)

    # All shapes in each env get the same friction value (broadcast from per-env sample)
    per_env_friction = static_friction[:, 0].cpu().numpy()

    # Small tolerance for floating-point round-trip through PhysX
    lo, hi = FRICTION_RANGE
    tol = 1e-4

    print(f"\n  Static friction range check:")
    print(f"    Min: {per_env_friction.min():.4f}  (expected >= {lo})")
    print(f"    Max: {per_env_friction.max():.4f}  (expected <= {hi})")

    assert np.all(per_env_friction >= lo - tol), (
        f"Friction below range: min={per_env_friction.min():.4f} < {lo}"
    )
    assert np.all(per_env_friction <= hi + tol), (
        f"Friction above range: max={per_env_friction.max():.4f} > {hi}"
    )


def test_friction_dynamic_is_lower(env):
    """Verify dynamic friction = static × 0.9 (physically consistent)."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    randomize_friction(env, env_ids, friction_range=FRICTION_RANGE)

    robot = env.scene["robot"]
    materials = robot.root_physx_view.get_material_properties()

    static_f = materials[:, 0, 0].cpu().numpy()
    dynamic_f = materials[:, 0, 1].cpu().numpy()

    expected_dynamic = static_f * 0.9
    max_diff = np.abs(dynamic_f - expected_dynamic).max()

    print(f"\n  Dynamic friction consistency:")
    print(f"    Max |dynamic - 0.9*static|: {max_diff:.6f}")

    assert max_diff < 1e-4, (
        f"Dynamic friction deviates from 0.9*static by {max_diff:.4f}"
    )


def test_friction_varies_across_envs(env):
    """Verify different environments receive different friction values.

    With 64 envs drawn from Uniform(0.3, 1.2), the probability that all
    are identical is negligibly small.
    """
    env_ids = torch.arange(env.num_envs, device=env.device)
    randomize_friction(env, env_ids, friction_range=FRICTION_RANGE)

    robot = env.scene["robot"]
    materials = robot.root_physx_view.get_material_properties()

    per_env_friction = materials[:, 0, 0].cpu().numpy()
    n_unique = len(np.unique(np.round(per_env_friction, 4)))

    print(f"\n  Friction diversity:")
    print(f"    Unique values: {n_unique} / {env.num_envs}")

    # With 64 envs, expect at least 50 unique values
    assert n_unique > env.num_envs // 2, (
        f"Only {n_unique} unique friction values out of {env.num_envs} envs. "
        "Randomization may not be working."
    )


def test_friction_partial_env_randomization(env):
    """Verify randomize_friction only modifies the specified env_ids."""
    # First set ALL envs to a known friction
    all_ids = torch.arange(env.num_envs, device=env.device)
    randomize_friction(env, all_ids, friction_range=(0.5, 0.5))

    robot = env.scene["robot"]
    before = robot.root_physx_view.get_material_properties()[:, 0, 0].cpu().numpy().copy()

    # Now randomize only the first half
    half_ids = torch.arange(env.num_envs // 2, device=env.device)
    randomize_friction(env, half_ids, friction_range=(0.8, 1.2))

    after = robot.root_physx_view.get_material_properties()[:, 0, 0].cpu().numpy()

    # The second half should be unchanged
    second_half_diff = np.abs(after[env.num_envs // 2:] - before[env.num_envs // 2:]).max()

    print(f"\n  Partial randomization check:")
    print(f"    Unchanged envs max diff: {second_half_diff:.6f}")

    assert second_half_diff < 1e-4, (
        f"Unselected envs changed friction by {second_half_diff:.4f}"
    )


def test_friction_empty_env_ids_no_crash(env):
    """Verify empty env_ids does not cause an error."""
    empty_ids = torch.tensor([], dtype=torch.long, device=env.device)
    # Should return immediately without error
    randomize_friction(env, empty_ids, friction_range=FRICTION_RANGE)
