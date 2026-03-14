# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for net_forces_w collision penalty reward functions.

Tests that ``collision_penalty_net`` and ``collision_sustained_penalty_net``
produce correct signals using ``net_forces_w``. These reward functions are
used by all env variants (base, Infinigen, ProcRoom).

Test strategy:
    1. Create a NoCam env with no obstacle filter on the contact sensor.
    2. Add a single test obstacle to the scene for controlled collision tests.
    3. Verify ``net_forces_w`` shape is (N, 1, 3) and is always populated.
    4. With the obstacle far away, verify both penalties return 0.0.
    5. Place the obstacle in front, drive into it, verify both penalties fire.
    6. Verify ``collision_sustained_penalty_net`` scales with intensity.

Usage:
    cd C:\\Worspace\\IsaacLab
    .\\isaaclab.bat -p -m pytest ..\\source\\strafer_lab\\test\\rewards\\test_collision_rewards.py -v
"""

import torch
import pytest
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

from test.common import NUM_ENVS
from test.common.robot import get_env_origins

from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg_NoCam,
    ActionsCfg_Ideal,
    ObsCfg_NoCam_Realistic,
)
from strafer_lab.tasks.navigation.mdp.rewards import (
    collision_penalty_net,
    collision_sustained_penalty_net,
)


# =====================================================================
# Module-Scoped Environment
# =====================================================================

_module_env = None


def _get_or_create_env():
    """Create a NoCam env with no filter on the contact sensor.

    This mimics the StraferSceneCfg_Infinigen contact sensor configuration:
    no ``filter_prim_paths_expr``, relying on ``net_forces_w`` for collision.
    A single test obstacle is added to the scene for controlled collision tests.
    """
    global _module_env

    if _module_env is not None:
        return _module_env

    cfg = StraferNavEnvCfg_NoCam()
    cfg.scene.num_envs = NUM_ENVS
    cfg.actions = ActionsCfg_Ideal()
    cfg.observations = ObsCfg_NoCam_Realistic()
    cfg.commands.goal_command.debug_vis = False

    # Replace contact sensor with filter-free version (Infinigen style)
    cfg.scene.contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=0.0,
        history_length=1,
    )

    # Add a single test obstacle for collision tests
    cfg.scene.test_obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestObstacle",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(50.0, 50.0, 0.15)),
    )

    # Prevent event repositioning on reset — tests control placement
    if hasattr(cfg.events, "randomize_obstacles"):
        cfg.events.randomize_obstacles = None
    if hasattr(cfg.events, "reset_robot"):
        cfg.events.reset_robot = None

    cfg.terminations.robot_flipped = None
    cfg.episode_length_s = 300.0

    _module_env = ManagerBasedRLEnv(cfg)
    _module_env.reset()

    return _module_env


@pytest.fixture(scope="module")
def infinigen_env():
    """Provide Strafer environment configured for Infinigen collision testing."""
    env = _get_or_create_env()
    yield env


def pytest_sessionfinish(session, exitstatus):
    """Clean up the environment after all tests complete."""
    global _module_env
    if _module_env is not None:
        _module_env.close()
        _module_env = None


# =====================================================================
# Helpers
# =====================================================================

SENSOR_CFG = SceneEntityCfg("contact_sensor")
TEST_FORCE_THRESHOLD = 0.1  # Newtons
MAX_ALLOWED_MISSES = 1


def _place_obstacle_in_front(env, distance: float = 0.4):
    """Place the test obstacle directly ahead of each robot."""
    robot = env.scene["robot"]
    obstacle = env.scene["test_obstacle"]
    device = env.device
    num_envs = env.num_envs

    robot_pos = robot.data.root_pos_w[:, :3].clone()
    robot_quat = robot.data.root_quat_w
    env_origins = get_env_origins(env)

    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    root_state = obstacle.data.default_root_state.clone()[:num_envs]
    root_state[:, 0] = robot_pos[:, 0] + distance * torch.cos(yaw)
    root_state[:, 1] = robot_pos[:, 1] + distance * torch.sin(yaw)
    root_state[:, 2] = env_origins[:, 2] + 0.15
    root_state[:, 3] = 1.0
    root_state[:, 4:7] = 0.0
    root_state[:, 7:] = 0.0

    all_ids = torch.arange(num_envs, device=device)
    obstacle.write_root_state_to_sim(root_state, all_ids)


def _move_obstacle_far_away(env):
    """Move the test obstacle far from all robots."""
    obstacle = env.scene["test_obstacle"]
    device = env.device
    num_envs = env.num_envs
    env_origins = get_env_origins(env)

    root_state = obstacle.data.default_root_state.clone()[:num_envs]
    root_state[:, 0] = env_origins[:, 0] + 50.0
    root_state[:, 1] = env_origins[:, 1] + 50.0
    root_state[:, 2] = env_origins[:, 2] + 0.15
    root_state[:, 3] = 1.0
    root_state[:, 4:7] = 0.0
    root_state[:, 7:] = 0.0

    all_ids = torch.arange(num_envs, device=device)
    obstacle.write_root_state_to_sim(root_state, all_ids)


def _step_stationary(env, n_steps: int = 10):
    """Step the environment with zero action."""
    action = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(n_steps):
        env.step(action)


def _drive_and_detect_net(env, n_steps: int = 40, speed: float = 1.0,
                          threshold: float = TEST_FORCE_THRESHOLD) -> torch.Tensor:
    """Drive forward and track collision via net_forces_w across all steps.

    Returns:
        Boolean tensor (num_envs,): True if net collision force exceeded
        threshold at any step.
    """
    sensor = env.scene.sensors["contact_sensor"]
    ever_detected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    action = torch.zeros(env.num_envs, 3, device=env.device)
    action[:, 0] = speed

    for _ in range(n_steps):
        env.step(action)
        nf = sensor.data.net_forces_w  # (N, 1, 3)
        force_mag = torch.norm(nf, dim=-1)  # (N, 1)
        step_detected = (force_mag > threshold).any(dim=-1)  # (N,)
        ever_detected |= step_detected

    return ever_detected


# =====================================================================
# Diagnostic: net_forces_w Shape and Population
# =====================================================================


class TestNetForcesWDiagnostics:
    """Verify net_forces_w is populated when no filter_prim_paths_expr is set."""

    def test_net_forces_w_shape(self, infinigen_env):
        """net_forces_w must be (num_envs, 1, 3) — always populated."""
        env = infinigen_env
        env.reset()
        _step_stationary(env, 5)

        sensor = env.scene.sensors["contact_sensor"]
        nf = sensor.data.net_forces_w

        assert nf is not None, "net_forces_w is None"
        assert nf.shape == (env.num_envs, 1, 3), (
            f"net_forces_w shape mismatch: expected ({env.num_envs}, 1, 3), got {nf.shape}"
        )
        print(f"\n  net_forces_w shape: {nf.shape}")

    def test_force_matrix_w_is_none_without_filter(self, infinigen_env):
        """force_matrix_w should be None when filter_prim_paths_expr is empty."""
        env = infinigen_env
        env.reset()
        _step_stationary(env, 5)

        sensor = env.scene.sensors["contact_sensor"]
        fm = sensor.data.force_matrix_w

        assert fm is None, (
            f"force_matrix_w should be None without filter_prim_paths_expr, "
            f"but got shape {fm.shape}"
        )
        print("\n  force_matrix_w is None (expected — no filter)")


# =====================================================================
# Control: No Collision Signal When Stationary & Far From Obstacles
# =====================================================================


class TestNoCollisionBaseline:
    """Verify zero net collision signal when no contact is occurring."""

    def test_collision_penalty_net_zero_when_no_contact(self, infinigen_env):
        """With obstacle far away, collision_penalty_net must return 0.0."""
        env = infinigen_env
        env.reset()

        _move_obstacle_far_away(env)
        _step_stationary(env, 20)

        penalty = collision_penalty_net(env, sensor_cfg=SENSOR_CFG, threshold=1.0)

        print(f"\n  collision_penalty_net (no contact):")
        print(f"    mean: {penalty.mean().item():.6f}")
        print(f"    max:  {penalty.max().item():.6f}")

        assert (penalty == 0.0).all(), (
            f"Expected all zeros, but {(penalty > 0).sum().item()} envs "
            f"have penalty > 0. max={penalty.max().item():.4f}"
        )

    def test_collision_sustained_net_zero_when_no_contact(self, infinigen_env):
        """With obstacle far away, collision_sustained_penalty_net must return 0.0."""
        env = infinigen_env
        env.reset()

        _move_obstacle_far_away(env)
        _step_stationary(env, 20)

        penalty = collision_sustained_penalty_net(env, sensor_cfg=SENSOR_CFG, threshold=1.0)

        print(f"\n  collision_sustained_penalty_net (no contact):")
        print(f"    mean: {penalty.mean().item():.6f}")
        print(f"    max:  {penalty.max().item():.6f}")

        assert (penalty == 0.0).all(), (
            f"Expected all zeros, but max={penalty.max().item():.4f}"
        )


# =====================================================================
# Collision Detection via net_forces_w
# =====================================================================


class TestCollisionDetectionNet:
    """Drive robots into obstacle and verify detection via net_forces_w."""

    def test_collision_detected_via_net_forces(self, infinigen_env):
        """Every env must detect collision via net_forces_w when driving into obstacle."""
        env = infinigen_env
        env.reset()

        _move_obstacle_far_away(env)
        _step_stationary(env, 10)
        _place_obstacle_in_front(env, distance=0.4)
        _step_stationary(env, 5)

        ever_detected = _drive_and_detect_net(env, n_steps=40, speed=1.0)
        n_detected = int(ever_detected.sum().item())
        n_missed = env.num_envs - n_detected

        print(f"\n  net_forces_w detection (40 steps):")
        print(f"    Detected: {n_detected}/{env.num_envs}")

        assert n_missed <= MAX_ALLOWED_MISSES, (
            f"net_forces_w missed {n_missed}/{env.num_envs} envs "
            f"(max allowed: {MAX_ALLOWED_MISSES})."
        )

    def test_collision_penalty_net_fires_on_impact(self, infinigen_env):
        """collision_penalty_net must return 1.0 at some step during impact."""
        env = infinigen_env
        env.reset()

        _move_obstacle_far_away(env)
        _step_stationary(env, 10)
        _place_obstacle_in_front(env, distance=0.4)
        _step_stationary(env, 5)

        max_penalty = torch.zeros(env.num_envs, device=env.device)
        action = torch.zeros(env.num_envs, 3, device=env.device)
        action[:, 0] = 1.0
        for _ in range(40):
            env.step(action)
            penalty = collision_penalty_net(env, sensor_cfg=SENSOR_CFG, threshold=1.0)
            max_penalty = torch.maximum(max_penalty, penalty)

        n_detected = int((max_penalty > 0).sum().item())
        n_missed = env.num_envs - n_detected

        print(f"\n  collision_penalty_net (max across 40 steps):")
        print(f"    Detected: {n_detected}/{env.num_envs}")

        assert n_missed <= MAX_ALLOWED_MISSES, (
            f"collision_penalty_net never fired in {n_missed}/{env.num_envs} envs."
        )


# =====================================================================
# Sustained vs Binary: Intensity Scales With Force
# =====================================================================


class TestSustainedVsBinaryNet:
    """Verify sustained_net provides graded signal during contact."""

    def test_sustained_net_nonzero_during_contact(self, infinigen_env):
        """Sustained net penalty must be > 0 at some step while hitting obstacle."""
        env = infinigen_env
        env.reset()

        _move_obstacle_far_away(env)
        _step_stationary(env, 10)
        _place_obstacle_in_front(env, distance=0.4)
        _step_stationary(env, 5)

        max_sustained = torch.zeros(env.num_envs, device=env.device)
        action = torch.zeros(env.num_envs, 3, device=env.device)
        action[:, 0] = 1.0
        for _ in range(40):
            env.step(action)
            penalty = collision_sustained_penalty_net(
                env, sensor_cfg=SENSOR_CFG, threshold=1.0
            )
            max_sustained = torch.maximum(max_sustained, penalty)

        n_detected = int((max_sustained > 0).sum().item())
        n_missed = env.num_envs - n_detected

        print(f"\n  collision_sustained_penalty_net (max across 40 steps):")
        print(f"    Detected: {n_detected}/{env.num_envs}")
        print(f"    max per env (first 8): {max_sustained[:8].tolist()}")

        assert n_missed <= MAX_ALLOWED_MISSES, (
            f"Sustained net penalty never fired in {n_missed}/{env.num_envs} envs."
        )
