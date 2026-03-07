# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for collision penalty reward functions.

Tests that the contact sensor detects obstacle collisions and that the
reward functions ``collision_penalty`` and ``collision_sustained_penalty``
produce correct signals.

Test strategy:
    1. Place an obstacle directly in front of each robot (0.4 m ahead).
    2. Drive the robot forward into it for N steps.
    3. Verify ``force_matrix_w`` reports nonzero obstacle contact forces.
    4. Verify ``collision_penalty`` returns 1.0 (binary detection).
    5. Verify ``collision_sustained_penalty`` returns > 0 (force-scaled).
    6. As a control, verify both return 0.0 when the robot is stationary
       and no obstacle is nearby.

Each test uses the shared ``collision_env`` fixture (module-scoped).

Usage:
    cd C:\\Worspace\\IsaacLab
    .\\isaaclab.bat -p -m pytest ..\\source\\strafer_lab\\test\\rewards\\test_collision_rewards.py -v
"""

import math
import torch
import pytest
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from test.common import NUM_ENVS
from test.common.robot import get_env_origins

from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg_NoCam,
    ActionsCfg_Ideal,
    ObsCfg_NoCam_Realistic,
)
from strafer_lab.tasks.navigation.mdp.rewards import (
    collision_penalty,
    collision_sustained_penalty,
)


# =====================================================================
# Module-Scoped Environment
# =====================================================================

_module_env = None


def _get_or_create_env():
    """Create an environment suitable for collision testing.

    Uses NoCam config with Ideal actions (no motor dynamics) so that
    commanded velocities translate directly to motion.

    Obstacle randomization events are disabled so that test-placed
    obstacles stay where the test put them across resets and steps.
    Episode length is extended to prevent time-out resets mid-test.
    """
    global _module_env

    if _module_env is not None:
        return _module_env

    cfg = StraferNavEnvCfg_NoCam()
    cfg.scene.num_envs = NUM_ENVS
    cfg.actions = ActionsCfg_Ideal()
    cfg.observations = ObsCfg_NoCam_Realistic()
    cfg.commands.goal_command.debug_vis = False

    # Prevent obstacle repositioning on reset — tests control placement
    if hasattr(cfg.events, "randomize_obstacles"):
        cfg.events.randomize_obstacles = None

    # Prevent robot repositioning on reset — tests control robot pose
    if hasattr(cfg.events, "reset_robot"):
        cfg.events.reset_robot = None

    # Disable terminations that could trigger mid-test resets
    # (robot may flip when ramming into kinematic obstacles)
    cfg.terminations.robot_flipped = None

    # Long episode to prevent time-out resets mid-test
    cfg.episode_length_s = 300.0

    _module_env = ManagerBasedRLEnv(cfg)
    _module_env.reset()

    return _module_env


@pytest.fixture(scope="module")
def collision_env():
    """Provide Strafer environment configured for collision testing."""
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

# Maximum allowed misses in a deterministic collision scenario.
# Every env has an identical obstacle placement, so the expected
# detection count is num_envs.  We allow at most 1 miss to account
# for floating-point contact-boundary edge cases.
MAX_ALLOWED_MISSES = 1

# Force threshold for tests (just above PhysX solver noise floor).
# Lower than the reward threshold (1.0N) because tests verify the
# sensor pipeline works, not whether contacts are "meaningful."
TEST_FORCE_THRESHOLD = 0.1  # Newtons


def _place_obstacle_in_front(env, obstacle_name: str, distance: float = 0.4):
    """Place a specific obstacle directly in front of each robot.

    The robot's forward direction is determined by its current yaw.
    The obstacle is placed ``distance`` meters ahead at chassis height.

    Args:
        env: The environment.
        obstacle_name: Scene key, e.g. "obstacle_0".
        distance: How far ahead to place the obstacle (meters).
    """
    robot = env.scene["robot"]
    obstacle = env.scene[obstacle_name]
    device = env.device
    num_envs = env.num_envs

    robot_pos = robot.data.root_pos_w[:, :3].clone()
    robot_quat = robot.data.root_quat_w
    env_origins = get_env_origins(env)

    # Robot yaw
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Place obstacle ahead in the robot's forward direction
    # z=0.15 puts the center of the 0.3m cube at 0.15m (spans 0-0.3m),
    # overlapping with chassis rails at ~0.06-0.12m height.
    root_state = obstacle.data.default_root_state.clone()[:num_envs]
    root_state[:, 0] = robot_pos[:, 0] + distance * torch.cos(yaw)
    root_state[:, 1] = robot_pos[:, 1] + distance * torch.sin(yaw)
    root_state[:, 2] = env_origins[:, 2] + 0.15  # obstacle cube center
    root_state[:, 3] = 1.0  # identity quaternion
    root_state[:, 4:7] = 0.0
    root_state[:, 7:] = 0.0

    all_ids = torch.arange(num_envs, device=device)
    obstacle.write_root_state_to_sim(root_state, all_ids)


def _move_obstacle_far_away(env, obstacle_name: str):
    """Move an obstacle far from all robots (50m away)."""
    obstacle = env.scene[obstacle_name]
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


def _step_forward(env, n_steps: int = 20, speed: float = 1.0):
    """Drive all robots forward (body-frame +X) for n_steps.

    Args:
        env: The environment.
        n_steps: Number of simulation steps to take.
        speed: Forward velocity command magnitude.
    """
    action = torch.zeros(env.num_envs, 3, device=env.device)
    action[:, 0] = speed  # vx forward
    for _ in range(n_steps):
        env.step(action)


def _drive_and_detect(env, n_steps: int = 40, speed: float = 1.0,
                      threshold: float = TEST_FORCE_THRESHOLD) -> torch.Tensor:
    """Drive forward and track which envs detected collision at ANY step.

    Contact forces are instantaneous — a robot may hit an obstacle at step
    10 then bounce away.  Checking only the final step misses transient
    contacts.  This function accumulates a boolean mask across all steps.

    Returns:
        Boolean tensor of shape (num_envs,): True if collision was detected
        at any step during the approach.
    """
    sensor = env.scene.sensors["contact_sensor"]
    ever_detected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    action = torch.zeros(env.num_envs, 3, device=env.device)
    action[:, 0] = speed

    for _ in range(n_steps):
        env.step(action)

        fm = sensor.data.force_matrix_w
        if fm is not None:
            force_mag = torch.norm(fm, dim=-1)
            step_detected = (force_mag > threshold).any(dim=-1).any(dim=-1)
            ever_detected |= step_detected

    return ever_detected


def _step_stationary(env, n_steps: int = 10):
    """Step the environment with zero action."""
    action = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(n_steps):
        env.step(action)


# =====================================================================
# Diagnostic: Contact Sensor Data Population
# =====================================================================


class TestContactSensorDiagnostics:
    """Verify the contact sensor is producing data at all."""

    def test_force_matrix_w_is_populated(self, collision_env):
        """``force_matrix_w`` must not be None when filter_prim_paths_expr is set."""
        env = collision_env
        env.reset()
        _step_stationary(env, 5)

        sensor = env.scene.sensors["contact_sensor"]

        assert sensor.data.force_matrix_w is not None, (
            "force_matrix_w is None — filter_prim_paths_expr may not have resolved. "
            "Check that ContactSensorCfg.filter_prim_paths_expr matches obstacle prims."
        )
        print(f"\n  force_matrix_w shape: {sensor.data.force_matrix_w.shape}")
        print(f"  Expected: ({env.num_envs}, 1, N_obstacles, 3)")

    def test_force_matrix_w_shape(self, collision_env):
        """``force_matrix_w`` shape must be (num_envs, num_bodies, num_filter_prims, 3)."""
        env = collision_env
        env.reset()
        _step_stationary(env, 5)

        sensor = env.scene.sensors["contact_sensor"]
        fm = sensor.data.force_matrix_w

        assert fm is not None, "force_matrix_w is None"

        # num_bodies = 1 (body_link only)
        # num_filter_prims = 8 (Obstacle_0 through Obstacle_7)
        assert fm.ndim == 4, f"Expected 4D tensor, got {fm.ndim}D: shape={fm.shape}"
        assert fm.shape[0] == env.num_envs, (
            f"Dim 0 should be num_envs={env.num_envs}, got {fm.shape[0]}"
        )
        assert fm.shape[1] >= 1, f"Dim 1 (num_bodies) should be >= 1, got {fm.shape[1]}"
        assert fm.shape[3] == 3, f"Dim 3 should be 3 (xyz forces), got {fm.shape[3]}"

        print(f"\n  force_matrix_w shape: {fm.shape}")
        print(f"    num_envs={fm.shape[0]}, num_bodies={fm.shape[1]}, "
              f"num_filter_prims={fm.shape[2]}, xyz=3")

    def test_net_forces_w_is_populated(self, collision_env):
        """``net_forces_w`` should always be populated (regardless of filter)."""
        env = collision_env
        env.reset()
        _step_stationary(env, 5)

        sensor = env.scene.sensors["contact_sensor"]
        nf = sensor.data.net_forces_w

        assert nf is not None, "net_forces_w is None"
        assert nf.shape == (env.num_envs, 1, 3), (
            f"net_forces_w shape mismatch: expected ({env.num_envs}, 1, 3), got {nf.shape}"
        )
        print(f"\n  net_forces_w shape: {nf.shape}")


# =====================================================================
# Control: No Collision When Stationary & Far From Obstacles
# =====================================================================


class TestNoCollisionBaseline:
    """Verify zero collision signal when no contact is occurring."""

    def test_collision_penalty_zero_when_no_contact(self, collision_env):
        """With obstacles far away, ``collision_penalty`` must return 0.0."""
        env = collision_env
        env.reset()

        # Move all obstacles far away
        for i in range(8):
            _move_obstacle_far_away(env, f"obstacle_{i}")

        _step_stationary(env, 20)

        penalty = collision_penalty(env, sensor_cfg=SENSOR_CFG, threshold=1.0)

        print(f"\n  Collision penalty (no contact):")
        print(f"    mean: {penalty.mean().item():.6f}")
        print(f"    max:  {penalty.max().item():.6f}")
        print(f"    nonzero count: {(penalty > 0).sum().item()}/{env.num_envs}")

        assert (penalty == 0.0).all(), (
            f"Expected all zeros with no obstacles nearby, "
            f"but {(penalty > 0).sum().item()} envs have penalty > 0. "
            f"max={penalty.max().item():.4f}"
        )

    def test_collision_sustained_zero_when_no_contact(self, collision_env):
        """With obstacles far away, ``collision_sustained_penalty`` must return 0.0."""
        env = collision_env
        env.reset()

        for i in range(8):
            _move_obstacle_far_away(env, f"obstacle_{i}")

        _step_stationary(env, 20)

        penalty = collision_sustained_penalty(env, sensor_cfg=SENSOR_CFG, threshold=1.0)

        print(f"\n  Sustained collision penalty (no contact):")
        print(f"    mean: {penalty.mean().item():.6f}")
        print(f"    max:  {penalty.max().item():.6f}")

        assert (penalty == 0.0).all(), (
            f"Expected all zeros with no obstacles nearby, "
            f"but max sustained penalty = {penalty.max().item():.4f}"
        )


# =====================================================================
# Collision Detection: Drive Into Obstacle
# =====================================================================


class TestCollisionDetection:
    """Drive robots into obstacles and verify detection."""

    def test_collision_detected_on_impact(self, collision_env):
        """Every env must detect collision at some point while driving into obstacle.

        Contact forces are instantaneous — a robot may hit the obstacle and
        bounce back.  We track detection across ALL steps, so a transient
        contact still counts.  Every env has an identical obstacle 0.4m ahead,
        so ALL must detect at some step.
        """
        env = collision_env
        env.reset()

        for i in range(8):
            _move_obstacle_far_away(env, f"obstacle_{i}")

        _step_stationary(env, 10)
        _place_obstacle_in_front(env, "obstacle_0", distance=0.4)
        _step_stationary(env, 5)

        ever_detected = _drive_and_detect(env, n_steps=40, speed=1.0)
        n_detected = int(ever_detected.sum().item())
        n_missed = env.num_envs - n_detected

        print(f"\n  Collision detection (accumulated over 40 steps):")
        print(f"    Detected: {n_detected}/{env.num_envs}")

        assert n_missed <= MAX_ALLOWED_MISSES, (
            f"collision detection missed {n_missed}/{env.num_envs} envs "
            f"(max allowed: {MAX_ALLOWED_MISSES}). "
            f"Every env has an obstacle directly ahead — misses indicate "
            f"a broken sensor pipeline."
        )

    def test_force_matrix_nonzero_during_contact(self, collision_env):
        """Raw ``force_matrix_w`` must show nonzero forces at some step.

        This is the most fundamental test — if force_matrix_w is never
        nonzero across all 40 steps of driving into an obstacle, the
        sensor pipeline is broken.
        """
        env = collision_env
        env.reset()

        for i in range(8):
            _move_obstacle_far_away(env, f"obstacle_{i}")

        _step_stationary(env, 10)
        _place_obstacle_in_front(env, "obstacle_0", distance=0.4)
        _step_stationary(env, 5)

        # Drive into obstacle, track max force seen per env across all steps
        sensor = env.scene.sensors["contact_sensor"]
        max_force_ever = torch.zeros(env.num_envs, device=env.device)

        action = torch.zeros(env.num_envs, 3, device=env.device)
        action[:, 0] = 1.0
        for _ in range(40):
            env.step(action)
            fm = sensor.data.force_matrix_w
            if fm is not None:
                force_mag = torch.norm(fm, dim=-1)
                step_max = force_mag.max(dim=-1).values.max(dim=-1).values
                max_force_ever = torch.maximum(max_force_ever, step_max)

        n_with_force = int((max_force_ever > 0.1).sum().item())

        print(f"\n  Raw force_matrix_w (max across 40 steps):")
        print(f"    max force per env (first 8): {max_force_ever[:8].tolist()}")
        print(f"    envs with force > 0.1N: {n_with_force}/{env.num_envs}")

        assert n_with_force == env.num_envs, (
            f"force_matrix_w never exceeded 0.1N in {env.num_envs - n_with_force} "
            f"envs across 40 steps of driving into an obstacle. "
            f"The contact sensor is not detecting obstacle contacts."
        )


# =====================================================================
# Directional Collision Tests
# =====================================================================


class TestCollisionFromMultipleAngles:
    """Test collision detection from different approach angles.

    This ensures the contact sensor works regardless of which face of
    the obstacle the robot hits.
    """

    @pytest.mark.parametrize("angle_deg", [0, 45, 90, 135, 180, -45, -90])
    def test_collision_detected_from_angle(self, collision_env, angle_deg):
        """Robot approaching obstacle from ``angle_deg`` must trigger detection.

        The robot is placed 0.5m away from obstacle_0 at the specified angle,
        then driven straight toward it.
        """
        env = collision_env
        env.reset()

        for i in range(8):
            _move_obstacle_far_away(env, f"obstacle_{i}")

        _step_stationary(env, 10)

        robot = env.scene["robot"]
        obstacle = env.scene["obstacle_0"]
        device = env.device
        num_envs = env.num_envs
        env_origins = get_env_origins(env)

        angle_rad = math.radians(angle_deg)

        # Place obstacle at each env's origin
        obs_state = obstacle.data.default_root_state.clone()[:num_envs]
        obs_state[:, 0] = env_origins[:, 0]
        obs_state[:, 1] = env_origins[:, 1]
        obs_state[:, 2] = env_origins[:, 2] + 0.15
        obs_state[:, 3] = 1.0
        obs_state[:, 4:7] = 0.0
        obs_state[:, 7:] = 0.0

        all_ids = torch.arange(num_envs, device=device)
        obstacle.write_root_state_to_sim(obs_state, all_ids)

        # Place robot 0.5m away from obstacle, pointing toward it
        approach_dist = 0.5
        robot_state = robot.data.default_root_state.clone()[:num_envs]
        robot_state[:, 0] = env_origins[:, 0] + approach_dist * math.cos(angle_rad + math.pi)
        robot_state[:, 1] = env_origins[:, 1] + approach_dist * math.sin(angle_rad + math.pi)
        robot_state[:, 2] = env_origins[:, 2] + 0.1

        # Orient robot to face toward the obstacle
        # yaw = angle_rad (pointing from robot position toward obstacle at origin)
        half_yaw = angle_rad / 2.0
        robot_state[:, 3] = math.cos(half_yaw)  # quat w
        robot_state[:, 4] = 0.0                   # quat x
        robot_state[:, 5] = 0.0                   # quat y
        robot_state[:, 6] = math.sin(half_yaw)    # quat z
        robot_state[:, 7:] = 0.0

        robot.write_root_state_to_sim(robot_state, all_ids)

        _step_stationary(env, 5)

        # Drive forward, accumulating detections across all steps
        ever_detected = _drive_and_detect(env, n_steps=40, speed=1.0)
        n_detected = int(ever_detected.sum().item())
        n_missed = env.num_envs - n_detected

        print(f"\n  Collision from {angle_deg} deg:")
        print(f"    Detected: {n_detected}/{env.num_envs}")

        assert n_missed <= MAX_ALLOWED_MISSES, (
            f"Collision from {angle_deg} deg: {n_missed}/{env.num_envs} "
            f"envs missed (max allowed: {MAX_ALLOWED_MISSES}). "
            f"Setup is identical per env — misses indicate a pipeline issue."
        )


# =====================================================================
# Sustained vs Binary: Intensity Scales With Force
# =====================================================================


class TestSustainedVsBinary:
    """Verify that sustained penalty provides a richer signal than binary."""

    def test_sustained_nonzero_during_contact(self, collision_env):
        """Sustained penalty must be > 0 at some step while pushing into obstacle.

        Since the sustained penalty is also instantaneous, we accumulate
        the max value seen across all driving steps.
        """
        env = collision_env
        env.reset()

        for i in range(8):
            _move_obstacle_far_away(env, f"obstacle_{i}")
        _step_stationary(env, 10)
        _place_obstacle_in_front(env, "obstacle_0", distance=0.4)
        _step_stationary(env, 5)

        max_sustained = torch.zeros(env.num_envs, device=env.device)
        action = torch.zeros(env.num_envs, 3, device=env.device)
        action[:, 0] = 1.0
        for _ in range(40):
            env.step(action)
            penalty = collision_sustained_penalty(
                env, sensor_cfg=SENSOR_CFG, threshold=1.0
            )
            max_sustained = torch.maximum(max_sustained, penalty)

        n_detected = int((max_sustained > 0).sum().item())
        n_missed = env.num_envs - n_detected

        print(f"\n  Sustained penalty (max across 40 steps):")
        print(f"    Detected: {n_detected}/{env.num_envs}")
        print(f"    max per env (first 8): {max_sustained[:8].tolist()}")

        assert n_missed <= MAX_ALLOWED_MISSES, (
            f"Sustained penalty never fired in {n_missed}/{env.num_envs} envs "
            f"across 40 steps of driving into an obstacle."
        )
