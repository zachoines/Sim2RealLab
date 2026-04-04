# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that collision events produce statistically detectable signals
in IMU and body velocity observations.

The depth camera has a 0.4m near-field blind zone.  Once an obstacle is
closer than 0.4m, the policy must rely on proprioceptive signals to
detect and react to contact.  These tests verify two such signals:

1. **IMU jerk at impact**: The moment of collision produces a sudden change
   in IMU acceleration magnitude (deceleration transient).  We compare
   the maximum per-step IMU change during collision vs during free driving.

2. **Body velocity drop**: During collision the robot's body velocity drops
   to near-zero even though the action commands full forward.  We compare
   body velocity distributions between free driving and collision.

Note: Wheel encoder velocity is NOT a reliable collision signal with
kinematic obstacles — the wheels continue spinning against the immovable
surface.  Body velocity (derived from root state) captures the stall.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/sensors/test_imu_collision.py -v
"""

import torch
import numpy as np
import pytest

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

from test.common import NUM_ENVS, CONFIDENCE_LEVEL, IMU_ACCEL_MAX, BODY_VEL_MAX
from test.common.stats import one_sample_t_test, welch_t_test
from test.common.robot import get_env_origins

import warp as wp
from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg_NoCam,
    ActionsCfg_Ideal,
    ObsCfg_NoCam_Ideal,
)

# Obs layout (NoCam, 19 dims):
# [0:3]   IMU accel (normalized by IMU_ACCEL_MAX)
# [3:6]   IMU gyro
# [6:10]  Encoder velocities
# [10:12] Goal relative
# [12:13] Goal distance
# [13:14] Goal heading
# [14:16] Body velocity XY (normalized by BODY_VEL_MAX)
# [16:19] Last action
IMU_ACCEL_SLICE = slice(0, 3)
BODY_VEL_SLICE = slice(14, 16)


# =====================================================================
# Constants
# =====================================================================

GRAVITY = 9.81  # m/s²
GRAVITY_TOLERANCE = 0.3  # Acceptable deviation from expected gravity


# =====================================================================
# Module-Scoped Environment
# =====================================================================

_module_env = None


def _get_or_create_env():
    """Create a NoCam env with a test obstacle for collision tests."""
    global _module_env

    if _module_env is not None:
        return _module_env

    cfg = StraferNavEnvCfg_NoCam()
    cfg.scene.num_envs = NUM_ENVS
    cfg.actions = ActionsCfg_Ideal()
    cfg.observations = ObsCfg_NoCam_Ideal()
    cfg.commands.goal_command.debug_vis = False

    # Add a single kinematic test obstacle
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

    # Disable resets that would interfere
    if hasattr(cfg.events, "reset_robot"):
        cfg.events.reset_robot = None
    cfg.terminations.robot_flipped = None
    cfg.terminations.sustained_collision = None
    cfg.episode_length_s = 300.0

    _module_env = ManagerBasedRLEnv(cfg)
    _module_env.reset()

    return _module_env


@pytest.fixture(scope="module")
def collision_env():
    """Provide environment configured for collision testing."""
    env = _get_or_create_env()
    yield env


def pytest_sessionfinish(session, exitstatus):
    global _module_env
    if _module_env is not None:
        _module_env.close()
        _module_env = None


# =====================================================================
# Helpers
# =====================================================================


def _place_obstacle_in_front(env, distance: float = 0.4):
    """Place the test obstacle directly ahead of each robot."""
    robot = env.scene["robot"]
    obstacle = env.scene["test_obstacle"]
    device = env.device
    num_envs = env.num_envs

    robot_pos = wp.to_torch(robot.data.root_pos_w)[:, :3].clone()
    robot_quat = wp.to_torch(robot.data.root_quat_w)
    env_origins = get_env_origins(env)

    x, y, z, w = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    root_pose = wp.to_torch(obstacle.data.default_root_pose).clone()[:num_envs]
    root_pose[:, 0] = robot_pos[:, 0] + distance * torch.cos(yaw)
    root_pose[:, 1] = robot_pos[:, 1] + distance * torch.sin(yaw)
    root_pose[:, 2] = env_origins[:, 2] + 0.15
    root_pose[:, 3:6] = 0.0
    root_pose[:, 6] = 1.0

    all_ids = torch.arange(num_envs, device=device)
    obstacle.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=all_ids)
    obstacle.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(num_envs, 6, device=device), env_ids=all_ids
    )


def _move_obstacle_far_away(env):
    """Move the test obstacle far from all robots."""
    obstacle = env.scene["test_obstacle"]
    device = env.device
    num_envs = env.num_envs
    env_origins = get_env_origins(env)

    root_pose = wp.to_torch(obstacle.data.default_root_pose).clone()[:num_envs]
    root_pose[:, 0] = env_origins[:, 0] + 50.0
    root_pose[:, 1] = env_origins[:, 1] + 50.0
    root_pose[:, 2] = env_origins[:, 2] + 0.15
    root_pose[:, 3:6] = 0.0  # quat x, y, z = 0
    root_pose[:, 6] = 1.0    # quat w = 1 (XYZW identity)

    all_ids = torch.arange(num_envs, device=device)
    obstacle.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=all_ids)
    obstacle.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(num_envs, 6, device=device), env_ids=all_ids
    )


def _collect_obs_per_step(
    env, action: torch.Tensor, n_steps: int
) -> list[torch.Tensor]:
    """Step environment and collect raw policy obs each step.

    Returns:
        List of tensors, each (num_envs, obs_dim). Length = n_steps.
    """
    obs_list = []
    for _ in range(n_steps):
        obs_dict, _, _, _, _ = env.step(action)
        obs_list.append(obs_dict["policy"].clone())
    return obs_list


def _imu_magnitudes_from_obs(obs_list: list[torch.Tensor]) -> np.ndarray:
    """Extract IMU acceleration magnitudes from obs list.

    Returns:
        2D array of shape (n_steps, num_envs).
    """
    mags = []
    for obs in obs_list:
        imu_norm = obs[:, IMU_ACCEL_SLICE]  # (num_envs, 3)
        imu_raw = imu_norm * IMU_ACCEL_MAX
        mag = torch.norm(imu_raw, dim=-1)  # (num_envs,)
        mags.append(mag.cpu().numpy())
    return np.stack(mags, axis=0)  # (n_steps, num_envs)


def _body_speed_from_obs(obs_list: list[torch.Tensor]) -> np.ndarray:
    """Extract body speed (XY magnitude) from obs list.

    Returns:
        2D array of shape (n_steps, num_envs).
    """
    speeds = []
    for obs in obs_list:
        bv_norm = obs[:, BODY_VEL_SLICE]  # (num_envs, 2)
        bv_raw = bv_norm * BODY_VEL_MAX
        speed = torch.norm(bv_raw, dim=-1)  # (num_envs,)
        speeds.append(speed.cpu().numpy())
    return np.stack(speeds, axis=0)  # (n_steps, num_envs)


# =====================================================================
# Tests
# =====================================================================


class TestCollisionProprioception:
    """Verify proprioceptive collision signals are statistically
    detectable via observations the policy has access to."""

    def test_baseline_imu_consistent_with_gravity(self, collision_env):
        """Stationary IMU acceleration CI must be consistent with gravity.

        Uses a one-sample t-test to verify the mean IMU magnitude is within
        a physics-motivated tolerance of 9.81 m/s².
        """
        env = collision_env
        env.reset()

        _move_obstacle_far_away(env)
        zero_action = torch.zeros(env.num_envs, 3, device=env.device)

        # Settle
        for _ in range(20):
            env.step(zero_action)

        obs_list = _collect_obs_per_step(env, zero_action, n_steps=50)
        mags = _imu_magnitudes_from_obs(obs_list)
        samples = mags.flatten()

        result = one_sample_t_test(samples, null_value=GRAVITY)
        gravity_error = abs(result.mean - GRAVITY)

        print(f"\n  IMU baseline (stationary):")
        print(f"    N samples: {result.n_samples:,}")
        print(f"    Mean magnitude: {result.mean:.3f} m/s²")
        print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI: [{result.ci_low:.3f}, {result.ci_high:.3f}] m/s²")
        print(f"    Deviation from {GRAVITY}: {gravity_error:.4f} m/s²")

        assert gravity_error < GRAVITY_TOLERANCE, (
            f"Stationary IMU mean ({result.mean:.3f} m/s²) deviates from "
            f"expected gravity ({GRAVITY} m/s²) by {gravity_error:.3f} m/s², "
            f"exceeding tolerance ±{GRAVITY_TOLERANCE}. "
            f"CI: [{result.ci_low:.3f}, {result.ci_high:.3f}]"
        )

    def test_collision_imu_mean_differs_from_free(self, collision_env):
        """IMU acceleration distribution during collision must be statistically
        different from free driving (Welch's t-test, two-sided).

        During free driving, motor forces + vibrations produce high mean IMU
        (~40+ m/s²).  During collision, the body is stuck against the obstacle
        and IMU drops closer to gravity (~25 m/s²), though still noisy from
        the wheels grinding against the surface.

        This test verifies the IMU distribution shift is detectable, confirming
        the policy has a proprioceptive signal that changes during collision.
        """
        env = collision_env
        env.reset()

        drive_action = torch.zeros(env.num_envs, 3, device=env.device)
        drive_action[:, 0] = 1.0

        # --- Free driving: reach cruising speed, then collect steady state ---
        _move_obstacle_far_away(env)
        for _ in range(40):
            env.step(drive_action)  # Reach cruising speed

        free_obs = _collect_obs_per_step(env, drive_action, n_steps=30)
        free_mags = _imu_magnitudes_from_obs(free_obs).flatten()

        # --- Collision: place obstacle, drive into it, collect stuck phase ---
        env.reset()
        _place_obstacle_in_front(env, distance=0.4)
        zero_action = torch.zeros(env.num_envs, 3, device=env.device)
        for _ in range(5):
            env.step(zero_action)  # Let obstacle settle

        # First 15 steps: approach + impact. Last 25 steps: stuck against wall.
        for _ in range(15):
            env.step(drive_action)

        collision_obs = _collect_obs_per_step(env, drive_action, n_steps=25)
        collision_mags = _imu_magnitudes_from_obs(collision_obs).flatten()

        # --- Welch's t-test: two-sided (distributions differ) ---
        result = welch_t_test(free_mags, collision_mags, alternative="two-sided")

        print(f"\n  IMU mean shift on collision (Welch's t-test, two-sided):")
        print(f"    Free driving:  mean={result['mean_a']:.2f}, std={result['std_a']:.2f} m/s² "
              f"(n={result['n_a']})")
        print(f"    Collision:     mean={result['mean_b']:.2f}, std={result['std_b']:.2f} m/s² "
              f"(n={result['n_b']})")
        print(f"    t-statistic: {result['t_statistic']:.2f}")
        print(f"    p-value: {result['p_value']:.2e}")
        print(f"    Cohen's d: {result['cohens_d']:.2f}")
        print(f"    Reject null (distributions equal): {result['reject_null']}")

        assert result["reject_null"], (
            f"IMU distribution not significantly different during collision. "
            f"Free mean: {result['mean_a']:.2f} m/s², "
            f"Collision mean: {result['mean_b']:.2f} m/s², "
            f"p={result['p_value']:.4f} (need < {1 - CONFIDENCE_LEVEL})"
        )

    def test_body_velocity_drops_on_collision(self, collision_env):
        """Body velocity during collision must be statistically lower than
        during free driving (Welch's t-test, one-sided).

        When the robot hits a kinematic obstacle, its body velocity drops
        to near-zero even though the action commands full forward and the
        wheels continue spinning.  This velocity-action mismatch is a
        strong proprioceptive collision signal available to the policy.
        """
        env = collision_env
        env.reset()

        drive_action = torch.zeros(env.num_envs, 3, device=env.device)
        drive_action[:, 0] = 1.0

        # --- Free driving: reach cruising speed, then collect ---
        _move_obstacle_far_away(env)
        for _ in range(40):
            env.step(drive_action)

        free_obs = _collect_obs_per_step(env, drive_action, n_steps=30)
        free_speed = _body_speed_from_obs(free_obs).flatten()

        # --- Collision: drive into obstacle, collect after impact ---
        env.reset()
        _place_obstacle_in_front(env, distance=0.4)
        zero_action = torch.zeros(env.num_envs, 3, device=env.device)
        for _ in range(5):
            env.step(zero_action)

        # Drive into obstacle — first 15 steps include approach + impact,
        # collect the last 25 steps where robot should be stuck
        for _ in range(15):
            env.step(drive_action)

        collision_obs = _collect_obs_per_step(env, drive_action, n_steps=25)
        collision_speed = _body_speed_from_obs(collision_obs).flatten()

        # --- Welch's t-test: free speed > collision speed (one-sided) ---
        result = welch_t_test(free_speed, collision_speed, alternative="greater")

        print(f"\n  Body velocity drop on collision (Welch's t-test):")
        print(f"    Free driving speed:  mean={result['mean_a']:.4f}, std={result['std_a']:.4f} m/s "
              f"(n={result['n_a']})")
        print(f"    Collision speed:     mean={result['mean_b']:.4f}, std={result['std_b']:.4f} m/s "
              f"(n={result['n_b']})")
        print(f"    t-statistic: {result['t_statistic']:.2f}")
        print(f"    p-value: {result['p_value']:.2e}")
        print(f"    Cohen's d: {result['cohens_d']:.2f}")
        print(f"    Reject null (free <= collision): {result['reject_null']}")

        assert result["reject_null"], (
            f"Body velocity drop not statistically significant. "
            f"Free speed: {result['mean_a']:.4f} m/s, "
            f"Collision speed: {result['mean_b']:.4f} m/s, "
            f"p={result['p_value']:.4f} (need < {1 - CONFIDENCE_LEVEL})"
        )

        # Effect size should be large — body fully stops vs cruising
        assert result["cohens_d"] > 0.5, (
            f"Body velocity drop has weak effect size: "
            f"Cohen's d = {result['cohens_d']:.2f} (need > 0.5)"
        )
