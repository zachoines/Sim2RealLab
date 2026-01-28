# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared test utilities for depth noise integration tests.

This module provides:
- Constants shared across all depth noise tests
- Dedicated test scene configuration with controlled geometry
- Environment creation and cleanup utilities
- Observation collection helpers
- Theoretical variance calculation functions

IMPORTANT: Isaac Sim must be launched BEFORE importing Isaac Lab modules.
Each test file must launch the app at module level before importing from here.

The test scene provides:
- A wall at known distance (TEST_WALL_DISTANCE) for non-max pixels
- Open space behind the robot for max-range pixels
- Controlled geometry independent of training scene content
"""

import torch
import numpy as np
from scipy import stats
from pathlib import Path

# =============================================================================
# Test Configuration Constants
# =============================================================================

# Environment settings
# Use single environment to avoid GridCloner layout issues with walls
# Statistical power comes from many timesteps rather than parallel envs
NUM_ENVS = 1
N_SAMPLES_STEPS = 500        # More steps to compensate for single env
N_SETTLE_STEPS = 10          # Steps to let physics settle initially
DEVICE = "cuda:0"

# Statistical thresholds
CONFIDENCE_LEVEL = 0.95      # For hypothesis tests

# Observation term indices for Depth config
# imu_accel(3) + imu_gyro(3) + encoders(4) + goal(2) + action(3) + depth(N)
# Depth starts at index 15 and continues to end
DEPTH_START_IDX = 15

# Depth camera parameters (from REAL_ROBOT_CONTRACT)
DEPTH_MAX_RANGE = 6.0
DEPTH_MIN_RANGE = 0.2

# Test scene geometry
TEST_WALL_DISTANCE = 2.0     # meters - wall in front of robot
TEST_WALL_DEPTH_NORMALIZED = TEST_WALL_DISTANCE / DEPTH_MAX_RANGE  # ~0.333

# Debug output directory
DEBUG_OUTPUT_DIR = Path(__file__).parent / "debug_output"


# =============================================================================
# Debug Utilities
# =============================================================================

def save_depth_image(
    env,
    filename: str = "depth_debug.png",
    face_wall: bool = True,
    annotate: bool = True,
):
    """Capture and save the current depth camera image for debugging.

    Saves both the raw depth values (as .npy) and a visualization (as .png).

    Args:
        env: The Isaac Lab environment
        filename: Base filename (without extension) for output files
        face_wall: If True, orient robot toward wall first. If False, face away.
        annotate: If True, add text annotations showing depth statistics

    Returns:
        Tuple of (depth_tensor, save_path) for further inspection
    """
    import matplotlib.pyplot as plt

    # Ensure output directory exists
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reset robot orientation
    reset_robot_pose(env, face_wall=face_wall)

    # Step simulation to update camera
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(N_SETTLE_STEPS):
        env.step(zero_action)

    # Get observation
    obs_dict, _, _, _, _ = env.step(zero_action)
    obs = obs_dict["policy"]

    # Extract depth (for env 0)
    depth_flat = obs[0, DEPTH_START_IDX:].cpu().numpy()

    # Get camera dimensions from scene config
    camera = env.scene["d555_camera"]
    height = camera.cfg.height
    width = camera.cfg.width

    # Reshape to image
    if len(depth_flat) == height * width:
        depth_img = depth_flat.reshape(height, width)
    else:
        # Fallback: try to infer dimensions
        total_pixels = len(depth_flat)
        # Common aspect ratios
        for h, w in [(60, 80), (48, 64), (120, 160), (240, 320)]:
            if h * w == total_pixels:
                depth_img = depth_flat.reshape(h, w)
                height, width = h, w
                break
        else:
            # Last resort: make it square-ish
            side = int(np.sqrt(total_pixels))
            depth_img = depth_flat[:side*side].reshape(side, side)
            height = width = side

    # Save raw numpy array
    base_name = Path(filename).stem
    npy_path = DEBUG_OUTPUT_DIR / f"{base_name}.npy"
    np.save(npy_path, depth_img)

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Show depth image (normalized values: 0=min_range, 1=max_range)
    im = ax.imshow(depth_img, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Normalized Depth (0=near, 1=far/max_range)')

    # Add statistics
    orientation = "FACING WALL" if face_wall else "FACING AWAY"
    title = f"Depth Camera Debug - {orientation}\n"
    title += f"Shape: {height}x{width}, "
    title += f"Min: {depth_img.min():.3f}, Max: {depth_img.max():.3f}, "
    title += f"Mean: {depth_img.mean():.3f}"
    ax.set_title(title)

    if annotate:
        # Count pixels at different depth ranges
        wall_mask = np.abs(depth_img - TEST_WALL_DEPTH_NORMALIZED) < 0.05
        max_mask = depth_img > 0.99
        near_mask = depth_img < 0.1

        stats_text = (
            f"Wall pixels (~{TEST_WALL_DEPTH_NORMALIZED:.3f}): {wall_mask.sum()}\n"
            f"Max-range pixels (>0.99): {max_mask.sum()}\n"
            f"Near pixels (<0.1): {near_mask.sum()}\n"
            f"Expected wall depth: {TEST_WALL_DEPTH_NORMALIZED:.3f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save figure
    png_path = DEBUG_OUTPUT_DIR / f"{base_name}.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"  Debug image saved to: {png_path}")
    print(f"  Raw depth saved to: {npy_path}")
    print(f"  Depth stats: min={depth_img.min():.3f}, max={depth_img.max():.3f}, mean={depth_img.mean():.3f}")

    return depth_img, png_path


def debug_camera_orientation(env):
    """Save depth images for both orientations to debug camera direction.

    This captures depth images with the robot facing toward and away from the wall,
    helping diagnose if the camera orientation matches expectations.

    Expected results:
    - Facing wall: Most pixels at ~0.333 (wall at 2m / 6m max_range)
    - Facing away: Most pixels at 1.0 (max_range / open space)
    """
    print("\n" + "=" * 60)
    print("DEBUG: Capturing depth images for both orientations")
    print("=" * 60)

    # Print robot and wall info
    robot = env.scene["robot"]
    print(f"\n  Robot root state shape: {robot.data.root_state_w.shape}")
    print(f"  Robot position: {robot.data.root_state_w[0, :3].cpu().numpy()}")
    print(f"  Robot quaternion (w,x,y,z): {robot.data.root_state_w[0, 3:7].cpu().numpy()}")

    # Print wall position if available
    if "test_wall" in env.scene.keys():
        wall = env.scene["test_wall"]
        print(f"  Wall position: {wall.data.root_state_w[0, :3].cpu().numpy()}")

    print("\n1. Robot facing TOWARD wall (expected: wall pixels ~0.333)")
    depth_toward, path_toward = save_depth_image(
        env, "debug_facing_wall", face_wall=True
    )

    # Print robot pose after reset
    print(f"  After reset - Robot quaternion: {robot.data.root_state_w[0, 3:7].cpu().numpy()}")

    print("\n2. Robot facing AWAY from wall (expected: max-range pixels ~1.0)")
    depth_away, path_away = save_depth_image(
        env, "debug_facing_away", face_wall=False
    )

    print(f"  After reset - Robot quaternion: {robot.data.root_state_w[0, 3:7].cpu().numpy()}")

    print("\n" + "=" * 60)
    print("DEBUG SUMMARY:")
    print(f"  Facing wall - mean depth: {depth_toward.mean():.3f} (expected ~0.333)")
    print(f"  Facing away - mean depth: {depth_away.mean():.3f} (expected ~1.0)")

    if depth_toward.mean() > 0.9 and depth_away.mean() < 0.5:
        print("\n  WARNING: Results are INVERTED!")
        print("  Camera may be facing opposite to expected direction.")
        print("  Consider swapping the quaternion in reset_robot_pose().")
    elif depth_toward.mean() > 0.8:
        print("\n  WARNING: Not seeing wall when facing toward it.")
        print("  Check camera offset and wall position in test_scene_cfg.py.")

    print("=" * 60 + "\n")

    return depth_toward, depth_away


# =============================================================================
# Theoretical Variance Calculations
# =============================================================================

def clamped_normal_variance(z: float) -> float:
    """Compute variance of a clamped standard normal: Y = min(X, z) where X ~ N(0, 1).

    This models the effect of clamping depth values at max_range. When noise pushes
    a depth reading above max_range, the value is clamped, reducing the effective
    variance of the noise.

    DERIVATION:
    -----------
    Let X ~ N(0, 1) and Y = min(X, z). We need Var(Y) = E[Y^2] - E[Y]^2.

    For Y = min(X, z):
      - If X < z: Y = X (occurs with probability Phi(z))
      - If X >= z: Y = z (occurs with probability 1 - Phi(z))

    E[Y] = integral_{-inf}^{z} x * phi(x) dx + z * P(X >= z)
         = -phi(z) + z * (1 - Phi(z))

    E[Y^2] = integral_{-inf}^{z} x^2 * phi(x) dx + z^2 * (1 - Phi(z))
           = Phi(z) - z * phi(z) + z^2 * (1 - Phi(z))

    Therefore: Var(Y) = E[Y^2] - E[Y]^2

    KEY VALUES:
    -----------
    - z = 0 (mean at boundary): Var(Y) = 0.3408
      This is our case! When depth = max_range, noise can only decrease the reading.
    - z = 1: Var(Y) = 0.7511
    - z = 2: Var(Y) = 0.9602
    - z -> inf: Var(Y) -> 1.0 (no clamping effect)

    Args:
        z: Distance from mean to clamp boundary in standard deviations.
           z = (boundary - mean) / std = (max_range - depth) / noise_std

    Returns:
        Variance of the clamped distribution (between 0 and 1 for standard normal input).
    """
    from scipy.stats import norm

    if z > 5:
        # Far from boundary: clamping has negligible effect
        return 1.0
    if z < -3:
        # Mean is well above boundary: severe clamping, most values get clamped
        return 0.01

    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)

    # E[Y] = -phi(z) + z * (1 - Phi(z))
    E_Y = -phi_z + z * (1 - Phi_z)

    # E[Y^2] = Phi(z) - z * phi(z) + z^2 * (1 - Phi(z))
    E_Y_sq = Phi_z - z * phi_z + z**2 * (1 - Phi_z)

    # Var(Y) = E[Y^2] - E[Y]^2
    var_Y = E_Y_sq - E_Y**2

    return var_Y


def hole_diff_variance(mean_depth_normalized: float, hole_probability: float) -> float:
    """Calculate expected variance of first-differences when only holes are present.

    DERIVATION:
    With only holes enabled (no Gaussian noise), at each timestep:
        y[t] = d  (true depth, normalized) with probability (1 - p)
        y[t] = 1  (hole = max_range normalized) with probability p

    First differences y[t] - y[t-1]:
        - Both normal:    diff = 0,     P = (1-p)^2
        - Both holes:     diff = 0,     P = p^2
        - Normal -> Hole: diff = 1 - d, P = (1-p) * p
        - Hole -> Normal: diff = d - 1, P = p * (1-p)

    E[diff] = 0 (symmetric)

    Var(diff) = E[diff^2]
              = 2 * p * (1-p) * (1 - d)^2

    Args:
        mean_depth_normalized: Mean depth in [0, 1] normalized space
        hole_probability: Probability of a pixel being a hole

    Returns:
        Expected variance of first-differences from holes alone
    """
    if hole_probability <= 0:
        return 0.0

    p = hole_probability
    d = mean_depth_normalized

    # Probability of a transition in either direction
    p_transition = 2 * p * (1 - p)

    # Jump magnitude when transitioning
    jump_size = abs(1.0 - d)

    return p_transition * (jump_size ** 2)


# =============================================================================
# Pixel Classification Helpers
# =============================================================================

def identify_wall_pixels(
    depth_obs: torch.Tensor,
    tolerance: float = 0.05,
    max_std: float = None,
    *,
    height: int | None = None,
    width: int | None = None,
    expected_depth: float = TEST_WALL_DEPTH_NORMALIZED,
) -> torch.Tensor:
    """Identify pixels that consistently show the wall (not max-range or edge pixels).

    Geometry-aware if height/width are provided: uses known wall size/pose and camera
    pose to compute the min/max wall depth band (with tolerance margin) and selects
    pixels whose median/min depth fall within that band. Falls back to statistical
    selection if the geometry band yields no pixels.

    Statistical (no height/width): median +- tolerance and std < max_std on flattened depth.
    """
    if max_std is None:
        max_std = tolerance

    if height is not None and width is not None:
        from test_scene_cfg import TEST_WALL_WIDTH, TEST_WALL_HEIGHT, TEST_WALL_THICKNESS, CAMERA_Z_OFFSET, CAMERA_Y_OFFSET

        # Camera pose (world) when reset_robot_pose(face_wall=True) is used
        # Robot origin at (0,0,0.1), camera offset (0, CAMERA_Y_OFFSET, CAMERA_Z_OFFSET)
        # CAMERA_Y_OFFSET is -0.15 (camera 15cm forward in -Y direction)
        cam_pos = torch.tensor([0.0, CAMERA_Y_OFFSET, 0.1 + CAMERA_Z_OFFSET], device=depth_obs.device)

        # Wall AABB in world (front face toward +Y of camera looking -Y)
        # Wall center Y = CAMERA_Y_OFFSET - TEST_WALL_DISTANCE - TEST_WALL_THICKNESS/2
        wall_xmin = -TEST_WALL_WIDTH / 2
        wall_xmax = TEST_WALL_WIDTH / 2
        wall_y_center = CAMERA_Y_OFFSET - TEST_WALL_DISTANCE - TEST_WALL_THICKNESS / 2
        wall_ymin = wall_y_center - TEST_WALL_THICKNESS / 2
        wall_ymax = wall_y_center + TEST_WALL_THICKNESS / 2
        wall_zmin = 0.0
        wall_zmax = TEST_WALL_HEIGHT

        # Min distance from camera to box
        nearest = torch.tensor([
            torch.clamp(cam_pos[0], wall_xmin, wall_xmax),
            torch.clamp(cam_pos[1], wall_ymin, wall_ymax),
            torch.clamp(cam_pos[2], wall_zmin, wall_zmax),
        ], device=depth_obs.device)
        min_dist = torch.linalg.norm(cam_pos - nearest)

        # Max distance: farthest corner
        corners = torch.tensor([
            [wall_xmin, wall_ymin, wall_zmin],
            [wall_xmin, wall_ymin, wall_zmax],
            [wall_xmin, wall_ymax, wall_zmin],
            [wall_xmin, wall_ymax, wall_zmax],
            [wall_xmax, wall_ymin, wall_zmin],
            [wall_xmax, wall_ymin, wall_zmax],
            [wall_xmax, wall_ymax, wall_zmin],
            [wall_xmax, wall_ymax, wall_zmax],
        ], device=depth_obs.device)
        max_dist = torch.linalg.norm(corners - cam_pos, dim=1).max()

        # Normalize distance band (use tolerance to set margin in meters)
        margin_m = tolerance * DEPTH_MAX_RANGE
        depth_min_norm = max((min_dist - margin_m) / DEPTH_MAX_RANGE, 0.0)
        depth_max_norm = min((max_dist + margin_m) / DEPTH_MAX_RANGE, 1.0)

        depth_img = depth_obs.reshape(-1, height, width)
        median_img = depth_img.median(dim=0).values
        min_img = depth_img.min(dim=0).values

        wall_mask_img = (
            (median_img > depth_min_norm)
            & (median_img < depth_max_norm)
            & (min_img > depth_min_norm)
            & (min_img < depth_max_norm)
        )

        wall_mask_flat = wall_mask_img.flatten()
        if wall_mask_flat.any():
            return wall_mask_flat
        # Fallback to statistical mask if geometry filter is too strict
        print("    [identify_wall_pixels] geometry mask empty; falling back to statistical mask")

    # Statistical mask on flattened depth
    median_depth = depth_obs.median(dim=0).values
    std_depth = depth_obs.std(dim=0)

    is_near_wall = (median_depth - expected_depth).abs() < tolerance
    is_stable = std_depth < max_std

    return is_near_wall & is_stable

def identify_stable_pixels(depth_obs: torch.Tensor, max_std: float = 0.02) -> torch.Tensor:
    """Identify pixels with stable (low variance) depth readings.

    Stable pixels have consistent depth readings, indicating they hit a flat
    surface without edge effects or rendering artifacts.

    Args:
        depth_obs: Depth observations tensor of shape (n_steps, depth_dim) for one env
        max_std: Maximum allowed standard deviation (normalized units)

    Returns:
        Boolean mask of shape (depth_dim,) where True = stable pixel
    """
    std_depth = depth_obs.std(dim=0)
    return std_depth < max_std


def identify_wall_pixels_with_holes(depth_obs: torch.Tensor, tolerance: float = 0.05) -> torch.Tensor:
    """Identify wall pixels in the presence of hole noise.

    With hole noise, wall pixels oscillate between:
    - Wall depth (~0.333 normalized) when no hole
    - Max range (1.0 normalized) when a hole occurs

    We identify wall pixels by their MINIMUM depth, which reveals the true
    wall depth (when not a hole). This is different from identify_wall_pixels()
    which uses median and requires low std.

    Args:
        depth_obs: Depth observations tensor of shape (n_steps, depth_dim) for one env
        tolerance: Allowed deviation from expected wall depth (normalized units)

    Returns:
        Boolean mask of shape (depth_dim,) where True = wall pixel
    """
    expected_wall_depth = TEST_WALL_DEPTH_NORMALIZED

    # Use minimum depth - this shows the true depth when not a hole
    min_depth = depth_obs.min(dim=0).values

    # Wall pixels: minimum depth near expected wall depth
    is_wall_pixel = (
        (min_depth > expected_wall_depth - tolerance) &
        (min_depth < expected_wall_depth + tolerance)
    )

    return is_wall_pixel


# =============================================================================
# Robot Pose Control
# =============================================================================

def reset_robot_pose(env, face_wall: bool = True):
    """Reset the robot to a known pose.

    Args:
        env: The Isaac Lab environment
        face_wall: If True, camera faces the wall (-Y direction).
                   If False, camera faces away from wall (+Y direction) to see max-range.

    This ensures consistent camera orientation regardless of how
    Isaac Lab's GridCloner initially placed the environment.
    """
    robot = env.scene["robot"]

    # Get current root state shape
    num_envs = env.num_envs
    device = env.device

    # Create desired root state
    # Root state format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
    root_state = torch.zeros(num_envs, 13, device=device)

    # Position: (0, 0, 0.1) - slightly above ground
    root_state[:, 0] = 0.0   # x
    root_state[:, 1] = 0.0   # y
    root_state[:, 2] = 0.1   # z

    if face_wall:
        # Identity quaternion (w, x, y, z) = (1, 0, 0, 0)
        # Robot forward is -Y in USD local frame, which aligns with world -Y
        # where the wall is located. Camera rotation on the robot handles
        # pointing the camera in the robot's forward direction.
        root_state[:, 3] = 1.0   # quat_w
        root_state[:, 4] = 0.0   # quat_x
        root_state[:, 5] = 0.0   # quat_y
        root_state[:, 6] = 0.0   # quat_z
    else:
        # 180° rotation around Z axis: (w, x, y, z) = (0, 0, 0, 1)
        # Robot forward (-Y local) now points to world +Y (away from wall)
        root_state[:, 3] = 0.0   # quat_w
        root_state[:, 4] = 0.0   # quat_x
        root_state[:, 5] = 0.0   # quat_y
        root_state[:, 6] = 1.0   # quat_z

    # Velocities: all zero (stationary)
    # root_state[:, 7:13] already zero

    # Write root state to simulation
    robot.write_root_state_to_sim(root_state)

    # Also reset joint positions and velocities to zero
    joint_pos = torch.zeros(num_envs, robot.num_joints, device=device)
    joint_vel = torch.zeros(num_envs, robot.num_joints, device=device)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


# =============================================================================
# Observation Collection
# =============================================================================

def collect_stationary_observations(
    env, n_steps: int, n_settle_steps: int = N_SETTLE_STEPS, face_wall: bool = True
) -> torch.Tensor:
    """Collect observations from a stationary robot over multiple steps.

    With zero actions and settled physics, any observation variance
    comes from sensor noise.

    Args:
        env: The Isaac Lab environment
        n_steps: Number of observation steps to collect
        n_settle_steps: Number of steps to let physics settle before collecting
        face_wall: If True (default), robot faces the wall (-Y direction).
                   If False, robot faces away from wall (+Y direction) to see max-range.

    Returns:
        Tensor of shape (n_steps, num_envs, obs_dim)
    """
    # Reset robot to known pose with specified orientation
    reset_robot_pose(env, face_wall=face_wall)

    # Zero action (stationary)
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)

    # Let physics settle after reset
    for _ in range(n_settle_steps):
        env.step(zero_action)

    # Collect observations
    observations = []
    for _ in range(n_steps):
        obs_dict, _, _, _, _ = env.step(zero_action)
        observations.append(obs_dict["policy"].clone())

    return torch.stack(observations, dim=0)


def collect_stationary_observations_facing_away(
    env, n_steps: int, n_settle_steps: int = N_SETTLE_STEPS
) -> torch.Tensor:
    """Collect observations with robot facing away from wall (for max-range tests).

    This is a convenience wrapper around collect_stationary_observations that
    faces the robot away from the wall so camera sees open space (max-range).

    Args:
        env: The Isaac Lab environment
        n_steps: Number of observation steps to collect
        n_settle_steps: Number of steps to let physics settle before collecting

    Returns:
        Tensor of shape (n_steps, num_envs, obs_dim)
    """
    return collect_stationary_observations(env, n_steps, n_settle_steps, face_wall=False)


# =============================================================================
# Environment Management
# =============================================================================

# Module-level environment storage
# Each test file manages its own _module_env to avoid cross-file interference
_module_env = None
_simulation_app = None


def set_simulation_app(app):
    """Store reference to the simulation app for cleanup."""
    global _simulation_app
    _simulation_app = app


def get_simulation_app():
    """Get the stored simulation app reference."""
    return _simulation_app


def create_depth_test_env(noise_cfg, num_envs: int = NUM_ENVS, use_test_scene: bool = False):
    """Create a depth camera test environment with specified noise configuration.

    Args:
        noise_cfg: The DepthNoiseModelCfg to apply to the depth observation
        num_envs: Number of environments to create
        use_test_scene: If True, use dedicated test scene with wall at known distance.
                        If False (default), use standard training scene.

    Returns:
        ManagerBasedRLEnv configured for depth noise testing

    Note:
        The test scene (use_test_scene=True) provides controlled geometry:
        - A wall at TEST_WALL_DISTANCE (2.0m) in front of the robot
        - Open space for max-range pixels
        - Independent of training scene content changes

        The standard scene (use_test_scene=False) uses the current training
        scene, which may change as assets are added for sim-to-real training.
    """
    # These imports require Isaac Sim to be running
    from isaaclab.envs import ManagerBasedRLEnv
    from strafer_lab.tasks.navigation.strafer_env_cfg import (
        StraferNavEnvCfg_Depth,
        ActionsCfg_Ideal,
        ObsCfg_Depth_Realistic,
    )

    cfg = StraferNavEnvCfg_Depth()
    cfg.scene.num_envs = num_envs
    cfg.actions = ActionsCfg_Ideal()  # Ideal actions for predictability
    cfg.observations = ObsCfg_Depth_Realistic()

    # Optionally use dedicated test scene with controlled geometry
    if use_test_scene:
        from test_scene_cfg import DepthNoiseTestSceneCfg, ENV_SPACING
        cfg.scene = DepthNoiseTestSceneCfg(num_envs=num_envs, env_spacing=ENV_SPACING)

    # Apply the specified noise configuration
    cfg.observations.policy.depth_image.noise = noise_cfg

    env = ManagerBasedRLEnv(cfg)
    env.reset()

    return env


def cleanup_env(env):
    """Clean up an environment instance."""
    if env is not None:
        env.close()


def cleanup_simulation():
    """Clean up the simulation app (call at session end)."""
    global _simulation_app
    if _simulation_app is not None:
        _simulation_app.close()
        _simulation_app = None


# =============================================================================
# Noise Configuration Factories
# =============================================================================

def create_gaussian_only_noise_cfg(
    base_noise_std: float = 0.001,
    depth_noise_coeff: float = 0.002,
):
    """Create depth noise config with ONLY Gaussian noise enabled.

    Holes and frame drops are disabled to isolate Gaussian component.

    Args:
        base_noise_std: Base noise standard deviation in meters
        depth_noise_coeff: Depth-dependent noise coefficient (m/m)

    Returns:
        DepthNoiseModelCfg with only Gaussian noise enabled
    """
    from strafer_lab.tasks.navigation.mdp.noise_models import DepthNoiseModelCfg
    from isaaclab.utils.noise import GaussianNoiseCfg

    return DepthNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=base_noise_std),
        base_noise_std=base_noise_std,
        depth_noise_coeff=depth_noise_coeff,
        hole_probability=0.0,  # Disabled
        min_range=DEPTH_MIN_RANGE,
        max_range=DEPTH_MAX_RANGE,
        frame_drop_prob=0.0,  # Disabled
    )


def create_holes_only_noise_cfg(hole_probability: float = 0.05):
    """Create depth noise config with ONLY holes enabled.

    Gaussian noise and frame drops are disabled to isolate hole component.

    Args:
        hole_probability: Probability of a pixel being set to max_range

    Returns:
        DepthNoiseModelCfg with only hole noise enabled
    """
    from strafer_lab.tasks.navigation.mdp.noise_models import DepthNoiseModelCfg
    from isaaclab.utils.noise import GaussianNoiseCfg

    return DepthNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=0.0),  # No Gaussian noise
        base_noise_std=0.0,
        depth_noise_coeff=0.0,
        hole_probability=hole_probability,
        min_range=DEPTH_MIN_RANGE,
        max_range=DEPTH_MAX_RANGE,
        frame_drop_prob=0.0,  # Disabled
    )


def create_frame_drops_with_gaussian_cfg(
    frame_drop_prob: float = 0.10,
    base_noise_std: float = 0.005,
):
    """Create depth noise config with frame drops AND small Gaussian noise.

    We need Gaussian noise to make frame drops detectable - without it,
    all frames are identical and drops are invisible.
    Holes are disabled to avoid confounding.

    Args:
        frame_drop_prob: Probability of dropping a frame
        base_noise_std: Small Gaussian noise to make drops detectable

    Returns:
        DepthNoiseModelCfg with frame drops and detection noise
    """
    from strafer_lab.tasks.navigation.mdp.noise_models import DepthNoiseModelCfg
    from isaaclab.utils.noise import GaussianNoiseCfg

    return DepthNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=base_noise_std),
        base_noise_std=base_noise_std,
        depth_noise_coeff=0.0,  # No depth-dependent noise for simplicity
        hole_probability=0.0,  # Disabled
        min_range=DEPTH_MIN_RANGE,
        max_range=DEPTH_MAX_RANGE,
        frame_drop_prob=frame_drop_prob,
    )
