# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared test utilities for depth noise tests.

This module provides:
- Constants specific to depth noise tests
- Environment creation utilities
- Observation collection helpers
- Theoretical variance calculation functions
- Pixel classification helpers

IMPORTANT: Isaac Sim is launched by the root conftest.py. Do NOT launch
AppLauncher in individual test files or in this module.

The test scene provides:
- A wall at known distance (TEST_WALL_DISTANCE) for non-max pixels
- Open space behind the robot for max-range pixels
- Controlled geometry independent of training scene content

Shared statistical utilities are imported from test.common.
"""

import torch
import numpy as np
from pathlib import Path

# Import shared constants from common module
from test.common.constants import (
    CONFIDENCE_LEVEL,
    NUM_ENVS,
    N_SETTLE_STEPS,
    N_SAMPLES_STEPS,
    DEVICE,
    DEPTH_MAX_RANGE,
    DEPTH_MIN_RANGE,
    D555_BASELINE_M,
    D555_FOCAL_LENGTH_PX,
    D555_DISPARITY_NOISE_PX,
)

# Re-export statistical utilities from common module for use by test files
from test.common.stats import (
    variance_ratio_test,
    binomial_test,
    one_sample_t_test,
    VarianceTestResult,
    BinomialTestResult,
    TTestResult,
)

# =============================================================================
# Test Configuration Constants
# =============================================================================
# Note: NUM_ENVS, N_SETTLE_STEPS, N_SAMPLES_STEPS are imported from common.constants

# Observation term indices for Depth config
# imu_accel(3) + imu_gyro(3) + encoders(4) + goal(2) + action(3) + depth(N)
# Depth starts at index 15 and continues to end
DEPTH_START_IDX = 15

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


def stereo_depth_noise_std(
    depth_m: float,
    baseline_m: float = D555_BASELINE_M,
    focal_length_px: float = D555_FOCAL_LENGTH_PX,
    disparity_noise_px: float = D555_DISPARITY_NOISE_PX,
) -> float:
    """Calculate stereo depth noise standard deviation at a given depth.

    Uses the Intel RealSense stereo error propagation formula:
        σ_z = (z² / (f · B)) · σ_d

    Where:
        z = depth in meters
        f = focal length in pixels (at native camera resolution)
        B = stereo baseline in meters
        σ_d = subpixel disparity noise in pixels
    See: https://openaccess.thecvf.com/content_cvpr_2017_workshops/w15/papers/Keselman_Intel_RealSense_Stereoscopic_CVPR_2017_paper.pdf

    This formula arises from error propagation of the stereo depth equation:
        z = f · B / d  (where d is disparity in pixels)

    Taking the derivative: dz/dd = -z²/(f·B), so σ_z = |dz/dd| · σ_d

    Args:
        depth_m: Depth in meters
        baseline_m: Stereo baseline in meters (default: 0.095m for D555)
        focal_length_px: Focal length in pixels at native resolution (default: 673 for D555)
        disparity_noise_px: Subpixel disparity noise in pixels (default: 0.08)

    Returns:
        Depth noise standard deviation in meters

    Example:
        At 2.0m with D555 defaults:
        σ_z = (2.0² / (673 · 0.095)) · 0.08 = 5.01mm
    """
    return (depth_m ** 2) * disparity_noise_px / (focal_length_px * baseline_m)


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

def compute_geometric_wall_mask(
    height: int,
    width: int,
    focal_length: float,
    horizontal_aperture: float,
    camera_height: float,
    wall_distance: float,
    wall_width: float,
    wall_height: float,
    wall_bottom_z: float = 0.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute a mask of pixels that geometrically hit the wall based on ray casting.

    This provides a precise pixel selection based on camera intrinsics and wall geometry,
    avoiding the need for statistical selection that may include/exclude wrong pixels.

    CAMERA MODEL:
    We use a pinhole camera model where:
    - focal_length and horizontal_aperture define the horizontal FOV
    - The camera looks along -Y axis (ROS convention after rotation)
    - Pixel (0,0) is top-left, (width-1, height-1) is bottom-right

    RAY CASTING:
    For each pixel (u, v), we compute the ray direction and check if it intersects
    the wall plane at Y = -wall_distance (relative to camera).

    Args:
        height: Image height in pixels
        width: Image width in pixels
        focal_length: Camera focal length in mm
        horizontal_aperture: Camera horizontal aperture in mm
        camera_height: Camera height above ground in meters
        wall_distance: Distance from camera to wall surface in meters
        wall_width: Wall width in meters (centered at X=0)
        wall_height: Wall height in meters
        wall_bottom_z: Z coordinate of wall bottom edge in meters
        device: Torch device

    Returns:
        Boolean tensor of shape (height * width,) where True = wall pixel
    """
    # Compute FOV from camera intrinsics
    # horizontal_fov = 2 * atan(horizontal_aperture / (2 * focal_length))
    h_fov = 2 * np.arctan(horizontal_aperture / (2 * focal_length))

    # Vertical FOV based on aspect ratio
    aspect = width / height
    v_fov = 2 * np.arctan(np.tan(h_fov / 2) / aspect)

    # Create pixel coordinate grids (normalized to [-1, 1])
    # u goes from -1 (left) to +1 (right)
    # v goes from -1 (top) to +1 (bottom)
    u = torch.linspace(-1, 1, width, device=device)
    v = torch.linspace(-1, 1, height, device=device)
    vv, uu = torch.meshgrid(v, u, indexing='ij')

    # Convert to ray directions in camera frame
    # Camera looks along +X in world frame (after ROS rotation applied)
    # ROS camera: Z-forward → rotated to align with +X world
    #
    # Ray direction components (unnormalized):
    # - X component: fixed at +1 (camera looks at +X)
    # - Y component: proportional to horizontal pixel offset (left-right)
    # - Z component: proportional to vertical pixel offset (up-down)

    # Half-angle tangents give the extent at unit depth
    tan_h_fov_2 = np.tan(h_fov / 2)
    tan_v_fov_2 = np.tan(v_fov / 2)

    # Ray directions (unnormalized, relative to camera looking at +X)
    ray_x = torch.ones_like(uu)      # Forward direction (+X)
    ray_y = -uu * tan_h_fov_2        # Horizontal spread (negated: image right → world -Y)
    ray_z = -vv * tan_v_fov_2        # Vertical spread (negated: image down → world -Z)

    # Normalize ray directions
    ray_norm = torch.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
    ray_x = ray_x / ray_norm
    ray_y = ray_y / ray_norm
    ray_z = ray_z / ray_norm

    # Ray-plane intersection
    # Wall is at X = +wall_distance (relative to camera at X=0)
    # Ray: P = O + t*D where O is camera origin
    # Plane: X = wall_distance
    # Intersection: t = wall_distance / ray_x

    t = wall_distance / ray_x  # t > 0 since ray_x > 0

    # Intersection points
    hit_y = ray_y * t
    hit_z = camera_height + ray_z * t  # Camera is at height camera_height

    # Check if intersection is within wall bounds
    # Wall is centered at Y=0, width wall_width (along Y axis)
    # Wall is from Z=wall_bottom_z to Z=wall_bottom_z+wall_height
    wall_y_min = -wall_width / 2
    wall_y_max = wall_width / 2
    wall_z_min = wall_bottom_z
    wall_z_max = wall_bottom_z + wall_height

    is_wall = (
        (hit_y >= wall_y_min) & (hit_y <= wall_y_max) &
        (hit_z >= wall_z_min) & (hit_z <= wall_z_max)
    )

    # Flatten to match observation format
    return is_wall.flatten()


def get_geometric_wall_mask(env, device: str = "cuda:0") -> torch.Tensor:
    """Get a geometric wall mask for the test scene using camera intrinsics.

    This is a convenience wrapper around compute_geometric_wall_mask that
    extracts parameters from the environment's camera configuration.

    Args:
        env: The Isaac Lab environment with the test scene
        device: Torch device

    Returns:
        Boolean tensor of shape (height * width,) where True = wall pixel
    """
    # Get camera config
    camera = env.scene["d555_camera"]
    height = camera.cfg.height
    width = camera.cfg.width

    # Get camera intrinsics from spawn config
    spawn_cfg = camera.cfg.spawn
    focal_length = spawn_cfg.focal_length
    horizontal_aperture = spawn_cfg.horizontal_aperture

    # Get camera offset (height above robot body)
    # The camera is mounted at CAMERA_Z_OFFSET above body_link
    # When robot is at z=0.1 (slight elevation), camera is at z=0.1 + offset
    camera_offset = camera.cfg.offset.pos
    camera_z_offset = camera_offset[2]  # Already in meters
    robot_elevation = 0.1  # From reset_robot_pose
    camera_height = robot_elevation + camera_z_offset

    # Get wall geometry from test scene constants
    from test.sensors.depth_noise.scene_cfg import (
        TEST_WALL_DISTANCE, TEST_WALL_WIDTH, TEST_WALL_HEIGHT
    )

    return compute_geometric_wall_mask(
        height=height,
        width=width,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        camera_height=camera_height,
        wall_distance=TEST_WALL_DISTANCE,
        wall_width=TEST_WALL_WIDTH,
        wall_height=TEST_WALL_HEIGHT,
        wall_bottom_z=0.0,  # Wall sits on ground
        device=device,
    )


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

    Uses statistical selection: pixels whose median depth is within tolerance of
    expected_depth AND whose std is below max_std.

    NOTE: For more precise pixel selection, consider using get_geometric_wall_mask()
    which computes the expected wall pixels based on camera intrinsics and wall geometry.

    Args:
        depth_obs: Depth observations tensor of shape (n_steps, depth_dim) for one env
        tolerance: Allowed deviation from expected wall depth (normalized units)
        max_std: Maximum allowed standard deviation (normalized units)
        height: Image height (currently unused, kept for API compatibility)
        width: Image width (currently unused, kept for API compatibility)
        expected_depth: Expected normalized depth of wall pixels

    Returns:
        Boolean mask of shape (depth_dim,) where True = wall pixel
    """
    if max_std is None:
        max_std = tolerance

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
# These functions are imported from the common module for consistency across tests.

from test.common.robot import (
    get_env_origins,
    reset_robot_pose,
    freeze_robot_in_place as _freeze_robot_in_place,
    clear_frozen_state,
)


# =============================================================================
# Observation Collection
# =============================================================================

def reset_noise_models(env):
    """Reset all noise models in the observation manager to clear internal state.

    This is important when repositioning the robot, as noise models like
    DepthNoiseModel maintain internal state (e.g., previous frame buffer)
    that should be cleared when starting a new observation collection.

    Args:
        env: The Isaac Lab environment
    """
    try:
        obs_manager = env.observation_manager
        # Iterate over all observation term groups
        for group_name, terms in obs_manager._group_obs_term_names.items():
            for term_name in terms:
                term = obs_manager._terms[group_name][term_name]
                # Check if the term has a noise model with a reset method
                if hasattr(term, '_noise_model') and term._noise_model is not None:
                    noise_model = term._noise_model
                    if hasattr(noise_model, 'reset'):
                        noise_model.reset()
    except Exception as e:
        # Silently continue if we can't access noise models
        # This is a best-effort reset
        pass


def collect_stationary_observations(
    env, n_steps: int, n_settle_steps: int = N_SETTLE_STEPS, face_wall: bool = True,
    freeze_robot: bool = True
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
        freeze_robot: If True (default), continuously zero out robot velocities during
                      observation collection to eliminate micro-settling noise. This ensures
                      any measured variance comes purely from sensor noise, not robot movement.

    Returns:
        Tensor of shape (n_steps, num_envs, obs_dim)
    """
    # Full environment reset to ensure clean state
    # This clears any episode management state that might interfere with pose setting
    env.reset()

    # Reset robot to known pose with specified orientation
    # Must be called AFTER env.reset() since reset() may randomize positions
    reset_robot_pose(env, face_wall=face_wall)

    # Clear frozen state so it captures fresh pose on next freeze call
    clear_frozen_state()

    # Reset noise models to clear any stale internal state
    reset_noise_models(env)

    # Zero action (stationary)
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)

    # Let physics settle after reset
    for _ in range(n_settle_steps):
        env.step(zero_action)
        if freeze_robot:
            _freeze_robot_in_place(env)

    # Collect observations
    observations = []

    for _ in range(n_steps):
        obs_dict, _, _, _, _ = env.step(zero_action)
        observations.append(obs_dict["policy"].clone())

        if freeze_robot:
            _freeze_robot_in_place(env)

    obs_tensor = torch.stack(observations, dim=0)
    return obs_tensor


# =============================================================================
# Environment Management
# =============================================================================

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
        from test.sensors.depth_noise.scene_cfg import DepthNoiseTestSceneCfg, ENV_SPACING
        cfg.scene = DepthNoiseTestSceneCfg(num_envs=num_envs, env_spacing=ENV_SPACING)

    # Apply the specified noise configuration
    cfg.observations.policy.depth_image.noise = noise_cfg

    env = ManagerBasedRLEnv(cfg)
    env.reset()

    return env


# =============================================================================
# Noise Configuration Factories
# =============================================================================

def create_gaussian_only_noise_cfg(
    baseline_m: float = D555_BASELINE_M,
    focal_length_px: float = D555_FOCAL_LENGTH_PX,
    disparity_noise_px: float = D555_DISPARITY_NOISE_PX,
):
    """Create depth noise config with ONLY stereo Gaussian noise enabled.

    Holes and frame drops are disabled to isolate the Gaussian component.

    Uses Intel RealSense stereo error propagation: σ_z = (z²/(f·B))·σ_d

    Args:
        baseline_m: Stereo baseline in meters (default: 0.095m for D555)
        focal_length_px: Focal length in pixels at native resolution (default: 673)
        disparity_noise_px: Subpixel disparity noise in pixels (default: 0.08)

    Returns:
        DepthNoiseModelCfg with only stereo Gaussian noise enabled
    """
    from strafer_lab.tasks.navigation.mdp.noise_models import DepthNoiseModelCfg
    from isaaclab.utils.noise import GaussianNoiseCfg

    # Compute noise at 1m for informational GaussianNoiseCfg
    noise_at_1m = stereo_depth_noise_std(1.0, baseline_m, focal_length_px, disparity_noise_px)

    return DepthNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=noise_at_1m),
        baseline_m=baseline_m,
        focal_length_px=focal_length_px,
        disparity_noise_px=disparity_noise_px,
        hole_probability=0.0,  # Disabled
        min_range=DEPTH_MIN_RANGE,
        max_range=DEPTH_MAX_RANGE,
        frame_drop_prob=0.0,  # Disabled
    )


def create_holes_only_noise_cfg(hole_probability: float = 0.05):
    """Create depth noise config with ONLY holes enabled.

    Stereo Gaussian noise and frame drops are disabled to isolate hole component.

    Args:
        hole_probability: Probability of a pixel being set to max_range

    Returns:
        DepthNoiseModelCfg with only hole noise enabled
    """
    from strafer_lab.tasks.navigation.mdp.noise_models import DepthNoiseModelCfg
    from isaaclab.utils.noise import GaussianNoiseCfg

    return DepthNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=0.0),  # No Gaussian noise
        baseline_m=D555_BASELINE_M,
        focal_length_px=D555_FOCAL_LENGTH_PX,
        disparity_noise_px=0.0,  # Disabled - no stereo noise
        hole_probability=hole_probability,
        min_range=DEPTH_MIN_RANGE,
        max_range=DEPTH_MAX_RANGE,
        frame_drop_prob=0.0,  # Disabled
    )


def create_frame_drops_with_gaussian_cfg(
    frame_drop_prob: float = 0.10,
    disparity_noise_px: float = D555_DISPARITY_NOISE_PX,
):
    """Create depth noise config with frame drops AND stereo Gaussian noise.

    We need Gaussian noise to make frame drops detectable - without it,
    all frames are identical and drops are invisible.
    Holes are disabled to avoid confounding.

    Args:
        frame_drop_prob: Probability of dropping a frame
        disparity_noise_px: Subpixel disparity noise (controls Gaussian noise magnitude)

    Returns:
        DepthNoiseModelCfg with frame drops and stereo detection noise
    """
    from strafer_lab.tasks.navigation.mdp.noise_models import DepthNoiseModelCfg
    from isaaclab.utils.noise import GaussianNoiseCfg

    # Compute noise at 1m for informational GaussianNoiseCfg
    noise_at_1m = stereo_depth_noise_std(1.0, D555_BASELINE_M, D555_FOCAL_LENGTH_PX, disparity_noise_px)

    return DepthNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=noise_at_1m),
        baseline_m=D555_BASELINE_M,
        focal_length_px=D555_FOCAL_LENGTH_PX,
        disparity_noise_px=disparity_noise_px,
        hole_probability=0.0,  # Disabled
        min_range=DEPTH_MIN_RANGE,
        max_range=DEPTH_MAX_RANGE,
        frame_drop_prob=frame_drop_prob,
    )


def create_no_noise_cfg():
    """Create depth noise config with ALL noise disabled.

    Used to measure baseline render variance from Isaac Sim's depth buffer.
    Any variance observed with this config comes from:
    - Raytracer floating-point precision
    - Robot micro-movements from physics settling
    - Temporal aliasing in the renderer

    Returns:
        DepthNoiseModelCfg with all noise sources disabled
    """
    from strafer_lab.tasks.navigation.mdp.noise_models import DepthNoiseModelCfg
    from isaaclab.utils.noise import GaussianNoiseCfg

    return DepthNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=0.0),  # No base noise
        baseline_m=D555_BASELINE_M,
        focal_length_px=D555_FOCAL_LENGTH_PX,
        disparity_noise_px=0.0,               # No stereo noise
        hole_probability=0.0,                  # No holes
        min_range=DEPTH_MIN_RANGE,
        max_range=DEPTH_MAX_RANGE,
        frame_drop_prob=0.0,                   # No frame drops
    )
