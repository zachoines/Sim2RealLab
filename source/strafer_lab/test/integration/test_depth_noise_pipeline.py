# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for depth camera noise through the full environment pipeline.

These tests validate that depth sensor noise is correctly applied during environment
stepping, testing the full pipeline: RAW meters → noise → scale → clip.

Note: These tests require cameras to be enabled, which is a different runtime
configuration than the proprioceptive-only noise tests.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/integration/test_depth_noise_pipeline.py -v
"""

# Isaac Sim must be launched BEFORE importing Isaac Lab modules
# Enable cameras for depth sensor tests
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

# --- Imports that require Isaac Sim runtime ---

import torch
import numpy as np
import pytest
from scipy import stats

from isaaclab.envs import ManagerBasedRLEnv

from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg_Depth,
    ActionsCfg_Ideal,
    ObsCfg_Depth_Ideal,
    ObsCfg_Depth_Realistic,
    DEPTH_MAX,
)
from strafer_lab.tasks.navigation.sim_real_cfg import (
    REAL_ROBOT_CONTRACT,
    get_depth_noise,
)


# =============================================================================
# Test Configuration
# =============================================================================

NUM_ENVS = 8                 # Fewer envs for camera tests (memory intensive)
N_SAMPLES_STEPS = 500        # Steps to collect (more samples = tighter CI)
N_SETTLE_STEPS = 10          # Steps to let physics settle initially
DEVICE = "cuda:0"

# Statistical thresholds
# With n=500 samples, 95% CI for std is roughly ±6.6% from sampling alone
# We use hypothesis testing rather than arbitrary tolerance
CONFIDENCE_LEVEL = 0.95      # For confidence intervals

# Observation term indices for Depth config
# imu_accel(3) + imu_gyro(3) + encoders(4) + goal(2) + action(3) + depth(N)
# Depth starts at index 15 and continues to end
DEPTH_START_IDX = 15


# =============================================================================
# Depth Camera Noise Theoretical Bounds
# =============================================================================

_DEPTH_NOISE_CFG = get_depth_noise(REAL_ROBOT_CONTRACT)

# Depth noise config values
if _DEPTH_NOISE_CFG:
    DEPTH_BASE_NOISE_STD = _DEPTH_NOISE_CFG.base_noise_std  # meters
    DEPTH_NOISE_COEFF = _DEPTH_NOISE_CFG.depth_noise_coeff  # meters per meter
    DEPTH_HOLE_PROB = _DEPTH_NOISE_CFG.hole_probability
    DEPTH_FRAME_DROP_PROB = _DEPTH_NOISE_CFG.frame_drop_prob
    DEPTH_MIN_RANGE = _DEPTH_NOISE_CFG.min_range
    DEPTH_MAX_RANGE = _DEPTH_NOISE_CFG.max_range
else:
    DEPTH_BASE_NOISE_STD = 0.0
    DEPTH_NOISE_COEFF = 0.0
    DEPTH_HOLE_PROB = 0.0
    DEPTH_FRAME_DROP_PROB = 0.0
    DEPTH_MIN_RANGE = 0.2
    DEPTH_MAX_RANGE = 6.0


def _calculate_depth_theoretical_std(
    depth_m: float,
    base_noise_std: float,
    depth_noise_coeff: float,
    max_range: float,
) -> float:
    """Calculate theoretical std for depth noise at a given depth in NORMALIZED space.
    
    Pipeline: raw obs (meters) → noise (in raw space) → scale (1/max) → output
    
    Noise is applied to RAW data, then scaled:
        noise_std_raw = base_noise_std + depth_noise_coeff * depth
        noise_std_normalized = noise_std_raw / max_range
    
    Args:
        depth_m: Depth in meters
        base_noise_std: Base noise std in meters
        depth_noise_coeff: Noise increase per meter depth
        max_range: Max depth for normalization
        
    Returns:
        Theoretical std of depth noise in normalized space
    """
    raw_std = base_noise_std + depth_noise_coeff * depth_m
    return raw_std / max_range


# =============================================================================
# Module-scoped Fixtures
# =============================================================================

_module_env = None


def _get_or_create_depth_env(use_noise: bool = True):
    """Get or create the shared depth test environment.
    
    Args:
        use_noise: If True, use Realistic config with noise enabled.
                   If False, use Ideal config without noise.
    """
    global _module_env
    
    if _module_env is not None:
        return _module_env
    
    cfg = StraferNavEnvCfg_Depth()
    cfg.scene.num_envs = NUM_ENVS
    cfg.actions = ActionsCfg_Ideal()  # Always ideal actions for predictability
    
    if use_noise:
        cfg.observations = ObsCfg_Depth_Realistic()
    else:
        cfg.observations = ObsCfg_Depth_Ideal()
    
    _module_env = ManagerBasedRLEnv(cfg)
    _module_env.reset()
    
    return _module_env


@pytest.fixture(scope="module")
def noisy_depth_env():
    """Provide Strafer environment with depth camera and realistic noise enabled."""
    env = _get_or_create_depth_env(use_noise=True)
    yield env


def pytest_sessionfinish(session, exitstatus):
    """Clean up the environment after all tests complete."""
    global _module_env
    if _module_env is not None:
        _module_env.close()
        _module_env = None
    simulation_app.close()


# =============================================================================
# Helper Functions
# =============================================================================

def collect_stationary_observations(env, n_steps: int) -> torch.Tensor:
    """Collect observations from a stationary robot over multiple steps.
    
    With zero actions and settled physics, any observation variance
    comes from sensor noise.
    
    Returns:
        Tensor of shape (n_steps, num_envs, obs_dim)
    """
    # Zero action (stationary)
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)
    
    # Let physics settle
    for _ in range(N_SETTLE_STEPS):
        env.step(zero_action)
    
    # Collect observations
    observations = []
    for _ in range(n_steps):
        obs_dict, _, _, _, _ = env.step(zero_action)
        observations.append(obs_dict["policy"].clone())
    
    return torch.stack(observations, dim=0)


# =============================================================================
# Tests: Depth Camera Noise
# =============================================================================

class TestDepthNoiseFromEnv:
    """Test depth camera noise characteristics from environment observations.
    
    These tests validate the full depth noise pipeline:
    1. depth_image() returns RAW meters
    2. DepthNoiseModel applies noise in RAW space
    3. ObsTerm.scale normalizes to [0, 1]
    4. Output is clipped to valid range
    """
    
    def test_depth_noise_is_present(self, noisy_depth_env):
        """Verify depth observations have temporal variance (noise is applied).
        
        We measure variance within a SINGLE environment over time to avoid
        confounding from scene differences between environments.
        """
        obs = collect_stationary_observations(noisy_depth_env, N_SAMPLES_STEPS)
        
        # Depth starts at index 15, analyze env 0 only
        depth_obs_env0 = obs[:, 0, DEPTH_START_IDX:]  # (n_steps, depth_dim)
        
        # Compute temporal variance per pixel, then average
        # This measures how much each pixel varies over time (should be noise)
        temporal_var_per_pixel = depth_obs_env0.var(dim=0)  # (depth_dim,)
        mean_temporal_var = temporal_var_per_pixel.mean().item()
        
        print(f"\n  Depth observation shape (env 0): {depth_obs_env0.shape}")
        print(f"  Mean temporal variance per pixel: {mean_temporal_var:.6f}")
        
        # With noise enabled, variance should be non-zero
        assert mean_temporal_var > 1e-8, \
            f"Depth temporal variance too low ({mean_temporal_var:.2e}), noise may not be applied"
    
    def test_depth_noise_std_reasonable(self, noisy_depth_env):
        """Verify depth noise std matches theoretical prediction using chi-squared test.
        
        Depth noise is depth-dependent: noise_std = base_std + coeff * depth
        
        We use FIRST DIFFERENCES to cleanly measure noise variance:
            Δy[t] = y[t] - y[t-1]
            Var(Δy) = 2 * noise_var  (for independent noise at each timestep)
        
        Statistical approach:
        - For Gaussian noise, (n-1) * S² / σ² ~ χ²(n-1)
        - We compute a confidence interval for our measured std
        - Test passes if theoretical std falls within this CI
        - This is more rigorous than arbitrary tolerance bounds
        
        Note: We analyze a SINGLE environment to avoid confounding from
        scene differences between environments.
        """
        obs = collect_stationary_observations(noisy_depth_env, N_SAMPLES_STEPS)
        
        # Extract depth observations for ENV 0 ONLY to avoid cross-env scene differences
        depth_obs_env0 = obs[:, 0, DEPTH_START_IDX:]  # (n_steps, depth_dim)
        
        # Compute first differences: y[t] - y[t-1]
        first_diffs = depth_obs_env0[1:] - depth_obs_env0[:-1]  # (n_steps-1, depth_dim)
        
        # For independent noise: Var(Δy) = 2 * noise_var
        # Measure variance of first-diffs (flattened across all pixels and timesteps)
        diff_var = first_diffs.var().item()
        measured_noise_var = diff_var / 2.0
        measured_noise_std = np.sqrt(measured_noise_var)
        
        # Get mean depth to estimate expected noise level
        # Depth is normalized: actual_depth = observed * max_range
        mean_normalized_depth = depth_obs_env0.mean().item()
        mean_depth_m = mean_normalized_depth * DEPTH_MAX_RANGE
        
        # Theoretical std at mean depth (in normalized space)
        expected_std = _calculate_depth_theoretical_std(
            mean_depth_m,
            DEPTH_BASE_NOISE_STD,
            DEPTH_NOISE_COEFF,
            DEPTH_MAX_RANGE,
        )
        expected_var = expected_std ** 2
        
        # Compute confidence interval using chi-squared distribution
        # For first-diffs: we have (n_steps - 1) differences
        # The sample variance has df = n - 1 degrees of freedom
        n_samples = first_diffs.numel()  # total number of first-diff samples
        df = n_samples - 1
        
        # CI for variance: [S² * df / χ²_upper, S² * df / χ²_lower]
        alpha = 1 - CONFIDENCE_LEVEL
        chi2_lower = stats.chi2.ppf(alpha / 2, df)
        chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)
        
        # Note: We're measuring diff_var, and noise_var = diff_var / 2
        # CI for noise_var
        noise_var_ci_lower = (diff_var * df / chi2_upper) / 2.0
        noise_var_ci_upper = (diff_var * df / chi2_lower) / 2.0
        noise_std_ci_lower = np.sqrt(noise_var_ci_lower)
        noise_std_ci_upper = np.sqrt(noise_var_ci_upper)
        
        # Account for holes: they cause large jumps that inflate variance
        # If holes are enabled, the upper bound should be relaxed
        expected_std_upper = expected_std * 2.0 if DEPTH_HOLE_PROB > 0 else expected_std
        
        print(f"\n  Depth noise analysis (chi-squared test, env 0 only):")
        print(f"    Config values:")
        print(f"      base_noise_std: {DEPTH_BASE_NOISE_STD:.4f} m")
        print(f"      depth_noise_coeff: {DEPTH_NOISE_COEFF:.4f} m/m")
        print(f"      hole_probability: {DEPTH_HOLE_PROB:.4f}")
        print(f"      max_range: {DEPTH_MAX_RANGE:.1f} m")
        print(f"    Scene analysis:")
        print(f"      Mean normalized depth: {mean_normalized_depth:.3f}")
        print(f"      Mean actual depth: {mean_depth_m:.2f} m")
        print(f"    Statistical analysis (n={n_samples:,} samples, df={df:,}):")
        print(f"      First-diff variance: {diff_var:.8f}")
        print(f"      Measured noise std: {measured_noise_std:.6f}")
        print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI for noise std: [{noise_std_ci_lower:.6f}, {noise_std_ci_upper:.6f}]")
        print(f"      Expected noise std: {expected_std:.6f}")
        print(f"      Expected range (with holes): [{expected_std:.6f}, {expected_std_upper:.6f}]")
        
        # Test: Does the theoretical value fall within our measured CI?
        # Or equivalently: does our CI overlap with the expected range?
        ci_overlaps_expected = (
            noise_std_ci_lower <= expected_std_upper and 
            noise_std_ci_upper >= expected_std
        )
        
        print(f"    Result: CI {'overlaps' if ci_overlaps_expected else 'does NOT overlap'} expected range")
        
        assert ci_overlaps_expected, (
            f"Measured noise std {CONFIDENCE_LEVEL*100:.0f}% CI [{noise_std_ci_lower:.6f}, {noise_std_ci_upper:.6f}] "
            f"does not overlap expected range [{expected_std:.6f}, {expected_std_upper:.6f}]"
        )
    
    def test_depth_noise_independent_per_env(self, noisy_depth_env):
        """Verify depth noise is independent across environments using correlation test.
        
        We use FIRST DIFFERENCES to isolate the noise component, avoiding
        contamination from scene differences between environments.
        
        Statistical approach:
        - Compute Pearson correlation between first-diffs of env 0 and env 1
        - Under null hypothesis of independence, r ~ N(0, 1/sqrt(n-3)) approximately
        - We use a proper hypothesis test rather than arbitrary threshold
        """
        obs = collect_stationary_observations(noisy_depth_env, 200)  # Fewer steps OK for correlation
        
        # Extract depth observations
        depth_obs = obs[:, :, DEPTH_START_IDX:]
        
        # Compute first differences per environment to isolate noise
        # This removes the static scene signal, leaving only temporal noise
        first_diffs = depth_obs[1:] - depth_obs[:-1]  # (n_steps-1, num_envs, depth_dim)
        
        # Sample a subset of pixels for correlation analysis
        n_pixels = min(500, first_diffs.shape[2])
        
        # Compare noise (first-diffs) between env 0 and env 1
        env0_diffs = first_diffs[:, 0, :n_pixels].cpu().numpy().flatten()
        env1_diffs = first_diffs[:, 1, :n_pixels].cpu().numpy().flatten()
        
        # Compute Pearson correlation and p-value
        correlation, p_value = stats.pearsonr(env0_diffs, env1_diffs)
        n = len(env0_diffs)
        
        # Standard error of correlation under null hypothesis
        se_r = 1.0 / np.sqrt(n - 3)
        
        print(f"\n  Depth noise independence test (correlation analysis):")
        print(f"    Sample size: n={n:,}")
        print(f"    Correlation between env 0 and env 1: r={correlation:.4f}")
        print(f"    Standard error under H0 (independence): {se_r:.6f}")
        print(f"    Two-tailed p-value: {p_value:.4e}")
        print(f"    Note: Low |r| with high p-value indicates independent noise")
        
        # Test: correlation should not be significantly different from zero
        # We allow p > 0.001 (very conservative) - if p < 0.001, noise is correlated
        # Also check that absolute correlation is reasonably small
        assert p_value > 0.001 or abs(correlation) < 0.1, (
            f"Depth noise appears correlated across environments: "
            f"r={correlation:.4f}, p={p_value:.4e}"
        )
    
    def test_depth_values_in_valid_range(self, noisy_depth_env):
        """Verify all depth values are within [0, 1] after normalization."""
        obs = collect_stationary_observations(noisy_depth_env, 100)  # Fewer steps OK for range check
        
        # Extract depth observations
        depth_obs = obs[:, :, DEPTH_START_IDX:]
        
        min_val = depth_obs.min().item()
        max_val = depth_obs.max().item()
        
        print(f"\n  Depth observation range check:")
        print(f"    Min value: {min_val:.6f}")
        print(f"    Max value: {max_val:.6f}")
        
        # After normalization, values should be clipped to [0, 1]
        # Allow tiny epsilon for floating point
        assert min_val >= -1e-6, f"Depth values below 0: {min_val}"
        assert max_val <= 1.0 + 1e-6, f"Depth values above 1: {max_val}"


# =============================================================================
# Run module cleanup on import (for running outside pytest)
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
