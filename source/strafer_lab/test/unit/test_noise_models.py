# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for sensor noise models.

These tests verify the statistical properties of individual noise model classes
in isolation, without running a full Isaac Sim environment.

Tests validate:
1. White noise has correct standard deviation
2. Bias is sampled within configured range
3. Bias drift accumulates correctly (random walk)
4. Quantization produces integer values
5. Tick error probabilities match config
6. Noise is independent per environment

Usage:
    pytest test/unit/test_noise_models.py -v
    
Note: These tests require Isaac Sim to be launched for torch device access,
but do not create a full simulation environment.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import numpy as np
from scipy import stats
import pytest

from strafer_lab.tasks.navigation.mdp.noise_models import (
    IMUNoiseModel,
    IMUNoiseModelCfg,
    EncoderNoiseModel,
    EncoderNoiseModelCfg,
    DepthNoiseModel,
    DepthNoiseModelCfg,
    RGBNoiseModel,
    RGBNoiseModelCfg,
)
from isaaclab.utils.noise import GaussianNoiseCfg


# =============================================================================
# Test Configuration
# =============================================================================

N_SAMPLES = 10000        # Samples for statistical tests
N_ENVS = 32              # Parallel environments
DEVICE = "cuda:0"
TOLERANCE = 0.15         # 15% tolerance for statistical tests


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def device():
    return DEVICE


# =============================================================================
# IMU Noise Model - White Noise Tests
# =============================================================================

class TestIMUWhiteNoise:
    """Test IMU white noise characteristics (bias disabled)."""
    
    def test_accel_noise_std_matches_config(self, device):
        """White noise std should match configured value when bias is disabled."""
        expected_std = 0.05
        
        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=expected_std),
            accel_noise_std=expected_std,
            accel_bias_range=(0.0, 0.0),  # No bias
            accel_bias_drift_rate=0.0,    # No drift
            output_size=3,
        )
        
        model = IMUNoiseModel(cfg, N_ENVS, device)
        
        # Collect samples
        samples = []
        base = torch.zeros(N_ENVS, 3, device=device)
        for _ in range(N_SAMPLES // N_ENVS):
            samples.append(model(base).cpu())
        
        data = torch.cat(samples, dim=0).numpy().flatten()
        measured_std = np.std(data)
        
        print(f"\n  Expected std: {expected_std:.4f}")
        print(f"  Measured std: {measured_std:.4f}")
        print(f"  Error: {abs(measured_std - expected_std) / expected_std * 100:.1f}%")
        
        assert abs(measured_std - expected_std) / expected_std < TOLERANCE
    
    def test_noise_has_zero_mean(self, device):
        """White noise should have zero mean."""
        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.1),
            accel_noise_std=0.1,
            accel_bias_range=(0.0, 0.0),
            accel_bias_drift_rate=0.0,
            output_size=3,
        )
        
        model = IMUNoiseModel(cfg, N_ENVS, device)
        
        samples = []
        base = torch.zeros(N_ENVS, 3, device=device)
        for _ in range(N_SAMPLES // N_ENVS):
            samples.append(model(base).cpu())
        
        data = torch.cat(samples, dim=0).numpy().flatten()
        mean = np.mean(data)
        sem = np.std(data) / np.sqrt(len(data))
        
        print(f"\n  Mean: {mean:.6f}")
        print(f"  Standard error: {sem:.6f}")
        print(f"  |mean| / std: {abs(mean) / np.std(data):.4f}")
        
        # Mean should be within 3 standard errors of zero
        assert abs(mean) < 3 * sem
    
    def test_noise_is_gaussian(self, device):
        """White noise should follow Gaussian distribution."""
        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.1),
            accel_noise_std=0.1,
            accel_bias_range=(0.0, 0.0),
            accel_bias_drift_rate=0.0,
            output_size=3,
        )
        
        model = IMUNoiseModel(cfg, N_ENVS, device)
        
        samples = []
        base = torch.zeros(N_ENVS, 3, device=device)
        for _ in range(N_SAMPLES // N_ENVS):
            samples.append(model(base).cpu())
        
        data = torch.cat(samples, dim=0).numpy().flatten()
        std = np.std(data)
        
        # Check percentiles match Gaussian
        within_1std = np.mean(np.abs(data) < std)
        within_2std = np.mean(np.abs(data) < 2 * std)
        
        print(f"\n  % within 1σ: {within_1std*100:.1f}% (expected: 68.3%)")
        print(f"  % within 2σ: {within_2std*100:.1f}% (expected: 95.4%)")
        
        assert 0.65 < within_1std < 0.72  # 68.3% ± ~4%
        assert 0.93 < within_2std < 0.98  # 95.4% ± ~2.5%
    
    def test_noise_is_independent_per_env(self, device):
        """Each environment should receive independent noise samples."""
        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.1),
            accel_noise_std=0.1,
            accel_bias_range=(0.0, 0.0),
            accel_bias_drift_rate=0.0,
            output_size=3,
        )
        
        model = IMUNoiseModel(cfg, N_ENVS, device)
        
        # Collect time series for env 0 and env 1
        env0_samples = []
        env1_samples = []
        base = torch.zeros(N_ENVS, 3, device=device)
        
        for _ in range(500):
            out = model(base)
            env0_samples.append(out[0, 0].item())
            env1_samples.append(out[1, 0].item())
        
        correlation = np.corrcoef(env0_samples, env1_samples)[0, 1]
        
        print(f"\n  Correlation between env 0 and env 1: {correlation:.4f}")
        print(f"  (Independent noise should have correlation near 0)")
        
        # Independent samples should have near-zero correlation
        assert abs(correlation) < 0.15


# =============================================================================
# IMU Noise Model - Bias Tests
# =============================================================================

class TestIMUBias:
    """Test IMU bias initialization and drift behavior."""
    
    def test_bias_within_configured_range(self, device):
        """Initial bias should be sampled within configured range."""
        bias_range = (-0.1, 0.1)
        
        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            accel_noise_std=0.0,  # No white noise
            accel_bias_range=bias_range,
            accel_bias_drift_rate=0.0,  # No drift
            output_size=3,
        )
        
        # Sample many models to test bias distribution
        all_biases = []
        for _ in range(100):
            model = IMUNoiseModel(cfg, N_ENVS, device)
            # With no noise, output = bias
            out = model(torch.zeros(N_ENVS, 3, device=device))
            all_biases.append(out.cpu())
        
        biases = torch.cat(all_biases, dim=0).numpy()
        
        print(f"\n  Configured range: [{bias_range[0]:.3f}, {bias_range[1]:.3f}]")
        print(f"  Observed range: [{biases.min():.4f}, {biases.max():.4f}]")
        print(f"  Mean bias: {biases.mean():.4f} (expected: 0)")
        
        # All biases should be within range
        assert biases.min() >= bias_range[0] - 0.01
        assert biases.max() <= bias_range[1] + 0.01
        
        # Mean should be centered (approximately 0)
        assert abs(biases.mean()) < 0.02
    
    def test_bias_drift_increases_variance(self, device):
        """Bias drift (random walk) should increase variance over time."""
        drift_rate = 0.01
        
        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            accel_noise_std=0.0,
            accel_bias_range=(-0.5, 0.5),  # Room for drift
            accel_bias_drift_rate=drift_rate,
            output_size=3,
        )
        
        model = IMUNoiseModel(cfg, N_ENVS, device)
        model._bias.zero_()  # Start from known state
        
        # Collect outputs over time
        outputs = []
        base = torch.zeros(N_ENVS, 3, device=device)
        for _ in range(200):
            outputs.append(model(base).cpu())
        
        outputs = torch.stack(outputs, dim=0)  # (time, envs, 3)
        
        # Variance should increase with time (random walk property)
        early_var = outputs[:50].var().item()
        late_var = outputs[150:].var().item()
        
        print(f"\n  Drift rate: {drift_rate}")
        print(f"  Early variance (steps 0-50): {early_var:.6f}")
        print(f"  Late variance (steps 150-200): {late_var:.6f}")
        print(f"  Ratio: {late_var / max(early_var, 1e-9):.2f}x")
        
        assert late_var > early_var, "Variance should grow with drift"
    
    def test_bias_clamped_to_2x_range(self, device):
        """Bias should be clamped to 2x the configured range."""
        bias_range = (-0.1, 0.1)
        clamp_limit = 0.2  # 2x bias_range
        
        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            accel_noise_std=0.0,
            accel_bias_range=bias_range,
            accel_bias_drift_rate=0.1,  # High drift to hit clamp
            output_size=3,
        )
        
        model = IMUNoiseModel(cfg, N_ENVS, device)
        
        # Run for many steps to let drift hit clamp
        base = torch.zeros(N_ENVS, 3, device=device)
        max_observed = 0.0
        for _ in range(1000):
            out = model(base)
            max_observed = max(max_observed, out.abs().max().item())
        
        print(f"\n  Clamp limit: ±{clamp_limit:.3f}")
        print(f"  Max observed bias: {max_observed:.4f}")
        
        # Should be clamped (with some tolerance for white noise if any)
        assert max_observed <= clamp_limit + 0.01


# =============================================================================
# Encoder Noise Model Tests
# =============================================================================

class TestEncoderNoise:
    """Test encoder noise model."""
    
    def test_velocity_noise_std(self, device):
        """Velocity noise should scale by max_velocity."""
        noise_frac = 0.05  # 5% of max
        max_vel = 5000.0
        expected_std = noise_frac * max_vel
        
        cfg = EncoderNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=noise_frac),
            velocity_noise_std=noise_frac,
            max_velocity=max_vel,
            enable_quantization=False,
            missed_tick_prob=0.0,
            extra_tick_prob=0.0,
        )
        
        model = EncoderNoiseModel(cfg, N_ENVS, device)
        
        # Apply to constant velocity
        base = torch.ones(N_ENVS, 4, device=device) * 1000.0
        samples = []
        for _ in range(N_SAMPLES // N_ENVS):
            samples.append((model(base) - base).cpu())
        
        noise = torch.cat(samples, dim=0).numpy().flatten()
        measured_std = np.std(noise)
        
        print(f"\n  Expected std: {expected_std:.2f}")
        print(f"  Measured std: {measured_std:.2f}")
        
        assert abs(measured_std - expected_std) / expected_std < TOLERANCE
    
    def test_quantization_produces_integers(self, device):
        """Quantization should produce integer values."""
        cfg = EncoderNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.1),
            velocity_noise_std=0.1,
            enable_quantization=True,
            missed_tick_prob=0.0,
            extra_tick_prob=0.0,
        )
        
        model = EncoderNoiseModel(cfg, N_ENVS, device)
        
        # Non-integer input
        base = torch.ones(N_ENVS, 4, device=device) * 123.456
        
        all_integer = True
        for _ in range(100):
            out = model(base)
            if not torch.all(torch.abs(out - out.round()) < 1e-5):
                all_integer = False
                break
        
        print(f"\n  All outputs are integers: {all_integer}")
        assert all_integer
    
    def test_missed_tick_probability(self, device):
        """Missed ticks should occur at configured probability."""
        expected_prob = 0.02
        
        cfg = EncoderNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            velocity_noise_std=0.0,
            enable_quantization=False,
            missed_tick_prob=expected_prob,
            extra_tick_prob=0.0,
        )
        
        model = EncoderNoiseModel(cfg, N_ENVS, device)
        
        base = torch.ones(N_ENVS, 4, device=device) * 100.0
        
        total = 0
        missed = 0
        for _ in range(1000):
            out = model(base)
            missed += (out < base).sum().item()
            total += out.numel()
        
        observed_prob = missed / total
        
        print(f"\n  Expected miss prob: {expected_prob:.4f}")
        print(f"  Observed miss prob: {observed_prob:.4f}")
        
        assert abs(observed_prob - expected_prob) < expected_prob * 0.3


# =============================================================================
# Depth Camera Noise Model Tests  
# =============================================================================

class TestDepthNoise:
    """Test depth camera noise model.
    
    Note: DepthNoiseModel now works on RAW meters (not normalized).
    Normalization is applied afterwards via ObsTerm.scale.
    """
    
    def test_depth_dependent_noise(self, device):
        """Noise should increase with depth (depth-dependent noise)."""
        base_std = 0.005
        depth_coeff = 0.01
        max_range = 10.0
        
        cfg = DepthNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=base_std),
            base_noise_std=base_std,
            depth_noise_coeff=depth_coeff,
            hole_probability=0.0,
            frame_drop_prob=0.0,
            max_range=max_range,
        )
        
        model = DepthNoiseModel(cfg, N_ENVS, device)
        
        # Test at different depths (in METERS, not normalized)
        depths_m = [1.0, 3.0, 5.0]
        measured_stds = []
        
        for depth in depths_m:
            model.reset()
            # Input is RAW meters
            base = torch.ones(N_ENVS, 500, device=device) * depth
            
            samples = []
            for _ in range(200):
                # Output is also RAW meters, measure noise directly
                samples.append((model(base) - depth).cpu())
            
            noise = torch.cat(samples, dim=0).numpy().flatten()
            measured_stds.append(np.std(noise))
        
        print(f"\n  Depth (m) | Expected Std | Measured Std")
        print(f"  " + "-" * 40)
        for d, m in zip(depths_m, measured_stds):
            expected = base_std + depth_coeff * d
            print(f"  {d:.1f}m      | {expected:.4f}m      | {m:.4f}m")
        
        # Verify noise increases with depth
        assert measured_stds[1] > measured_stds[0]
        assert measured_stds[2] > measured_stds[1]
    
    def test_hole_probability(self, device):
        """Holes should occur at configured probability."""
        expected_prob = 0.03
        max_range = 6.0
        
        cfg = DepthNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            base_noise_std=0.0,
            depth_noise_coeff=0.0,
            hole_probability=expected_prob,
            frame_drop_prob=0.0,
            max_range=max_range,
        )
        
        model = DepthNoiseModel(cfg, N_ENVS, device)
        
        # Mid-range depth in METERS
        base = torch.ones(N_ENVS, 1000, device=device) * 3.0  # 3m
        
        total = 0
        holes = 0
        for _ in range(500):
            out = model(base)
            # Holes are set to max_range (in meters)
            holes += (out >= max_range - 0.01).sum().item()
            total += out.numel()
        
        observed_prob = holes / total
        
        print(f"\n  Expected hole prob: {expected_prob:.4f}")
        print(f"  Observed hole prob: {observed_prob:.4f}")
        
        assert abs(observed_prob - expected_prob) < expected_prob * 0.3


# =============================================================================
# RGB Camera Noise Model Tests
# =============================================================================

class TestRGBNoise:
    """Test RGB camera noise model.
    
    Note: RGBNoiseModel now works on [0,1] RGB data (not mean-centered).
    The observation function returns [0,1] directly from uint8/255.0 conversion.
    """
    
    def test_pixel_noise_std(self, device):
        """Pixel noise should match configured std."""
        expected_std = 0.04
        
        cfg = RGBNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=expected_std),
            pixel_noise_std=expected_std,
            brightness_range=(1.0, 1.0),  # No brightness variation
            frame_drop_prob=0.0,
        )
        
        model = RGBNoiseModel(cfg, N_ENVS, device)
        
        # Input is [0,1] RGB data
        base = torch.ones(N_ENVS, 3000, device=device) * 0.5
        
        samples = []
        for _ in range(500):
            samples.append((model(base) - 0.5).cpu())
        
        # Need to account for brightness being 1.0
        noise = torch.cat(samples, dim=0).numpy().flatten()
        measured_std = np.std(noise)
        
        print(f"\n  Expected std: {expected_std:.4f}")
        print(f"  Measured std: {measured_std:.4f}")
        
        assert abs(measured_std - expected_std) / expected_std < TOLERANCE
    
    def test_brightness_variation(self, device):
        """Brightness should vary within configured range."""
        b_min, b_max = 0.8, 1.2
        
        cfg = RGBNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            pixel_noise_std=0.0,
            brightness_range=(b_min, b_max),
            frame_drop_prob=0.0,
        )
        
        model = RGBNoiseModel(cfg, N_ENVS, device)
        
        # Input is [0,1] RGB data
        base = torch.ones(N_ENVS, 100, device=device) * 0.5
        
        brightness_factors = []
        for _ in range(500):
            out = model(base)
            # brightness = output / input (clamped to [0,1], so may be truncated)
            factor = out.mean(dim=1) / 0.5
            brightness_factors.append(factor.cpu())
        
        factors = torch.cat(brightness_factors, dim=0).numpy()
        
        print(f"\n  Expected range: [{b_min:.2f}, {b_max:.2f}]")
        print(f"  Observed range: [{factors.min():.3f}, {factors.max():.3f}]")
        print(f"  Mean: {factors.mean():.3f}")
        
        # Note: high brightness values may get clamped to 1.0, so max factor is limited
        assert factors.min() >= b_min - 0.02
        # Upper bound limited by clamping: 0.5 * 1.2 = 0.6, clamped at 1.0 gives factor = 2.0
        # But actual output is clamped, so factor = 1.0 / 0.5 = 2.0 max for this test
        assert factors.max() <= b_max + 0.02
    
    def test_output_clamped_to_valid_range(self, device):
        """Output should be clamped to [0, 1]."""
        cfg = RGBNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.2),  # High noise to test clamping
            pixel_noise_std=0.2,
            brightness_range=(0.5, 1.5),  # Extreme brightness
            frame_drop_prob=0.0,
        )
        
        model = RGBNoiseModel(cfg, N_ENVS, device)
        
        # Test near boundaries
        base_low = torch.ones(N_ENVS, 1000, device=device) * 0.1
        base_high = torch.ones(N_ENVS, 1000, device=device) * 0.9
        
        for _ in range(100):
            out_low = model(base_low)
            out_high = model(base_high)
            
            # All values should be in [0, 1]
            assert out_low.min() >= 0.0, f"Output below 0: {out_low.min()}"
            assert out_low.max() <= 1.0, f"Output above 1: {out_low.max()}"
            assert out_high.min() >= 0.0, f"Output below 0: {out_high.min()}"
            assert out_high.max() <= 1.0, f"Output above 1: {out_high.max()}"
        
        print(f"\n  All outputs clamped to [0, 1]: PASSED")


# =============================================================================
# Cleanup
# =============================================================================

def pytest_sessionfinish(session, exitstatus):
    """Clean up simulation after tests."""
    simulation_app.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
