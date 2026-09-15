# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for depth camera noise model in isolation.

These tests validate the DepthNoiseModel class directly without a
simulation environment. They verify:
- Depth-dependent stereo noise (σ ∝ z²) with chi-squared test
- Hole probability matches configured rate
- The near-field convention: an image the observation term has already
  filled survives the model instead of being inverted to the far clamp

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test_sim/noise_models/test_depth_noise.py -v
"""

import pytest
import torch
import numpy as np

from test_sim.common import (
    chi_squared_variance_test,
    binomial_test,
    DEVICE,
)

# -- imports resolved after AppLauncher (root conftest) --
from strafer_lab.tasks.navigation.mdp.noise_models import (
    DepthNoiseModel,
    DepthNoiseModelCfg,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES = 50_000
N_ENVS = 32
# Deliberately not square: a transposed shape declaration would satisfy the
# model's pixel-count check, and only a non-square frame makes the neighbourhood
# tests notice.
TEST_HEIGHT = 12
TEST_WIDTH = 16
N_PIXELS = TEST_HEIGHT * TEST_WIDTH  # 12×16 depth image, flattened

TEST_BASELINE_M = 0.095
TEST_FOCAL_PX = 673.0
TEST_DISPARITY_NOISE_PX = 0.08
TEST_HOLE_PROBABILITY = 0.03
TEST_MAX_RANGE = 6.0
TEST_MIN_RANGE = 0.2


def _make_depth_model(
    hole_probability: float = 0.0,
    disparity_noise_px: float = TEST_DISPARITY_NOISE_PX,
    **convention,
) -> DepthNoiseModel:
    """Create a DepthNoiseModel for testing.

    ``convention`` forwards ``too_close_fill`` / ``hole_fill``; the image
    dimensions match N_PIXELS so the neighbourhood median can unflatten.
    """
    cfg = DepthNoiseModelCfg(
        baseline_m=TEST_BASELINE_M,
        focal_length_px=TEST_FOCAL_PX,
        disparity_noise_px=disparity_noise_px,
        hole_probability=hole_probability,
        min_range=TEST_MIN_RANGE,
        max_range=TEST_MAX_RANGE,
        height=TEST_HEIGHT,
        width=TEST_WIDTH,
        latency_steps=0,
        **convention,
    )
    return DepthNoiseModel(cfg, N_ENVS, DEVICE)


# =============================================================================
# Tests
# =============================================================================


def test_depth_dependent_noise():
    """Verify depth noise increases with distance (stereo noise model).

    The stereo noise formula is: σ = z² × σ_d / (f × B)
    at distances [1m, 2m, 4m], noise std should scale as z².

    Tests both:
    1. Monotone ordering: std(1m) < std(2m) < std(4m)
    2. Chi-squared variance test at a reference depth against theoretical σ
    """
    model = _make_depth_model(hole_probability=0.0)
    stereo_coeff = TEST_DISPARITY_NOISE_PX / (TEST_FOCAL_PX * TEST_BASELINE_M)

    # Use enough samples for stable chi-squared but not so many that float32
    # GPU precision causes false rejections.  A 99% CI keeps the test stable.
    n_chi2 = 10_000
    chi2_confidence = 0.99
    test_depths = [1.0, 2.0, 4.0]
    measured_stds = {}

    for z in test_depths:
        # Flattened depth image: (N_ENVS, N_PIXELS)
        clean = torch.full((N_ENVS, N_PIXELS), z, device=DEVICE)
        samples = []
        for _ in range(n_chi2):
            noisy = model(clean.clone())
            # Single env, single pixel to keep chi-squared df reasonable
            samples.append(noisy[0, 0].cpu().item() - z)

        flat = np.array(samples)
        measured_stds[z] = np.std(flat)

    # 1. Monotone ordering
    for i in range(len(test_depths) - 1):
        z_near, z_far = test_depths[i], test_depths[i + 1]
        assert measured_stds[z_near] < measured_stds[z_far], (
            f"Noise should increase with depth: "
            f"std({z_near}m)={measured_stds[z_near]:.4f} >= "
            f"std({z_far}m)={measured_stds[z_far]:.4f}"
        )

    # 2. Chi-squared test at reference depth (2m)
    ref_depth = 2.0
    expected_std = ref_depth**2 * stereo_coeff
    clean_ref = torch.full((N_ENVS, N_PIXELS), ref_depth, device=DEVICE)
    ref_samples = []
    for _ in range(n_chi2):
        noisy = model(clean_ref.clone())
        # Single env, single pixel
        ref_samples.append(noisy[0, 0].cpu().item() - ref_depth)
    ref_flat = np.array(ref_samples)

    result = chi_squared_variance_test(ref_flat, expected_std**2, confidence_level=chi2_confidence)

    print(f"\n  Depth-dependent noise test:")
    for z in test_depths:
        expected = z**2 * stereo_coeff
        print(f"    z={z}m: measured σ={measured_stds[z]:.6f}, expected σ={expected:.6f}")
    print(f"    Chi-squared at 2m: ratio={result.ratio:.4f}, in_ci={result.in_ci}")
    print(f"    CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")

    assert result.in_ci, (
        f"Noise at reference depth doesn't match stereo formula. "
        f"Expected σ²={expected_std**2:.6f}, got {result.measured_var:.6f} "
        f"(ratio={result.ratio:.4f}, CI=[{result.ci_low:.4f}, {result.ci_high:.4f}])"
    )


def test_hole_probability():
    """Verify hole insertion rate matches configured probability.

    Pinned to ``hole_fill="max"`` so a hole is identifiable by its value:
    the rate is what this test measures, not the value written.
    """
    model = _make_depth_model(hole_probability=TEST_HOLE_PROBABILITY, hole_fill="max")

    # Mid-range depth unlikely to trigger max_range by noise alone
    clean = torch.full((N_ENVS, N_PIXELS), 3.0, device=DEVICE)

    n_total = 0
    n_holes = 0
    for _ in range(N_SAMPLES):
        noisy = model(clean.clone())
        # Holes are replaced with max_range
        n_holes += int((noisy >= TEST_MAX_RANGE - 1e-3).sum().item())
        n_total += N_ENVS * N_PIXELS

    # Use 99% confidence: with ~409M observations the binomial test has
    # extreme power — even sub-0.01% deviations cause rejection at 95%.
    hole_confidence = 0.99
    result = binomial_test(n_holes, n_total, TEST_HOLE_PROBABILITY,
                           confidence_level=hole_confidence)

    print(f"\n  Hole probability test:")
    print(f"    Expected rate: {TEST_HOLE_PROBABILITY}")
    print(f"    Observed rate: {n_holes / n_total:.4f}")
    print(f"    Binomial p-value: {result.p_value:.4f}")

    assert not result.reject_null, (
        f"Hole rate doesn't match config. "
        f"Expected {TEST_HOLE_PROBABILITY}, got {n_holes / n_total:.4f} "
        f"(p={result.p_value:.4f})"
    )


def test_nearfield_fill_survives_the_model():
    """An image the observation term has already filled must survive.

    ``depth_image`` replaces every sub-``nearfield_clip`` pixel with
    ``nearfield_fill``, which is the same number as ``min_range``. The
    ``too_close`` comparison therefore fires on the fill value itself, and a
    symmetric dither about a threshold placed on the population's own value
    sends half of it across. Writing the fill value back makes the comparison
    idempotent, so the class survives whole.
    """
    model = _make_depth_model(hole_probability=0.0)
    filled = torch.full((N_ENVS, N_PIXELS), TEST_MIN_RANGE, device=DEVICE)

    far = 0
    n_total = 0
    for _ in range(200):
        noisy = model(filled.clone())
        far += int((noisy >= TEST_MAX_RANGE - 1e-3).sum().item())
        n_total += N_ENVS * N_PIXELS
        assert bool((noisy >= TEST_MIN_RANGE).all()), (
            "the near-fill write must leave no pixel below min_range"
        )

    share = far / n_total
    print(f"\n  Near-field survival: far-clamp share {share:.6f} of {n_total} pixels")
    assert share == 0.0, (
        f"{share:.4f} of an already-filled image reached the far clamp; the "
        f"near-field convention inverts the observation term's own fill"
    )


def test_nearfield_fill_inverts_under_the_far_clamp_convention():
    """The retained convention reproduces the defect the fix removes.

    Half the filled class crosses a threshold placed on its own value, and the
    share is a property of the threshold's position rather than of the noise
    magnitude — so it holds at both realism tiers.
    """
    filled = torch.full((N_ENVS, N_PIXELS), TEST_MIN_RANGE, device=DEVICE)

    for disparity in (TEST_DISPARITY_NOISE_PX, 2 * TEST_DISPARITY_NOISE_PX):
        model = _make_depth_model(
            disparity_noise_px=disparity, too_close_fill="max"
        )

        far = 0
        n_total = 0
        for _ in range(200):
            noisy = model(filled.clone())
            far += int((noisy >= TEST_MAX_RANGE - 1e-3).sum().item())
            n_total += N_ENVS * N_PIXELS

        share = far / n_total
        print(f"    disparity {disparity}: far-clamp share {share:.4f}")
        assert 0.45 < share < 0.55, (
            f"expected about half the filled class to cross, got {share:.4f}"
        )


def test_hole_fill_reads_the_median_of_its_valid_neighbours():
    """A hole takes the median of what surrounds it, not a constant.

    Eight distinct neighbours pin the reduction exactly: a mean would give 4.5,
    copying any single neighbour would give that neighbour, and the upper median
    would give 5.0. The lower median is deliberate — on an even count it biases
    the pixel towards the nearer surface, which is the conservative direction
    for a policy that has to avoid what it cannot resolve.
    """
    model = _make_depth_model(hole_probability=0.0, disparity_noise_px=0.0)
    image = torch.zeros((1, N_PIXELS), device=DEVICE)
    centre = 8 * TEST_WIDTH + 8
    neighbours = [
        (7, 7), (7, 8), (7, 9),
        (8, 7),         (8, 9),
        (9, 7), (9, 8), (9, 9),
    ]
    for value, (row, col) in enumerate(neighbours, start=1):
        image[0, row * TEST_WIDTH + col] = float(value)
    image[0, centre] = 99.0

    invalid = torch.zeros((1, N_PIXELS), dtype=torch.bool, device=DEVICE)
    invalid[0, centre] = True
    filled = model._neighbourhood_median(image.clone(), invalid)

    print(f"\n  Hole median: neighbours 1..8 -> {filled[0, centre].item()}")
    assert filled[0, centre].item() == 4.0, (
        f"expected the lower median of 1..8, got {filled[0, centre].item()}"
    )
    # Nothing but the hole may move.
    untouched = image.clone()
    untouched[0, centre] = 4.0
    assert torch.equal(filled, untouched), "the rescue wrote outside the hole"


def test_hole_fill_at_the_frame_edge_reads_only_real_neighbours():
    """A corner hole has three neighbours, and each votes once.

    Padding the frame to keep the window square must not let a replicated cell
    vote: two far neighbours and one near one would otherwise tie and resolve
    to the near value, moving an edge pixel off its surface.
    """
    model = _make_depth_model(hole_probability=0.0, disparity_noise_px=0.0)
    image = torch.zeros((1, N_PIXELS), device=DEVICE)
    image[0, 0] = 99.0
    image[0, 1] = 5.0
    image[0, TEST_WIDTH] = 5.0
    image[0, TEST_WIDTH + 1] = TEST_MIN_RANGE
    invalid = torch.zeros((1, N_PIXELS), dtype=torch.bool, device=DEVICE)
    invalid[0, 0] = True

    corner = model._neighbourhood_median(image.clone(), invalid)[0, 0].item()
    print(f"  Corner hole, neighbours (5.0, 5.0, {TEST_MIN_RANGE}) -> {corner}")
    assert corner == 5.0, (
        f"expected the median of the three real neighbours, got {corner}; a "
        f"padded cell is voting"
    )


def test_hole_fill_is_invisible_on_a_uniform_surface():
    """On one flat surface every neighbourhood median is the surface itself."""
    depth = 3.0
    model = _make_depth_model(hole_probability=0.05, disparity_noise_px=0.0)
    clean = torch.full((N_ENVS, N_PIXELS), depth, device=DEVICE)

    worst = 0.0
    holes_seen = 0
    for _ in range(50):
        noisy = model(clean.clone())
        worst = max(worst, float((noisy - depth).abs().max().item()))
        holes_seen += int((noisy != depth).sum().item())

    print(f"  Hole median on a uniform {depth} m surface: worst deviation "
          f"{worst:.3e} m")
    assert worst == 0.0, f"a hole deviated by {worst:.4e} m from its surface"


def test_hole_fill_falls_back_where_the_whole_neighbourhood_is_invalid():
    """An isolated invalid 3x3 has nothing to read, so it takes the near fill.

    Distinguishable from a constant near fill because the ring around the
    cluster keeps its surface value.
    """
    model = _make_depth_model(hole_probability=0.0, disparity_noise_px=0.0)
    image = torch.full((1, N_PIXELS), 3.0, device=DEVICE)
    invalid = torch.zeros((1, N_PIXELS), dtype=torch.bool, device=DEVICE)
    for row in (7, 8, 9):
        for col in (7, 8, 9):
            invalid[0, row * TEST_WIDTH + col] = True

    filled = model._neighbourhood_median(image.clone(), invalid)
    centre = filled[0, 8 * TEST_WIDTH + 8].item()
    corner = filled[0, 7 * TEST_WIDTH + 7].item()
    outside = filled[0, 6 * TEST_WIDTH + 8].item()

    print(f"  All-invalid 3x3: centre {centre}, corner {corner}, "
          f"outside {outside}")
    assert centre == pytest.approx(TEST_MIN_RANGE), (
        f"a fully invalid neighbourhood should take the near fill, got {centre}"
    )
    assert corner == 3.0, (
        "a corner of the cluster still has valid neighbours and should read the "
        f"surface, got {corner}"
    )
    assert outside == 3.0, "the rescue wrote outside the invalid cluster"


def test_the_retired_convention_reproduces_the_pre_fix_stream_exactly():
    """The selectable constants are a baseline, so they must not drift.

    Both arms draw the same randomness in the same order — the median consumes
    none — so a refactor that reorders the draws is caught here rather than in
    a statistic that tolerates it.
    """
    clean = torch.full((N_ENVS, N_PIXELS), 2.0, device=DEVICE)

    states = []
    outputs = []
    for convention in ({"too_close_fill": "max", "hole_fill": "max"},
                       {"too_close_fill": "near", "hole_fill": "median"}):
        torch.manual_seed(11)
        model = _make_depth_model(hole_probability=0.05, **convention)
        outputs.append(model(clean.clone()))
        states.append(torch.random.get_rng_state().clone())

    assert torch.equal(states[0], states[1]), (
        "the two conventions consumed different randomness; the retired arm is "
        "no longer a comparable baseline"
    )
    assert not torch.equal(outputs[0], outputs[1]), (
        "the two conventions produced identical frames, so neither is being "
        "selected"
    )


def test_the_frame_shape_has_to_match_the_declared_one():
    """A declared shape that disagrees with the frame is a configuration error.

    Checked on every frame, not only when the median runs, so a stale shape
    cannot sit inert until a tier that uses holes reaches it.
    """
    model = _make_depth_model(hole_probability=0.0)
    with pytest.raises(ValueError, match="declares"):
        model(torch.full((N_ENVS, N_PIXELS + 1), 2.0, device=DEVICE))


@pytest.mark.parametrize(
    "convention",
    [{"too_close_fill": "nearr"}, {"too_close_fill": ""},
     {"hole_fill": "mean"}, {"hole_fill": "MEDIAN"}],
)
def test_an_unknown_convention_is_refused(convention):
    """Falling through to a default would silently restore the retired arm."""
    with pytest.raises(ValueError, match="must be one of"):
        _make_depth_model(**convention)
