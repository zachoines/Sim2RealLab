# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Circular statistics utilities for angular data.

This module provides circular (directional) statistics for testing motion
directions, orientations, and other angular quantities that wrap around
at ±π.

Standard statistics fail for angles because they don't handle wraparound
correctly. For example, the arithmetic mean of 170° and -170° is 0°,
but the circular mean is 180° (or -180°).

References:
    - Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics.
    - Fisher, N.I. (1993). Statistical Analysis of Circular Data.
"""

import numpy as np
from scipy import stats

from .constants import CONFIDENCE_LEVEL


def circular_mean(angles: np.ndarray) -> float:
    """Compute circular mean of angles using vector averaging.

    Converts angles to unit vectors, averages them, and converts back.
    This correctly handles angle wrapping at ±π.

    Args:
        angles: Array of angles in radians

    Returns:
        Circular mean angle in radians [-π, π]

    Example:
        >>> circular_mean(np.array([np.pi - 0.1, -np.pi + 0.1]))
        # Returns approximately π (or -π), not 0
    """
    mean_cos = np.mean(np.cos(angles))
    mean_sin = np.mean(np.sin(angles))
    return np.arctan2(mean_sin, mean_cos)


def circular_variance(angles: np.ndarray) -> float:
    """Compute circular variance of angles.

    Circular variance V = 1 - R, where R is the mean resultant length.
    V ranges from 0 (all angles identical) to 1 (uniform distribution).

    Args:
        angles: Array of angles in radians

    Returns:
        Circular variance in [0, 1]
    """
    mean_cos = np.mean(np.cos(angles))
    mean_sin = np.mean(np.sin(angles))
    R = np.sqrt(mean_cos**2 + mean_sin**2)  # Mean resultant length
    return 1 - R


def circular_std(angles: np.ndarray) -> float:
    """Compute circular standard deviation.

    Uses the formula: σ_circ = sqrt(-2 * ln(R)) where R is mean resultant length.
    This approximates the wrapped normal distribution's std for small dispersions.

    Args:
        angles: Array of angles in radians

    Returns:
        Circular standard deviation in radians
    """
    mean_cos = np.mean(np.cos(angles))
    mean_sin = np.mean(np.sin(angles))
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    # Clamp R to avoid log(0) or negative values from numerical error
    R = np.clip(R, 1e-10, 1.0 - 1e-10)
    return np.sqrt(-2 * np.log(R))


def mean_resultant_length(angles: np.ndarray) -> float:
    """Compute mean resultant length (concentration measure).

    R = |mean(e^(i*θ))| measures how concentrated the angles are.
    R = 1: all angles identical
    R = 0: uniformly distributed around circle

    Args:
        angles: Array of angles in radians

    Returns:
        Mean resultant length in [0, 1]
    """
    mean_cos = np.mean(np.cos(angles))
    mean_sin = np.mean(np.sin(angles))
    return np.sqrt(mean_cos**2 + mean_sin**2)


def circular_confidence_interval(
    angles: np.ndarray,
    confidence_level: float = CONFIDENCE_LEVEL
) -> tuple[float, float, float]:
    """Compute confidence interval for circular mean using Fisher's method.

    For concentrated distributions (high κ), the circular mean approximately
    follows a wrapped normal distribution. Uses the large-sample approximation
    for the CI based on the angular standard error.

    Args:
        angles: Array of angles in radians
        confidence_level: Confidence level (default 0.95)

    Returns:
        Tuple of (mean_angle, ci_half_width, mean_resultant_length)
    """
    n = len(angles)
    mean_cos = np.mean(np.cos(angles))
    mean_sin = np.mean(np.sin(angles))
    R = np.sqrt(mean_cos**2 + mean_sin**2)  # Mean resultant length
    mean_angle = np.arctan2(mean_sin, mean_cos)

    # For large samples with concentrated distribution, use normal approximation
    # Angular standard error (Mardia & Jupp, 2000, eq. 4.4.6)
    if R > 0.01:
        se_angle = np.sqrt((1 - R**2) / (n * R**2 + 1e-10))
    else:
        # Very dispersed data - use conservative estimate
        se_angle = np.pi / np.sqrt(n)

    alpha = 1 - confidence_level
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_half_width = z_crit * se_angle

    return mean_angle, ci_half_width, R


def angle_in_circular_ci(
    angle: float,
    mean: float,
    half_width: float
) -> bool:
    """Check if angle falls within circular confidence interval.

    Handles the wraparound at ±π correctly by computing the shortest
    angular distance on the circle.

    Args:
        angle: Angle to test (radians)
        mean: Center of CI (radians)
        half_width: Half-width of CI (radians)

    Returns:
        True if angle is within [mean - half_width, mean + half_width] on the circle
    """
    # Compute angular distance (shortest path on circle)
    diff = angle - mean
    # Wrap to [-π, π]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return abs(diff) <= half_width


def angular_distance(angle1: float, angle2: float) -> float:
    """Compute shortest angular distance between two angles.

    Args:
        angle1: First angle in radians
        angle2: Second angle in radians

    Returns:
        Absolute angular distance in [0, π]
    """
    diff = angle1 - angle2
    # Wrap to [-π, π] and take absolute value
    return abs(np.arctan2(np.sin(diff), np.cos(diff)))


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π] range.

    Args:
        angle: Angle in radians

    Returns:
        Angle wrapped to [-π, π]
    """
    return np.arctan2(np.sin(angle), np.cos(angle))
