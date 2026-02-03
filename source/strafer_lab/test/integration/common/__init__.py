# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Common utilities for integration tests.

This module provides shared utilities across all integration test suites:
- Statistical testing functions (chi-squared, t-tests, circular stats)
- Common constants (confidence levels, sample sizes)
- Fixture patterns for Isaac Sim environments

Usage:
    from test.integration.common import CONFIDENCE_LEVEL, chi_squared_variance_test
    from test.integration.common.circular import circular_mean, circular_ci
"""

from .constants import (
    CONFIDENCE_LEVEL,
    DEFAULT_NUM_ENVS,
    DEFAULT_SETTLE_STEPS,
    DEVICE,
)

from .stats import (
    chi_squared_variance_test,
    chi_squared_variance_ci,
    variance_ratio_test,
    one_sample_t_test,
    welch_t_test,
    binomial_test,
    VarianceTestResult,
    TTestResult,
    BinomialTestResult,
)

from .circular import (
    circular_mean,
    circular_variance,
    circular_std,
    circular_confidence_interval,
    angle_in_circular_ci,
)

__all__ = [
    # Constants
    "CONFIDENCE_LEVEL",
    "DEFAULT_NUM_ENVS",
    "DEFAULT_SETTLE_STEPS",
    "DEVICE",
    # Statistical tests
    "chi_squared_variance_test",
    "chi_squared_variance_ci",
    "variance_ratio_test",
    "one_sample_t_test",
    "welch_t_test",
    "binomial_test",
    # Result types
    "VarianceTestResult",
    "TTestResult",
    "BinomialTestResult",
    # Circular statistics
    "circular_mean",
    "circular_variance",
    "circular_std",
    "circular_confidence_interval",
    "angle_in_circular_ci",
]
