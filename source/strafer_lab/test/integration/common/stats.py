# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Statistical testing utilities for integration tests.

This module provides common statistical testing functions used across
integration tests for validating noise models, motor dynamics, and
sensor characteristics.

All functions follow a consistent interface returning result dictionaries
with standardized keys for easy assertion and logging.
"""

import numpy as np
from scipy import stats
from typing import NamedTuple

from .constants import CONFIDENCE_LEVEL


# =============================================================================
# Result Types
# =============================================================================

class VarianceTestResult(NamedTuple):
    """Result of a variance test."""
    measured_var: float
    expected_var: float
    ratio: float
    ci_low: float
    ci_high: float
    in_ci: bool
    n_samples: int
    df: int


class TTestResult(NamedTuple):
    """Result of a t-test."""
    mean: float
    t_statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    n_samples: int
    reject_null: bool


class BinomialTestResult(NamedTuple):
    """Result of a binomial test."""
    observed_count: int
    total_count: int
    observed_rate: float
    expected_rate: float
    p_value: float
    reject_null: bool


# =============================================================================
# Variance Tests (Chi-Squared)
# =============================================================================

def chi_squared_variance_test(
    samples: np.ndarray,
    expected_var: float,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> VarianceTestResult:
    """Test if sample variance matches expected variance using chi-squared test.

    Uses the chi-squared distribution for variance testing:
        (n-1) * s² / σ² ~ χ²(n-1)

    where s² is sample variance and σ² is expected variance.

    Args:
        samples: 1D array of samples
        expected_var: Expected (theoretical) variance
        confidence_level: Confidence level for the test (default 0.95)

    Returns:
        VarianceTestResult with test statistics and CI
    """
    samples = np.asarray(samples).flatten()
    n = len(samples)
    df = n - 1
    measured_var = np.var(samples, ddof=1)  # Unbiased estimator

    if expected_var <= 0:
        raise ValueError("expected_var must be positive")

    ratio = measured_var / expected_var

    # Chi-squared confidence interval for variance ratio
    alpha = 1 - confidence_level
    chi2_low = stats.chi2.ppf(alpha / 2, df)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df)

    # CI for ratio: multiply by df/chi2 to get CI bounds
    ci_low = chi2_low / df
    ci_high = chi2_high / df

    in_ci = ci_low <= ratio <= ci_high

    return VarianceTestResult(
        measured_var=measured_var,
        expected_var=expected_var,
        ratio=ratio,
        ci_low=ci_low,
        ci_high=ci_high,
        in_ci=in_ci,
        n_samples=n,
        df=df,
    )


def chi_squared_variance_ci(
    samples: np.ndarray,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float, float]:
    """Compute confidence interval for variance.

    Args:
        samples: 1D array of samples
        confidence_level: Confidence level for CI (default 0.95)

    Returns:
        Tuple of (variance, ci_lower, ci_upper)
    """
    samples = np.asarray(samples).flatten()
    n = len(samples)
    df = n - 1
    var = np.var(samples, ddof=1)

    alpha = 1 - confidence_level
    chi2_low = stats.chi2.ppf(alpha / 2, df)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df)

    # CI for variance: (n-1)s²/χ²_upper < σ² < (n-1)s²/χ²_lower
    ci_lower = df * var / chi2_high
    ci_upper = df * var / chi2_low

    return var, ci_lower, ci_upper


# =============================================================================
# T-Tests
# =============================================================================

def one_sample_t_test(
    samples: np.ndarray,
    null_value: float = 0.0,
    alternative: str = "two-sided",
    confidence_level: float = CONFIDENCE_LEVEL,
) -> TTestResult:
    """Perform one-sample t-test.

    Tests whether the population mean differs from null_value.

    Args:
        samples: 1D array of samples
        null_value: Value to test against (default 0)
        alternative: "two-sided", "greater", or "less"
        confidence_level: Confidence level for CI

    Returns:
        TTestResult with test statistics
    """
    samples = np.asarray(samples).flatten()
    n = len(samples)
    mean = np.mean(samples)
    sem = stats.sem(samples)

    t_stat, p_two_sided = stats.ttest_1samp(samples, null_value)

    # Adjust p-value for one-sided alternatives
    if alternative == "greater":
        p_value = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    elif alternative == "less":
        p_value = p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2
    else:
        p_value = p_two_sided

    # Confidence interval for mean
    alpha = 1 - confidence_level
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    ci_low = mean - t_crit * sem
    ci_high = mean + t_crit * sem

    reject_null = p_value < (1 - confidence_level)

    return TTestResult(
        mean=mean,
        t_statistic=t_stat,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n_samples=n,
        reject_null=reject_null,
    )


def welch_t_test(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    alternative: str = "two-sided",
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict:
    """Perform Welch's t-test (unequal variance t-test).

    Compares means of two independent samples without assuming equal variances.

    Args:
        samples_a: First sample array
        samples_b: Second sample array
        alternative: "two-sided", "greater" (a > b), or "less" (a < b)
        confidence_level: Confidence level for effect size interpretation

    Returns:
        Dict with t_statistic, p_value, cohens_d, mean_a, mean_b, etc.
    """
    samples_a = np.asarray(samples_a).flatten()
    samples_b = np.asarray(samples_b).flatten()

    t_stat, p_two_sided = stats.ttest_ind(samples_a, samples_b, equal_var=False)

    # Adjust p-value for one-sided alternatives
    if alternative == "greater":
        p_value = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    elif alternative == "less":
        p_value = p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2
    else:
        p_value = p_two_sided

    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(samples_a) + np.var(samples_b)) / 2)
    cohens_d = (np.mean(samples_a) - np.mean(samples_b)) / (pooled_std + 1e-9)

    reject_null = p_value < (1 - confidence_level)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "mean_a": np.mean(samples_a),
        "mean_b": np.mean(samples_b),
        "std_a": np.std(samples_a),
        "std_b": np.std(samples_b),
        "n_a": len(samples_a),
        "n_b": len(samples_b),
        "reject_null": reject_null,
    }


# =============================================================================
# Binomial Tests
# =============================================================================

def binomial_test(
    successes: int,
    trials: int,
    expected_prob: float,
    alternative: str = "two-sided",
    confidence_level: float = CONFIDENCE_LEVEL,
) -> BinomialTestResult:
    """Perform binomial test for proportion.

    Tests whether observed success rate matches expected probability.

    Args:
        successes: Number of successes (e.g., holes observed)
        trials: Total number of trials
        expected_prob: Expected probability under null hypothesis
        alternative: "two-sided", "greater", or "less"
        confidence_level: Confidence level for reject decision

    Returns:
        BinomialTestResult with test statistics
    """
    result = stats.binomtest(successes, trials, expected_prob, alternative=alternative)
    observed_rate = successes / trials

    alpha = 1 - confidence_level
    reject_null = result.pvalue < alpha

    return BinomialTestResult(
        observed_count=successes,
        total_count=trials,
        observed_rate=observed_rate,
        expected_rate=expected_prob,
        p_value=result.pvalue,
        reject_null=reject_null,
    )
