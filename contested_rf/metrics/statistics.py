"""Paired statistics for the estimator comparison.

All estimators see the same data within a replication, so comparisons are
paired: we analyse the per-replication difference in RMSE, not the marginal
distributions. Bootstrap CIs make no distributional assumption; Cohen's d on
the paired differences reports practical significance.
"""
import numpy as np


def paired_bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    """Bootstrap CI on the mean of paired differences.

    Returns (mean, lo, hi). If the CI excludes 0 the difference is significant.
    """
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.array([rng.choice(diffs, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diffs.mean()), float(lo), float(hi)


def cohens_d_paired(diffs):
    """Cohen's d for paired differences: mean / SD(differences)."""
    diffs = np.asarray(diffs, dtype=float)
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 1e-12 else 0.0
