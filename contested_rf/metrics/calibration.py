"""Uncertainty-calibration metric.

An estimator is well calibrated if its credible intervals contain the truth at
the stated rate. Assessed over the SINR grid using predictive samples, since
the map from a Gaussian on interference power to SINR is nonlinear.

    Coverage(alpha) = fraction of grid points whose true SINR lies inside the
                      central alpha-credible interval of the predictive samples

Calibration error is the mean |Coverage(alpha) - alpha| over the assessed
levels; a reliability curve plots Coverage(alpha) against alpha.
"""
import numpy as np


def coverage_curve(sinr_samples, sinr_true, alphas=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    """Empirical coverage at each nominal level.

    Args:
        sinr_samples: (M, S) predictive SINR samples per grid point.
        sinr_true: (M,) true SINR at each grid point.
        alphas: nominal central-interval levels.

    Returns:
        (coverage dict {alpha: observed}, calibration_error float).
    """
    sinr_samples = np.asarray(sinr_samples, dtype=float)
    sinr_true = np.asarray(sinr_true, dtype=float).ravel()
    cov = {}
    for a in alphas:
        lo = np.quantile(sinr_samples, (1.0 - a) / 2.0, axis=1)
        hi = np.quantile(sinr_samples, (1.0 + a) / 2.0, axis=1)
        inside = (sinr_true >= lo) & (sinr_true <= hi)
        cov[a] = float(np.mean(inside))
    cal_err = float(np.mean([abs(cov[a] - a) for a in alphas]))
    return cov, cal_err
