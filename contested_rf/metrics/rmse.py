"""Reconstruction RMSE over the SINR grid, the primary metric.

    RMSE = sqrt( mean_m ( SINR_true(x_m) - SINR_est(x_m) )^2 )

Computed in dB, which weights low-SINR regions where communication is most
threatened.
"""
import numpy as np


def grid_rmse(sinr_true, sinr_est):
    """Root-mean-square error between two SINR maps (same shape), in dB."""
    a = np.asarray(sinr_true, dtype=float).ravel()
    b = np.asarray(sinr_est, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def convergence_step(rmse_history, obs_counts, threshold, window=100):
    """First observation count where the moving-average RMSE drops below
    `threshold` and stays below for the remainder of the survey.

    Args:
        rmse_history: (K,) RMSE values at the checkpoints in obs_counts.
        obs_counts: (K,) observation counts for each RMSE value.
        threshold: RMSE threshold in dB (e.g. 4.0 flat, 3.0 real terrain).
        window: moving-average window in *checkpoints* (kept small since
            checkpoints are already sparse).

    Returns:
        observation count (int) at first sustained crossing, or -1 if never.
    """
    rmse_history = np.asarray(rmse_history, dtype=float)
    obs_counts = np.asarray(obs_counts)
    k = len(rmse_history)
    if k == 0:
        return -1
    w = max(1, min(window, k))
    # Causal trailing moving average: smooth[i] is the mean of the up-to-w most
    # recent RMSE values (inclusive of i). A centred np.convolve(mode="same")
    # would (a) look into the future and (b) divide edge windows by the full w
    # even though fewer terms are summed, dragging the tail spuriously toward
    # zero and reporting a "convergence" for a series that never crosses the
    # threshold. The explicit trailing mean avoids both.
    smooth = np.array([rmse_history[max(0, i - w + 1):i + 1].mean()
                       for i in range(k)])
    below = smooth < threshold
    for i in range(k):
        if below[i] and below[i:].all():
            return int(obs_counts[i])
    return -1
