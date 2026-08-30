"""Scenario 2 reconstruction: composed two-jammer interference field.

Jammer A is a known omnidirectional background source; jammer B is the unknown
directional emitter whose 6D state the filter infers. The interference field is
the linear sum of both received powers, with B's clamped Gaussian antenna gain,
mirroring the Scenario 2 likelihood over a grid.

The signal, SINR, pure-GP and no-jamming parts are reused from
contested_rf.estimators.combined; only the parametric interference differs.
"""
from __future__ import annotations

import numpy as np

from contested_rf.estimators import combined as C
from contested_rf.estimators.gp_residual import GPResidual
from contested_rf.propagation.antenna import directional_gain
from contested_rf.propagation.path_loss import free_space_path_loss


def s2_interference_dBm(points, B_params, jammer_a, scenario, n=C.N_LEARNER):
    """Total received interference power (dBm) from known A + directional B.

    Args:
        points: (M, 2) query locations.
        B_params: 6-vector [x, y, P_tx, theta_main_deg, theta_3db_deg, F_b_dB].
        jammer_a: the known omni Jammer (position + power read).
        scenario: for carrier frequency.
        n: path-loss exponent (learner n=2.0 for reconstruction; 2.5 for truth).
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    pl_d0 = free_space_path_loss(1.0, scenario.operating_freq_Hz)

    # Known omni A.
    ax, ay = jammer_a.position
    dA = np.maximum(np.hypot(points[:, 0] - ax, points[:, 1] - ay), 1.0)
    P_A = jammer_a.power_dBm - (pl_d0 + 10.0 * n * np.log10(dA))
    PA_mW = 10.0 ** (P_A / 10.0)

    # Directional B.
    x, y, ptx, tm, t3, fb = B_params
    dB = np.maximum(np.hypot(points[:, 0] - x, points[:, 1] - y), 1.0)
    plB = pl_d0 + 10.0 * n * np.log10(dB)
    bearing = np.degrees(np.arctan2(points[:, 1] - y, points[:, 0] - x))
    g = directional_gain(bearing, tm, t3, G0_dB=0.0, front_back_ratio_dB=fb)
    PB_mW = 10.0 ** ((ptx - plB + g) / 10.0)

    return 10.0 * np.log10(PA_mW + PB_mW)


def parametric_sinr_map_s2(B_hat, jammer_a, scenario, grid_shape=(60, 60),
                           n=C.N_LEARNER):
    X, Y, pts = C._grid_points(scenario, grid_shape)
    P_interf = s2_interference_dBm(pts, B_hat, jammer_a, scenario, n)
    P_signal = C.learner_signal_dBm(scenario, pts)
    return X, Y, C._sinr_from_interference(P_interf, P_signal).reshape(X.shape)


def combined_sinr_map_s2(B_hat, gp, jammer_a, scenario, grid_shape=(60, 60),
                         n=C.N_LEARNER):
    X, Y, pts = C._grid_points(scenario, grid_shape)
    P_interf = s2_interference_dBm(pts, B_hat, jammer_a, scenario, n) \
        + gp.predict(pts, return_var=False)
    P_signal = C.learner_signal_dBm(scenario, pts)
    return X, Y, C._sinr_from_interference(P_interf, P_signal).reshape(X.shape)


def fit_residual_gp_s2(B_hat, jammer_a, sensor_positions, observations,
                       scenario, n=C.N_LEARNER, warm_gp=None, restarts=1,
                       **gp_kwargs):
    f = s2_interference_dBm(sensor_positions, B_hat, jammer_a, scenario, n)
    gp = warm_gp if warm_gp is not None else GPResidual(**gp_kwargs)
    gp.set_data(sensor_positions, observations - f)
    gp.optimize(restarts=restarts)
    return gp
