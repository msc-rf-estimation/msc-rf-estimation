"""The combined estimator and its three comparators.

All four reconstruct the SINR map over the operating grid and are scored by
RMSE against the ground-truth map. They differ only in how they predict the
jammer interference field; the base-station signal and noise floor are known
to all of them.

    SINR_est(x) = P_signal_learner(x) / (P_interference_est(x) + P_noise)

    combined        parametric jammer field from the SMC posterior mean, plus
                    a GP correction learned on the residuals
    pure_parametric parametric field only
    pure_gp         GP fitted directly to the raw observations, constant mean
    no_jamming      zero interference; an upper bound, not a competitor

The learner uses n = 2.0 and no diffraction term against a truth of n = 2.5;
that gap is what the residual layer exists to absorb.
"""
from __future__ import annotations

import numpy as np

from contested_rf.estimators.gp_residual import GPResidual
from contested_rf.propagation.antenna import directional_gain
from contested_rf.propagation.path_loss import free_space_path_loss
from contested_rf.propagation.sinr import sinr_from_powers

N_LEARNER = 2.0            # learner path-loss exponent for the JAMMER (impoverished)
N_SIGNAL_KNOWN = 2.5       # base station is a KNOWN cooperative transmitter: its
                           # coverage is pre-characterised, so its signal field is
                           # reconstructed with the true propagation. Without this,
                           # the base-station model error (~14 dB) dominates the
                           # SINR RMSE and masks all estimator differences.
NOISE_FLOOR_DBM = -100.0   # known thermal/receiver noise floor
_D0 = 1.0
_MIN_INTERFERENCE_DBM = -400.0  # ~0 mW, for the no-jamming baseline


def _pl_d0(scenario):
    return free_space_path_loss(_D0, scenario.operating_freq_Hz)


def parametric_power_dBm(theta, points, scenario, n=N_LEARNER):
    """Learner-predicted received jammer power (dBm) at a set of points.

    Single omnidirectional jammer (Scenario 1): theta = (x, y, P_tx_dBm).

    Args:
        theta: (3,) jammer state (x, y, P_tx_dBm).
        points: (M, 2) query locations.
        scenario: Scenario (for carrier frequency).
        n: learner path-loss exponent.

    Returns:
        (M,) predicted received power in dBm.
    """
    points = np.atleast_2d(points)
    x, y, p_tx = theta
    dx = points[:, 0] - x
    dy = points[:, 1] - y
    d = np.maximum(np.sqrt(dx ** 2 + dy ** 2), _D0)
    pl = _pl_d0(scenario) + 10.0 * n * np.log10(d / _D0)
    return p_tx - pl


def learner_signal_dBm(scenario, points, n=N_SIGNAL_KNOWN):
    """Base-station received signal power in dBm at points.

    The base station is a known cooperative transmitter, so its field is
    reconstructed with the true propagation exponent; only the jammer field is
    subject to the impoverished learner model. Assumes an omnidirectional base
    station.
    """
    points = np.atleast_2d(points)
    bs = scenario.base_station
    bx, by = bs.position
    d = np.maximum(np.sqrt((points[:, 0] - bx) ** 2 + (points[:, 1] - by) ** 2), _D0)
    pl = _pl_d0(scenario) + 10.0 * n * np.log10(d / _D0)
    if bs.is_directional:
        bearings = np.degrees(np.arctan2(points[:, 1] - by, points[:, 0] - bx))
        gain = directional_gain(
            bearings,
            theta_main=bs.theta_main_deg,
            theta_3db=bs.theta_3db_deg,
            G0_dB=bs.peak_gain_dB,
            front_back_ratio_dB=getattr(bs, "front_back_ratio_dB", None),
        )
    else:
        gain = bs.peak_gain_dB
    return bs.power_dBm - pl + gain


def _grid_points(scenario, grid_shape):
    xmin, xmax = scenario.grid_xlim
    ymin, ymax = scenario.grid_ylim
    nx, ny = grid_shape
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    return X, Y, pts


def _sinr_from_interference(P_interf_dBm, P_signal_dBm):
    return sinr_from_powers(
        p_signal_dbm=P_signal_dBm,
        p_interference_dbm=P_interf_dBm,
        p_noise_dbm=NOISE_FLOOR_DBM,
    )


# --- Estimator SINR-map reconstructions -----------------------------------
def parametric_sinr_map(theta_hat, scenario, grid_shape=(60, 60), n=N_LEARNER):
    """Pure-parametric SINR map from the SMC weighted mean only."""
    X, Y, pts = _grid_points(scenario, grid_shape)
    P_interf = parametric_power_dBm(theta_hat, pts, scenario, n)
    P_signal = learner_signal_dBm(scenario, pts)  # known BS, true propagation
    sinr = _sinr_from_interference(P_interf, P_signal).reshape(X.shape)
    return X, Y, sinr


def combined_sinr_map(theta_hat, gp, scenario, grid_shape=(60, 60), n=N_LEARNER):
    """Combined SINR map: parametric jammer power + GP residual correction."""
    X, Y, pts = _grid_points(scenario, grid_shape)
    P_param = parametric_power_dBm(theta_hat, pts, scenario, n)
    resid = gp.predict(pts, return_var=False)
    P_interf = P_param + resid
    P_signal = learner_signal_dBm(scenario, pts)  # known BS, true propagation
    sinr = _sinr_from_interference(P_interf, P_signal).reshape(X.shape)
    return X, Y, sinr


def pure_gp_sinr_map(gp, gp_mean, scenario, grid_shape=(60, 60), n=N_LEARNER):
    """Pure-GP SINR map: GP fit to raw power observations (constant-mean)."""
    X, Y, pts = _grid_points(scenario, grid_shape)
    P_interf = gp.predict(pts, return_var=False) + gp_mean
    P_signal = learner_signal_dBm(scenario, pts)  # known BS, true propagation
    sinr = _sinr_from_interference(P_interf, P_signal).reshape(X.shape)
    return X, Y, sinr


def no_jamming_sinr_map(scenario, grid_shape=(60, 60), n=N_LEARNER):
    """No-jamming baseline: interference = 0, so SINR = signal / noise."""
    X, Y, pts = _grid_points(scenario, grid_shape)
    P_signal = learner_signal_dBm(scenario, pts)  # known BS, true propagation
    P_interf = np.full(pts.shape[0], _MIN_INTERFERENCE_DBM)
    sinr = _sinr_from_interference(P_interf, P_signal).reshape(X.shape)
    return X, Y, sinr


def fit_residual_gp(theta_hat, sensor_positions, observations, scenario,
                    n=N_LEARNER, warm_gp=None, optimize=True, restarts=1,
                    **gp_kwargs):
    """Fit the residual GP on observations seen so far, given theta_hat.

    r_k = z_k - f_learner(x_k, theta_hat). Returns a fitted GPResidual.
    """
    f_learner = parametric_power_dBm(theta_hat, sensor_positions, scenario, n)
    residuals = observations - f_learner
    gp = warm_gp if warm_gp is not None else GPResidual(**gp_kwargs)
    gp.set_data(sensor_positions, residuals)
    if optimize:
        gp.optimize(restarts=restarts)
    else:
        gp._build_cache()
    return gp


def fit_pure_gp(sensor_positions, observations, warm_gp=None, optimize=True,
                restarts=1, **gp_kwargs):
    """Fit the pure-GP baseline on raw power observations (constant mean).

    Returns (gp, gp_mean) where predictions are gp.predict(...) + gp_mean.
    """
    gp_mean = float(np.mean(observations))
    gp = warm_gp if warm_gp is not None else GPResidual(**gp_kwargs)
    gp.set_data(sensor_positions, observations - gp_mean)
    if optimize:
        gp.optimize(restarts=restarts)
    else:
        gp._build_cache()
    return gp, gp_mean
