"""Tests for the combined estimator, baselines, and RMSE metric."""
import numpy as np

from contested_rf.estimators import combined as C
from contested_rf.metrics.rmse import grid_rmse, convergence_step
from contested_rf.simulation.scenario import SCENARIO_1


def test_parametric_power_decreases_with_distance():
    theta = (1000.0, 1000.0, 30.0)
    near = C.parametric_power_dBm(theta, np.array([[1010.0, 1000.0]]), SCENARIO_1)
    far = C.parametric_power_dBm(theta, np.array([[1500.0, 1000.0]]), SCENARIO_1)
    assert near[0] > far[0]


def test_sinr_maps_have_grid_shape():
    theta = (1200.0, 800.0, 30.0)
    X, Y, sinr = C.parametric_sinr_map(theta, SCENARIO_1, grid_shape=(30, 30))
    assert sinr.shape == (30, 30)
    _, _, sinr_nj = C.no_jamming_sinr_map(SCENARIO_1, grid_shape=(30, 30))
    # No-jamming SINR should exceed jammed SINR everywhere (no interference).
    assert np.all(sinr_nj >= sinr - 1e-9)


def test_combined_equals_parametric_when_gp_is_zero():
    # A GP with no data predicts 0 mean -> combined == parametric.
    theta = (1200.0, 800.0, 30.0)
    gp = C.GPResidual()
    _, _, sinr_c = C.combined_sinr_map(theta, gp, SCENARIO_1, grid_shape=(20, 20))
    _, _, sinr_p = C.parametric_sinr_map(theta, SCENARIO_1, grid_shape=(20, 20))
    assert np.allclose(sinr_c, sinr_p)


def test_grid_rmse_zero_for_identical_maps():
    a = np.random.default_rng(0).normal(size=(20, 20))
    assert grid_rmse(a, a) == 0.0
    assert grid_rmse(a, a + 2.0) == 2.0  # constant 2 dB offset -> RMSE 2


def test_convergence_step_detects_sustained_crossing():
    counts = [50, 100, 200, 400, 800]
    rmse = [8.0, 6.0, 3.5, 3.0, 2.8]  # crosses 4 dB at 200 and stays below
    assert convergence_step(rmse, counts, threshold=4.0, window=1) == 200
    never = [8.0, 7.0, 6.0, 5.0, 4.5]
    assert convergence_step(never, counts, threshold=4.0, window=1) == -1


def test_convergence_step_no_false_positive_on_plateau_with_window():
    """Regression: a series that plateaus ABOVE the threshold must never report
    a convergence. The previous np.convolve(mode="same") smoother divided edge
    windows by the full window width, dragging the plateau's tail spuriously
    below threshold and returning a (false) convergence at the last checkpoint."""
    counts = [500, 1000, 2000, 4000, 8000]
    plateau = [5.0, 5.0, 5.0, 5.0, 5.0]  # never crosses 4 dB
    assert convergence_step(plateau, counts, threshold=4.0, window=100) == -1
    # And a genuine sustained crossing is still detected under a real window.
    crossing = [9.0, 3.5, 3.4, 3.3, 3.2]
    assert convergence_step(crossing, counts, threshold=4.0, window=3) == 4000
