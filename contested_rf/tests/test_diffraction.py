"""Tests for knife-edge diffraction (ITU-R P.526)."""
import numpy as np

from contested_rf.propagation.diffraction import (
    fresnel_parameter,
    knife_edge_loss_dB,
    diffraction_loss_from_profile,
)


def test_no_loss_well_below_los():
    # Obstruction far below the line of sight -> v very negative -> 0 dB.
    v = fresnel_parameter(h=-50.0, d1=500.0, d2=500.0, freq_Hz=2.4e9)
    assert v < -0.78
    assert knife_edge_loss_dB(v) == 0.0


def test_grazing_incidence_about_6dB():
    # At v = 0 (obstruction exactly on the LoS) loss is ~6 dB.
    assert abs(knife_edge_loss_dB(0.0) - 6.0) < 1.0


def test_loss_increases_with_obstruction_height():
    f = 2.4e9
    v_low = fresnel_parameter(h=5.0, d1=500.0, d2=500.0, freq_Hz=f)
    v_high = fresnel_parameter(h=30.0, d1=500.0, d2=500.0, freq_Hz=f)
    assert knife_edge_loss_dB(v_high) > knife_edge_loss_dB(v_low)
    # A 20 m ridge midway on a 1 km 2.4 GHz path is a heavy obstruction.
    assert knife_edge_loss_dB(v_high) > 15.0


def test_monotonic_in_v():
    v = np.linspace(-0.78, 5.0, 50)
    loss = knife_edge_loss_dB(v)
    assert np.all(np.diff(loss) >= -1e-9)


def test_profile_helper_matches_direct():
    # A 20 m obstruction midway, endpoints at 50 m height each.
    loss = diffraction_loss_from_profile(
        tx_xy=(0.0, 0.0), rx_xy=(1000.0, 0.0), obstruction_xy=(500.0, 0.0),
        obstruction_height=70.0, tx_height=50.0, rx_height=50.0, freq_Hz=2.4e9)
    # LoS at midpoint = 50 m; obstruction 70 m -> h = 20 m -> heavy loss.
    assert loss > 15.0


def test_clear_path_no_loss():
    loss = diffraction_loss_from_profile(
        tx_xy=(0.0, 0.0), rx_xy=(1000.0, 0.0), obstruction_xy=(500.0, 0.0),
        obstruction_height=30.0, tx_height=50.0, rx_height=50.0, freq_Hz=2.4e9)
    assert loss == 0.0  # obstruction below LoS
