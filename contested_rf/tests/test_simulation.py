"""Tests for the simulation layer: Jammer dataclass and scenario definitions."""
import math

import numpy as np
import pytest

from contested_rf.simulation.base_station import BaseStation
from contested_rf.simulation.ground_truth import compute_sinr_map, generate_measurements
from contested_rf.simulation.jammer import Jammer
from contested_rf.simulation.power_map import compute_received_power_map
from contested_rf.simulation.scenario import (
    SCENARIO_1,
    SCENARIO_2,
    SCENARIO_3,
    SCENARIOS,
)
from contested_rf.simulation.uav import generate_uav_observations, lawnmower_trajectory


# ---------------------------------------------------------------------------
# Jammer.position_at — static
# ---------------------------------------------------------------------------

def test_jammer_static_position_ignores_time():
    """A static jammer (velocity 0) returns initial position at any t."""
    jammer = Jammer(name="A", position=(100.0, 200.0), power_dBm=30.0)

    assert jammer.position_at(0.0) == (100.0, 200.0)
    assert jammer.position_at(1000.0) == (100.0, 200.0)


# ---------------------------------------------------------------------------
# Jammer.position_at — dynamic
# ---------------------------------------------------------------------------

def test_jammer_dynamic_position_at_start():
    """At t=0 the dynamic jammer is still at its initial position."""
    jammer = Jammer(
        name="A",
        position=(0.0, 0.0),
        power_dBm=30.0,
        velocity_mps=2.0,
        target_position=(100.0, 0.0),
    )

    assert jammer.position_at(0.0) == (0.0, 0.0)


def test_jammer_dynamic_position_at_midpoint():
    """At t=25s with v=2 m/s toward (100,0), jammer should be at (50, 0)."""
    jammer = Jammer(
        name="A",
        position=(0.0, 0.0),
        power_dBm=30.0,
        velocity_mps=2.0,
        target_position=(100.0, 0.0),
    )

    x, y = jammer.position_at(25.0)
    assert x == pytest.approx(50.0)
    assert y == pytest.approx(0.0)


def test_jammer_dynamic_stops_at_target():
    """Once the jammer has travelled past the target distance, it sits at the target."""
    jammer = Jammer(
        name="A",
        position=(0.0, 0.0),
        power_dBm=30.0,
        velocity_mps=2.0,
        target_position=(100.0, 0.0),
    )

    # Distance to target is 100 m at 2 m/s => arrival at t=50 s. Anything later
    # should still report the target position, not overshoot.
    assert jammer.position_at(60.0) == (100.0, 0.0)
    assert jammer.position_at(1000.0) == (100.0, 0.0)


# ---------------------------------------------------------------------------
# Jammer.gain_toward — omni
# ---------------------------------------------------------------------------

def test_jammer_omni_gain_independent_of_direction():
    """Omni jammers radiate peak_gain_dB in every direction."""
    jammer = Jammer(name="A", position=(0.0, 0.0), power_dBm=30.0)

    # Default peak_gain_dB is 0 dB.
    assert jammer.gain_toward((100.0, 0.0)) == 0.0
    assert jammer.gain_toward((0.0, 100.0)) == 0.0
    assert jammer.gain_toward((-50.0, -50.0)) == 0.0


# ---------------------------------------------------------------------------
# Jammer.gain_toward — directional
# ---------------------------------------------------------------------------

def test_jammer_directional_gain_peak_at_main_beam():
    """A sensor directly along the main beam direction gets peak gain."""
    jammer = Jammer(
        name="B",
        position=(0.0, 0.0),
        power_dBm=27.0,
        is_directional=True,
        theta_main_deg=0.0,  # main beam pointing east (along +x)
        theta_3db_deg=60.0,
        peak_gain_dB=10.0,
    )

    # Sensor due east of the jammer.
    assert jammer.gain_toward((100.0, 0.0)) == pytest.approx(10.0)


def test_jammer_directional_gain_minus_3db_at_half_beamwidth():
    """A sensor at +30 degrees off boresight (half of a 60-deg beamwidth) sees -3 dB."""
    jammer = Jammer(
        name="B",
        position=(0.0, 0.0),
        power_dBm=27.0,
        is_directional=True,
        theta_main_deg=0.0,  # boresight east
        theta_3db_deg=60.0,
        peak_gain_dB=10.0,
    )

    # Sensor at 30 degrees off east: bearing = atan2(sin(30), cos(30))
    sensor = (math.cos(math.radians(30.0)) * 100.0, math.sin(math.radians(30.0)) * 100.0)
    assert jammer.gain_toward(sensor) == pytest.approx(10.0 - 3.0, abs=0.05)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def test_scenario_1_has_one_static_omni_jammer():
    """Scenario 1: single static omni at (1200, 800), 30 dBm."""
    assert len(SCENARIO_1.jammers) == 1
    j = SCENARIO_1.jammers[0]
    assert j.position == (1200.0, 800.0)
    assert j.power_dBm == 30.0
    assert not j.is_directional
    assert j.velocity_mps == 0.0


def test_scenario_2_has_one_omni_and_one_directional():
    """Scenario 2: A omni, B directional with 60-deg beamwidth."""
    assert len(SCENARIO_2.jammers) == 2

    a, b = SCENARIO_2.jammers
    assert a.name == "A"
    assert not a.is_directional
    assert a.power_dBm == 30.0

    assert b.name == "B"
    assert b.is_directional
    assert b.power_dBm == 27.0
    assert b.theta_3db_deg == 60.0


def test_scenario_3_has_dynamic_jammer():
    """Scenario 3: single dynamic jammer moving from (500,500) to (1500,1500) at 2 m/s."""
    assert len(SCENARIO_3.jammers) == 1
    j = SCENARIO_3.jammers[0]
    assert j.velocity_mps == 2.0
    assert j.target_position == (1500.0, 1500.0)


def test_scenarios_lookup_table_contains_three():
    """The SCENARIOS dict provides integer lookup for the three scenarios."""
    assert set(SCENARIOS.keys()) == {1, 2, 3}
    assert SCENARIOS[1] is SCENARIO_1
    assert SCENARIOS[2] is SCENARIO_2
    assert SCENARIOS[3] is SCENARIO_3


# ---------------------------------------------------------------------------
# Integration: 2D received power map
# ---------------------------------------------------------------------------

def test_power_map_returns_correct_shapes():
    """X, Y, and P should match the requested grid_shape."""
    X, Y, P = compute_received_power_map(
        SCENARIO_1, grid_shape=(50, 50), seed=42
    )

    assert X.shape == (50, 50)
    assert Y.shape == (50, 50)
    assert P.shape == (50, 50)


def test_power_map_is_stronger_near_jammer_than_at_corner():
    """Received power near the jammer must exceed power at a far corner."""
    # 50x50 keeps the Cholesky on the shadow fading covariance matrix tractable.
    X, Y, P = compute_received_power_map(
        SCENARIO_1, grid_shape=(50, 50), seed=42
    )

    jx, jy = SCENARIO_1.jammers[0].position
    d_sq = (X - jx) ** 2 + (Y - jy) ** 2
    near_idx = np.unravel_index(d_sq.argmin(), d_sq.shape)

    # Path loss difference between ~10 m and ~1.5 km swamps shadow fading.
    assert P[near_idx] > P[0, 0]
    assert P[near_idx] > P[-1, -1]


def test_power_map_reproducible_with_seed():
    """Same seed must give identical power maps."""
    _, _, P1 = compute_received_power_map(
        SCENARIO_1, grid_shape=(30, 30), seed=7
    )
    _, _, P2 = compute_received_power_map(
        SCENARIO_1, grid_shape=(30, 30), seed=7
    )

    np.testing.assert_array_equal(P1, P2)


def test_power_map_works_for_directional_scenario():
    """Scenario 2 contains a directional jammer; the function should run and
    return a finite map."""
    _, _, P = compute_received_power_map(
        SCENARIO_2, grid_shape=(30, 30), seed=1
    )

    assert P.shape == (30, 30)
    assert np.all(np.isfinite(P))


def test_power_map_dynamic_jammer_position_changes_with_time():
    """For Scenario 3, the power map should differ between t=0 and t=500s
    because the jammer has moved substantially."""
    _, _, P_t0 = compute_received_power_map(
        SCENARIO_3, grid_shape=(30, 30), seed=1, t_sec=0.0
    )
    _, _, P_t500 = compute_received_power_map(
        SCENARIO_3, grid_shape=(30, 30), seed=1, t_sec=500.0
    )

    # Same seed -> same shadow fading. Different jammer position -> different
    # path loss field. Maps must differ.
    assert not np.array_equal(P_t0, P_t500)


# ---------------------------------------------------------------------------
# Ground truth simulator: generate_measurements
# ---------------------------------------------------------------------------

def test_generate_measurements_returns_one_value_per_sensor():
    """Output length must equal number of sensor positions."""
    sensors = np.array([[100.0, 100.0], [500.0, 500.0], [1500.0, 1500.0]])
    z = generate_measurements(SCENARIO_1, sensors, seed=42)

    assert z.shape == (3,)


def test_generate_measurements_reproducible_with_seed():
    """Same seed must give identical measurements."""
    sensors = np.array([[100.0, 100.0], [500.0, 500.0], [1500.0, 1500.0]])
    z1 = generate_measurements(SCENARIO_1, sensors, seed=7)
    z2 = generate_measurements(SCENARIO_1, sensors, seed=7)

    np.testing.assert_array_equal(z1, z2)


def test_generate_measurements_stronger_at_near_sensor_than_far_sensor():
    """A sensor close to the jammer reads higher power than one far away."""
    jx, jy = SCENARIO_1.jammers[0].position  # (1200, 800)
    sensors = np.array([[jx + 10.0, jy], [0.0, 0.0]])

    z = generate_measurements(SCENARIO_1, sensors, seed=42)

    assert z[0] > z[1]


# ---------------------------------------------------------------------------
# Scenario: base station presence
# ---------------------------------------------------------------------------

def test_scenarios_have_default_base_station():
    """All three scenarios should have a base station with sensible defaults."""
    for scenario in (SCENARIO_1, SCENARIO_2, SCENARIO_3):
        assert isinstance(scenario.base_station, BaseStation)
        assert scenario.base_station.power_dBm == 30.0
        assert scenario.base_station.position == (1000.0, 1000.0)


# ---------------------------------------------------------------------------
# SINR map
# ---------------------------------------------------------------------------

def test_sinr_map_returns_correct_shapes():
    """X, Y, SINR should match the requested grid_shape."""
    X, Y, SINR = compute_sinr_map(SCENARIO_1, grid_shape=(30, 30))

    assert X.shape == (30, 30)
    assert Y.shape == (30, 30)
    assert SINR.shape == (30, 30)


def test_sinr_map_returns_finite_values():
    """SINR map should be finite everywhere (no inf/nan)."""
    _, _, SINR = compute_sinr_map(SCENARIO_1, grid_shape=(30, 30))

    assert np.all(np.isfinite(SINR))


def test_sinr_map_higher_near_base_station_than_near_jammer():
    """SINR near the base station should exceed SINR near the jammer."""
    X, Y, SINR = compute_sinr_map(SCENARIO_1, grid_shape=(50, 50))

    bx, by = SCENARIO_1.base_station.position  # (1000, 1000)
    jx, jy = SCENARIO_1.jammers[0].position  # (1200, 800)

    d_bs_sq = (X - bx) ** 2 + (Y - by) ** 2
    d_j_sq = (X - jx) ** 2 + (Y - jy) ** 2

    near_bs = np.unravel_index(d_bs_sq.argmin(), d_bs_sq.shape)
    near_jammer = np.unravel_index(d_j_sq.argmin(), d_j_sq.shape)

    assert SINR[near_bs] > SINR[near_jammer]


def test_sinr_map_works_for_two_jammer_scenario():
    """Scenario 2's directional jammer should not break the SINR computation."""
    _, _, SINR = compute_sinr_map(SCENARIO_2, grid_shape=(30, 30))

    assert SINR.shape == (30, 30)
    assert np.all(np.isfinite(SINR))


# ---------------------------------------------------------------------------
# UAV trajectory + observations
# ---------------------------------------------------------------------------

def test_lawnmower_trajectory_returns_consistent_shapes():
    """Positions should be (N, 2), timestamps (N,), with N > 0."""
    positions, timestamps = lawnmower_trajectory(
        x_range=(0.0, 1000.0),
        y_range=(0.0, 1000.0),
        track_spacing=500.0,  # 2 tracks for speed
        seed=1,
    )

    assert positions.ndim == 2
    assert positions.shape[1] == 2
    assert timestamps.shape == (positions.shape[0],)
    assert positions.shape[0] > 0


def test_lawnmower_trajectory_timestamps_monotonic():
    """Timestamps should be monotonically increasing along the flight path."""
    _, timestamps = lawnmower_trajectory(
        x_range=(0.0, 1000.0),
        y_range=(0.0, 1000.0),
        track_spacing=500.0,
        seed=1,
    )

    assert np.all(np.diff(timestamps) > 0)


def test_lawnmower_trajectory_reproducible_with_seed():
    """Same seed must produce identical trajectories."""
    p1, t1 = lawnmower_trajectory(
        x_range=(0.0, 1000.0), y_range=(0.0, 1000.0), track_spacing=500.0, seed=7
    )
    p2, t2 = lawnmower_trajectory(
        x_range=(0.0, 1000.0), y_range=(0.0, 1000.0), track_spacing=500.0, seed=7
    )

    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(t1, t2)


def test_uav_observations_have_four_uavs():
    """generate_uav_observations should produce observations from 4 distinct UAVs."""
    positions, timestamps, observations, uav_ids = generate_uav_observations(
        SCENARIO_1, track_spacing=500.0, seed=42
    )

    assert set(uav_ids) == {0, 1, 2, 3}
    assert positions.shape[0] == observations.shape[0]
    assert positions.shape[0] == timestamps.shape[0]
    assert positions.shape[0] == uav_ids.shape[0]


def test_uav_observations_are_finite():
    """All UAV-collected powers should be finite."""
    _, _, observations, _ = generate_uav_observations(
        SCENARIO_1, track_spacing=500.0, seed=42
    )

    assert np.all(np.isfinite(observations))


def test_uav_observations_reproducible_with_seed():
    """Same seed should produce identical UAV observations."""
    _, _, obs1, _ = generate_uav_observations(SCENARIO_1, track_spacing=500.0, seed=7)
    _, _, obs2, _ = generate_uav_observations(SCENARIO_1, track_spacing=500.0, seed=7)

    np.testing.assert_array_equal(obs1, obs2)


def test_uav_observations_quadrants_cover_distinct_regions():
    """Each UAV should be operating in its own quadrant (mean x and y separate quadrants)."""
    positions, _, _, uav_ids = generate_uav_observations(
        SCENARIO_1, track_spacing=500.0, seed=42
    )

    # SW quadrant should have low x and low y on average.
    sw_mask = uav_ids == 0
    ne_mask = uav_ids == 3

    sw_mean_x = positions[sw_mask, 0].mean()
    sw_mean_y = positions[sw_mask, 1].mean()
    ne_mean_x = positions[ne_mask, 0].mean()
    ne_mean_y = positions[ne_mask, 1].mean()

    assert sw_mean_x < ne_mean_x
    assert sw_mean_y < ne_mean_y
