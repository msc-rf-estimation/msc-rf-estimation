# Import packages
import pytest
import numpy as np
from contested_rf.propagation.path_loss import free_space_path_loss
from contested_rf.propagation.path_loss import log_distance_path_loss
from contested_rf.propagation.shadow_fading import sample_shadowing_2d
from contested_rf.propagation.antenna import directional_gain

def test_fspl_at_reference_distance():
    """Test that FSPL at reference distance equals 40.0 dB, from hand calcs."""
    # Arrange - set up the inputs
    d = 1.0
    freq = 2.4e9

    # Act - call the function
    result = free_space_path_loss(d, freq)

    # Assert - check the output matches expectations
    assert result == pytest.approx(40.0, abs=0.1)

def test_fspl_at_100m():
    """Test that FSPL at 100m equals 80.0 dB, from hand calcs."""
    # Arrange - set up the inputs
    d = 100.0
    freq = 2.4e9
    
    # Act - call the function
    result = free_space_path_loss(d, freq)

    # Assert - check the output matches expectations
    assert result == pytest.approx(80.0, abs=0.1)
    
def test_log_distance_at_100m_n25():
    """Test that log path loss at 100m is 90.0 dB, from hand calcs."""
    # Arrange - set up the inputs
    d = 100.0
    freq = 2.4e9
    d0 = 1.0
    n = 2.5
    pl_d0 = free_space_path_loss(d0, freq)

    # Act - call the function
    result = log_distance_path_loss(d, d0, n, pl_d0)

    # Assert - check the output matches expectations
    assert result == pytest.approx(90.0, abs=0.1)

def test_log_distance_at_500m_n25():
    """Test that log path loss at 500m is 107.5 dB, from hand calcs."""
    # Arrange - set up the inputs
    d = 500.0
    freq = 2.4e9
    d0 = 1.0
    n = 2.5
    pl_d0 = free_space_path_loss(d0, freq)

    # Act - call the function
    result = log_distance_path_loss(d, d0, n, pl_d0)

    # Assert - check the output matches expectations
    assert result == pytest.approx(107.5, abs=0.1)

def test_log_distance_at_1km_n25():
    """Test that log path loss at 1km is 115 dB, from hand calcs."""
    # Arrange - set up the inputs
    d = 1000.0
    freq = 2.4e9
    d0 = 1.0
    n = 2.5
    pl_d0 = free_space_path_loss(d0, freq)

    # Act - call the function
    result = log_distance_path_loss(d, d0, n, pl_d0)

    # Assert - check the ouput matches expectations
    assert result == pytest.approx(115, abs=0.1)

def test_log_distance_at_100m_n20():
    """Test that log path loss at 100m is 80.0 dB, from hand calcs."""
    # Arrange - set up the inputs
    d = 100.0
    freq = 2.4e9
    d0 = 1.0
    n = 2.0
    pl_d0 = free_space_path_loss(d0, freq)

    # Act - call the function
    result = log_distance_path_loss(d, d0, n, pl_d0)

    # Assert - check the ouput matches expectations
    assert result == pytest.approx(80.0, abs=0.1)

def test_log_distance_at_500m_n20():
    """Test that log path loss at 500m is 94.0 dB, from hand calcs."""
    # Arrange - set up the inputs
    d = 500.0
    freq = 2.4e9
    d0 = 1.0
    n = 2.0
    pl_d0 = free_space_path_loss(d0, freq)

    # Act - call the function
    result = log_distance_path_loss(d, d0, n, pl_d0)

    # Assert - check the ouput matches expectations
    assert result == pytest.approx(94.0, abs=0.1)

def test_log_distance_at_1km_n20():
    """Test that log path loss at 1km is 90 dB, from hand calcs."""
    # Arrange - set up the inputs
    d = 1000.0
    freq = 2.4e9
    d0 = 1.0
    n = 2.0
    pl_d0 = free_space_path_loss(d0, freq)

    # Act - call the function
    result = log_distance_path_loss(d, d0, n, pl_d0)

    # Assert - check the ouput matches expectations
    assert result == pytest.approx(100.0, abs=0.1)

def test_log_distance_with_n2_equals_fpsl():
    "Test that log-distance reproduces free-space behaviour at any distance."
    # Arrange - set up the inputs
    d = 500.00
    freq = 2.4e9
    d0 = 1.0
    n = 2.0
    pl_d0 = free_space_path_loss(d0, freq)

    # Act - call the function
    result_log_distance = log_distance_path_loss(d, d0, n, pl_d0)
    result_fspl = free_space_path_loss(d, freq)
    
    # Assert - check the output matches expectations
    assert result_log_distance == pytest.approx(result_fspl, abs=0.1)

def test_residual_grows_with_distance():
    """Test that the residual grows with distance."""
    # Arrange - set up the inputs
    freq = 2.4e9
    d0 = 1.0
    pl_d0 = free_space_path_loss(d0, freq)

    # Act - call the functions
    pl_truth_100 = log_distance_path_loss(100, d0, 2.5, pl_d0)
    pl_learner_100 = log_distance_path_loss(100, d0, 2.0, pl_d0)

    pl_truth_500 = log_distance_path_loss(500, d0, 2.5, pl_d0)
    pl_learner_500 = log_distance_path_loss(500, d0, 2.0, pl_d0)

    pl_truth_1000 = log_distance_path_loss(1000, d0, 2.5, pl_d0)
    pl_learner_1000 = log_distance_path_loss(1000, d0, 2.0, pl_d0)
    
    # Assert - check the output matches expectations 
    assert pl_truth_100 - pl_learner_100 == pytest.approx(10.0, abs=0.1)
    assert pl_truth_500 - pl_learner_500 == pytest.approx(13.5, abs=0.1)
    assert pl_truth_1000 - pl_learner_1000 == pytest.approx(15.0, abs=0.1)


def test_path_loss_increases_with_distance():
    """Prove integrity of model by testing that path loss increases with distance."""
    # Arrange - set up the inputs
    freq = 2.4e9
    d0 = 1.0
    n = 2.0
    pl_d0 = free_space_path_loss(d0, freq)

    # Act - call the functions
    pl_1000 = log_distance_path_loss(1000, d0, n, pl_d0)
    pl_500 = log_distance_path_loss(500, d0, n, pl_d0)

    # Assert - check the output matches expectations 
    assert pl_1000 > pl_500

def test_path_loss_increases_with_frequency():
    """Prove integrity of model by testing that path loss increases with frequency."""
    # Arrange - set up the inputs
    d = 1000.0
    d0 = 1.0
    n = 2.0
    pl_d0_hf = free_space_path_loss(d0, 4.8e9)
    pl_d0_lf = free_space_path_loss(d0, 2.4e9)

    # Act - call the function
    pl_hf = log_distance_path_loss(d, d0, n, pl_d0_hf)
    pl_lf = log_distance_path_loss(d, d0, n, pl_d0_lf)

    # Assert - check the output matches expectations 
    assert pl_hf > pl_lf

def test_fspl_accepts_array():
    """Test that FSPL accepts a NumPy array of distances and returns an array."""
    # Arrange - set up the inputs
    distances = np.array([100, 400, 1000])
    freq = 2.4e9

    # Act - call the function(s)
    result = free_space_path_loss(distances, freq)

    # Assert - check the output matches expectations
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)

def test_log_distance_accepts_2d_grid():
    """Test that log distance accepts a NumPy array of distances and returns an array."""
    # Arrange - set up the inputs
    distances = np.array([100, 500, 1000])
    freq = 2.4e9
    d0 = 1.0
    n = 2.0
    pl_d0 = free_space_path_loss(d0, freq)

    # Act - call the function(s)
    result = log_distance_path_loss(distances, d0, n, pl_d0)

    # Assert - check the output matches expectations
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_shadow_fading_output_shape():
    """Test that sample_shadowing_2d returns one value per input point."""
    # Arrange - set up a small grid of 2D query points
    query_points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    sigma_sq = 16.0     # σ² = 16 → σ = 4 dB
    L = 50.0
    
    # Act - call the function(s)
    result = sample_shadowing_2d(query_points, sigma_sq=sigma_sq, L=L, seed=42)

    # Assert - output should be a 1D array wiith one value per input point
    assert result.shape == (5,)


def test_shadow_fading_seeds_control_outputs():
    """Test that sample_shadowing_2d returns the same output with the same seed."""
    
    # Arrange - set up a small grid of 2D query points
    query_points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    sigma_sq = 16.0     # σ² = 16 → σ = 4 dB
    L = 50.0

    # Act - call the function(s)
    arr1 = sample_shadowing_2d(query_points, sigma_sq=sigma_sq, L=L, seed=52)
    arr2 = sample_shadowing_2d(query_points, sigma_sq=sigma_sq, L=L, seed=52)
    arr3 = sample_shadowing_2d(query_points, sigma_sq=sigma_sq, L=L, seed=48)

    # Assert - check the output matches expectations
    assert np.array_equal(arr1, arr2)
    assert not np.array_equal(arr1, arr3)


def test_shadow_fading_mean_across_realisations():
    """Test that the mean of many realisations at a fixed point is approximately zero."""
    
    # Arrange - set up the inputs
    query_points = np.array([[0.0, 0.0]])
    sigma_sq = 16.0
    L = 50.0
    n_realisations = 200
    
    # Act - collect the value at the fixed point across many seeds
    samples = []
    for i in range(n_realisations):
        result = sample_shadowing_2d(query_points, sigma_sq=sigma_sq, L=L, seed=i)
        samples.append(result[0])
    samples = np.array(samples)

    # Assert - sample mean should be near zero (tolerance from standard error)
    assert abs(samples.mean()) < 0.5


def test_shadow_fading_std_across_realisations():
    """Test that the spread of realisations is approximately σ."""
    
    # Arrange - set up the inputs
    query_points = np.array([[0.0, 0.0]])
    sigma_sq = 16.0
    L = 50.0
    n_realisations = 200

    # Act - call the function(s)
    samples = []
    for i in range(n_realisations):
        result = sample_shadowing_2d(query_points, sigma_sq=sigma_sq, L=L, seed=i)
        samples.append(result[0])
    samples = np.array(samples)

    # Assert - check the output matches expeectations
    assert abs(samples.std() - np.sqrt(sigma_sq)) < 0.4


# ---------------------------------------------------------------------------
# Antenna directional gain tests
# ---------------------------------------------------------------------------

def test_antenna_peak_gain_at_main_beam():
    """Test that the gain equals G0 exactly when theta equals theta_main."""
    # Arrange
    theta_main = 45.0
    theta_3db = 60.0
    G0 = 27.0

    # Act
    result = directional_gain(theta_main, theta_main, theta_3db, G0_dB=G0)

    # Assert
    assert result == pytest.approx(G0, abs=1e-9)


def test_antenna_minus_3db_at_half_beamwidth():
    """Test that gain drops by 3 dB at theta = theta_main + theta_3db / 2."""
    theta_main = 0.0
    theta_3db = 60.0
    G0 = 20.0

    result = directional_gain(theta_main + theta_3db / 2, theta_main, theta_3db, G0_dB=G0)

    assert result == pytest.approx(G0 - 3.0, abs=0.05)


def test_antenna_minus_3db_at_negative_half_beamwidth():
    """Test that gain drops by 3 dB at theta = theta_main - theta_3db / 2 (symmetry)."""
    theta_main = 0.0
    theta_3db = 60.0
    G0 = 20.0

    result = directional_gain(theta_main - theta_3db / 2, theta_main, theta_3db, G0_dB=G0)

    assert result == pytest.approx(G0 - 3.0, abs=0.05)


def test_antenna_handles_angle_wrap():
    """Test that wraparound at 0/360 degrees is handled correctly."""
    # Main beam at 10 deg. A query at 350 deg is 20 deg away (not 340), so the
    # gain at 350 deg should match the gain at -10 deg.
    theta_main = 10.0
    theta_3db = 60.0
    G0 = 0.0

    result_wrapped = directional_gain(350.0, theta_main, theta_3db, G0_dB=G0)
    result_direct = directional_gain(-10.0, theta_main, theta_3db, G0_dB=G0)

    assert result_wrapped == pytest.approx(result_direct, abs=1e-9)


def test_antenna_vectorised_over_array():
    """Test that the function returns the right shape and values for an array of angles."""
    angles = np.array([0.0, 30.0, 60.0, 90.0])
    result = directional_gain(angles, theta_main=0.0, theta_3db=60.0, G0_dB=20.0)

    # Shape preserved
    assert result.shape == (4,)
    # Peak at boresight
    assert result[0] == pytest.approx(20.0, abs=1e-9)
    # -3 dB at half-beamwidth (30 deg)
    assert result[1] == pytest.approx(17.0, abs=0.05)


def test_antenna_back_lobe_clamped_to_front_back_ratio():
    """With a finite front-to-back ratio the back lobe sits exactly F_b below
    peak, instead of the bare Gaussian's unphysical ~-108 dB null. The clamped
    depth is what keeps ground truth consistent with the SMC's clamped model."""
    G0, theta_3db, fb = 0.0, 60.0, 30.0
    # 180 deg off boresight: bare Gaussian is ~-108 dB, clamp holds it at -30.
    bare = directional_gain(180.0, theta_main=0.0, theta_3db=theta_3db, G0_dB=G0)
    clamped = directional_gain(180.0, theta_main=0.0, theta_3db=theta_3db,
                               G0_dB=G0, front_back_ratio_dB=fb)
    assert bare < -100.0
    assert clamped == pytest.approx(-fb, abs=1e-9)
    # Inside the main lobe the clamp is inactive: -3 dB at the half-beamwidth.
    near = directional_gain(theta_3db / 2, theta_main=0.0, theta_3db=theta_3db,
                            G0_dB=G0, front_back_ratio_dB=fb)
    assert near == pytest.approx(-3.0, abs=0.05)


def test_jammer_directional_gain_uses_front_back_ratio():
    """A Jammer configured with front_back_ratio_dB clamps its own back lobe,
    so ground-truth simulators and the estimator model agree on the antenna."""
    from contested_rf.simulation.jammer import Jammer
    j = Jammer(name="B", position=(0.0, 0.0), power_dBm=30.0, is_directional=True,
               theta_main_deg=0.0, theta_3db_deg=60.0, front_back_ratio_dB=30.0)
    # Sensor directly behind the boresight (to the -x side) is ~180 deg off.
    back = j.gain_toward((-100.0, 0.0))
    assert back == pytest.approx(-30.0, abs=1e-6)
