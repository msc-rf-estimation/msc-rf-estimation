"""Tests for the SMC particle filter estimator."""
import numpy as np
import pytest

from contested_rf.estimators.evaluation import (
    evaluate_pf_result,
    format_evaluation_summary,
)
from contested_rf.estimators.particle_filter import (
    effective_sample_size,
    init_particles,
    init_particles_gaussian,
    jitter_particles,
    log_likelihood_per_particle,
    run_particle_filter,
    systematic_resample,
    update_log_weights,
)
from contested_rf.simulation.scenario import SCENARIO_1
from contested_rf.simulation.uav import generate_uav_observations
from contested_rf.propagation.path_loss import (
    free_space_path_loss,
    log_distance_path_loss,
)


# ---------------------------------------------------------------------------
# Prior sampling
# ---------------------------------------------------------------------------

def test_init_particles_returns_correct_shape():
    """N particles with 3 columns (x, y, P_tx)."""
    particles = init_particles(
        n_particles=500,
        x_range=(0.0, 2000.0),
        y_range=(0.0, 2000.0),
        power_dBm_range=(20.0, 40.0),
        seed=42,
    )

    assert particles.shape == (500, 3)


def test_init_particles_within_specified_ranges():
    """All particles must lie within the specified bounds."""
    particles = init_particles(
        n_particles=1000,
        x_range=(0.0, 2000.0),
        y_range=(0.0, 2000.0),
        power_dBm_range=(20.0, 40.0),
        seed=42,
    )

    assert (particles[:, 0] >= 0.0).all() and (particles[:, 0] <= 2000.0).all()
    assert (particles[:, 1] >= 0.0).all() and (particles[:, 1] <= 2000.0).all()
    assert (particles[:, 2] >= 20.0).all() and (particles[:, 2] <= 40.0).all()


def test_init_particles_reproducible_with_seed():
    """Same seed must produce identical particle clouds."""
    a = init_particles(100, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), seed=7)
    b = init_particles(100, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), seed=7)

    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Gaussian prior sampler
# ---------------------------------------------------------------------------

def test_init_particles_gaussian_returns_correct_shape():
    """N particles with 3 columns."""
    particles = init_particles_gaussian(
        n_particles=500,
        means=(1200.0, 800.0, 30.0),
        stds=(100.0, 100.0, 2.0),
        seed=42,
    )

    assert particles.shape == (500, 3)


def test_init_particles_gaussian_empirical_mean_matches_specified():
    """Empirical column means should approximate the specified means."""
    particles = init_particles_gaussian(
        n_particles=10000,
        means=(1200.0, 800.0, 30.0),
        stds=(50.0, 50.0, 1.0),
        seed=42,
    )

    empirical = particles.mean(axis=0)
    expected = np.array([1200.0, 800.0, 30.0])
    np.testing.assert_allclose(empirical, expected, atol=2.0)


def test_init_particles_gaussian_empirical_std_matches_specified():
    """Empirical column stds should approximate the specified stds."""
    particles = init_particles_gaussian(
        n_particles=10000,
        means=(1200.0, 800.0, 30.0),
        stds=(50.0, 50.0, 1.0),
        seed=42,
    )

    empirical = particles.std(axis=0)
    expected = np.array([50.0, 50.0, 1.0])
    np.testing.assert_allclose(empirical, expected, rtol=0.05)


def test_init_particles_gaussian_reproducible_with_seed():
    """Same seed must produce identical particle clouds."""
    a = init_particles_gaussian(100, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), seed=7)
    b = init_particles_gaussian(100, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), seed=7)

    np.testing.assert_array_equal(a, b)


def test_run_particle_filter_accepts_initial_particles():
    """When initial_particles is supplied, the PF must use it directly and
    infer N from its shape."""
    positions, timestamps, observations, _ = generate_uav_observations(
        SCENARIO_1, track_spacing=600.0, seed=1,
    )
    positions = positions[:20]
    timestamps = timestamps[:20]
    observations = observations[:20]

    # Supply a Gaussian prior centred near truth.
    initial = init_particles_gaussian(
        n_particles=300,
        means=(1200.0, 800.0, 30.0),
        stds=(100.0, 100.0, 2.0),
        seed=42,
    )

    result = run_particle_filter(
        SCENARIO_1, positions, timestamps, observations,
        initial_particles=initial,
        seed=1,
    )

    # Final particle count must equal whatever we supplied.
    assert result["particles_final"].shape == (300, 3)
    # Posterior summary should also have the right length.
    assert result["posterior_means"].shape == (20, 3)


# ---------------------------------------------------------------------------
# Likelihood evaluation
# ---------------------------------------------------------------------------

def test_log_likelihood_returns_one_value_per_particle():
    """Should return (N,) when given N particles."""
    particles = np.array([
        [100.0, 100.0, 30.0],
        [500.0, 500.0, 30.0],
        [1000.0, 1000.0, 30.0],
    ])
    log_lik = log_likelihood_per_particle(
        particles,
        sensor_position=(0.0, 0.0),
        observed_power_dBm=-50.0,
        operating_freq_Hz=2.4e9,
    )

    assert log_lik.shape == (3,)


def _predicted_power(particle_x, particle_y, particle_p_tx, sensor, freq, n=2.5):
    """Helper: compute the deterministic predicted power for one particle."""
    d = np.sqrt((sensor[0] - particle_x) ** 2 + (sensor[1] - particle_y) ** 2)
    pl_d0 = free_space_path_loss(1.0, freq)
    pl = log_distance_path_loss(d, 1.0, n, pl_d0)
    return particle_p_tx - pl


def test_log_likelihood_higher_when_particle_predicts_observation_exactly():
    """A particle whose prediction matches the observation should score
    higher than a particle whose prediction is off by 20 dB in transmit power."""
    sensor = (0.0, 0.0)
    freq = 2.4e9

    # Truth particle's predicted power at the sensor.
    truth_predicted = _predicted_power(500.0, 500.0, 30.0, sensor, freq)
    observed = truth_predicted  # observation matches truth exactly

    truth = np.array([[500.0, 500.0, 30.0]])
    wrong = np.array([[500.0, 500.0, 10.0]])  # 20 dB too quiet

    # _predicted_power generates `observed` at n=2.5, so the likelihood must be
    # evaluated at the same exponent for the matched particle to score residual 0.
    log_lik_truth = log_likelihood_per_particle(truth, sensor, observed, freq,
                                                path_loss_exponent=2.5)
    log_lik_wrong = log_likelihood_per_particle(wrong, sensor, observed, freq,
                                                path_loss_exponent=2.5)

    # Truth particle has residual 0 → log-likelihood = 0 (we drop the constant
    # Gaussian normaliser).
    assert log_lik_truth[0] == pytest.approx(0.0, abs=1e-9)
    # Wrong particle has a non-zero residual → strictly worse log-likelihood.
    assert log_lik_wrong[0] < log_lik_truth[0]


def test_log_likelihood_symmetric_in_residual_sign():
    """Residuals of +δ and -δ around the truth should give identical log-likelihoods."""
    sensor = (1000.0, 1000.0)
    freq = 2.4e9

    # Truth particle's predicted power; we use this as the observation so the
    # two test particles sit symmetrically either side of it in power.
    centre_predicted = _predicted_power(500.0, 500.0, 30.0, sensor, freq)
    observed = centre_predicted

    p_above = np.array([[500.0, 500.0, 35.0]])  # residual = -5 dB
    p_below = np.array([[500.0, 500.0, 25.0]])  # residual = +5 dB

    log_lik_above = log_likelihood_per_particle(p_above, sensor, observed, freq,
                                                path_loss_exponent=2.5)
    log_lik_below = log_likelihood_per_particle(p_below, sensor, observed, freq,
                                                path_loss_exponent=2.5)

    assert log_lik_above[0] == pytest.approx(log_lik_below[0], abs=1e-9)


def test_log_likelihood_vectorised_over_particles():
    """A particle very close to the sensor should score worse than one at the
    correct distance, all else equal."""
    sensor = (0.0, 0.0)
    freq = 2.4e9

    # Two particles with the same transmit power but at very different
    # distances. One predicts very strong signal, the other very weak. With a
    # fixed observed power, the one that predicts closer to the observation
    # wins.
    very_close = np.array([[10.0, 0.0, 30.0]])  # predicted ~+30 - 65 = -35 dBm
    far = np.array([[1500.0, 0.0, 30.0]])  # predicted ~+30 - 110 = -80 dBm

    observed = -80.0  # matches the "far" particle's prediction.

    log_lik_close = log_likelihood_per_particle(very_close, sensor, observed, freq)
    log_lik_far = log_likelihood_per_particle(far, sensor, observed, freq)

    assert log_lik_far[0] > log_lik_close[0]


# ---------------------------------------------------------------------------
# Weight update + ESS
# ---------------------------------------------------------------------------

def test_update_log_weights_produces_normalised_weights():
    """After update, exp(log_weights).sum() should be 1."""
    n = 100
    log_weights = np.full(n, -np.log(n))  # uniform initial weights
    log_likelihoods = np.random.default_rng(0).normal(0.0, 2.0, size=n)

    new_log_weights = update_log_weights(log_weights, log_likelihoods)

    assert np.exp(new_log_weights).sum() == pytest.approx(1.0, abs=1e-10)


def test_update_log_weights_favours_higher_likelihood_particles():
    """After update, the particle with the highest log-likelihood should
    have the highest new weight."""
    n = 5
    log_weights = np.full(n, -np.log(n))
    log_likelihoods = np.array([-10.0, -5.0, 0.0, -3.0, -8.0])

    new_log_weights = update_log_weights(log_weights, log_likelihoods)

    # Particle index 2 has the highest log-likelihood (0) so should get the
    # largest new weight.
    assert np.argmax(new_log_weights) == 2


def test_update_log_weights_handles_extreme_likelihoods():
    """Log-likelihoods that span 1000s of dB shouldn't overflow."""
    n = 3
    log_weights = np.full(n, -np.log(n))
    # Extreme range — one particle dominates massively.
    log_likelihoods = np.array([-1500.0, -1000.0, -100.0])

    new_log_weights = update_log_weights(log_weights, log_likelihoods)

    # Sum should still be 1, no NaN or inf.
    assert np.all(np.isfinite(new_log_weights))
    assert np.exp(new_log_weights).sum() == pytest.approx(1.0, abs=1e-10)


def test_ess_equals_n_when_weights_are_uniform():
    """A perfectly uniform swarm has ESS = N."""
    n = 250
    log_weights = np.full(n, -np.log(n))

    ess = effective_sample_size(log_weights)

    assert ess == pytest.approx(float(n), rel=1e-9)


def test_ess_approaches_one_when_one_particle_dominates():
    """If one particle has almost all the weight, ESS should approach 1."""
    n = 100
    # Put nearly all mass on particle 0.
    log_weights = np.full(n, -100.0)
    log_weights[0] = 0.0
    log_weights = log_weights - np.log(np.exp(log_weights).sum())  # normalise

    ess = effective_sample_size(log_weights)

    assert ess == pytest.approx(1.0, abs=0.01)


def test_ess_is_bounded_between_one_and_n():
    """ESS should always lie in [1, N] for any valid weight distribution."""
    n = 50
    rng = np.random.default_rng(7)
    log_weights = rng.normal(0.0, 3.0, size=n)
    # Normalise.
    log_weights = log_weights - np.log(np.exp(log_weights).sum())

    ess = effective_sample_size(log_weights)

    assert 1.0 <= ess <= float(n)


# ---------------------------------------------------------------------------
# Systematic resampling
# ---------------------------------------------------------------------------

def test_resample_preserves_particle_count():
    """Output particle count must equal input."""
    n = 100
    particles = np.arange(n * 3, dtype=float).reshape(n, 3)
    log_weights = np.full(n, -np.log(n))

    new_p, new_w = systematic_resample(particles, log_weights, seed=42)

    assert new_p.shape == (n, 3)
    assert new_w.shape == (n,)


def test_resample_resets_weights_to_uniform():
    """After resampling, all log-weights should be -log(N)."""
    n = 100
    particles = np.arange(n * 3, dtype=float).reshape(n, 3)
    log_weights = np.random.default_rng(0).normal(0.0, 1.0, size=n)
    log_weights = log_weights - np.log(np.exp(log_weights).sum())

    _, new_w = systematic_resample(particles, log_weights, seed=42)

    expected = np.full(n, -np.log(n))
    np.testing.assert_allclose(new_w, expected, atol=1e-12)


def test_resample_only_returns_input_particles():
    """Every resampled particle must equal one of the input particles."""
    n = 50
    particles = np.arange(n * 3, dtype=float).reshape(n, 3)
    log_weights = np.random.default_rng(0).normal(0.0, 2.0, size=n)
    log_weights = log_weights - np.log(np.exp(log_weights).sum())

    new_p, _ = systematic_resample(particles, log_weights, seed=42)

    # Every resampled particle's row should appear in the original set.
    input_rows = {tuple(row) for row in particles}
    for row in new_p:
        assert tuple(row) in input_rows


def test_resample_concentrates_on_high_weight_particles():
    """Particles with overwhelming weight should appear most often."""
    n = 1000
    particles = np.arange(n * 3, dtype=float).reshape(n, 3)
    # Put 99% of weight on particle 0.
    log_weights = np.full(n, -50.0)
    log_weights[0] = 0.0
    log_weights = log_weights - np.log(np.exp(log_weights).sum())

    new_p, _ = systematic_resample(particles, log_weights, seed=42)

    # Almost all resampled particles should be copies of particle 0
    # (i.e., have its row signature).
    matches_particle_0 = (new_p == particles[0]).all(axis=1).sum()
    assert matches_particle_0 > 0.95 * n


def test_resample_with_uniform_weights_returns_original_set():
    """With uniform weights, systematic resampling returns one copy of each
    particle (because the deterministic grid lines up with the inverse CDF)."""
    n = 50
    particles = np.arange(n * 3, dtype=float).reshape(n, 3)
    log_weights = np.full(n, -np.log(n))

    new_p, _ = systematic_resample(particles, log_weights, seed=42)

    # Resampled set should be a permutation (in this case, exactly the input
    # order) of the original — no particle is lost or duplicated.
    np.testing.assert_array_equal(np.sort(new_p, axis=0), np.sort(particles, axis=0))


def test_resample_reproducible_with_seed():
    """Same seed must produce identical resampled particles."""
    n = 100
    particles = np.arange(n * 3, dtype=float).reshape(n, 3)
    log_weights = np.random.default_rng(0).normal(0.0, 1.0, size=n)
    log_weights = log_weights - np.log(np.exp(log_weights).sum())

    p1, _ = systematic_resample(particles, log_weights, seed=7)
    p2, _ = systematic_resample(particles, log_weights, seed=7)

    np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Jitter
# ---------------------------------------------------------------------------

def test_jitter_preserves_particle_shape():
    """Jitter shouldn't change the array shape."""
    n = 100
    particles = np.zeros((n, 3))
    jittered = jitter_particles(particles, jitter_sigmas=[5.0, 5.0, 0.2], seed=42)

    assert jittered.shape == (n, 3)


def test_jitter_with_zero_sigmas_returns_input_unchanged():
    """With zero jitter, every particle should be unchanged."""
    n = 50
    particles = np.random.default_rng(0).normal(0.0, 100.0, size=(n, 3))
    jittered = jitter_particles(particles, jitter_sigmas=[0.0, 0.0, 0.0], seed=42)

    np.testing.assert_array_equal(jittered, particles)


def test_jitter_perturbs_each_dimension_by_specified_std():
    """The std of the per-dimension noise should match jitter_sigmas."""
    n = 10000  # large N for tight empirical std
    particles = np.zeros((n, 3))
    sigmas = np.array([5.0, 10.0, 0.5])

    jittered = jitter_particles(particles, jitter_sigmas=sigmas, seed=42)

    empirical_std = jittered.std(axis=0)
    np.testing.assert_allclose(empirical_std, sigmas, rtol=0.05)


def test_jitter_breaks_identical_duplicates():
    """The clinical case: N identical duplicates become N distinct particles."""
    n = 100
    # Every particle is the same point.
    particles = np.tile(np.array([1000.0, 800.0, 30.0]), (n, 1))

    jittered = jitter_particles(particles, jitter_sigmas=[5.0, 5.0, 0.2], seed=42)

    # No two jittered particles should be exactly identical.
    # Check by counting unique rows.
    unique_rows = np.unique(jittered, axis=0)
    assert unique_rows.shape[0] == n


def test_jitter_reproducible_with_seed():
    """Same seed must produce identical jittered particles."""
    particles = np.zeros((50, 3))
    j1 = jitter_particles(particles, [1.0, 1.0, 0.1], seed=7)
    j2 = jitter_particles(particles, [1.0, 1.0, 0.1], seed=7)

    np.testing.assert_array_equal(j1, j2)


# ---------------------------------------------------------------------------
# Main PF loop — integration tests on Scenario 1
# ---------------------------------------------------------------------------

def test_run_particle_filter_returns_expected_keys():
    """Output dict must contain all documented keys."""
    # Tiny synthetic dataset: 5 sensor readings, 200 particles.
    positions, timestamps, observations, _ = generate_uav_observations(
        SCENARIO_1, track_spacing=600.0, seed=1,
    )
    # Take a small slice for speed.
    positions = positions[:20]
    timestamps = timestamps[:20]
    observations = observations[:20]

    result = run_particle_filter(
        SCENARIO_1,
        sensor_positions=positions,
        timestamps=timestamps,
        observations=observations,
        n_particles=200,
        seed=1,
    )

    expected_keys = {
        "particles_final", "log_weights_final",
        "posterior_means", "posterior_stds",
        "ess_history", "resample_steps",
    }
    assert set(result.keys()) == expected_keys


def test_run_particle_filter_converges_on_scenario_1():
    """Posterior mean should be within ~200 m of the true jammer location
    after a reasonable number of observations."""
    positions, timestamps, observations, _ = generate_uav_observations(
        SCENARIO_1, track_spacing=300.0, seed=1,
    )
    # Use a chunk of observations — enough for the filter to converge.
    positions = positions[:200]
    timestamps = timestamps[:200]
    observations = observations[:200]

    # Matched physics: observations are generated at n=2.5, so the filter must
    # run at n=2.5 to converge on the true location. (The experiments run the
    # filter impoverished at n=2.0 against n=2.5 truth by design; that mismatch
    # is the studied effect, not a convergence test.)
    result = run_particle_filter(
        SCENARIO_1,
        sensor_positions=positions,
        timestamps=timestamps,
        observations=observations,
        n_particles=2000,
        path_loss_exponent=2.5,
        seed=1,
    )

    true_x, true_y = SCENARIO_1.jammers[0].position
    true_p = SCENARIO_1.jammers[0].power_dBm

    final_mean = result["posterior_means"][-1]
    spatial_error = np.sqrt((final_mean[0] - true_x) ** 2 + (final_mean[1] - true_y) ** 2)
    power_error = abs(final_mean[2] - true_p)

    # Loose tolerance — even with 200 obs the posterior won't be pinpoint.
    assert spatial_error < 300.0
    assert power_error < 4.0


def test_run_particle_filter_ess_history_is_bounded():
    """ESS values must lie in [1, N] throughout the run."""
    positions, timestamps, observations, _ = generate_uav_observations(
        SCENARIO_1, track_spacing=600.0, seed=1,
    )
    positions = positions[:20]
    timestamps = timestamps[:20]
    observations = observations[:20]

    result = run_particle_filter(
        SCENARIO_1, positions, timestamps, observations,
        n_particles=200, seed=1,
    )

    assert (result["ess_history"] >= 1.0).all()
    assert (result["ess_history"] <= 200.0 + 1e-6).all()


def test_run_particle_filter_reproducible_with_seed():
    """Same seed must give identical posterior means."""
    positions, timestamps, observations, _ = generate_uav_observations(
        SCENARIO_1, track_spacing=600.0, seed=1,
    )
    positions = positions[:20]
    timestamps = timestamps[:20]
    observations = observations[:20]

    r1 = run_particle_filter(SCENARIO_1, positions, timestamps, observations,
                              n_particles=200, seed=42)
    r2 = run_particle_filter(SCENARIO_1, positions, timestamps, observations,
                              n_particles=200, seed=42)

    np.testing.assert_array_equal(r1["posterior_means"], r2["posterior_means"])


# ---------------------------------------------------------------------------
# Evaluation module
# ---------------------------------------------------------------------------

def _fake_pf_result(n_steps=10, n_particles=50, final_mean=(1000.0, 1000.0, 30.0),
                    spread=100.0, seed=0):
    """Build a synthetic PF result dict for testing the evaluator."""
    rng = np.random.default_rng(seed)
    posterior_means = np.tile(np.array(final_mean), (n_steps, 1))
    posterior_means += rng.normal(0.0, 10.0, size=posterior_means.shape)
    posterior_stds = np.full((n_steps, 3), 20.0)
    ess_history = np.full(n_steps, n_particles * 0.8)
    particles = np.column_stack([
        rng.normal(final_mean[0], spread, size=n_particles),
        rng.normal(final_mean[1], spread, size=n_particles),
        rng.normal(final_mean[2], 1.0, size=n_particles),
    ])
    log_weights = np.full(n_particles, -np.log(n_particles))
    return {
        "particles_final": particles,
        "log_weights_final": log_weights,
        "posterior_means": posterior_means,
        "posterior_stds": posterior_stds,
        "ess_history": ess_history,
        "resample_steps": [3, 7],
    }


def test_evaluate_pf_result_returns_expected_keys():
    """The metrics dict should contain all documented keys."""
    result = _fake_pf_result()
    metrics = evaluate_pf_result(result, true_params=(1000.0, 1000.0, 30.0))

    required = {
        "spatial_error_final", "spatial_error_history",
        "power_error_final", "power_error_history",
        "ess_min", "ess_mean", "ess_p50",
        "n_resamples", "convergence_step_50m",
    }
    assert required.issubset(metrics.keys())
    # And at least one coverage key by default.
    assert any(k.startswith("coverage_at_") for k in metrics)


def test_evaluate_pf_result_spatial_error_matches_manual_calculation():
    """spatial_error_final should equal the Euclidean distance from the
    final posterior mean to the truth."""
    final_mean = (1234.0, 567.0, 30.0)
    result = _fake_pf_result(final_mean=final_mean, spread=0.01)
    # Use a final_mean with very tight noise so posterior_means[-1] ≈ final_mean.
    result["posterior_means"][-1] = np.array(final_mean)

    true_params = (1000.0, 500.0, 30.0)
    metrics = evaluate_pf_result(result, true_params=true_params)

    expected = np.sqrt((1234 - 1000) ** 2 + (567 - 500) ** 2)
    assert metrics["spatial_error_final"] == pytest.approx(expected, abs=1e-6)


def test_evaluate_pf_result_coverage_one_when_radius_huge():
    """All particles inside a 10 km radius → coverage 1.0."""
    result = _fake_pf_result()
    metrics = evaluate_pf_result(
        result, true_params=(1000.0, 1000.0, 30.0), coverage_radii_m=(10000.0,)
    )

    assert metrics["coverage_at_10000m"] == pytest.approx(1.0, abs=1e-9)


def test_evaluate_pf_result_coverage_zero_when_truth_far():
    """Truth 1e9 m from the particle cloud → coverage 0."""
    result = _fake_pf_result()
    metrics = evaluate_pf_result(
        result, true_params=(1e9, 1e9, 30.0), coverage_radii_m=(100.0,)
    )

    assert metrics["coverage_at_100m"] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_pf_result_convergence_step_never():
    """If the spatial error never drops below 50 m, convergence_step_50m = -1."""
    result = _fake_pf_result(final_mean=(1e6, 1e6, 30.0))
    metrics = evaluate_pf_result(result, true_params=(0.0, 0.0, 30.0))

    assert metrics["convergence_step_50m"] == -1


def test_format_evaluation_summary_includes_key_numbers():
    """The pretty-print should mention error, ESS, and resamples."""
    result = _fake_pf_result()
    metrics = evaluate_pf_result(result, true_params=(1000.0, 1000.0, 30.0))

    summary = format_evaluation_summary(metrics, scenario_name="TEST")

    assert "TEST" in summary
    assert "spatial error" in summary.lower()
    assert "resampling events" in summary.lower()
    assert "ess" in summary.lower()
