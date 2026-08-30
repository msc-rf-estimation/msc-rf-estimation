"""Bootstrap particle filter for 3D jammer state estimation.

State space: theta = (x_jammer, y_jammer, P_tx_dBm) — a static jammer at an
unknown location with an unknown transmit power. This module is built
chunk-by-chunk; see git log for the incremental history.
"""
import numpy as np

from contested_rf.propagation.path_loss import (
    free_space_path_loss,
    log_distance_path_loss,
)


def init_particles(
    n_particles,
    x_range,
    y_range,
    power_dBm_range,
    seed=None,
):
    """Sample N particles from a uniform prior over (x, y, P_tx).

    Args:
        n_particles: number of particles.
        x_range, y_range: prior bounds in metres.
        power_dBm_range: prior bounds on transmit power, dBm.
        seed: RNG seed.

    Returns:
        (N, 3) array with columns [x, y, P_tx_dBm].
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(x_range[0], x_range[1], size=n_particles)
    y = rng.uniform(y_range[0], y_range[1], size=n_particles)
    power = rng.uniform(power_dBm_range[0], power_dBm_range[1], size=n_particles)

    return np.column_stack([x, y, power])


def init_particles_gaussian(
    n_particles,
    means,
    stds,
    seed=None,
):
    """Sample N particles from a Gaussian prior with diagonal covariance.

    Used by the prior-sensitivity experiment.

    Args:
        n_particles: number of particles.
        means: prior means for (x, y, P_tx) in (m, m, dBm).
        stds: prior standard deviations, same order, in (m, m, dB).
        seed: RNG seed.

    Returns:
        (N, 3) array with columns [x, y, P_tx_dBm].
    """
    rng = np.random.default_rng(seed)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    # Standard normal noise scaled per-dimension. Equivalent to drawing from
    # N(means, diag(stds**2)).
    noise = rng.normal(0.0, 1.0, size=(n_particles, 3))
    return means + noise * stds


def log_likelihood_per_particle(
    particles,
    sensor_position,
    observed_power_dBm,
    operating_freq_Hz,
    path_loss_exponent=2.0,
    noise_sigma_dB=4.5,
):
    """Gaussian log-likelihood of one observation under N particles.

        predicted_i = P_tx_i - PL(d_i)
        log L_i    proportional to -(observed - predicted_i)^2 / (2 sigma^2)

    Evaluated in log space; the Gaussian normaliser is dropped since it is
    constant across particles and cancels at normalisation. noise_sigma_dB is
    shadow and measurement noise combined in quadrature. Omnidirectional
    emitters only; the directional case is in particle_filter_s2.

    Args:
        particles: (N, 3) array, [x, y, P_tx_dBm] per row.
        sensor_position: (x, y) in metres.
        observed_power_dBm: measured power at that sensor.
        operating_freq_Hz: carrier frequency, for the path-loss reference.
        path_loss_exponent: log-distance n.
        noise_sigma_dB: total noise standard deviation, dB.

    Returns:
        (N,) un-normalised log-likelihoods.
    """
    d0 = 1.0
    pl_d0 = free_space_path_loss(d0, operating_freq_Hz)

    # Distance from each particle's hypothesised jammer location to the sensor.
    dx = sensor_position[0] - particles[:, 0]
    dy = sensor_position[1] - particles[:, 1]
    d = np.maximum(np.sqrt(dx ** 2 + dy ** 2), d0)

    pl = log_distance_path_loss(d, d0=d0, n=path_loss_exponent, pl_d0=pl_d0)

    # Predicted received power: P_tx - path loss, in dBm.
    predicted_dBm = particles[:, 2] - pl

    residual = observed_power_dBm - predicted_dBm

    return -0.5 * (residual ** 2) / (noise_sigma_dB ** 2)


def _logsumexp(x):
    """Numerically stable log(sum(exp(x))), factoring out max(x)."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))


def update_log_weights(log_weights, log_likelihoods):
    """Bootstrap weight update for one observation, normalised in log space.

    Args:
        log_weights: (N,) current normalised log-weights.
        log_likelihoods: (N,) per-particle log-likelihood of the new observation.

    Returns:
        (N,) normalised log-weights.
    """
    new_log_weights = log_weights + log_likelihoods
    log_normaliser = _logsumexp(new_log_weights)
    return new_log_weights - log_normaliser


def effective_sample_size(log_weights):
    """Effective sample size, 1 / sum(w_i^2), computed from normalised log-weights.

    Ranges from 1 (full degeneracy) to N (uniform weights).

    Args:
        log_weights: (N,) normalised log-weights.

    Returns:
        ESS in [1, N].
    """
    return float(np.exp(-_logsumexp(2.0 * log_weights)))


def systematic_resample(particles, log_weights, seed=None):
    """Resample N particles with replacement and reset weights to uniform.

    Systematic resampling draws a single uniform on [0, 1/N) and walks a
    deterministic grid, giving lower variance than multinomial for the same
    expected duplicate counts.

    Args:
        particles: (N, D) current particles.
        log_weights: (N,) normalised log-weights.
        seed: RNG seed for the single uniform draw.

    Returns:
        (N, D) resampled particles and (N,) uniform log-weights.
    """
    n = particles.shape[0]
    rng = np.random.default_rng(seed)

    weights = np.exp(log_weights)
    cumulative = np.cumsum(weights)

    # Single random offset for the deterministic grid of N sample points.
    u = rng.uniform(0.0, 1.0 / n)
    sample_points = u + np.arange(n) / n

    indices = np.searchsorted(cumulative, sample_points)
    # Guard the upper edge against floating-point rounding pushing past 1.
    indices = np.clip(indices, 0, n - 1)

    new_particles = particles[indices]
    new_log_weights = np.full(n, -np.log(n))

    return new_particles, new_log_weights


def jitter_particles(particles, jitter_sigmas, seed=None):
    """Perturb each particle with per-dimension Gaussian noise.

    The emitters here are static, so there is no motion model to separate
    duplicated particles after resampling; jitter does that instead. Scale is
    a few percent of the prior range per dimension.

    Args:
        particles: (N, D) array.
        jitter_sigmas: scalar or (D,) per-dimension standard deviation.
        seed: RNG seed.

    Returns:
        (N, D) perturbed particles.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=particles.shape) * np.asarray(jitter_sigmas)
    return particles + noise


def run_particle_filter(
    scenario,
    sensor_positions,
    timestamps,
    observations,
    n_particles=5000,
    initial_particles=None,
    x_range=None,
    y_range=None,
    power_dBm_range=(20.0, 40.0),
    path_loss_exponent=2.0,
    noise_sigma_dB=4.5,
    ess_threshold_fraction=0.5,
    jitter_sigmas=(5.0, 5.0, 0.2),
    seed=None,
):
    """Run a bootstrap particle filter over a stream of observations.

    State is theta = (x, y, P_tx_dBm) for a static emitter. Per observation:
    weight by log-likelihood, renormalise, and resample with jitter when the
    effective sample size falls below the threshold.

    Args:
        scenario: Scenario, for carrier frequency and default prior bounds.
        sensor_positions: (N_obs, 2) sensor positions in metres.
        timestamps: (N_obs,) observation times; unused for a static emitter.
        observations: (N_obs,) received powers in dBm.
        n_particles: particle count; ignored if initial_particles is given.
        initial_particles: (N, 3) pre-computed particles, for informative priors.
        x_range, y_range, power_dBm_range: uniform prior bounds; default to the
            scenario grid limits.
        path_loss_exponent: log-distance n.
        noise_sigma_dB: total noise standard deviation for the likelihood.
        ess_threshold_fraction: resample when ESS < fraction * N.
        jitter_sigmas: per-dimension jitter applied after resampling.
        seed: master RNG seed.

    Returns:
        dict with particles_final (N, 3), log_weights_final (N,),
        posterior_means and posterior_stds (N_obs, 3), ess_history (N_obs,)
        and resample_steps.
    """
    rng = np.random.default_rng(seed)

    if x_range is None:
        x_range = scenario.grid_xlim
    if y_range is None:
        y_range = scenario.grid_ylim

    # Initialise particles from the prior. If the caller has supplied a
    # pre-computed particle set (e.g. from init_particles_gaussian for the
    # informative-prior experiment in §Y.Z), use that directly; otherwise
    # default to the uniform sampler.
    if initial_particles is not None:
        particles = np.asarray(initial_particles, dtype=float)
        n_particles = particles.shape[0]
    else:
        particles = init_particles(
            n_particles, x_range, y_range, power_dBm_range,
            seed=int(rng.integers(0, 2 ** 31)),
        )
    log_weights = np.full(n_particles, -np.log(n_particles))

    n_obs = len(observations)
    posterior_means = np.zeros((n_obs, 3))
    posterior_stds = np.zeros((n_obs, 3))
    ess_history = np.zeros(n_obs)
    resample_steps = []

    threshold = ess_threshold_fraction * n_particles

    for t in range(n_obs):
        # Update step: per-particle log-likelihood, then re-weight.
        log_lik = log_likelihood_per_particle(
            particles,
            sensor_position=sensor_positions[t],
            observed_power_dBm=observations[t],
            operating_freq_Hz=scenario.operating_freq_Hz,
            path_loss_exponent=path_loss_exponent,
            noise_sigma_dB=noise_sigma_dB,
        )
        log_weights = update_log_weights(log_weights, log_lik)

        # ESS check.
        ess = effective_sample_size(log_weights)
        ess_history[t] = ess

        # Resample + jitter if degenerate.
        if ess < threshold:
            particles, log_weights = systematic_resample(
                particles, log_weights, seed=int(rng.integers(0, 2 ** 31)),
            )
            particles = jitter_particles(
                particles, jitter_sigmas, seed=int(rng.integers(0, 2 ** 31)),
            )
            # Constrain jittered particles to the prior support. Jitter is an
            # unbounded Gaussian nudge, so without this the swarm can drift
            # outside the physical domain (off-grid positions, implausible
            # transmit powers). The 6-D S2 filter already clips its state; this
            # brings the 3-D S1 filter into line.
            particles[:, 0] = np.clip(particles[:, 0], x_range[0], x_range[1])
            particles[:, 1] = np.clip(particles[:, 1], y_range[0], y_range[1])
            particles[:, 2] = np.clip(particles[:, 2],
                                      power_dBm_range[0], power_dBm_range[1])
            resample_steps.append(t)

        # Posterior summary: weighted mean and weighted std per dimension.
        weights = np.exp(log_weights)
        mean = (weights[:, None] * particles).sum(axis=0)
        posterior_means[t] = mean
        diffs = particles - mean
        posterior_stds[t] = np.sqrt((weights[:, None] * diffs ** 2).sum(axis=0))

    return {
        "particles_final": particles,
        "log_weights_final": log_weights,
        "posterior_means": posterior_means,
        "posterior_stds": posterior_stds,
        "ess_history": ess_history,
        "resample_steps": resample_steps,
    }


if __name__ == "__main__":
    # First headline demo: run the bootstrap PF on Scenario 1 UAV data,
    # save a three-panel diagnostic figure for the briefing note.
    import os

    import matplotlib.pyplot as plt

    from contested_rf.simulation.ground_truth import compute_sinr_map
    from contested_rf.simulation.scenario import SCENARIO_1
    from contested_rf.simulation.uav import generate_uav_observations

    # --- Simulate the UAV dataset ---
    positions, timestamps, observations, _ = generate_uav_observations(
        SCENARIO_1, seed=42
    )

    # --- Run the PF ---
    result = run_particle_filter(
        SCENARIO_1,
        sensor_positions=positions,
        timestamps=timestamps,
        observations=observations,
        n_particles=5000,
        seed=1,
    )

    true_x, true_y = SCENARIO_1.jammers[0].position
    true_p = SCENARIO_1.jammers[0].power_dBm

    final_mean = result["posterior_means"][-1]
    final_std = result["posterior_stds"][-1]
    spatial_error = np.sqrt(
        (final_mean[0] - true_x) ** 2 + (final_mean[1] - true_y) ** 2
    )

    print(f"Observations: {len(observations)}")
    print(f"True jammer (x, y, P_tx): ({true_x:.0f}, {true_y:.0f}, {true_p:.1f})")
    print(f"Posterior mean:         ({final_mean[0]:.0f}, {final_mean[1]:.0f}, {final_mean[2]:.1f})")
    print(f"Posterior std:          ({final_std[0]:.0f}, {final_std[1]:.0f}, {final_std[2]:.2f})")
    print(f"Final spatial error:    {spatial_error:.1f} m")
    print(f"Resampling fired at:    {len(result['resample_steps'])} time steps")

    # --- Three-panel figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: SINR map with particle cloud, posterior mean, true position
    X, Y, SINR = compute_sinr_map(SCENARIO_1, grid_shape=(100, 100))
    im = ax1.pcolormesh(X, Y, SINR, shading="auto", cmap="RdYlGn",
                        vmin=-30, vmax=30, alpha=0.6)
    plt.colorbar(im, ax=ax1, label="SINR (dB)")

    particles_final = result["particles_final"]
    weights_final = np.exp(result["log_weights_final"])
    # Subsample for plot clarity if N is large.
    n_show = min(500, len(particles_final))
    idx = np.random.default_rng(0).choice(len(particles_final), n_show, replace=False)
    ax1.scatter(particles_final[idx, 0], particles_final[idx, 1],
                s=weights_final[idx] * 5000 + 2, c="purple", alpha=0.4,
                label=f"Particle cloud ({n_show} of {len(particles_final)})")

    ax1.plot(*SCENARIO_1.jammers[0].position, "kx", markersize=18, markeredgewidth=3,
             label="True jammer")
    ax1.plot(final_mean[0], final_mean[1], "y*", markersize=20, markeredgewidth=1.5,
             markeredgecolor="black", label=f"Posterior mean (err = {spatial_error:.0f} m)")

    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Final posterior over SINR ground truth")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper left", fontsize=9)

    # Panel 2: spatial error vs time
    means = result["posterior_means"]
    errors = np.sqrt((means[:, 0] - true_x) ** 2 + (means[:, 1] - true_y) ** 2)
    ax2.plot(errors, "b-", linewidth=1.2)
    ax2.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Observation step")
    ax2.set_ylabel("Distance from truth (m)")
    ax2.set_title("Posterior mean convergence")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Panel 3: ESS over time, with resample markers
    ess = result["ess_history"]
    n_part = particles_final.shape[0]
    ax3.plot(ess, "g-", linewidth=1.2, label="ESS")
    ax3.axhline(0.5 * n_part, color="orange", linestyle="--", alpha=0.7,
                label="Resample threshold (N/2)")
    ax3.axhline(n_part, color="black", linestyle=":", alpha=0.3,
                label=f"N = {n_part}")
    for step in result["resample_steps"]:
        ax3.axvline(step, color="red", alpha=0.15, linewidth=0.5)
    ax3.set_xlabel("Observation step")
    ax3.set_ylabel("Effective sample size")
    ax3.set_title(f"ESS history ({len(result['resample_steps'])} resamples)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out_path = "figures/scenario_1_pf_diagnostics.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")
