"""Bootstrap particle filter for Scenario 2: 6D directional jammer.

State is theta = (x_B, y_B, P_tx_B, theta_main_B, theta_3dB_B, F_b_B).
Jammer A is a known omnidirectional background contributor taken from the
Scenario object and is not estimated.

Reuses the dimension-agnostic helpers in particle_filter.py and adds a 6D
prior sampler, a composed likelihood that sums A and B in linear power, a
jitter step with angular wrap and positivity clipping, and per-step
mode-collapse diagnostics.
"""
import numpy as np

from contested_rf.estimators.particle_filter import (
    _logsumexp,
    effective_sample_size,
    systematic_resample,
    update_log_weights,
)
from contested_rf.propagation.antenna import directional_gain
from contested_rf.propagation.path_loss import (
    free_space_path_loss,
    log_distance_path_loss,
)


# Column indices into the 6D state vector.
IX_X = 0
IX_Y = 1
IX_PTX = 2
IX_THETA_MAIN = 3
IX_THETA_3DB = 4
IX_FB = 5


def init_particles_6d(
    n_particles,
    x_range,
    y_range,
    power_dBm_range=(20.0, 40.0),
    theta_main_range=(0.0, 360.0),
    theta_3db_range=(10.0, 120.0),
    fb_range=(10.0, 40.0),
    seed=None,
):
    """Sample N particles from a uniform 6D prior over jammer B.

    theta_main covers the full circle: the prior is deliberately ignorant of
    beam direction. theta_3dB bounds exclude needle beams below 10 degrees and
    effectively omnidirectional beams above 120. F_b bounds span a poor (10 dB)
    to a well-designed (40 dB) antenna.

    Args:
        n_particles: number of particles.
        x_range, y_range: prior bounds on position, metres.
        power_dBm_range: prior bounds on transmit power, dBm.
        theta_main_range: prior bounds on main-beam direction, degrees.
        theta_3db_range: prior bounds on half-power beamwidth, degrees.
        fb_range: prior bounds on front-to-back ratio, dB.
        seed: RNG seed.

    Returns:
        (N, 6) array, columns [x, y, P_tx_dBm, theta_main, theta_3dB, F_b].
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(x_range[0], x_range[1], size=n_particles)
    y = rng.uniform(y_range[0], y_range[1], size=n_particles)
    p = rng.uniform(power_dBm_range[0], power_dBm_range[1], size=n_particles)
    tm = rng.uniform(theta_main_range[0], theta_main_range[1], size=n_particles)
    t3 = rng.uniform(theta_3db_range[0], theta_3db_range[1], size=n_particles)
    fb = rng.uniform(fb_range[0], fb_range[1], size=n_particles)

    return np.column_stack([x, y, p, tm, t3, fb])


def _known_omni_power_mW_at(sensor_position, jammer_a, operating_freq_Hz,
                            path_loss_exponent):
    """Predicted received power from the known omni jammer, in linear mW.

    Particle-independent, so computed once per observation and broadcast.

    Args:
        sensor_position: (x, y) in metres.
        jammer_a: Jammer object; position and power_dBm are read.
        operating_freq_Hz: carrier frequency.
        path_loss_exponent: log-distance n.

    Returns:
        Scalar power in mW.
    """
    d0 = 1.0
    pl_d0 = free_space_path_loss(d0, operating_freq_Hz)

    ax, ay = jammer_a.position
    dx = sensor_position[0] - ax
    dy = sensor_position[1] - ay
    d_a = max(np.hypot(dx, dy), d0)
    pl_a = log_distance_path_loss(d_a, d0=d0, n=path_loss_exponent,
                                  pl_d0=pl_d0)
    p_a_dbm = jammer_a.power_dBm - pl_a
    return 10.0 ** (p_a_dbm / 10.0)


def log_likelihood_per_particle_s2(
    particles,
    sensor_position,
    observed_power_dBm,
    jammer_a,
    operating_freq_Hz,
    path_loss_exponent=2.0,
    noise_sigma_dB=4.5,  # NB: experiments pass N_LEARNER=2.0 explicitly
):
    """Composed log-likelihood for one observation under N 6D particles.

    For each particle, the directional contribution from B is

        P_B_dBm = P_tx_B - PL(d_B) + G_dB(bearing, theta_main, theta_3dB, F_b)

    A's known contribution is then added in LINEAR mW, not in dBm; summing dBm
    would multiply linear powers. The simulator follows the same convention and
    the likelihood must match it.

        predicted_dBm = 10 log10(P_A_mW + 10**(P_B_dBm / 10))

    The Gaussian normaliser is dropped as it cancels at normalisation.

    Args:
        particles: (N, 6) array, [x, y, P_tx, theta_main, theta_3dB, F_b].
        sensor_position: (x, y) in metres.
        observed_power_dBm: measured power at that sensor.
        jammer_a: Jammer object for the known omni source.
        operating_freq_Hz: carrier frequency.
        path_loss_exponent: log-distance n.
        noise_sigma_dB: total noise standard deviation, dB.

    Returns:
        (N,) un-normalised log-likelihoods.
    """
    d0 = 1.0
    pl_d0 = free_space_path_loss(d0, operating_freq_Hz)

    # --- Particle-dependent: jammer B contribution ---
    dx = sensor_position[0] - particles[:, IX_X]
    dy = sensor_position[1] - particles[:, IX_Y]
    d_b = np.maximum(np.sqrt(dx ** 2 + dy ** 2), d0)
    pl_b = log_distance_path_loss(d_b, d0=d0, n=path_loss_exponent,
                                  pl_d0=pl_d0)

    # Bearing from each particle's hypothesised jammer B position to the
    # sensor. atan2 returns radians in (-pi, pi]; convert to degrees so the
    # convention matches directional_gain's theta argument.
    bearings_deg = np.degrees(np.arctan2(dy, dx))

    g_dB = directional_gain(
        bearings_deg,
        theta_main=particles[:, IX_THETA_MAIN],
        theta_3db=particles[:, IX_THETA_3DB],
        G0_dB=0.0,
        front_back_ratio_dB=particles[:, IX_FB],
    )

    p_b_dbm = particles[:, IX_PTX] - pl_b + g_dB
    p_b_mW = 10.0 ** (p_b_dbm / 10.0)

    # --- Particle-independent: jammer A contribution (broadcast) ---
    p_a_mW = _known_omni_power_mW_at(
        sensor_position, jammer_a, operating_freq_Hz, path_loss_exponent
    )

    # --- Linear sum, back to dBm, Gaussian residual ---
    p_total_mW = p_a_mW + p_b_mW
    predicted_dbm = 10.0 * np.log10(p_total_mW)

    residual = observed_power_dBm - predicted_dbm
    return -0.5 * (residual ** 2) / (noise_sigma_dB ** 2)


# ---------------------------------------------------------------------------
# Jitter for the 6D state — wraps theta_main into [0, 360), clips positive-
# only parameters (theta_3dB, F_b) to physically sensible bounds.
# ---------------------------------------------------------------------------

def jitter_particles_6d(
    particles,
    jitter_sigmas,
    theta_3db_bounds=(5.0, 180.0),
    fb_bounds=(3.0, 60.0),
    seed=None,
):
    """Perturb each 6D particle, with domain corrections.

    theta_main is circular and is wrapped back into [0, 360) after jittering.
    theta_3dB and F_b are positive-only and are clipped to physical bounds, since
    Gaussian jitter can otherwise drive them to zero or negative. x, y and P_tx
    are unconstrained.

    Args:
        particles: (N, 6) array.
        jitter_sigmas: (6,) per-dimension standard deviation, same column order.
        theta_3db_bounds: (min, max) clip for beamwidth.
        fb_bounds: (min, max) clip for front-to-back ratio.
        seed: RNG seed.

    Returns:
        (N, 6) perturbed particles.
    """
    rng = np.random.default_rng(seed)
    sigmas = np.asarray(jitter_sigmas, dtype=float)
    if sigmas.shape != (6,):
        raise ValueError(f"jitter_sigmas must have shape (6,); got {sigmas.shape}")

    noise = rng.normal(0.0, 1.0, size=particles.shape) * sigmas
    out = particles + noise

    # Circular wrap on the main-beam direction.
    out[:, IX_THETA_MAIN] = out[:, IX_THETA_MAIN] % 360.0

    # Positivity clips on beamwidth and front-to-back.
    out[:, IX_THETA_3DB] = np.clip(
        out[:, IX_THETA_3DB], theta_3db_bounds[0], theta_3db_bounds[1]
    )
    out[:, IX_FB] = np.clip(out[:, IX_FB], fb_bounds[0], fb_bounds[1])

    return out


def _weighted_posterior_summary(particles, log_weights):
    """Weighted posterior mean and standard deviation per dimension.

    theta_main is circular, so its mean is the angle of the resultant vector
    R = sum_i w_i exp(i theta_i) rather than the arithmetic mean, and its spread
    is derived from |R| and rescaled to degrees for uniform output units.

    Args:
        particles: (N, 6) array.
        log_weights: (N,) normalised log-weights.

    Returns:
        means (6,) and stds (6,).
    """
    weights = np.exp(log_weights)
    means = np.zeros(6)
    stds = np.zeros(6)

    # Linear dimensions
    for j in (IX_X, IX_Y, IX_PTX, IX_THETA_3DB, IX_FB):
        m = (weights * particles[:, j]).sum()
        v = (weights * (particles[:, j] - m) ** 2).sum()
        means[j] = m
        stds[j] = float(np.sqrt(v))

    # Circular dimension: theta_main
    theta_rad = np.radians(particles[:, IX_THETA_MAIN])
    Cx = (weights * np.cos(theta_rad)).sum()
    Cy = (weights * np.sin(theta_rad)).sum()
    R = np.hypot(Cx, Cy)
    mean_rad = np.arctan2(Cy, Cx)
    means[IX_THETA_MAIN] = np.degrees(mean_rad) % 360.0

    # Circular std (Mardia & Jupp): sqrt(-2 ln R) in radians, converted to
    # degrees. For R close to 1 (concentrated) this approaches 0; for R = 0
    # (uniform) it diverges. Clip R away from 0 to keep the number finite.
    R_safe = max(R, 1e-12)
    stds[IX_THETA_MAIN] = float(np.degrees(np.sqrt(-2.0 * np.log(R_safe))))

    return means, stds


def run_particle_filter_s2(
    scenario,
    sensor_positions,
    timestamps,
    observations,
    n_particles=8000,
    initial_particles=None,
    x_range=None,
    y_range=None,
    power_dBm_range=(20.0, 40.0),
    theta_main_range=(0.0, 360.0),
    theta_3db_range=(10.0, 120.0),
    fb_range=(10.0, 40.0),
    path_loss_exponent=2.0,
    noise_sigma_dB=4.5,
    ess_threshold_fraction=0.5,
    jitter_sigmas=(15.0, 15.0, 0.5, 4.0, 2.0, 1.0),
    seed=None,
):
    """Run a 6D bootstrap particle filter over Scenario 2 observations.

    scenario.jammers[0] is the known omnidirectional background; jammers[1] is
    the directional emitter whose parameters are inferred.

    Args:
        scenario: Scenario with two jammers (A omni, B directional).
        sensor_positions: (N_obs, 2) sensor positions, metres.
        timestamps: (N_obs,) observation times; unused for a static emitter.
        observations: (N_obs,) received powers, dBm.
        n_particles: particle count; 8000 is the default for the 6D state.
        initial_particles: optional (N, 6) starting cloud.
        x_range, y_range, power_dBm_range, theta_main_range, theta_3db_range,
            fb_range: uniform prior bounds; spatial defaults follow the scenario.
        path_loss_exponent, noise_sigma_dB: as for the 3D filter.
        ess_threshold_fraction: resample when ESS < fraction * N.
        jitter_sigmas: (6,) per-dimension jitter, small relative to the priors.
        seed: master RNG seed.

    Returns:
        dict with particles_final, log_weights_final, posterior_means and
        posterior_stds (N_obs, 6), ess_history and resample_steps.
    """
    rng = np.random.default_rng(seed)

    if x_range is None:
        x_range = scenario.grid_xlim
    if y_range is None:
        y_range = scenario.grid_ylim

    if len(scenario.jammers) < 2:
        raise ValueError("Scenario 2 PF expects a scenario with at least two "
                         "jammers; jammer[0] omni A (known), jammer[1] "
                         "directional B (estimated).")
    jammer_a = scenario.jammers[0]

    if initial_particles is not None:
        particles = np.asarray(initial_particles, dtype=float)
        n_particles = particles.shape[0]
    else:
        particles = init_particles_6d(
            n_particles,
            x_range=x_range, y_range=y_range,
            power_dBm_range=power_dBm_range,
            theta_main_range=theta_main_range,
            theta_3db_range=theta_3db_range,
            fb_range=fb_range,
            seed=int(rng.integers(0, 2 ** 31)),
        )
    log_weights = np.full(n_particles, -np.log(n_particles))

    n_obs = len(observations)
    posterior_means = np.zeros((n_obs, 6))
    posterior_stds = np.zeros((n_obs, 6))
    ess_history = np.zeros(n_obs)
    resample_steps = []

    threshold = ess_threshold_fraction * n_particles

    for t in range(n_obs):
        log_lik = log_likelihood_per_particle_s2(
            particles,
            sensor_position=sensor_positions[t],
            observed_power_dBm=observations[t],
            jammer_a=jammer_a,
            operating_freq_Hz=scenario.operating_freq_Hz,
            path_loss_exponent=path_loss_exponent,
            noise_sigma_dB=noise_sigma_dB,
        )
        log_weights = update_log_weights(log_weights, log_lik)

        ess = effective_sample_size(log_weights)
        ess_history[t] = ess

        if ess < threshold:
            particles, log_weights = systematic_resample(
                particles, log_weights,
                seed=int(rng.integers(0, 2 ** 31)),
            )
            particles = jitter_particles_6d(
                particles, jitter_sigmas,
                seed=int(rng.integers(0, 2 ** 31)),
            )
            resample_steps.append(t)

        means, stds = _weighted_posterior_summary(particles, log_weights)
        posterior_means[t] = means
        posterior_stds[t] = stds

    return {
        "particles_final": particles,
        "log_weights_final": log_weights,
        "posterior_means": posterior_means,
        "posterior_stds": posterior_stds,
        "ess_history": ess_history,
        "resample_steps": resample_steps,
    }


# ---------------------------------------------------------------------------
# Mode-collapse diagnostics for theta_main.
#
# With a finite front-to-back ratio, a reading explained by a main lobe
# pointing at the sensor is also explained, less well, by a back lobe pointing
# away from it. Where the survey geometry does not resolve that, the posterior
# splits into two antipodal clusters and its arithmetic mean falls between
# them, in a region of zero density.
#
# Two statistics are recorded per step from the weighted cloud:
#   circular variance  V = 1 - |R|, R = sum_i w_i exp(i theta_i), in [0, 1]
#   hedge score        H = min(w_front, w_back) / (w_front + w_back), in
#                      [0, 0.5], over +/- 30 degree wedges about angle(R) and
#                      its antipode
#
# High V with high H indicates front/back ambiguity specifically; high V alone
# is also consistent with a merely diffuse posterior.
# ---------------------------------------------------------------------------

def circular_variance(theta_deg, weights):
    """Weighted circular variance over angles in degrees.

    Returns 1 - |R| where R = sum_i w_i * exp(i*theta_i). Range [0, 1].
    Zero for a delta concentration at one angle; one for any distribution
    whose resultant vector is zero (e.g. uniform on the circle, or exactly
    antipodal at 50/50).
    """
    theta_rad = np.radians(theta_deg)
    Cx = float((weights * np.cos(theta_rad)).sum())
    Cy = float((weights * np.sin(theta_rad)).sum())
    R = float(np.hypot(Cx, Cy))
    return 1.0 - R


def antipodal_concentration(theta_deg, weights):
    """Doubled-angle concentration of a weighted angular swarm.

    A perfectly antipodal 50/50 swarm has resultant R = 0, so a wedge probe
    anchored on angle(R) is undefined. Mapping theta -> 2*theta collapses theta
    and theta+180 to the same point, so |R_2| measures concentration on the axis
    regardless of whether the cloud is unimodal or antipodal:

        unimodal   |R| ~ 1, |R_2| ~ 1
        antipodal  |R| ~ 0, |R_2| ~ 1
        diffuse    |R| ~ 0, |R_2| ~ 0

    Args:
        theta_deg: (N,) angles in degrees.
        weights: (N,) normalised linear weights.

    Returns:
        (axial_concentration, bimodality), both in [0, 1]; bimodality is
        1 - |R|/|R_2| clipped, so ~1 means axial but not unimodal.
    """
    theta_rad = np.radians(theta_deg)

    # Linear resultant
    Cx = float((weights * np.cos(theta_rad)).sum())
    Cy = float((weights * np.sin(theta_rad)).sum())
    R = float(np.hypot(Cx, Cy))

    # Doubled-angle resultant
    Cx2 = float((weights * np.cos(2.0 * theta_rad)).sum())
    Cy2 = float((weights * np.sin(2.0 * theta_rad)).sum())
    R2 = float(np.hypot(Cx2, Cy2))

    if R2 <= 1e-12:
        bimodality = 0.0
    else:
        bimodality = max(0.0, 1.0 - R / R2)
    return R2, bimodality


def _run_pf_s2_with_diagnostics(*args, **kwargs):
    """Run the 6D filter, recording mode-collapse diagnostics per step.

    Duplicates the main loop so that run_particle_filter_s2 remains free of
    diagnostic bookkeeping. Same arguments as run_particle_filter_s2.
    """
    # The cleanest implementation is to copy the loop and add four lines.
    # Re-import the helpers used by the bare function to avoid an
    # extra public re-entry point.
    rng = np.random.default_rng(kwargs.get("seed"))

    scenario = args[0] if args else kwargs["scenario"]
    sensor_positions = kwargs.get("sensor_positions",
                                  args[1] if len(args) > 1 else None)
    timestamps = kwargs.get("timestamps",
                            args[2] if len(args) > 2 else None)
    observations = kwargs.get("observations",
                              args[3] if len(args) > 3 else None)

    n_particles = kwargs.get("n_particles", 8000)
    initial_particles = kwargs.get("initial_particles")
    x_range = kwargs.get("x_range") or scenario.grid_xlim
    y_range = kwargs.get("y_range") or scenario.grid_ylim
    power_dBm_range = kwargs.get("power_dBm_range", (20.0, 40.0))
    theta_main_range = kwargs.get("theta_main_range", (0.0, 360.0))
    theta_3db_range = kwargs.get("theta_3db_range", (10.0, 120.0))
    fb_range = kwargs.get("fb_range", (10.0, 40.0))
    path_loss_exponent = kwargs.get("path_loss_exponent", 2.5)
    noise_sigma_dB = kwargs.get("noise_sigma_dB", 4.5)
    ess_threshold_fraction = kwargs.get("ess_threshold_fraction", 0.5)
    jitter_sigmas = kwargs.get("jitter_sigmas",
                               (15.0, 15.0, 0.5, 4.0, 2.0, 1.0))

    if len(scenario.jammers) < 2:
        raise ValueError("Scenario 2 PF expects at least two jammers.")
    jammer_a = scenario.jammers[0]

    if initial_particles is not None:
        particles = np.asarray(initial_particles, dtype=float)
        n_particles = particles.shape[0]
    else:
        particles = init_particles_6d(
            n_particles,
            x_range=x_range, y_range=y_range,
            power_dBm_range=power_dBm_range,
            theta_main_range=theta_main_range,
            theta_3db_range=theta_3db_range,
            fb_range=fb_range,
            seed=int(rng.integers(0, 2 ** 31)),
        )
    log_weights = np.full(n_particles, -np.log(n_particles))

    n_obs = len(observations)
    posterior_means = np.zeros((n_obs, 6))
    posterior_stds = np.zeros((n_obs, 6))
    ess_history = np.zeros(n_obs)
    circ_var_history = np.zeros(n_obs)
    axial_conc_history = np.zeros(n_obs)
    bimodality_history = np.zeros(n_obs)
    resample_steps = []
    collapse_flags = []

    threshold = ess_threshold_fraction * n_particles

    for t in range(n_obs):
        log_lik = log_likelihood_per_particle_s2(
            particles,
            sensor_position=sensor_positions[t],
            observed_power_dBm=observations[t],
            jammer_a=jammer_a,
            operating_freq_Hz=scenario.operating_freq_Hz,
            path_loss_exponent=path_loss_exponent,
            noise_sigma_dB=noise_sigma_dB,
        )
        log_weights = update_log_weights(log_weights, log_lik)

        ess = effective_sample_size(log_weights)
        ess_history[t] = ess

        if ess < threshold:
            particles, log_weights = systematic_resample(
                particles, log_weights,
                seed=int(rng.integers(0, 2 ** 31)),
            )
            particles = jitter_particles_6d(
                particles, jitter_sigmas,
                seed=int(rng.integers(0, 2 ** 31)),
            )
            resample_steps.append(t)

        # Posterior summary
        means, stds = _weighted_posterior_summary(particles, log_weights)
        posterior_means[t] = means
        posterior_stds[t] = stds

        # Mode-collapse diagnostics
        weights = np.exp(log_weights)
        circ_var = circular_variance(particles[:, IX_THETA_MAIN], weights)
        axial_conc, bimod = antipodal_concentration(
            particles[:, IX_THETA_MAIN], weights,
        )
        circ_var_history[t] = circ_var
        axial_conc_history[t] = axial_conc
        bimodality_history[t] = bimod

        # Flag a collapse when the swarm is genuinely concentrated on an
        # *axis* (axial_conc high) but split antipodally along it (bimod
        # near 1). Either alone has an innocent reading: high axial_conc
        # with low bimod is unimodal concentration (good); high bimod with
        # low axial_conc is just diffuse noise.
        if axial_conc >= 0.6 and bimod >= 0.5:
            collapse_flags.append(t)

    return {
        "particles_final": particles,
        "log_weights_final": log_weights,
        "posterior_means": posterior_means,
        "posterior_stds": posterior_stds,
        "ess_history": ess_history,
        "resample_steps": resample_steps,
        "circ_var_history": circ_var_history,
        "axial_conc_history": axial_conc_history,
        "bimodality_history": bimodality_history,
        "collapse_flags": collapse_flags,
    }


def run_particle_filter_s2_with_diagnostics(
    scenario, sensor_positions, timestamps, observations, **kwargs
):
    """Public entry point for the diagnostic PF — see _run_pf_s2_with_diagnostics."""
    return _run_pf_s2_with_diagnostics(
        scenario, sensor_positions, timestamps, observations, **kwargs
    )


if __name__ == "__main__":
    # End-to-end smoke for Scenario 2: run the diagnostic PF on the standard
    # UAV lawnmower dataset and save a six-panel figure summarising what
    # the filter recovered, where (or whether) it converged, and how the
    # bimodality diagnostics behaved.
    import os

    import matplotlib.pyplot as plt

    from contested_rf.simulation.ground_truth import compute_sinr_map
    from contested_rf.simulation.scenario import SCENARIO_2
    from contested_rf.simulation.uav import generate_uav_observations

    positions, timestamps, observations, _ = generate_uav_observations(
        SCENARIO_2, seed=42
    )
    print(f"Observations: {len(observations)}")

    result = run_particle_filter_s2_with_diagnostics(
        SCENARIO_2,
        sensor_positions=positions,
        timestamps=timestamps,
        observations=observations,
        n_particles=8000,
        seed=1,
    )

    jam_a = SCENARIO_2.jammers[0]
    jam_b = SCENARIO_2.jammers[1]

    final_mean = result["posterior_means"][-1]
    final_std = result["posterior_stds"][-1]
    spatial_err = float(np.hypot(
        final_mean[IX_X] - jam_b.position[0],
        final_mean[IX_Y] - jam_b.position[1],
    ))
    # Circular error on theta_main, signed in (-180, 180]
    theta_err = (final_mean[IX_THETA_MAIN] - jam_b.theta_main_deg + 180.0) % 360.0 - 180.0

    print(f"\nTrue B:           (x={jam_b.position[0]:.0f}, y={jam_b.position[1]:.0f}, "
          f"P_tx={jam_b.power_dBm:.1f}, theta_main={jam_b.theta_main_deg:.0f}, "
          f"theta_3dB={jam_b.theta_3db_deg:.0f})")
    print(f"Posterior mean B: (x={final_mean[IX_X]:.0f}, y={final_mean[IX_Y]:.0f}, "
          f"P_tx={final_mean[IX_PTX]:.1f}, theta_main={final_mean[IX_THETA_MAIN]:.0f}, "
          f"theta_3dB={final_mean[IX_THETA_3DB]:.0f}, F_b={final_mean[IX_FB]:.1f})")
    print(f"Posterior std:    (x={final_std[IX_X]:.0f}, y={final_std[IX_Y]:.0f}, "
          f"P_tx={final_std[IX_PTX]:.2f}, theta_main={final_std[IX_THETA_MAIN]:.0f}, "
          f"theta_3dB={final_std[IX_THETA_3DB]:.0f}, F_b={final_std[IX_FB]:.1f})")
    print(f"Spatial error (B): {spatial_err:.0f} m")
    print(f"Beam direction error: {theta_err:+.0f} deg")
    print(f"Resampling events: {len(result['resample_steps'])}")
    print(f"Steps flagged for bimodal collapse: {len(result['collapse_flags'])}"
          f" of {len(observations)}")

    # ----- six-panel figure -----
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    ax_map = fig.add_subplot(gs[:, 0])  # tall left panel
    ax_err = fig.add_subplot(gs[0, 1])
    ax_ess = fig.add_subplot(gs[1, 1])
    ax_theta = fig.add_subplot(gs[0, 2])
    ax_bimod = fig.add_subplot(gs[1, 2])

    # Panel 1: SINR map + posterior cloud + truth
    X, Y, SINR = compute_sinr_map(SCENARIO_2, grid_shape=(120, 120))
    im = ax_map.pcolormesh(X, Y, SINR, shading="auto", cmap="RdYlGn",
                            vmin=-30, vmax=30, alpha=0.55)
    plt.colorbar(im, ax=ax_map, label="SINR (dB)")

    particles_final = result["particles_final"]
    weights_final = np.exp(result["log_weights_final"])
    n_show = min(800, len(particles_final))
    rng_plot = np.random.default_rng(0)
    idx = rng_plot.choice(len(particles_final), n_show, replace=False)
    ax_map.scatter(particles_final[idx, IX_X], particles_final[idx, IX_Y],
                    s=weights_final[idx] * 8000 + 2, c="purple", alpha=0.4,
                    label=f"Particle cloud ({n_show}/{len(particles_final)})")

    bs = SCENARIO_2.base_station
    ax_map.plot(*bs.position, "b^", markersize=14, markeredgewidth=2,
                markeredgecolor="black", label=f"BS ({bs.name})")
    ax_map.plot(*jam_a.position, "kx", markersize=16, markeredgewidth=3,
                label=f"Jammer A (omni, known)")
    ax_map.plot(*jam_b.position, "k+", markersize=18, markeredgewidth=3,
                label=f"Jammer B (directional, true)")
    ax_map.plot(final_mean[IX_X], final_mean[IX_Y], "y*", markersize=20,
                markeredgewidth=1.5, markeredgecolor="black",
                label=f"Posterior mean B (err {spatial_err:.0f} m)")
    # Arrow showing posterior mean beam direction
    arrow_len = 300.0
    th = np.radians(final_mean[IX_THETA_MAIN])
    ax_map.annotate(
        "", xy=(final_mean[IX_X] + arrow_len * np.cos(th),
                final_mean[IX_Y] + arrow_len * np.sin(th)),
        xytext=(final_mean[IX_X], final_mean[IX_Y]),
        arrowprops=dict(arrowstyle="->", color="gold", lw=2.0),
    )
    # Arrow showing true beam direction
    th_true = np.radians(jam_b.theta_main_deg)
    ax_map.annotate(
        "", xy=(jam_b.position[0] + arrow_len * np.cos(th_true),
                jam_b.position[1] + arrow_len * np.sin(th_true)),
        xytext=jam_b.position,
        arrowprops=dict(arrowstyle="->", color="black", lw=2.0),
    )
    ax_map.set_xlabel("x (m)"); ax_map.set_ylabel("y (m)")
    ax_map.set_title("S2: 6D posterior over SINR ground truth")
    ax_map.set_aspect("equal")
    ax_map.legend(loc="upper left", fontsize=8)

    # Panel 2: spatial error over time
    means = result["posterior_means"]
    err_history = np.sqrt(
        (means[:, IX_X] - jam_b.position[0]) ** 2
        + (means[:, IX_Y] - jam_b.position[1]) ** 2
    )
    ax_err.plot(err_history, "b-", linewidth=1.0)
    ax_err.set_yscale("log")
    ax_err.set_xlabel("Observation step")
    ax_err.set_ylabel("Distance to true B (m)")
    ax_err.set_title("Spatial convergence on jammer B")
    ax_err.grid(True, alpha=0.3)

    # Panel 3: ESS over time
    ess = result["ess_history"]
    nP = particles_final.shape[0]
    ax_ess.plot(ess, "g-", linewidth=1.0, label="ESS")
    ax_ess.axhline(0.5 * nP, color="orange", linestyle="--", alpha=0.7,
                    label="Resample threshold")
    ax_ess.axhline(nP, color="black", linestyle=":", alpha=0.3,
                    label=f"N = {nP}")
    for step in result["resample_steps"]:
        ax_ess.axvline(step, color="red", alpha=0.10, linewidth=0.5)
    ax_ess.set_xlabel("Observation step")
    ax_ess.set_ylabel("Effective sample size")
    ax_ess.set_title(f"ESS ({len(result['resample_steps'])} resamples)")
    ax_ess.legend(loc="lower right", fontsize=8)
    ax_ess.grid(True, alpha=0.3)

    # Panel 4: theta_main posterior mean vs truth
    ax_theta.plot(means[:, IX_THETA_MAIN], "-", color="purple", linewidth=1.0,
                   label="posterior mean theta_main")
    ax_theta.axhline(jam_b.theta_main_deg, color="black", linestyle="--",
                     label=f"truth = {jam_b.theta_main_deg:.0f} deg")
    ax_theta.set_xlabel("Observation step")
    ax_theta.set_ylabel("Beam direction (deg, 0..360)")
    ax_theta.set_title("Recovered main-beam direction")
    ax_theta.set_ylim(-10, 370)
    ax_theta.legend(loc="lower right", fontsize=8)
    ax_theta.grid(True, alpha=0.3)

    # Panel 5: bimodality diagnostics
    axial = result["axial_conc_history"]
    bimod = result["bimodality_history"]
    ax_bimod.plot(axial, "-", color="teal", linewidth=1.0,
                   label="axial concentration |R_2|")
    ax_bimod.plot(bimod, "-", color="crimson", linewidth=1.0,
                   label="bimodality 1 - |R|/|R_2|")
    ax_bimod.axhline(0.5, color="black", linestyle=":", alpha=0.4)
    for step in result["collapse_flags"]:
        ax_bimod.axvline(step, color="orange", alpha=0.15, linewidth=0.5)
    ax_bimod.set_xlabel("Observation step")
    ax_bimod.set_ylabel("Diagnostic value")
    ax_bimod.set_title(
        f"Mode-collapse diagnostics "
        f"({len(result['collapse_flags'])} flagged steps)"
    )
    ax_bimod.set_ylim(-0.05, 1.05)
    ax_bimod.legend(loc="lower right", fontsize=8)
    ax_bimod.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/scenario_2_pf_diagnostics.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")
