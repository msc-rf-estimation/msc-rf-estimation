"""UAV trajectory generation and observation collection.

A fleet of UAVs flies lawnmower patterns over the scenario area, sampling
received jammer power at fixed intervals. Jammer positions are evaluated at
each observation's own timestamp, so a moving emitter is handled correctly.
"""
import numpy as np

from contested_rf.propagation.antenna import directional_gain
from contested_rf.propagation.path_loss import (
    free_space_path_loss,
    log_distance_path_loss,
)
from contested_rf.propagation.shadow_fading import sample_shadowing_2d


def lawnmower_trajectory(
    x_range,
    y_range,
    speed_mps=10.0,
    sample_dt=1.0,
    track_spacing=100.0,
    perturbation_sigma=20.0,
    t_start=0.0,
    seed=None,
):
    """Generate a single-UAV lawnmower pattern over a rectangle.

    Starts in the south-west corner and snakes east-north-west, with a small
    Gaussian perturbation on each waypoint to model flight imprecision.

    Args:
        x_range, y_range: extent of the flight area, metres.
        speed_mps: cruise speed.
        sample_dt: seconds between observations.
        track_spacing: distance between adjacent tracks, metres.
        perturbation_sigma: waypoint noise standard deviation, metres.
        t_start: time of the first observation.
        seed: RNG seed.

    Returns:
        positions (N, 2) and timestamps (N,).
    """
    rng = np.random.default_rng(seed)

    x_min, x_max = x_range
    y_min, y_max = y_range

    track_length = x_max - x_min
    samples_per_track = max(int(track_length / (speed_mps * sample_dt)), 1)
    n_tracks = max(int((y_max - y_min) / track_spacing), 1)

    positions = []
    timestamps = []
    t = t_start

    for i_track in range(n_tracks):
        y = y_min + (i_track + 0.5) * track_spacing
        # Alternate east-then-west to make a snake.
        if i_track % 2 == 0:
            x_path = np.linspace(x_min, x_max, samples_per_track)
        else:
            x_path = np.linspace(x_max, x_min, samples_per_track)

        for x in x_path:
            x_obs = x + rng.normal(0.0, perturbation_sigma)
            y_obs = y + rng.normal(0.0, perturbation_sigma)
            positions.append((x_obs, y_obs))
            timestamps.append(t)
            t += sample_dt

    return np.array(positions), np.array(timestamps)


def generate_uav_observations(
    scenario,
    n_uavs=4,
    speed_mps=10.0,
    sample_dt=1.0,
    track_spacing=100.0,
    perturbation_sigma=20.0,
    measurement_noise_sigma=2.0,
    path_loss_exponent=2.5,
    shadow_sigma=4.0,
    shadow_L=50.0,
    shadow_field=None,
    terrain=None,
    diffraction=None,
    seed=None,
):
    """Simulate UAV-collected observations across a scenario.

    The area is divided into n_uavs quadrants with one UAV each. The measurement
    model is

        observation_dBm = received_jammer_power + shadow_fading + measurement_noise

    where the deterministic term composes path loss and antenna gain from each
    jammer at its position at that timestamp. Shadow fading is one spatial
    realisation sampled across all positions; measurement noise is independent
    per observation.

    Args:
        scenario: Scenario with jammers and grid bounds.
        n_uavs: number of UAVs; currently must be 4.
        speed_mps, sample_dt, track_spacing, perturbation_sigma: passed to
            lawnmower_trajectory.
        measurement_noise_sigma: per-observation noise standard deviation, dB.
        path_loss_exponent, shadow_sigma, shadow_L: propagation parameters.
        seed: master RNG seed; sub-seeds derived from it.

    Returns:
        positions (N, 2), timestamps (N,), observations (N,) in dBm, and
        uav_ids (N,).
    """
    if n_uavs != 4:
        raise NotImplementedError(
            "Currently only n_uavs=4 (one per quadrant) supported."
        )

    rng = np.random.default_rng(seed)

    xmin, xmax = scenario.grid_xlim
    ymin, ymax = scenario.grid_ylim
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0

    quadrants = [
        ((xmin, xmid), (ymin, ymid)),  # UAV 0: SW
        ((xmid, xmax), (ymin, ymid)),  # UAV 1: SE
        ((xmin, xmid), (ymid, ymax)),  # UAV 2: NW
        ((xmid, xmax), (ymid, ymax)),  # UAV 3: NE
    ]

    all_positions = []
    all_timestamps = []
    all_uav_ids = []

    for uav_id, (x_range, y_range) in enumerate(quadrants):
        uav_seed = None if seed is None else seed * 1000 + uav_id
        positions, timestamps = lawnmower_trajectory(
            x_range=x_range,
            y_range=y_range,
            speed_mps=speed_mps,
            sample_dt=sample_dt,
            track_spacing=track_spacing,
            perturbation_sigma=perturbation_sigma,
            seed=uav_seed,
        )
        all_positions.append(positions)
        all_timestamps.append(timestamps)
        all_uav_ids.append(np.full(len(positions), uav_id))

    positions = np.vstack(all_positions)
    timestamps = np.concatenate(all_timestamps)
    uav_ids = np.concatenate(all_uav_ids)
    N = positions.shape[0]

    # Compose received jammer power, accounting for time-varying jammer
    # positions (we can't reuse generate_measurements directly because it
    # assumes a single t_sec for all sensors).
    d0 = 1.0
    pl_d0 = free_space_path_loss(d0, scenario.operating_freq_Hz)

    P_total_mW = np.zeros(N)

    for jammer in scenario.jammers:
        jammer_xy = np.array([jammer.position_at(t) for t in timestamps])

        dx = positions[:, 0] - jammer_xy[:, 0]
        dy = positions[:, 1] - jammer_xy[:, 1]
        d = np.maximum(np.sqrt(dx ** 2 + dy ** 2), d0)

        pl = log_distance_path_loss(d, d0=d0, n=path_loss_exponent, pl_d0=pl_d0)

        if jammer.is_directional:
            bearings_deg = np.degrees(np.arctan2(dy, dx))
            gain = directional_gain(
                bearings_deg,
                theta_main=jammer.theta_main_deg,
                theta_3db=jammer.theta_3db_deg,
                G0_dB=jammer.peak_gain_dB,
                front_back_ratio_dB=jammer.front_back_ratio_dB,
            )
        else:
            gain = jammer.peak_gain_dB

        P_jammer_dBm = jammer.power_dBm - pl + gain

        # Per-jammer path-dependent diffraction loss (static jammer position).
        if diffraction is not None:
            jxy = np.asarray(jammer.position, dtype=float)
            P_jammer_dBm = P_jammer_dBm - diffraction.loss(jxy, positions)

        P_total_mW += 10.0 ** (P_jammer_dBm / 10.0)

    P_total_dBm = 10.0 * np.log10(P_total_mW)

    # Structured terrain excess-loss (invisible to the learner's model).
    if terrain is not None:
        P_total_dBm = P_total_dBm - terrain.loss(positions)

    # Single spatial shadow fading realisation across all UAV positions.
    # If a shared ShadowField is supplied, query it so the observations and the
    # evaluation SINR map draw from the SAME realisation (design decision D1);
    # otherwise fall back to an independent per-call draw.
    if shadow_field is not None:
        shadow = shadow_field.evaluate(positions)
    else:
        shadow_seed = None if seed is None else seed + 9_999
        shadow = sample_shadowing_2d(
            positions,
            sigma_sq=shadow_sigma ** 2,
            L=shadow_L,
            seed=shadow_seed,
        )

    measurement_noise = rng.normal(0.0, measurement_noise_sigma, size=N)

    observations = P_total_dBm + shadow + measurement_noise

    return positions, timestamps, observations, uav_ids


if __name__ == "__main__":
    # Demo: Scenario 1 UAV dataset overlaid on the SINR map.
    import os

    import matplotlib.pyplot as plt

    from contested_rf.simulation.ground_truth import compute_sinr_map
    from contested_rf.simulation.scenario import SCENARIO_1

    X, Y, SINR = compute_sinr_map(SCENARIO_1, grid_shape=(100, 100))

    positions, timestamps, observations, uav_ids = generate_uav_observations(
        SCENARIO_1, seed=42
    )

    print(f"Total UAV observations: {len(observations)}")
    print(f"Per-UAV breakdown: {[int((uav_ids == k).sum()) for k in range(4)]}")
    print(f"Observation power range: {observations.min():.1f} to {observations.max():.1f} dBm")
    print(f"SINR range: {SINR.min():.1f} to {SINR.max():.1f} dB")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.pcolormesh(X, Y, SINR, shading="auto", cmap="RdYlGn",
                       vmin=-30, vmax=30, alpha=0.75)
    plt.colorbar(im, ax=ax, label="SINR (dB)")

    uav_colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#17becf"]
    for uav_id in range(4):
        mask = uav_ids == uav_id
        ax.plot(positions[mask, 0], positions[mask, 1],
                "-", color=uav_colors[uav_id], linewidth=0.7, alpha=0.85,
                label=f"UAV {uav_id} ({int(mask.sum())} obs)")

    bs = SCENARIO_1.base_station
    ax.plot(*bs.position, "b^", markersize=15, markeredgewidth=2,
            markeredgecolor="black", label=f"Base station ({bs.name})")
    for j in SCENARIO_1.jammers:
        ax.plot(*j.position, "kx", markersize=15, markeredgewidth=3,
                label=f"Jammer {j.name}")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"{SCENARIO_1.name} — SINR ground truth + UAV trajectories "
        f"({len(observations)} observations)"
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)

    os.makedirs("figures", exist_ok=True)
    out_path = "figures/scenario_1_uav_dataset.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")
