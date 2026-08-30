"""Ground-truth SINR maps and synthetic sensor measurements.

compute_sinr_map is the evaluation artefact: deterministic SINR over a grid
given the base station, jammers and noise floor. Shadow fading is excluded, so
the map is a property of the environment.

generate_measurements is the data the estimators see: received jammer power at
given sensor positions, with shadow fading applied. The base station and noise
floor do not appear, as they are not being inferred.
"""
import numpy as np

from contested_rf.propagation.antenna import directional_gain
from contested_rf.propagation.path_loss import (
    free_space_path_loss,
    log_distance_path_loss,
)
from contested_rf.propagation.shadow_fading import sample_shadowing_2d
from contested_rf.propagation.sinr import sinr_from_powers


def compute_sinr_map(
    scenario,
    grid_shape=(100, 100),
    path_loss_exponent=2.5,
    noise_floor_dBm=-100.0,
    t_sec=0.0,
    shadow_field=None,
    terrain=None,
    diffraction=None,
):
    """Deterministic SINR in dB at every grid point.

        SINR_linear = P_signal / (sum P_interference + P_noise)

    Interference from all jammers is summed in linear mW. Shadow fading is not
    applied here; randomness enters only at measurement.

    Args:
        scenario: Scenario with base_station and jammers.
        grid_shape: (nx, ny) grid points per axis.
        path_loss_exponent: log-distance n.
        noise_floor_dBm: thermal and receiver noise floor.
        t_sec: simulation time, for moving emitters.

    Returns:
        (X, Y, SINR_dB) with X and Y meshgrids of shape (ny, nx).
    """
    xmin, xmax = scenario.grid_xlim
    ymin, ymax = scenario.grid_ylim
    nx, ny = grid_shape

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    d0 = 1.0
    pl_d0 = free_space_path_loss(d0, scenario.operating_freq_Hz)

    # --- Signal from the base station ---
    bs = scenario.base_station
    bx, by = bs.position
    d_bs = np.maximum(np.sqrt((X - bx) ** 2 + (Y - by) ** 2), d0)
    pl_bs = log_distance_path_loss(d_bs, d0=d0, n=path_loss_exponent, pl_d0=pl_d0)

    if bs.is_directional:
        bearings_bs = np.degrees(np.arctan2(Y - by, X - bx))
        gain_bs = directional_gain(
            bearings_bs,
            theta_main=bs.theta_main_deg,
            theta_3db=bs.theta_3db_deg,
            G0_dB=bs.peak_gain_dB,
        )
    else:
        gain_bs = bs.peak_gain_dB

    P_signal_dBm = bs.power_dBm - pl_bs + gain_bs

    # --- Total interference from all jammers ---
    # Powers sum linearly (in mW), not in dB.
    P_interference_mW = np.zeros((ny, nx))

    for jammer in scenario.jammers:
        jx, jy = jammer.position_at(t_sec)
        d_j = np.maximum(np.sqrt((X - jx) ** 2 + (Y - jy) ** 2), d0)
        pl_j = log_distance_path_loss(d_j, d0=d0, n=path_loss_exponent, pl_d0=pl_d0)

        if jammer.is_directional:
            bearings_j = np.degrees(np.arctan2(Y - jy, X - jx))
            gain_j = directional_gain(
                bearings_j,
                theta_main=jammer.theta_main_deg,
                theta_3db=jammer.theta_3db_deg,
                G0_dB=jammer.peak_gain_dB,
                front_back_ratio_dB=jammer.front_back_ratio_dB,
            )
        else:
            gain_j = jammer.peak_gain_dB

        P_jammer_dBm = jammer.power_dBm - pl_j + gain_j

        # Path-dependent knife-edge diffraction loss from THIS jammer to every
        # grid point (per-jammer, since each emitter casts its own terrain
        # shadow). Invisible to the learner's forward model.
        if diffraction is not None:
            grid_pts_d = np.column_stack([X.ravel(), Y.ravel()])
            dl = diffraction.loss(np.array([jx, jy]), grid_pts_d).reshape(X.shape)
            P_jammer_dBm = P_jammer_dBm - dl

        P_interference_mW += 10.0 ** (P_jammer_dBm / 10.0)

    # Cap minimum to avoid log(0) when there are no jammers in this scenario
    # (purely defensive; current scenarios all have at least one).
    P_interference_mW = np.maximum(P_interference_mW, 1e-30)
    P_interference_dBm = 10.0 * np.log10(P_interference_mW)

    # Structured terrain excess-loss (invisible to the learner's model): applied
    # to the interference field so the ground truth carries the same terrain
    # shadows the UAV observations saw.
    if terrain is not None:
        grid_pts_t = np.column_stack([X.ravel(), Y.ravel()])
        P_interference_dBm = P_interference_dBm - terrain.loss(grid_pts_t).reshape(X.shape)

    # Shared shadow-fading realisation (design decision D1): if supplied, apply
    # the SAME field the UAV observations saw to the interference field, so the
    # evaluation target is the realised (not idealised) SINR. Applied to the
    # total jammer power in dB, matching the observation model in uav.py.
    if shadow_field is not None:
        grid_pts = np.column_stack([X.ravel(), Y.ravel()])
        shadow = shadow_field.evaluate(grid_pts).reshape(X.shape)
        P_interference_dBm = P_interference_dBm + shadow

    # --- SINR via the existing helper ---
    SINR_dB = sinr_from_powers(
        p_signal_dbm=P_signal_dBm,
        p_interference_dbm=P_interference_dBm,
        p_noise_dbm=noise_floor_dBm,
    )

    return X, Y, SINR_dB


def generate_measurements(
    scenario,
    sensor_positions,
    path_loss_exponent=2.5,
    shadow_sigma=4.0,
    shadow_L=50.0,
    t_sec=0.0,
    seed=None,
):
    """Received power in dBm at each sensor position.

    Args:
        scenario: Scenario with jammers, grid bounds and carrier frequency.
        sensor_positions: (N, 2) sensor positions in metres.
        path_loss_exponent: log-distance n.
        shadow_sigma: shadow standard deviation, dB.
        shadow_L: shadow decorrelation distance, metres.
        t_sec: simulation time, for moving emitters.
        seed: RNG seed for the shadow realisation.

    Returns:
        (N,) received power in dBm.
    """
    sensor_positions = np.asarray(sensor_positions, dtype=float)
    N = sensor_positions.shape[0]

    # Reference distance for the log-distance path loss model.
    d0 = 1.0
    pl_d0 = free_space_path_loss(d0, scenario.operating_freq_Hz)

    # Accumulate jammer contributions in linear (mW). Power from independent
    # emitters adds linearly, not in dB.
    P_total_mW = np.zeros(N)

    for jammer in scenario.jammers:
        jx, jy = jammer.position_at(t_sec)

        dx = sensor_positions[:, 0] - jx
        dy = sensor_positions[:, 1] - jy
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
        P_total_mW += 10.0 ** (P_jammer_dBm / 10.0)

    P_total_dBm = 10.0 * np.log10(P_total_mW)

    # One shadow fading realisation across *all* sensors. Sensors close together
    # share correlated shadow values; sensors far apart have nearly independent
    # values. This correlation is exactly what the SMC's "conditional indepen-
    # dence given the parameters" assumption is approximating — and why it
    # only holds cleanly when sensors are well-separated relative to shadow_L.
    shadow = sample_shadowing_2d(
        sensor_positions, sigma_sq=shadow_sigma ** 2, L=shadow_L, seed=seed
    )

    return P_total_dBm + shadow


if __name__ == "__main__":
    # Generate and save a Scenario 1 SINR map. This is the deterministic
    # ground truth that the SMC + GP combination eventually tries to
    # reconstruct from sparse UAV observations.
    import os

    import matplotlib.pyplot as plt

    from contested_rf.simulation.scenario import SCENARIO_1

    X, Y, SINR = compute_sinr_map(SCENARIO_1, grid_shape=(100, 100))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(X, Y, SINR, shading="auto", cmap="RdYlGn",
                       vmin=-30, vmax=30)
    plt.colorbar(im, ax=ax, label="SINR (dB)")

    # Mark base station and jammer positions.
    bs = SCENARIO_1.base_station
    ax.plot(*bs.position, "b^", markersize=14, markeredgewidth=2,
            markeredgecolor="black", label=f"Base station ({bs.name})")
    for j in SCENARIO_1.jammers:
        ax.plot(*j.position, "rx", markersize=14, markeredgewidth=3,
                label=f"Jammer {j.name}")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{SCENARIO_1.name} — SINR map (deterministic ground truth)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")

    os.makedirs("figures", exist_ok=True)
    out_path = "figures/scenario_1_sinr_map.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"SINR range: {SINR.min():.1f} dB to {SINR.max():.1f} dB")
