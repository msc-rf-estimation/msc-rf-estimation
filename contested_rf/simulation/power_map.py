"""Received power maps over a scenario grid, flat terrain.

Composes path loss, antenna gain and shadow fading. Run directly to plot a
Scenario 1 heatmap:

    python -m contested_rf.simulation.power_map
"""
import numpy as np

from contested_rf.propagation.antenna import directional_gain
from contested_rf.propagation.path_loss import (
    free_space_path_loss,
    log_distance_path_loss,
)
from contested_rf.propagation.shadow_fading import sample_shadowing_2d


def compute_received_power_map(
    scenario,
    grid_shape=(100, 100),
    path_loss_exponent=2.5,
    shadow_sigma=4.0,
    shadow_L=50.0,
    t_sec=0.0,
    seed=None,
):
    """Received power in dBm at every grid point for a scenario.

    Composes log-distance path loss, antenna gain and one shadow-fading
    realisation. Multiple jammers are summed in linear mW, since powers from
    independent emitters add linearly rather than in dB.

    Args:
        scenario: Scenario with jammers and grid bounds.
        grid_shape: (nx, ny) grid points per axis.
        path_loss_exponent: log-distance n.
        shadow_sigma: shadow standard deviation, dB.
        shadow_L: shadow decorrelation distance, metres.
        t_sec: simulation time, for moving emitters.
        seed: RNG seed for the shadow realisation.

    Returns:
        (X, Y, P_dBm) with X and Y meshgrids of shape (ny, nx).
    """
    xmin, xmax = scenario.grid_xlim
    ymin, ymax = scenario.grid_ylim
    nx, ny = grid_shape

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)  # both shape (ny, nx)

    # Path loss reference: free-space loss at 1 m at the carrier frequency.
    d0 = 1.0
    pl_d0 = free_space_path_loss(d0, scenario.operating_freq_Hz)

    # Accumulate jammer contributions in linear (mW) — we need linear because
    # the *powers* sum, not the dB values. Working in dB and summing would be
    # silently wrong for multi-jammer scenarios.
    P_total_mW = np.zeros((ny, nx))

    for jammer in scenario.jammers:
        jx, jy = jammer.position_at(t_sec)

        # Distance from this jammer to every grid point.
        d = np.sqrt((X - jx) ** 2 + (Y - jy) ** 2)
        # Clamp distance to d0 so log10(0) doesn't blow up at the jammer's
        # exact location. The path loss model is only valid for d >= d0 anyway.
        d = np.maximum(d, d0)

        # Path loss attenuation, per grid point.
        pl = log_distance_path_loss(d, d0=d0, n=path_loss_exponent, pl_d0=pl_d0)

        # Antenna gain. For directional jammers we compute bearings vectorised
        # and call directional_gain on the whole array. For omni jammers the
        # gain is a constant scalar and broadcasts.
        if jammer.is_directional:
            bearings_deg = np.degrees(np.arctan2(Y - jy, X - jx))
            gain = directional_gain(
                bearings_deg,
                theta_main=jammer.theta_main_deg,
                theta_3db=jammer.theta_3db_deg,
                G0_dB=jammer.peak_gain_dB,
            )
        else:
            gain = jammer.peak_gain_dB

        # Received power from this jammer at every grid point, in dBm.
        P_jammer_dBm = jammer.power_dBm - pl + gain

        # Convert to linear (mW) and accumulate. Powers sum linearly.
        P_total_mW += 10.0 ** (P_jammer_dBm / 10.0)

    # Convert summed linear power back to dBm.
    P_total_dBm = 10.0 * np.log10(P_total_mW)

    # Single shadow fading realisation across the whole grid. There is *one*
    # underlying field (the physical environment); every grid point reads its
    # value at its own location. Independent shadow per point would break the
    # spatial correlation and the conditional-independence story.
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    shadow = sample_shadowing_2d(
        grid_points, sigma_sq=shadow_sigma ** 2, L=shadow_L, seed=seed
    ).reshape(ny, nx)

    return X, Y, P_total_dBm + shadow


if __name__ == "__main__":
    # Generate and save a heatmap for Scenario 1.
    import os

    import matplotlib.pyplot as plt

    from contested_rf.simulation.scenario import SCENARIO_1

    X, Y, P = compute_received_power_map(SCENARIO_1, grid_shape=(100, 100), seed=42)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(X, Y, P, shading="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="Received power (dBm)")

    for j in SCENARIO_1.jammers:
        ax.plot(*j.position, "rx", markersize=12, markeredgewidth=2,
                label=f"Jammer {j.name}")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{SCENARIO_1.name} — received power map")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")

    os.makedirs("figures", exist_ok=True)
    out_path = "figures/scenario_1_power_map.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
