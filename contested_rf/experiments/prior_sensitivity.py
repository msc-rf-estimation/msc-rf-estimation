"""Prior-sensitivity experiment for the Scenario 1 particle filter.

Runs three prior configurations across two data regimes to test where prior
choice affects posterior precision. Posterior precision combines prior and
data precision additively, so prior choice should matter only where the two
are comparable.

Priors:
  A  uninformed uniform over the 2 km x 2 km area
  B  Gaussian, sigma_xy = 500 m, centred 224 m off truth
  C  Gaussian, sigma_xy = 100 m, centred 58 m off truth

Regimes:
  full    all 4000 observations
  sparse  every 20th observation, 200 total

Run from the project root:

    python -m contested_rf.experiments.prior_sensitivity
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from contested_rf.estimators.evaluation import evaluate_pf_result
from contested_rf.estimators.particle_filter import (
    init_particles_gaussian,
    run_particle_filter,
)
from contested_rf.simulation.scenario import SCENARIO_1
from contested_rf.simulation.uav import generate_uav_observations


# --- Shared configuration ---
TRUE_X, TRUE_Y = SCENARIO_1.jammers[0].position  # (1200, 800)
TRUE_P = SCENARIO_1.jammers[0].power_dBm  # 30 dBm
TRUE_PARAMS = (TRUE_X, TRUE_Y, TRUE_P)

N_PARTICLES = 5000
PF_SEED = 1

INTEL_GUESS_B = (1400.0, 700.0)  # 224 m off truth — moderate-quality intel
INTEL_GUESS_C = (1250.0, 770.0)  # 58 m off truth — high-quality intel


def _make_priors():
    """Construct the three prior particle clouds once, reuse across regimes."""
    prior_b = init_particles_gaussian(
        n_particles=N_PARTICLES,
        means=(INTEL_GUESS_B[0], INTEL_GUESS_B[1], 30.0),
        stds=(500.0, 500.0, 2.0),
        seed=PF_SEED,
    )
    prior_c = init_particles_gaussian(
        n_particles=N_PARTICLES,
        means=(INTEL_GUESS_C[0], INTEL_GUESS_C[1], 30.0),
        stds=(100.0, 100.0, 2.0),
        seed=PF_SEED,
    )
    return prior_b, prior_c


def _run_three_priors(positions, timestamps, observations, prior_b, prior_c):
    """Run the PF under three prior configurations, return (label, result) list."""
    result_a = run_particle_filter(
        SCENARIO_1,
        sensor_positions=positions,
        timestamps=timestamps,
        observations=observations,
        n_particles=N_PARTICLES,
        seed=PF_SEED,
    )
    result_b = run_particle_filter(
        SCENARIO_1,
        sensor_positions=positions,
        timestamps=timestamps,
        observations=observations,
        initial_particles=prior_b,
        seed=PF_SEED,
    )
    result_c = run_particle_filter(
        SCENARIO_1,
        sensor_positions=positions,
        timestamps=timestamps,
        observations=observations,
        initial_particles=prior_c,
        seed=PF_SEED,
    )
    return [
        ("A. Uninformed uniform", result_a),
        ("B. Gaussian sigma=500m", result_b),
        ("C. Gaussian sigma=100m", result_c),
    ]


def _print_summary(label, runs):
    """Print a one-block summary across the three priors."""
    print("\n" + "=" * 88)
    print(f"Regime: {label}")
    print("-" * 88)
    print(f"{'Prior':32s} | {'Final err':>10s} | {'Conv@50m':>9s} | "
          f"{'Cov100m':>9s} | {'Final std (x, y)':>20s}")
    print("-" * 88)
    for prior_label, result in runs:
        m = evaluate_pf_result(result, TRUE_PARAMS)
        final_std = result["posterior_stds"][-1]
        print(
            f"{prior_label:32s} | {m['spatial_error_final']:8.1f} m | "
            f"{m['convergence_step_50m']:>9d} | {m['coverage_at_100m']*100:7.1f}% | "
            f"{final_std[0]:6.1f}, {final_std[1]:6.1f} m"
        )
    print("=" * 88)


def _plot_regime(runs, regime_label, out_path):
    """Two-panel figure: spatial error and posterior std over time."""
    colors = {
        "A. Uninformed uniform": "C0",
        "B. Gaussian sigma=500m": "C1",
        "C. Gaussian sigma=100m": "C2",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for label, result in runs:
        m = evaluate_pf_result(result, TRUE_PARAMS)
        ax1.plot(m["spatial_error_history"], label=label,
                 color=colors[label], linewidth=1.4)
        stds_xy = np.sqrt(
            result["posterior_stds"][:, 0] * result["posterior_stds"][:, 1]
        )
        ax2.plot(stds_xy, label=label, color=colors[label], linewidth=1.4)

    ax1.set_yscale("log")
    ax1.set_xlabel("Observation step")
    ax1.set_ylabel("Spatial error to truth (m)")
    ax1.set_title(f"Convergence — {regime_label}")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(loc="upper right")

    ax2.set_yscale("log")
    ax2.set_xlabel("Observation step")
    ax2.set_ylabel(r"Geometric mean of posterior std$_{xy}$ (m)")
    ax2.set_title(f"Posterior spatial precision — {regime_label}")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    # Generate the full UAV dataset once.
    positions, timestamps, observations, _ = generate_uav_observations(
        SCENARIO_1, seed=42
    )

    # Build prior particle clouds once.
    prior_b, prior_c = _make_priors()

    # --- Data-rich regime: use all 4000 observations ---
    runs_full = _run_three_priors(
        positions, timestamps, observations, prior_b, prior_c
    )
    _print_summary("data-rich (4000 observations)", runs_full)
    _plot_regime(runs_full, "data-rich (N_obs = 4000)",
                 "figures/prior_sensitivity_full.png")

    # --- Data-sparse regime: every 20th observation (200 total) ---
    # Stride is preferable to first-N here because it gives a sample that
    # spans all four UAV quadrants, mimicking "early mission" coverage rather
    # than just one corner of the area.
    stride = 20
    pos_sparse = positions[::stride]
    ts_sparse = timestamps[::stride]
    obs_sparse = observations[::stride]

    runs_sparse = _run_three_priors(
        pos_sparse, ts_sparse, obs_sparse, prior_b, prior_c
    )
    _print_summary(f"data-sparse ({len(obs_sparse)} observations)", runs_sparse)
    _plot_regime(runs_sparse, f"data-sparse (N_obs = {len(obs_sparse)})",
                 "figures/prior_sensitivity_sparse.png")

    # --- Cross-regime comparison table ---
    print("\n" + "=" * 88)
    print("Cross-regime comparison — final spatial error (m)")
    print("-" * 88)
    print(f"{'Prior':32s} | {'Data-rich (4000)':>17s} | {'Data-sparse (200)':>18s} | "
          f"{'Sparse vs Rich':>16s}")
    print("-" * 88)
    for (label_full, result_full), (label_sparse, result_sparse) in zip(
        runs_full, runs_sparse
    ):
        m_full = evaluate_pf_result(result_full, TRUE_PARAMS)
        m_sparse = evaluate_pf_result(result_sparse, TRUE_PARAMS)
        ratio = m_sparse['spatial_error_final'] / m_full['spatial_error_final']
        print(
            f"{label_full:32s} | {m_full['spatial_error_final']:>15.1f} m | "
            f"{m_sparse['spatial_error_final']:>16.1f} m | {ratio:>15.1f}x"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
