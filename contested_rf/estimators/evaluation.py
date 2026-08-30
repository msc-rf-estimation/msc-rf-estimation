"""Quantitative evaluation of particle filter results against ground truth.

Designed to consume the dict returned by `run_particle_filter` plus the true
jammer parameters, returning a dict of metrics. Use across scenarios to
compare PF behaviour: "how well did the PF localise, how concentrated is the
posterior, how often did weights degenerate?"
"""
import numpy as np


def evaluate_pf_result(result, true_params, coverage_radii_m=(50.0, 100.0, 200.0)):
    """Summarise a particle-filter run against ground truth.

    Covers accuracy (posterior mean against truth, finally and over time),
    concentration (posterior weight within given radii of truth) and filter
    health (ESS statistics and resampling count).

    Args:
        result: dict returned by run_particle_filter.
        true_params: (x, y, P_tx) ground-truth emitter state.
        coverage_radii_m: radii in metres at which to compute weight coverage.

    Returns:
        dict with spatial_error_final and _history, power_error_final and
        _history, coverage_at_<r>m, ess_min/mean/p50, n_resamples and
        convergence_step_50m (-1 if never reached).
    """
    true_x, true_y, true_p = true_params

    means = result["posterior_means"]
    spatial_errors = np.sqrt(
        (means[:, 0] - true_x) ** 2 + (means[:, 1] - true_y) ** 2
    )
    power_errors = np.abs(means[:, 2] - true_p)

    # Coverage of truth by final posterior weight: how much probability mass
    # lies within R metres of the true (x, y).
    particles = result["particles_final"]
    weights = np.exp(result["log_weights_final"])
    d_final = np.sqrt(
        (particles[:, 0] - true_x) ** 2 + (particles[:, 1] - true_y) ** 2
    )
    coverage = {}
    for r in coverage_radii_m:
        coverage[f"coverage_at_{int(r)}m"] = float(weights[d_final < r].sum())

    ess = result["ess_history"]

    below_50 = np.where(spatial_errors < 50.0)[0]
    convergence_step_50m = int(below_50[0]) if len(below_50) > 0 else -1

    return {
        "spatial_error_final": float(spatial_errors[-1]),
        "spatial_error_history": spatial_errors,
        "power_error_final": float(power_errors[-1]),
        "power_error_history": power_errors,
        "ess_min": float(ess.min()),
        "ess_mean": float(ess.mean()),
        "ess_p50": float(np.median(ess)),
        "n_resamples": len(result["resample_steps"]),
        "convergence_step_50m": convergence_step_50m,
        **coverage,
    }


def format_evaluation_summary(metrics, scenario_name=""):
    """Pretty-print a metrics dict as a one-block summary."""
    lines = [
        f"--- PF evaluation {('— ' + scenario_name) if scenario_name else ''} ---",
        f"  Final spatial error:        {metrics['spatial_error_final']:.1f} m",
        f"  Final power error:          {metrics['power_error_final']:.2f} dB",
        f"  First time error < 50 m:    "
        + (f"step {metrics['convergence_step_50m']}"
           if metrics["convergence_step_50m"] >= 0 else "never"),
    ]
    coverage_keys = sorted(k for k in metrics if k.startswith("coverage_at_"))
    for k in coverage_keys:
        radius = k.replace("coverage_at_", "").replace("m", "")
        lines.append(f"  Posterior weight within {radius} m: {metrics[k]*100:.1f}%")
    lines.extend([
        f"  ESS  (min / median / mean): "
        f"{metrics['ess_min']:.0f} / {metrics['ess_p50']:.0f} / {metrics['ess_mean']:.0f}",
        f"  Resampling events:          {metrics['n_resamples']}",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    # Failure-mode demo: run the SAME (static-omni) PF on all three scenarios,
    # evaluate each against the appropriate truth, and save a comparison
    # figure. Expect S1 to work cleanly; S2 to localise somewhere biased by
    # the directional jammer's interference; S3 to localise somewhere
    # inconsistent because the static-jammer assumption is violated by motion.
    import os

    import matplotlib.pyplot as plt

    from contested_rf.estimators.particle_filter import run_particle_filter
    from contested_rf.simulation.ground_truth import compute_sinr_map
    from contested_rf.simulation.scenario import (
        SCENARIO_1,
        SCENARIO_2,
        SCENARIO_3,
    )
    from contested_rf.simulation.uav import generate_uav_observations

    def _truth_for(scenario, final_t):
        """Pick a sensible 'true' jammer state for evaluation.

        S1: the single static jammer.
        S2: the *omni* jammer A — the directional jammer B is treated as
            interference the simple PF isn't equipped to disentangle.
        S3: the jammer's position at the final timestamp.
        """
        if scenario is SCENARIO_1:
            j = scenario.jammers[0]
            return (j.position[0], j.position[1], j.power_dBm)
        if scenario is SCENARIO_2:
            j = scenario.jammers[0]  # jammer A, the omni one
            return (j.position[0], j.position[1], j.power_dBm)
        # SCENARIO_3
        j = scenario.jammers[0]
        x_final, y_final = j.position_at(final_t)
        return (x_final, y_final, j.power_dBm)

    scenarios = [
        ("S1", SCENARIO_1),
        ("S2", SCENARIO_2),
        ("S3", SCENARIO_3),
    ]

    fig, axes = plt.subplots(len(scenarios), 3, figsize=(18, 5 * len(scenarios)))
    all_metrics = {}

    for row, (label, scenario) in enumerate(scenarios):
        # Generate UAV dataset for this scenario.
        positions, timestamps, observations, _ = generate_uav_observations(
            scenario, seed=42
        )

        # Choose truth based on scenario type.
        true_params = _truth_for(scenario, final_t=timestamps[-1])

        # Run the same (static-omni) PF.
        result = run_particle_filter(
            scenario,
            sensor_positions=positions,
            timestamps=timestamps,
            observations=observations,
            n_particles=5000,
            seed=1,
        )

        metrics = evaluate_pf_result(result, true_params=true_params)
        all_metrics[label] = metrics

        print(format_evaluation_summary(metrics, scenario_name=f"{label} ({scenario.name})"))
        print()

        # Per-row diagnostic plots.
        ax_map, ax_err, ax_ess = axes[row]

        # Compute SINR map *at the final timestamp* so the cold spots match
        # the jammer marker positions (matters for dynamic scenarios).
        X, Y, SINR = compute_sinr_map(scenario, grid_shape=(80, 80),
                                       t_sec=timestamps[-1])
        im = ax_map.pcolormesh(X, Y, SINR, shading="auto", cmap="RdYlGn",
                               vmin=-30, vmax=30, alpha=0.5)
        plt.colorbar(im, ax=ax_map, label="SINR (dB)")

        # Subsample final particle cloud for visibility.
        particles_final = result["particles_final"]
        weights_final = np.exp(result["log_weights_final"])
        rng_plot = np.random.default_rng(0)
        n_show = min(500, len(particles_final))
        idx = rng_plot.choice(len(particles_final), n_show, replace=False)
        ax_map.scatter(particles_final[idx, 0], particles_final[idx, 1],
                       s=weights_final[idx] * 5000 + 2, c="purple", alpha=0.4,
                       label="Particle cloud")

        # Base station first (so it sits behind jammer markers).
        ax_map.plot(*scenario.base_station.position, "b^",
                    markersize=12, markeredgewidth=1.5, markeredgecolor="black",
                    label="Base station")

        # Eval target — drawn before jammer markers so X sits on top.
        ax_map.plot(true_params[0], true_params[1], "o",
                    markersize=18, markerfacecolor="none",
                    markeredgewidth=2.0, markeredgecolor="cyan",
                    label="Eval target")

        # Jammers — bigger, more prominent, with labels next to each. For
        # dynamic jammers also draw the trajectory line.
        for j in scenario.jammers:
            if j.velocity_mps > 0 and j.target_position is not None:
                t_samples = np.linspace(0.0, timestamps[-1], 60)
                traj = np.array([j.position_at(t) for t in t_samples])
                ax_map.plot(traj[:, 0], traj[:, 1], "k:", linewidth=1.2,
                            alpha=0.6, label=f"Jammer {j.name} trajectory")
            jx, jy = j.position_at(timestamps[-1])
            ax_map.plot(jx, jy, "kx", markersize=18, markeredgewidth=3.5)
            ax_map.annotate(
                f"  {j.name}", xy=(jx, jy), xytext=(10, 6),
                textcoords="offset points", fontsize=11, fontweight="bold",
            )

        # Posterior mean drawn last so star is on top.
        final_mean = result["posterior_means"][-1]
        ax_map.plot(final_mean[0], final_mean[1], "y*", markersize=18,
                    markeredgewidth=1.5, markeredgecolor="black",
                    label=f"Posterior mean (err {metrics['spatial_error_final']:.0f} m)")
        ax_map.set_xlabel("x (m)")
        ax_map.set_ylabel("y (m)")
        ax_map.set_title(f"{label}: {scenario.name}")
        ax_map.set_aspect("equal")
        ax_map.legend(loc="upper left", fontsize=8)

        # Convergence plot.
        errs = metrics["spatial_error_history"]
        ax_err.plot(errs, "b-", linewidth=1.1)
        ax_err.set_yscale("log")
        ax_err.set_xlabel("Observation step")
        ax_err.set_ylabel("Distance from truth (m)")
        ax_err.set_title(f"{label}: convergence")
        ax_err.grid(True, alpha=0.3)

        # ESS history.
        ess = result["ess_history"]
        n_part = particles_final.shape[0]
        ax_ess.plot(ess, "g-", linewidth=1.1)
        ax_ess.axhline(0.5 * n_part, color="orange", linestyle="--", alpha=0.6,
                       label="Resample threshold")
        ax_ess.axhline(n_part, color="black", linestyle=":", alpha=0.3,
                       label=f"N = {n_part}")
        ax_ess.set_xlabel("Observation step")
        ax_ess.set_ylabel("ESS")
        ax_ess.set_title(f"{label}: ESS ({metrics['n_resamples']} resamples)")
        ax_ess.legend(loc="upper right", fontsize=8)
        ax_ess.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out_path = "figures/pf_all_scenarios_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")

    # Print a single-line comparison table.
    print("\n--- Summary table ---")
    print(f"{'Scenario':<6} | {'Final err (m)':>14} | {'Power err (dB)':>14} | "
          f"{'Cov 100 m':>10} | {'Resamples':>10}")
    print("-" * 70)
    for label, m in all_metrics.items():
        cov = m.get("coverage_at_100m", 0.0)
        print(f"{label:<6} | {m['spatial_error_final']:>14.1f} | "
              f"{m['power_error_final']:>14.2f} | {cov * 100:>9.1f}% | "
              f"{m['n_resamples']:>10}")
