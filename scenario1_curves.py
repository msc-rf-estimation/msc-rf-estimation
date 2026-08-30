"""First convergence curves: all four estimators on Scenario 1, flat terrain.

Runs the SMC once (learner n=2.0), then at a set of observation checkpoints
reconstructs each estimator's SINR map and scores RMSE against the ground-truth
map. Produces the headline "does the core idea work?" figure.
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contested_rf.estimators.particle_filter import run_particle_filter
from contested_rf.estimators import combined as C
from contested_rf.metrics.rmse import grid_rmse, convergence_step
from contested_rf.simulation.ground_truth import compute_sinr_map
from contested_rf.simulation.scenario import SCENARIO_1
from contested_rf.simulation.shadow_field import ShadowField
from contested_rf.simulation.uav import generate_uav_observations

GRID = (60, 60)
TRUTH_N = 2.5
SEED = 42


def main():
    t0 = time.time()
    scenario = SCENARIO_1

    # --- Shared shadow realisation (design decision D1) ---
    # One field, queried by BOTH the observations and the evaluation map, so the
    # ground-truth SINR is the realised field the GP can legitimately learn.
    shadow = ShadowField(
        scenario.grid_xlim, scenario.grid_ylim, sigma=4.0, L=50.0, seed=7
    )

    # --- Data + ground truth ---
    positions, timestamps, observations, _ = generate_uav_observations(
        scenario, seed=SEED, shadow_field=shadow
    )
    n_obs = len(observations)
    _, _, sinr_true = compute_sinr_map(
        scenario, grid_shape=GRID, path_loss_exponent=TRUTH_N,
        noise_floor_dBm=C.NOISE_FLOOR_DBM, shadow_field=shadow,
    )

    # --- SMC (learner n = 2.0) ---
    result = run_particle_filter(
        scenario, sensor_positions=positions, timestamps=timestamps,
        observations=observations, n_particles=5000,
        path_loss_exponent=C.N_LEARNER, seed=1,
    )
    theta_hist = result["posterior_means"]  # (n_obs, 3)
    print(f"SMC done ({time.time()-t0:.1f}s). Final theta_hat = "
          f"{theta_hist[-1].round(1)}  (true = [1200, 800, 30])")

    # --- Checkpoints (log-ish spaced) ---
    checkpoints = [c for c in
                   [25, 50, 75, 100, 150, 200, 300, 400, 600, 800,
                    1200, 1600, 2400, n_obs]
                   if c <= n_obs]

    curves = {"combined": [], "pure_parametric": [], "pure_gp": [],
              "no_jamming": []}

    # No-jamming is constant (ignores observations).
    _, _, sinr_nj = C.no_jamming_sinr_map(scenario, grid_shape=GRID)
    rmse_nj = grid_rmse(sinr_true, sinr_nj)

    # Cap GP training size for speed on this first-look run: evenly thin the
    # observations seen so far to at most MAX_GP points. This is a subset-of-
    # data GP (legitimate sparse method); the full sweep will use the proper
    # sparse-inducing path. Convergence shape is preserved.
    MAX_GP = 500

    def _thin(Xp, Zk):
        if len(Zk) <= MAX_GP:
            return Xp, Zk
        idx = np.linspace(0, len(Zk) - 1, MAX_GP).astype(int)
        return Xp[idx], Zk[idx]

    warm_res = None
    warm_pure = None
    for k in checkpoints:
        theta_k = theta_hist[k - 1]
        Xp, Zk = _thin(positions[:k], observations[:k])

        # Combined: residual GP given theta_k.
        gp_res = C.fit_residual_gp(theta_k, Xp, Zk, scenario, warm_gp=warm_res)
        warm_res = gp_res
        _, _, sinr_c = C.combined_sinr_map(theta_k, gp_res, scenario, grid_shape=GRID)

        # Pure parametric.
        _, _, sinr_p = C.parametric_sinr_map(theta_k, scenario, grid_shape=GRID)

        # Pure GP on raw observations.
        gp_pure, gp_mean = C.fit_pure_gp(Xp, Zk, warm_gp=warm_pure)
        warm_pure = gp_pure
        _, _, sinr_g = C.pure_gp_sinr_map(gp_pure, gp_mean, scenario, grid_shape=GRID)

        curves["combined"].append(grid_rmse(sinr_true, sinr_c))
        curves["pure_parametric"].append(grid_rmse(sinr_true, sinr_p))
        curves["pure_gp"].append(grid_rmse(sinr_true, sinr_g))
        curves["no_jamming"].append(rmse_nj)
        print(f"  k={k:5d}: combined={curves['combined'][-1]:5.2f}  "
              f"param={curves['pure_parametric'][-1]:5.2f}  "
              f"pureGP={curves['pure_gp'][-1]:5.2f}  dB")

    # --- Convergence steps to 4 dB (flat-terrain threshold) ---
    print("\nObservations to RMSE < 4 dB (flat threshold):")
    for name in ("combined", "pure_parametric", "pure_gp"):
        step = convergence_step(curves[name], checkpoints, threshold=4.0, window=3)
        print(f"  {name:16s}: {'never' if step < 0 else str(step)}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 6))
    styles = {
        "combined": ("C0", "o", "Combined (SMC + GP)"),
        "pure_gp": ("C1", "s", "Pure GP"),
        "pure_parametric": ("C2", "^", "Pure parametric (SMC only)"),
        "no_jamming": ("0.5", "", "No-jamming baseline"),
    }
    for name, (color, marker, label) in styles.items():
        ls = "--" if name == "no_jamming" else "-"
        ax.plot(checkpoints, curves[name], ls, color=color, marker=marker,
                markersize=5, linewidth=1.8, label=label)
    ax.axhline(4.0, color="red", linestyle=":", alpha=0.6,
               label="4 dB threshold (flat)")
    ax.set_xscale("log")
    ax.set_xlabel("Observation count $k$")
    ax.set_ylabel("SINR RMSE (dB)")
    ax.set_title("Scenario 1 (flat terrain): SINR-map convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out = "figures/scenario1_convergence_curves.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}  (total {time.time()-t0:.1f}s)")
    return curves, checkpoints


if __name__ == "__main__":
    main()
