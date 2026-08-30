"""Synthetic-terrain test: does the decomposition beat pure-GP when there is
structured, unmodelled loss? Runs Scenario 1 flat vs. synthetic-terrain and
plots combined / pure-GP / pure-parametric convergence side by side.
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contested_rf.estimators.particle_filter import run_particle_filter
from contested_rf.estimators import combined as C
from contested_rf.metrics.rmse import grid_rmse
from contested_rf.simulation.ground_truth import compute_sinr_map
from contested_rf.simulation.scenario import SCENARIO_1
from contested_rf.simulation.shadow_field import ShadowField
from contested_rf.terrain.synthetic import SyntheticTerrain
from contested_rf.simulation.uav import generate_uav_observations

GRID = (60, 60)
SEED = 42
CHECKPOINTS = [25, 50, 100, 200, 400, 800, 1200, 1600, 2400]


def run_condition(terrain, tag):
    sc = SCENARIO_1
    shadow = ShadowField(sc.grid_xlim, sc.grid_ylim, sigma=4.0, L=50.0, seed=7)
    pos, ts, obs, _ = generate_uav_observations(
        sc, seed=SEED, shadow_field=shadow, terrain=terrain)
    n_obs = len(obs)
    _, _, sinr_true = compute_sinr_map(
        sc, grid_shape=GRID, path_loss_exponent=2.5,
        noise_floor_dBm=C.NOISE_FLOOR_DBM, shadow_field=shadow, terrain=terrain)

    res = run_particle_filter(sc, sensor_positions=pos, timestamps=ts,
                              observations=obs, n_particles=5000,
                              path_loss_exponent=C.N_LEARNER, seed=1)
    theta_hist = res["posterior_means"]

    cps = [c for c in CHECKPOINTS if c <= n_obs] + [n_obs]
    out = {"combined": [], "pure_parametric": [], "pure_gp": [], "cps": cps}

    def thin(Xp, Zk, m=500):
        if len(Zk) <= m:
            return Xp, Zk
        idx = np.linspace(0, len(Zk) - 1, m).astype(int)
        return Xp[idx], Zk[idx]

    wr = wp = None
    for k in cps:
        th = theta_hist[k - 1]
        Xp, Zk = thin(pos[:k], obs[:k])
        gr = C.fit_residual_gp(th, Xp, Zk, sc, warm_gp=wr); wr = gr
        _, _, sc_c = C.combined_sinr_map(th, gr, sc, grid_shape=GRID)
        _, _, sc_p = C.parametric_sinr_map(th, sc, grid_shape=GRID)
        gp, gm = C.fit_pure_gp(Xp, Zk, warm_gp=wp); wp = gp
        _, _, sc_g = C.pure_gp_sinr_map(gp, gm, sc, grid_shape=GRID)
        out["combined"].append(grid_rmse(sinr_true, sc_c))
        out["pure_parametric"].append(grid_rmse(sinr_true, sc_p))
        out["pure_gp"].append(grid_rmse(sinr_true, sc_g))

    print(f"[{tag}] final: combined={out['combined'][-1]:.2f}  "
          f"pureGP={out['pure_gp'][-1]:.2f}  param={out['pure_parametric'][-1]:.2f} dB  "
          f"| combined-vs-pureGP gap = {out['pure_gp'][-1]-out['combined'][-1]:+.2f} dB")
    return out


def main():
    t0 = time.time()
    flat = run_condition(None, "flat")
    terr = run_condition(SyntheticTerrain(), "terrain")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    styles = {"combined": ("C0", "o", "Combined (SMC+GP)"),
              "pure_gp": ("C1", "s", "Pure GP"),
              "pure_parametric": ("C2", "^", "Pure parametric")}
    for ax, data, title in [(axes[0], flat, "Flat terrain"),
                            (axes[1], terr, "Synthetic terrain (structured loss)")]:
        for name, (c, m, lab) in styles.items():
            ax.plot(data["cps"], data[name], "-", color=c, marker=m,
                    markersize=5, linewidth=1.8, label=lab)
        ax.axhline(4.0, color="red", linestyle=":", alpha=0.6, label="4 dB")
        ax.set_xscale("log"); ax.set_xlabel("Observation count $k$")
        ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    axes[0].set_ylabel("SINR RMSE (dB)")
    fig.suptitle("Scenario 1: does the decomposition win when error is structured?",
                 fontsize=13)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = "figures/scenario1_terrain_test.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
