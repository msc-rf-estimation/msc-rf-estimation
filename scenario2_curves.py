"""Scenario 2 (two jammers, one directional): does the decomposition win on the
harder multi-modal inference problem? Convergence of all estimators vs. a
consistent ground truth built from the same clamped forward model.
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contested_rf.estimators.particle_filter_s2 import run_particle_filter_s2
from contested_rf.estimators import combined as C
from contested_rf.estimators import combined_s2 as C2
from contested_rf.metrics.rmse import grid_rmse, convergence_step
from contested_rf.simulation.scenario import SCENARIO_2
from contested_rf.simulation.shadow_field import ShadowField
from contested_rf.terrain.synthetic import SyntheticTerrain
from contested_rf.simulation.uav import generate_uav_observations

GRID = (60, 60)
SEED = 42
TRUE_B = np.array([1400.0, 1300.0, 27.0, 225.0, 60.0, 30.0])  # x,y,Ptx,tm,t3,Fb
TERRAIN = None  # flat first — isolates the multi-jammer inference question


def build_obs_and_truth(scenario, shadow, terrain, rng):
    jammer_a = scenario.jammers[0]
    # UAV trajectory (positions/timestamps) — discard its observations; we
    # recompute them through the clamped forward model for consistency.
    pos, ts, _, _ = generate_uav_observations(
        scenario, seed=SEED, shadow_field=shadow, terrain=terrain)
    n = len(pos)

    # Observations: true B (n=2.5), + shadow + optional terrain loss + noise.
    interf = C2.s2_interference_dBm(pos, TRUE_B, jammer_a, scenario, n=2.5)
    if terrain is not None:
        interf = interf - terrain.loss(pos)
    obs = interf + shadow.evaluate(pos) + rng.normal(0, 2.0, size=n)

    # Truth SINR map (same forward model, no measurement noise).
    X, Y, pts = C._grid_points(scenario, GRID)
    g_interf = C2.s2_interference_dBm(pts, TRUE_B, jammer_a, scenario, n=2.5)
    if terrain is not None:
        g_interf = g_interf - terrain.loss(pts)
    g_interf = g_interf + shadow.evaluate(pts)
    P_signal = C.learner_signal_dBm(scenario, pts)
    sinr_true = C._sinr_from_interference(g_interf, P_signal).reshape(X.shape)
    return pos, ts, obs, sinr_true


def main():
    t0 = time.time()
    sc = SCENARIO_2
    jammer_a = sc.jammers[0]
    rng = np.random.default_rng(0)
    shadow = ShadowField(sc.grid_xlim, sc.grid_ylim, sigma=4.0, L=50.0, seed=7)
    pos, ts, obs, sinr_true = build_obs_and_truth(sc, shadow, TERRAIN, rng)
    n_obs = len(obs)

    # 6D SMC (learner n=2.0), inferring directional jammer B.
    r = run_particle_filter_s2(sc, sensor_positions=pos, timestamps=ts,
                               observations=obs, n_particles=12000,
                               path_loss_exponent=C.N_LEARNER, seed=1)
    B_hist = r["posterior_means"]
    print(f"SMC done ({time.time()-t0:.1f}s). B_hat = {B_hist[-1].round(1)}")
    print(f"true B                = {TRUE_B}")

    checkpoints = [c for c in [50, 100, 200, 400, 800, 1200, 1600, 2400] if c <= n_obs] + [n_obs]
    curves = {"combined": [], "pure_gp": [], "pure_parametric": []}

    def thin(Xp, Zk, m=500):
        if len(Zk) <= m:
            return Xp, Zk
        idx = np.linspace(0, len(Zk) - 1, m).astype(int)
        return Xp[idx], Zk[idx]

    for k in checkpoints:
        B_k = B_hist[k - 1]
        Xp, Zp = thin(pos[:k], obs[:k])
        gr = C2.fit_residual_gp_s2(B_k, jammer_a, Xp, Zp, sc)
        _, _, sc_c = C2.combined_sinr_map_s2(B_k, gr, jammer_a, sc, grid_shape=GRID)
        _, _, sc_p = C2.parametric_sinr_map_s2(B_k, jammer_a, sc, grid_shape=GRID)
        gp, gm = C.fit_pure_gp(Xp, Zp)
        _, _, sc_g = C.pure_gp_sinr_map(gp, gm, sc, grid_shape=GRID)
        curves["combined"].append(grid_rmse(sinr_true, sc_c))
        curves["pure_gp"].append(grid_rmse(sinr_true, sc_g))
        curves["pure_parametric"].append(grid_rmse(sinr_true, sc_p))
        print(f"  k={k:5d}: combined={curves['combined'][-1]:5.2f}  "
              f"pureGP={curves['pure_gp'][-1]:5.2f}  param={curves['pure_parametric'][-1]:5.2f} dB")

    gap = curves["pure_gp"][-1] - curves["combined"][-1]
    print(f"\nFinal combined-vs-pureGP gap = {gap:+.2f} dB "
          f"({'combined better' if gap > 0 else 'pure-GP better'})")

    fig, ax = plt.subplots(figsize=(9, 6))
    styles = {"combined": ("C0", "o", "Combined (SMC+GP)"),
              "pure_gp": ("C1", "s", "Pure GP"),
              "pure_parametric": ("C2", "^", "Pure parametric")}
    for name, (c, m, lab) in styles.items():
        ax.plot(checkpoints, curves[name], "-", color=c, marker=m, markersize=5,
                linewidth=1.8, label=lab)
    ax.axhline(4.0, color="red", linestyle=":", alpha=0.6, label="4 dB")
    ax.set_xscale("log"); ax.set_xlabel("Observation count $k$")
    ax.set_ylabel("SINR RMSE (dB)")
    ax.set_title("Scenario 2 (two jammers, one directional): convergence")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = "figures/scenario2_convergence.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
