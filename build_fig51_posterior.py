"""Scenario 2 posterior over jammer-B main-beam bearing (Figure 5.1).

Replicates the Scenario 2 flat-terrain dense-coverage condition of
run_factorial.py and saves the final weighted particle cloud, so the posterior
over theta_main can be plotted from data. Also records the circular variance
and antipodal hedge score defined in particle_filter_s2, which separate a
genuinely bimodal posterior from a filter that committed to one mode.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

from contested_rf.estimators.particle_filter_s2 import (
    run_particle_filter_s2, circular_variance, IX_THETA_MAIN)
from contested_rf.simulation.scenario import SCENARIO_2
from contested_rf.simulation.shadow_field import ShadowField

import run_factorial as RF

TRUE_THETA = 225.0
SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [6000, 24000]
OUT = "results/fig51_posterior.json"


def hedge_score(theta_deg, w):
    """min/max weight in two opposing +/-30 deg wedges about the resultant."""
    ang = np.radians(theta_deg)
    R = np.sum(w * np.exp(1j * ang))
    phi = np.degrees(np.angle(R)) % 360.0
    d_front = np.abs((theta_deg - phi + 180) % 360 - 180)
    d_back = np.abs((theta_deg - (phi + 180) % 360 + 180) % 360 - 180)
    wf, wb = w[d_front <= 30].sum(), w[d_back <= 30].sum()
    return float(min(wf, wb) / max(wf + wb, 1e-12)), float(phi)


def main():
    out = []
    for n_particles in BUDGETS:
        for seed in SEEDS:
            rng = np.random.default_rng(2000 + seed)
            shadow = ShadowField(SCENARIO_2.grid_xlim, SCENARIO_2.grid_ylim,
                                 sigma=4.0, L=50.0, seed=seed)
            pos, ts, obs, _truth, _A = RF.simulate_s2(SCENARIO_2, shadow, None, rng)
            r = run_particle_filter_s2(
                SCENARIO_2, sensor_positions=pos, timestamps=ts, observations=obs,
                n_particles=n_particles, path_loss_exponent=2.0, seed=1000 + seed)
            th = r["particles_final"][:, IX_THETA_MAIN]
            w = np.exp(r["log_weights_final"])
            w = w / w.sum()
            H, phi = hedge_score(th, w)
            rec = {
                "n_particles": n_particles, "seed": seed,
                "posterior_mean_theta": float(r["posterior_means"][-1][IX_THETA_MAIN]),
                "resultant_angle": phi,
                "circular_variance": float(circular_variance(th, w)),
                "hedge_score": H,
                "ess_final": float(r["ess_history"][-1]),
                "theta_hist": np.histogram(th, bins=72, range=(0, 360),
                                           weights=w)[0].tolist(),
            }
            out.append(rec)
            print(f"N={n_particles} seed={seed} mean={rec['posterior_mean_theta']:7.1f} "
                  f"Vcirc={rec['circular_variance']:.3f} H={H:.3f} "
                  f"ESS={rec['ess_final']:.0f}", flush=True)
    with open(OUT, "w") as f:
        json.dump({"true_theta": TRUE_THETA, "bin_edges_deg": list(range(0, 365, 5)),
                   "runs": out}, f, indent=1)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
