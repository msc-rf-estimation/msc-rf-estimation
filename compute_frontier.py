"""Accuracy against wall-clock compute cost.

SWaP-limited sensing makes compute a first-class constraint that RMSE on a
dense survey ignores. This experiment sweeps each estimator's compute budget
and plots reconstruction accuracy against cost.

  pure GP    batch learner, exact inference O(n^3) in training-set size; sweep n
  particle   recursive and streaming, cost roughly linear in particle count;
             accuracy plateaus at the parametric model's bias floor
  combined   pays the filter cost plus a GP fit on the residual

Scenario 1, flat terrain, averaged over several seeds. Run alone, since the
timings are wall-clock.
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
from contested_rf.simulation.uav import generate_uav_observations
import ckpt

GRID = (50, 50)
SEEDS = list(range(5))
GP_TRAIN_SIZES = [50, 100, 200, 350, 500]
PF_PARTICLES = [500, 1000, 2000, 5000, 10000]
CKPT = "results/frontier_ckpt.json"


def _world(seed):
    sc = SCENARIO_1
    shadow = ShadowField(sc.grid_xlim, sc.grid_ylim, sigma=4.0, L=50.0, seed=seed)
    pos, ts, obs, _ = generate_uav_observations(sc, seed=seed, shadow_field=shadow)
    X, Y, pts = C._grid_points(sc, GRID)
    _, _, truth = compute_sinr_map(sc, grid_shape=GRID, path_loss_exponent=2.5,
                                   noise_floor_dBm=C.NOISE_FLOOR_DBM, shadow_field=shadow)
    return sc, pos, ts, obs, pts, truth


def gp_point(sc, pos, obs, pts, truth, n_train):
    """Pure-GP RMSE and wall-time at a given training-set size."""
    idx = np.linspace(0, len(obs) - 1, min(n_train, len(obs))).astype(int)
    t0 = time.perf_counter()
    gp, gm = C.fit_pure_gp(pos[idx], obs[idx], restarts=1)
    _, _, m_g = C.pure_gp_sinr_map(gp, gm, sc, grid_shape=GRID)
    dt = time.perf_counter() - t0
    return grid_rmse(truth, m_g), dt


def pf_point(sc, pos, ts, obs, pts, truth, n_particles, pf_seed):
    """Pure-parametric and combined RMSE and wall-time at a given particle count.

    Combined adds a residual GP on a fixed 500-point thin, so its extra cost
    over the PF is constant across the particle sweep."""
    t0 = time.perf_counter()
    r = run_particle_filter(sc, sensor_positions=pos, timestamps=ts, observations=obs,
                            n_particles=n_particles, path_loss_exponent=C.N_LEARNER, seed=pf_seed)
    th = r["posterior_means"][-1]
    _, _, m_par = C.parametric_sinr_map(th, sc, grid_shape=GRID)
    t_pf = time.perf_counter() - t0

    idx = np.linspace(0, len(obs) - 1, min(500, len(obs))).astype(int)
    t1 = time.perf_counter()
    gr = C.fit_residual_gp(th, pos[idx], obs[idx], sc, restarts=1)
    _, _, m_comb = C.combined_sinr_map(th, gr, sc, grid_shape=GRID)
    t_comb = t_pf + (time.perf_counter() - t1)
    return grid_rmse(truth, m_par), t_pf, grid_rmse(truth, m_comb), t_comb


def compute_seed(seed):
    """One seed -> per-config (rmse, wall-time) for GP / parametric / combined."""
    sc, pos, ts, obs, pts, truth = _world(seed)
    out = {"gp": {}, "par": {}, "comb": {}}
    for n in GP_TRAIN_SIZES:
        rmse, dt = gp_point(sc, pos, obs, pts, truth, n)
        out["gp"][str(n)] = [rmse, dt]
    for n in PF_PARTICLES:
        r_par, t_par, r_c, t_c = pf_point(sc, pos, ts, obs, pts, truth, n, 1000 + seed)
        out["par"][str(n)] = [r_par, t_par]
        out["comb"][str(n)] = [r_c, t_c]
    return out


def main():
    t0 = time.time()
    os.makedirs("results", exist_ok=True)
    store = ckpt.load(CKPT)
    for seed in SEEDS:
        key = str(seed)
        if key not in store:
            store[key] = compute_seed(seed)
            ckpt.save(CKPT, store)
            print(f"  [frontier] seed {seed} done", flush=True)

    def curve(kind, keys):
        t = np.array([np.mean([store[str(s)][kind][str(k)][1] for s in SEEDS if str(s) in store])
                      for k in keys])
        r = np.array([np.mean([store[str(s)][kind][str(k)][0] for s in SEEDS if str(s) in store])
                      for k in keys])
        return t, r

    gp_t, gp_r = curve("gp", GP_TRAIN_SIZES)
    par_t, par_r = curve("par", PF_PARTICLES)
    comb_t, comb_r = curve("comb", PF_PARTICLES)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(gp_t, gp_r, "C1-s", lw=1.8, label="pure GP (batch, sweep n_train)")
    ax.plot(par_t, par_r, "C2-^", lw=1.8, label="pure parametric (PF, sweep particles)")
    ax.plot(comb_t, comb_r, "C0-o", lw=1.8, label="combined (PF + residual GP)")
    for t, r, n in zip(gp_t, gp_r, GP_TRAIN_SIZES):
        ax.annotate(f"n={n}", (t, r), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for t, r, n in zip(par_t, par_r, PF_PARTICLES):
        ax.annotate(f"{n//1000}k" if n >= 1000 else f"{n}", (t, r), fontsize=7,
                    xytext=(3, -8), textcoords="offset points")
    ax.set_xlabel("wall-clock cost per reconstruction (s)")
    ax.set_ylabel("SINR-map RMSE (dB)")
    ax.set_xscale("log")
    ax.set_title("Compute / SWaP frontier (S1, flat): accuracy vs wall-clock cost")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/compute_frontier.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/compute_frontier.png", flush=True)

    print("\n=== compute frontier (mean over seeds) ===")
    print("  pure GP (batch):")
    for n, t, r in zip(GP_TRAIN_SIZES, gp_t, gp_r):
        print(f"    n_train={n:4d}  {t*1000:7.1f} ms  RMSE {r:5.2f} dB", flush=True)
    print("  pure parametric (PF):")
    for n, t, r in zip(PF_PARTICLES, par_t, par_r):
        print(f"    particles={n:5d}  {t*1000:7.1f} ms  RMSE {r:5.2f} dB", flush=True)
    print("  combined (PF + residual GP):")
    for n, t, r in zip(PF_PARTICLES, comb_t, comb_r):
        print(f"    particles={n:5d}  {t*1000:7.1f} ms  RMSE {r:5.2f} dB", flush=True)
    print(f"\nDone ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
