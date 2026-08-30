"""Multi-seed robustness of the Scenario 1 result.

Replicates the head-to-head across N independent seeds, each a different
shadow realisation, survey trajectory and measurement noise draw, so the null
result carries error bars rather than resting on a single run. Reports final
SINR RMSE and calibration error per estimator, and the paired
(combined - pure_gp) difference.
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
from contested_rf.metrics.calibration import coverage_curve
from contested_rf.simulation.ground_truth import compute_sinr_map
from contested_rf.simulation.scenario import SCENARIO_1
from contested_rf.simulation.shadow_field import ShadowField
from contested_rf.terrain.synthetic import SyntheticTerrain
from contested_rf.simulation.uav import generate_uav_observations

GRID = (50, 50)
SEEDS = list(range(10))
sc = SCENARIO_1


def _sinr_samples(P_mean, P_std, P_signal, S, rng):
    z = rng.normal(size=(P_mean.shape[0], S))
    P_samp = P_mean[:, None] + P_std[:, None] * z
    s_mw = 10 ** (P_signal[:, None] / 10.0)
    i_mw = 10 ** (P_samp / 10.0)
    n_mw = 10 ** (C.NOISE_FLOOR_DBM / 10.0)
    return 10.0 * np.log10(s_mw / (i_mw + n_mw))


def one_seed(seed, terrain):
    rng = np.random.default_rng(1000 + seed)
    shadow = ShadowField(sc.grid_xlim, sc.grid_ylim, sigma=4.0, L=50.0, seed=seed)
    pos, ts, obs, _ = generate_uav_observations(
        sc, seed=seed, shadow_field=shadow, terrain=terrain)
    _, _, sinr_true = compute_sinr_map(
        sc, grid_shape=GRID, path_loss_exponent=2.5,
        noise_floor_dBm=C.NOISE_FLOOR_DBM, shadow_field=shadow, terrain=terrain)
    truth = sinr_true.ravel()

    r = run_particle_filter(sc, sensor_positions=pos, timestamps=ts,
                            observations=obs, n_particles=5000,
                            path_loss_exponent=C.N_LEARNER, seed=1)
    th = r["posterior_means"][-1]

    idx = np.linspace(0, len(obs) - 1, 500).astype(int)
    Xp, Zp = pos[idx], obs[idx]
    X, Y, pts = C._grid_points(sc, GRID)
    P_signal = C.learner_signal_dBm(sc, pts)

    # RMSE
    gr = C.fit_residual_gp(th, Xp, Zp, sc)
    _, _, sc_c = C.combined_sinr_map(th, gr, sc, grid_shape=GRID)
    _, _, sc_p = C.parametric_sinr_map(th, sc, grid_shape=GRID)
    gp, gm = C.fit_pure_gp(Xp, Zp)
    _, _, sc_g = C.pure_gp_sinr_map(gp, gm, sc, grid_shape=GRID)
    rmse = {"combined": grid_rmse(sinr_true, sc_c),
            "pure_gp": grid_rmse(sinr_true, sc_g),
            "pure_parametric": grid_rmse(sinr_true, sc_p)}

    # Calibration
    mu_r, var_r = gr.predict(pts, return_var=True)
    samp_c = _sinr_samples(C.parametric_power_dBm(th, pts, sc) + mu_r,
                           np.sqrt(var_r), P_signal, 200, rng)
    mu_g, var_g = gp.predict(pts, return_var=True)
    samp_g = _sinr_samples(mu_g + gm, np.sqrt(var_g), P_signal, 200, rng)
    particles = r["particles_final"]
    w = np.exp(r["log_weights_final"]); w /= w.sum()
    sidx = rng.choice(len(particles), 200, p=w)
    P_par = np.stack([C.parametric_power_dBm(particles[s], pts, sc) for s in sidx], axis=1)
    s_mw = 10 ** (P_signal[:, None] / 10.0)
    samp_p = 10.0 * np.log10(s_mw / (10 ** (P_par / 10.0) + 10 ** (C.NOISE_FLOOR_DBM / 10.0)))
    cal = {n: coverage_curve(s, truth)[1]
           for n, s in [("combined", samp_c), ("pure_gp", samp_g),
                        ("pure_parametric", samp_p)]}
    return rmse, cal


def summarise(label, terrain):
    rmses = {k: [] for k in ("combined", "pure_gp", "pure_parametric")}
    cals = {k: [] for k in ("combined", "pure_gp", "pure_parametric")}
    for s in SEEDS:
        rmse, cal = one_seed(s, terrain)
        for k in rmses:
            rmses[k].append(rmse[k]); cals[k].append(cal[k])
    print(f"\n=== {label} (n={len(SEEDS)} seeds) ===")
    print(f"{'estimator':16s} {'RMSE mean±std':>16s} {'calib mean±std':>16s}")
    for k in rmses:
        rm, rs = np.mean(rmses[k]), np.std(rmses[k])
        cm, cs = np.mean(cals[k]), np.std(cals[k])
        print(f"{k:16s} {rm:6.2f} ± {rs:4.2f} dB   {cm:6.3f} ± {cs:5.3f}")
    diff = np.array(rmses["combined"]) - np.array(rmses["pure_gp"])
    print(f"paired (combined - pure_gp) RMSE: {diff.mean():+.2f} ± {diff.std():.2f} dB "
          f"[min {diff.min():+.2f}, max {diff.max():+.2f}]")
    return rmses, cals


def main():
    t0 = time.time()
    flat = summarise("FLAT", None)
    terr = summarise("TERRAIN", SyntheticTerrain())

    # Figure: RMSE (mean±std) and calibration error (mean±std), flat vs terrain.
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))
    names = ["combined", "pure_gp", "pure_parametric"]
    labels = ["Combined", "Pure GP", "Pure param."]
    xpos = np.arange(len(names))
    for ax, metric_idx, title, ylab in [
            (axA, 0, "Final SINR RMSE (lower better)", "RMSE (dB)"),
            (axB, 1, "Calibration error (lower better)", "|coverage - nominal|")]:
        for off, (cond, data, col) in enumerate(
                [("flat", flat, "C0"), ("terrain", terr, "C3")]):
            src = data[metric_idx]
            means = [np.mean(src[n]) for n in names]
            stds = [np.std(src[n]) for n in names]
            ax.bar(xpos + (off - 0.5) * 0.4, means, width=0.38, yerr=stds,
                   capsize=4, label=cond, color=col, alpha=0.8)
        ax.set_xticks(xpos); ax.set_xticklabels(labels)
        ax.set_title(title); ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.3); ax.legend()
    fig.suptitle(f"Scenario 1 robustness across {len(SEEDS)} seeds", fontsize=13)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = "figures/scenario1_robustness.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
