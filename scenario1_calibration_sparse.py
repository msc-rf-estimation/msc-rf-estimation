"""Sample efficiency and calibration on Scenario 1.

Panel A, sample efficiency: with observations spread across the whole area,
does the parametric layer let the combined estimator reconstruct the field
from fewer observations than a pure GP?

Panel B, calibration: do the estimators' credible intervals contain the truth
at the stated rate? Terrain enabled, as the harder case.
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
SEED = 42
sc = SCENARIO_1


def _sinr_samples(P_mean, P_std, P_signal, S, rng):
    """(M,S) SINR samples from a Gaussian on interference power (dBm)."""
    z = rng.normal(size=(P_mean.shape[0], S))
    P_samp = P_mean[:, None] + P_std[:, None] * z
    s_mw = 10 ** (P_signal[:, None] / 10.0)
    i_mw = 10 ** (P_samp / 10.0)
    n_mw = 10 ** (C.NOISE_FLOOR_DBM / 10.0)
    return 10.0 * np.log10(s_mw / (i_mw + n_mw))


def sample_efficiency(terrain=None):
    shadow = ShadowField(sc.grid_xlim, sc.grid_ylim, sigma=4.0, L=50.0, seed=7)
    pos, ts, obs, _ = generate_uav_observations(
        sc, seed=SEED, shadow_field=shadow, terrain=terrain)
    n = len(obs)
    _, _, sinr_true = compute_sinr_map(
        sc, grid_shape=GRID, path_loss_exponent=2.5,
        noise_floor_dBm=C.NOISE_FLOOR_DBM, shadow_field=shadow, terrain=terrain)

    Ns = [15, 30, 60, 120, 250, 500, 1000]
    res = {"combined": [], "pure_gp": [], "pure_parametric": [], "Ns": Ns}
    for N in Ns:
        idx = np.linspace(0, n - 1, N).astype(int)  # spread across the survey
        Xp, Zp = pos[idx], obs[idx]
        r = run_particle_filter(sc, sensor_positions=Xp, timestamps=ts[idx],
                                observations=Zp, n_particles=4000,
                                path_loss_exponent=C.N_LEARNER, seed=1)
        th = r["posterior_means"][-1]
        gr = C.fit_residual_gp(th, Xp, Zp, sc)
        _, _, sc_c = C.combined_sinr_map(th, gr, sc, grid_shape=GRID)
        _, _, sc_p = C.parametric_sinr_map(th, sc, grid_shape=GRID)
        gp, gm = C.fit_pure_gp(Xp, Zp)
        _, _, sc_g = C.pure_gp_sinr_map(gp, gm, sc, grid_shape=GRID)
        res["combined"].append(grid_rmse(sinr_true, sc_c))
        res["pure_gp"].append(grid_rmse(sinr_true, sc_g))
        res["pure_parametric"].append(grid_rmse(sinr_true, sc_p))
        print(f"  N={N:5d}: combined={res['combined'][-1]:5.2f}  "
              f"pureGP={res['pure_gp'][-1]:5.2f}  param={res['pure_parametric'][-1]:5.2f} dB")
    return res


def calibration(terrain):
    rng = np.random.default_rng(0)
    shadow = ShadowField(sc.grid_xlim, sc.grid_ylim, sigma=4.0, L=50.0, seed=7)
    pos, ts, obs, _ = generate_uav_observations(
        sc, seed=SEED, shadow_field=shadow, terrain=terrain)
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

    # Combined: mean = f_param + mu_GP, std = sqrt(var_GP)
    gr = C.fit_residual_gp(th, Xp, Zp, sc)
    mu_r, var_r = gr.predict(pts, return_var=True)
    P_mean_c = C.parametric_power_dBm(th, pts, sc) + mu_r
    samp_c = _sinr_samples(P_mean_c, np.sqrt(var_r), P_signal, 300, rng)

    # Pure GP
    gp, gm = C.fit_pure_gp(Xp, Zp)
    mu_g, var_g = gp.predict(pts, return_var=True)
    samp_g = _sinr_samples(mu_g + gm, np.sqrt(var_g), P_signal, 300, rng)

    # Pure parametric: uncertainty from SMC particle spread (200 draws)
    particles = r["particles_final"]
    w = np.exp(r["log_weights_final"]); w /= w.sum()
    sidx = rng.choice(len(particles), 200, p=w)
    P_param_samples = np.stack(
        [C.parametric_power_dBm(particles[s], pts, sc) for s in sidx], axis=1)
    s_mw = 10 ** (P_signal[:, None] / 10.0)
    i_mw = 10 ** (P_param_samples / 10.0)
    n_mw = 10 ** (C.NOISE_FLOOR_DBM / 10.0)
    samp_p = 10.0 * np.log10(s_mw / (i_mw + n_mw))

    out = {}
    for name, samp in [("combined", samp_c), ("pure_gp", samp_g),
                       ("pure_parametric", samp_p)]:
        cov, cal_err = coverage_curve(samp, truth)
        out[name] = (cov, cal_err)
        print(f"  {name:16s} calibration error = {cal_err:.3f}")
    return out


def main():
    t0 = time.time()
    print("== Sample efficiency (flat) ==")
    eff = sample_efficiency(terrain=None)
    print("== Calibration (terrain) ==")
    cal = calibration(SyntheticTerrain())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    styles = {"combined": ("C0", "o", "Combined (SMC+GP)"),
              "pure_gp": ("C1", "s", "Pure GP"),
              "pure_parametric": ("C2", "^", "Pure parametric")}
    for name, (c, m, lab) in styles.items():
        axA.plot(eff["Ns"], eff[name], "-", color=c, marker=m, markersize=6,
                 linewidth=1.8, label=lab)
    axA.axhline(4.0, color="red", linestyle=":", alpha=0.6, label="4 dB")
    axA.set_xscale("log"); axA.set_xlabel("Number of (spread) observations")
    axA.set_ylabel("SINR RMSE (dB)")
    axA.set_title("A. Sample efficiency (sparse, spatially spread)")
    axA.grid(True, alpha=0.3); axA.legend(fontsize=9)

    alphas = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    axB.plot([0.5, 0.95], [0.5, 0.95], "k--", alpha=0.5, label="Perfect")
    for name, (c, m, lab) in styles.items():
        cov, cal_err = cal[name]
        axB.plot(list(alphas), [cov[a] for a in alphas], "-", color=c, marker=m,
                 markersize=6, linewidth=1.8, label=f"{lab} (err {cal_err:.2f})")
    axB.set_xlabel("Nominal coverage $\\alpha$")
    axB.set_ylabel("Observed coverage")
    axB.set_title("B. Calibration (terrain, full survey)")
    axB.grid(True, alpha=0.3); axB.legend(fontsize=9)

    fig.suptitle("Scenario 1: sample efficiency and calibration", fontsize=13)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = "figures/scenario1_calibration_sparse.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
