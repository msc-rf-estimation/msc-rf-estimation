"""Full factorial experiment: 2 scenarios x 2 terrains x 4 estimators x N seeds.

For each (scenario, terrain, seed): simulate a consistent world (shadow +
optional diffraction terrain), run the appropriate SMC, reconstruct all four
estimators at full data, and score SINR RMSE (all four) plus calibration error
(the two GP-based estimators). Aggregate with paired bootstrap CIs and Cohen's
d on the key combined-vs-pure-GP comparison.
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contested_rf.estimators.particle_filter import run_particle_filter
from contested_rf.estimators.particle_filter_s2 import run_particle_filter_s2
from contested_rf.estimators import combined as C
from contested_rf.estimators import combined_s2 as C2
from contested_rf.metrics.rmse import grid_rmse
from contested_rf.metrics.calibration import coverage_curve
from contested_rf.metrics.statistics import paired_bootstrap_ci, cohens_d_paired
from contested_rf.simulation.ground_truth import compute_sinr_map
from contested_rf.simulation.scenario import SCENARIO_1, SCENARIO_2
from contested_rf.simulation.shadow_field import ShadowField
from contested_rf.simulation.uav import generate_uav_observations
from contested_rf.terrain.dem import TerrainDEM
from contested_rf.terrain.profiles import DiffractionField

GRID = (50, 50)
N_SEEDS = 20
RELIEF = 70.0
TRUE_B = np.array([1400.0, 1300.0, 27.0, 225.0, 60.0, 30.0])
ESTIMATORS = ["combined", "pure_gp", "pure_parametric", "no_jamming"]


def _diffraction(scenario, seed):
    dem = TerrainDEM.procedural(scenario.grid_xlim, scenario.grid_ylim,
                                seed=seed, relief_m=RELIEF)
    return DiffractionField(dem, freq_Hz=scenario.operating_freq_Hz)


def simulate_s2(scenario, shadow, df, rng):
    """S2 obs + truth through the shared clamped forward model, per-jammer
    diffraction applied to both jammers."""
    A, B = scenario.jammers[0], scenario.jammers[1]
    pos, ts, _, _ = generate_uav_observations(scenario, seed=int(rng.integers(1e6)),
                                              shadow_field=shadow)

    # Compute A and B separately so diffraction can be applied per emitter.
    from contested_rf.propagation.path_loss import free_space_path_loss
    from contested_rf.propagation.antenna import directional_gain
    pl_d0 = free_space_path_loss(1.0, scenario.operating_freq_Hz)

    def powers_dBm(points):
        points = np.atleast_2d(points)
        dA = np.maximum(np.hypot(points[:, 0]-A.position[0], points[:, 1]-A.position[1]), 1.0)
        PA = A.power_dBm - (pl_d0 + 10*2.5*np.log10(dA))
        x, y, ptx, tm, t3, fb = TRUE_B
        dB = np.maximum(np.hypot(points[:, 0]-x, points[:, 1]-y), 1.0)
        g = directional_gain(np.degrees(np.arctan2(points[:, 1]-y, points[:, 0]-x)), tm, t3, 0.0, fb)
        PB = ptx - (pl_d0 + 10*2.5*np.log10(dB)) + g
        if df is not None:
            PA = PA - df.loss(np.asarray(A.position, float), points)
            PB = PB - df.loss(np.array([x, y], float), points)
        return 10.0*np.log10(10**(PA/10) + 10**(PB/10))

    obs = powers_dBm(pos) + shadow.evaluate(pos) + rng.normal(0, 2.0, size=len(pos))
    X, Y, pts = C._grid_points(scenario, GRID)
    g_int = powers_dBm(pts) + shadow.evaluate(pts)
    truth = C._sinr_from_interference(g_int, C.learner_signal_dBm(scenario, pts)).reshape(X.shape)
    return pos, ts, obs, truth, A


def evaluate_combo(scenario, terrain_on, seed):
    rng = np.random.default_rng(2000 + seed)
    shadow = ShadowField(scenario.grid_xlim, scenario.grid_ylim, sigma=4.0, L=50.0, seed=seed)
    df = _diffraction(scenario, seed) if terrain_on else None
    is_s2 = scenario is SCENARIO_2

    if is_s2:
        pos, ts, obs, truth, jammer_a = simulate_s2(scenario, shadow, df, rng)
    else:
        pos, ts, obs, _ = generate_uav_observations(scenario, seed=seed,
                                                    shadow_field=shadow, diffraction=df)
        _, _, truth = compute_sinr_map(scenario, grid_shape=GRID, path_loss_exponent=2.5,
                                       noise_floor_dBm=C.NOISE_FLOOR_DBM,
                                       shadow_field=shadow, diffraction=df)

    idx = np.linspace(0, len(obs) - 1, 500).astype(int)
    Xp, Zp = pos[idx], obs[idx]
    X, Y, pts = C._grid_points(scenario, GRID)
    P_signal = C.learner_signal_dBm(scenario, pts)

    pf_seed = 1000 + seed  # derive the filter's RNG from the replication seed so
                           # resampling/jitter randomness varies across seeds too
    if is_s2:
        r = run_particle_filter_s2(scenario, sensor_positions=pos, timestamps=ts,
                                   observations=obs, n_particles=6000,
                                   path_loss_exponent=C.N_LEARNER, seed=pf_seed)
        th = r["posterior_means"][-1]
        gr = C2.fit_residual_gp_s2(th, jammer_a, Xp, Zp, scenario, restarts=3)
        _, _, m_comb = C2.combined_sinr_map_s2(th, gr, jammer_a, scenario, grid_shape=GRID)
        _, _, m_par = C2.parametric_sinr_map_s2(th, jammer_a, scenario, grid_shape=GRID)
    else:
        r = run_particle_filter(scenario, sensor_positions=pos, timestamps=ts,
                                observations=obs, n_particles=5000,
                                path_loss_exponent=C.N_LEARNER, seed=pf_seed)
        th = r["posterior_means"][-1]
        gr = C.fit_residual_gp(th, Xp, Zp, scenario, restarts=3)
        _, _, m_comb = C.combined_sinr_map(th, gr, scenario, grid_shape=GRID)
        _, _, m_par = C.parametric_sinr_map(th, scenario, grid_shape=GRID)

    gp, gm = C.fit_pure_gp(Xp, Zp, restarts=3)
    _, _, m_gp = C.pure_gp_sinr_map(gp, gm, scenario, grid_shape=GRID)
    _, _, m_nj = C.no_jamming_sinr_map(scenario, grid_shape=GRID)

    rmse = {"combined": grid_rmse(truth, m_comb), "pure_gp": grid_rmse(truth, m_gp),
            "pure_parametric": grid_rmse(truth, m_par), "no_jamming": grid_rmse(truth, m_nj)}

    # --- Calibration (combined & pure_gp) via Monte-Carlo SINR samples. ---
    M = 200

    def sinr_from_P(P_samps):
        s = 10 ** (P_signal[:, None] / 10.0)
        nf = 10 ** (C.NOISE_FLOOR_DBM / 10.0)
        return 10.0 * np.log10(s / (10 ** (P_samps / 10.0) + nf))

    # Pure GP: the predictive variance is the entire uncertainty budget.
    mu_g, var_g = gp.predict(pts, return_var=True)
    Pg = (mu_g + gm)[:, None] + np.sqrt(var_g)[:, None] * rng.normal(size=(len(pts), M))
    cal_g = coverage_curve(sinr_from_P(Pg), truth.ravel())[1]

    # Combined: propagate BOTH the SMC parameter posterior (theta drawn from the
    # weighted particle cloud) AND the GP residual posterior. The old code used
    # the GP residual variance alone, ignoring the filter's own uncertainty and
    # so understating the combined estimator's predictive spread.
    w = np.exp(r["log_weights_final"])
    w = w / w.sum()
    theta_draws = r["particles_final"][rng.choice(len(w), size=M, p=w)]
    mu_r, var_r = gr.predict(pts, return_var=True)
    resid = mu_r[:, None] + np.sqrt(var_r)[:, None] * rng.normal(size=(len(pts), M))
    P_param = np.empty((len(pts), M))
    for m in range(M):
        if is_s2:
            P_param[:, m] = C2.s2_interference_dBm(pts, theta_draws[m], jammer_a, scenario)
        else:
            P_param[:, m] = C.parametric_power_dBm(theta_draws[m], pts, scenario)
    cal_c = coverage_curve(sinr_from_P(P_param + resid), truth.ravel())[1]
    return rmse, {"combined": cal_c, "pure_gp": cal_g}


def main():
    t0 = time.time()
    conditions = [("S1", SCENARIO_1, "flat", False), ("S1", SCENARIO_1, "real", True),
                  ("S2", SCENARIO_2, "flat", False), ("S2", SCENARIO_2, "real", True)]
    results = {}
    for sname, scenario, tname, ton in conditions:
        rmses = {e: [] for e in ESTIMATORS}
        cals = {"combined": [], "pure_gp": []}
        for s in range(N_SEEDS):
            rmse, cal = evaluate_combo(scenario, ton, s)
            for e in ESTIMATORS:
                rmses[e].append(rmse[e])
            cals["combined"].append(cal["combined"]); cals["pure_gp"].append(cal["pure_gp"])
        results[(sname, tname)] = (rmses, cals)
        diffs = np.array(rmses["combined"]) - np.array(rmses["pure_gp"])
        mean, lo, hi = paired_bootstrap_ci(diffs)
        d = cohens_d_paired(diffs)
        print(f"\n=== {sname} / {tname} (n={N_SEEDS}) ===")
        for e in ESTIMATORS:
            print(f"  {e:16s} RMSE {np.mean(rmses[e]):6.2f} ± {np.std(rmses[e]):.2f} dB")
        print(f"  calib: combined {np.mean(cals['combined']):.3f}  pure_gp {np.mean(cals['pure_gp']):.3f}")
        print(f"  combined-pure_gp RMSE diff {mean:+.2f} dB, 95% CI [{lo:+.2f},{hi:+.2f}], Cohen d {d:+.2f}")

    # Summary figure: grouped RMSE bars per condition.
    fig, ax = plt.subplots(figsize=(12, 6))
    conds = list(results.keys())
    xpos = np.arange(len(conds)); w = 0.2
    colors = {"combined": "C0", "pure_gp": "C1", "pure_parametric": "C2", "no_jamming": "0.6"}
    for i, e in enumerate(ESTIMATORS):
        means = [np.mean(results[c][0][e]) for c in conds]
        stds = [np.std(results[c][0][e]) for c in conds]
        ax.bar(xpos + (i - 1.5) * w, means, w, yerr=stds, capsize=3, label=e, color=colors[e])
    ax.set_xticks(xpos); ax.set_xticklabels([f"{s}\n{t}" for s, t in conds])
    ax.set_ylabel("SINR RMSE (dB)")
    ax.set_title(f"Factorial results: SINR RMSE by condition (n={N_SEEDS} seeds)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/factorial_rmse.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figures/factorial_rmse.png  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
