"""Partial-coverage extrapolation experiment (resumable).

Restricts the survey to a contiguous band containing the target jammer,
shrinks the band width (coverage fraction f), and scores SINR-map RMSE over
the full grid, the observed band and the unobserved flanks separately.

Two terrain arms: flat (free space plus shadow) as the primary, and procedural
knife-edge diffraction as a robustness check. Scenarios 1 and 2, both arms.
Per-seed checkpointing to results/coverage_ckpt.json.
"""
import math
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
from contested_rf.metrics.statistics import paired_bootstrap_ci, cohens_d_paired
from contested_rf.simulation.ground_truth import compute_sinr_map
from contested_rf.simulation.scenario import SCENARIO_1, SCENARIO_2
from contested_rf.simulation.shadow_field import ShadowField
from contested_rf.simulation.uav import generate_uav_observations
from contested_rf.propagation.path_loss import free_space_path_loss
from contested_rf.propagation.antenna import directional_gain
from contested_rf.terrain.dem import TerrainDEM
from contested_rf.terrain.profiles import DiffractionField
import ckpt

W = 2000.0
GRID = (50, 50)
COVERAGE = [1.0, 0.66, 0.5, 0.33, 0.25]
SEEDS = list(range(20))
RELIEF = 70.0
TRUE_B = np.array([1400.0, 1300.0, 27.0, 225.0, 60.0, 30.0])
TERRAIN_ARMS = [("flat", False), ("terrain", True)]
# Flat is the primary result (20 seeds). The terrain arm is a robustness check;
# effect sizes are large (|d| 2-10), so 12 seeds gives ample power there.
ARM_SEEDS = {"flat": list(range(20)), "terrain": list(range(12))}
ESTS = ("combined", "pure_gp", "pure_parametric")
REGIONS = ("full", "observed", "unobserved")
CKPT = "results/coverage_ckpt.json"


def _diffraction(scenario, seed):
    dem = TerrainDEM.procedural(scenario.grid_xlim, scenario.grid_ylim,
                                seed=seed, relief_m=RELIEF)
    return DiffractionField(dem, freq_Hz=scenario.operating_freq_Hz)


def band(center_x, f):
    half = f * W / 2.0
    lo = min(max(center_x - half, 0.0), W - f * W)
    return lo, lo + f * W


def subset_rmse(truth, est, mask):
    d = (truth.ravel() - est.ravel())[mask.ravel()]
    return float(np.sqrt(np.mean(d ** 2))) if mask.any() else float("nan")


def simulate_s2(seed, rng, df=None):
    sc = SCENARIO_2
    A = sc.jammers[0]
    shadow = ShadowField(sc.grid_xlim, sc.grid_ylim, sigma=4.0, L=50.0, seed=seed)
    pos, ts, _, _ = generate_uav_observations(sc, seed=seed, shadow_field=shadow)
    pl_d0 = free_space_path_loss(1.0, sc.operating_freq_Hz)

    def powers(points):
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
        return 10*np.log10(10**(PA/10) + 10**(PB/10))

    obs = powers(pos) + shadow.evaluate(pos) + rng.normal(0, 2.0, size=len(pos))
    X, Y, pts = C._grid_points(sc, GRID)
    truth = C._sinr_from_interference(powers(pts) + shadow.evaluate(pts),
                                      C.learner_signal_dBm(sc, pts)).reshape(X.shape)
    return sc, A, pos, ts, obs, truth, X, Y, pts


def reconstruct(scenario, is_s2, pos_m, obs_m, ts_m, pts, jammer_a=None, pf_seed=1):
    idx = np.linspace(0, len(obs_m)-1, min(500, len(obs_m))).astype(int)
    Xp, Zp = pos_m[idx], obs_m[idx]
    if is_s2:
        r = run_particle_filter_s2(scenario, sensor_positions=pos_m, timestamps=ts_m,
                                   observations=obs_m, n_particles=8000,
                                   path_loss_exponent=C.N_LEARNER, seed=pf_seed)
        th = r["posterior_means"][-1]
        gr = C2.fit_residual_gp_s2(th, jammer_a, Xp, Zp, scenario, restarts=3)
        _, _, m_c = C2.combined_sinr_map_s2(th, gr, jammer_a, scenario, grid_shape=GRID)
        _, _, m_p = C2.parametric_sinr_map_s2(th, jammer_a, scenario, grid_shape=GRID)
    else:
        r = run_particle_filter(scenario, sensor_positions=pos_m, timestamps=ts_m,
                                observations=obs_m, n_particles=5000,
                                path_loss_exponent=C.N_LEARNER, seed=pf_seed)
        th = r["posterior_means"][-1]
        gr = C.fit_residual_gp(th, Xp, Zp, scenario, restarts=3)
        _, _, m_c = C.combined_sinr_map(th, gr, scenario, grid_shape=GRID)
        _, _, m_p = C.parametric_sinr_map(th, scenario, grid_shape=GRID)
    gp, gm = C.fit_pure_gp(Xp, Zp, restarts=3)
    _, _, m_g = C.pure_gp_sinr_map(gp, gm, scenario, grid_shape=GRID)
    return m_c, m_g, m_p


def compute_seed(name, terrain_on, seed):
    """One seed -> {str(f): {region: {est: rmse_or_None}}}."""
    is_s2 = name == "S2"
    center_x = (SCENARIO_2.jammers[1].position[0] if is_s2
                else SCENARIO_1.jammers[0].position[0])
    rng = np.random.default_rng(3000 + seed)
    df = _diffraction(SCENARIO_2 if is_s2 else SCENARIO_1, seed) if terrain_on else None
    if is_s2:
        sc, A, pos, ts, obs, truth, X, Y, pts = simulate_s2(seed, rng, df=df)
    else:
        sc, A = SCENARIO_1, None
        shadow = ShadowField(sc.grid_xlim, sc.grid_ylim, sigma=4.0, L=50.0, seed=seed)
        pos, ts, obs, _ = generate_uav_observations(sc, seed=seed,
                                                    shadow_field=shadow, diffraction=df)
        X, Y, pts = C._grid_points(sc, GRID)
        _, _, truth = compute_sinr_map(sc, grid_shape=GRID, path_loss_exponent=2.5,
                                       noise_floor_dBm=C.NOISE_FLOOR_DBM,
                                       shadow_field=shadow, diffraction=df)
    gx = pts[:, 0].reshape(X.shape)
    out = {}
    for f in COVERAGE:
        lo, hi = band(center_x, f)
        obs_mask = (pos[:, 0] >= lo) & (pos[:, 0] <= hi)
        rec = {reg: {} for reg in REGIONS}
        if obs_mask.sum() < 30:
            out[str(f)] = rec
            continue
        m_c, m_g, m_p = reconstruct(sc, is_s2, pos[obs_mask], obs[obs_mask],
                                    ts[obs_mask], pts, jammer_a=A, pf_seed=1000 + seed)
        g_obs = (gx >= lo) & (gx <= hi)
        g_unobs = ~g_obs
        for est, m in [("combined", m_c), ("pure_gp", m_g), ("pure_parametric", m_p)]:
            rec["full"][est] = subset_rmse(truth, m, np.ones_like(g_obs, bool))
            rec["observed"][est] = subset_rmse(truth, m, g_obs)
            rec["unobserved"][est] = (subset_rmse(truth, m, g_unobs) if g_unobs.any() else None)
        out[str(f)] = rec
    return out


def aggregate(store, arm):
    res = {}
    for name in ("S1", "S2"):
        res[name] = {f: {reg: {e: [] for e in ESTS} for reg in REGIONS} for f in COVERAGE}
        for seed in ARM_SEEDS[arm]:
            sd = store.get(f"{arm}|{name}|{seed}")
            if sd is None:
                continue
            for f in COVERAGE:
                rec = sd.get(str(f), {})
                for reg in REGIONS:
                    for e in ESTS:
                        v = rec.get(reg, {}).get(e)
                        if v is not None and not (isinstance(v, float) and math.isnan(v)):
                            res[name][f][reg][e].append(v)
    return res


def _plot_arm(res, arm, t0):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, name in enumerate(("S1", "S2")):
        for row, region in enumerate(("full", "unobserved")):
            ax = axes[row, col]
            fs = [f for f in COVERAGE if res[name][f][region]["combined"]]
            for est, cstyle in [("combined", "C0-o"), ("pure_gp", "C1-s"),
                                ("pure_parametric", "C2-^")]:
                means = [np.mean(res[name][f][region][est]) for f in fs]
                stds = [np.std(res[name][f][region][est]) for f in fs]
                ax.errorbar(fs, means, yerr=stds, fmt=cstyle, capsize=3, label=est, linewidth=1.7)
            ax.set_title(f"{name} — {region} region RMSE ({arm})")
            ax.set_xlabel("coverage fraction f"); ax.set_ylabel("SINR RMSE (dB)")
            ax.invert_xaxis(); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Partial coverage ({arm}): does the decomposition win when the survey leaves gaps?",
                 fontsize=13)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = f"figures/coverage_sweep_{arm}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  ({time.time()-t0:.1f}s)", flush=True)


def _report_arm(res, arm):
    print(f"\n=== paired (combined - pure_gp) RMSE by coverage, UNOBSERVED region [{arm}] ===")
    for name in ("S1", "S2"):
        for f in COVERAGE:
            c = res[name][f]["unobserved"]["combined"]
            g = res[name][f]["unobserved"]["pure_gp"]
            if not c or len(c) != len(g) or not c:
                continue
            d = np.array(c) - np.array(g)
            m, lo, hi = paired_bootstrap_ci(d)
            print(f"  {name} f={f:.2f}: {m:+.2f} dB [{lo:+.2f},{hi:+.2f}] d={cohens_d_paired(d):+.2f}"
                  f"  (combined {'better' if m < 0 else 'worse'}, n={len(c)})", flush=True)


def main():
    t0 = time.time()
    os.makedirs("results", exist_ok=True)
    store = ckpt.load(CKPT)
    for arm, terrain_on in TERRAIN_ARMS:
        for name in ("S1", "S2"):
            for seed in ARM_SEEDS[arm]:
                key = f"{arm}|{name}|{seed}"
                if key in store:
                    continue
                store[key] = compute_seed(name, terrain_on, seed)
                ckpt.save(CKPT, store)
                print(f"  [{arm}/{name}] seed {seed} done", flush=True)
        res = aggregate(store, arm)
        _plot_arm(res, arm, t0)
        _report_arm(res, arm)
    print(f"\nDone ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
