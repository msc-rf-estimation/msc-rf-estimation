"""Standoff geometry: emitter outside the surveyed band (resumable).

The survey is confined to a contiguous band on one side of the map and the
jammer sits beyond its far edge, so reconstruction near the emitter is pure
extrapolation. Sweeps the survey far edge B, giving a standoff distance of
(jammer_x - B). Scenarios 1 and 2, flat terrain, scored in the unobserved
flank (x > B) and in a near-jammer window. Per-seed checkpointing to
results/standoff_ckpt.json.
"""
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contested_rf.estimators import combined as C
from contested_rf.metrics.statistics import paired_bootstrap_ci, cohens_d_paired
from contested_rf.simulation.ground_truth import compute_sinr_map
from contested_rf.simulation.scenario import SCENARIO_1, SCENARIO_2
from contested_rf.simulation.shadow_field import ShadowField
from contested_rf.simulation.uav import generate_uav_observations

import run_coverage as RC  # reuse reconstruct / simulate_s2 / subset_rmse
import ckpt

GRID = RC.GRID
# Stretch variant; effect sizes are large, so 12 seeds gives ample power.
SEEDS = list(range(12))
SURVEY_EDGE = [1000.0, 800.0, 600.0]
NEAR_WIN = 300.0
ESTS = ("combined", "pure_gp", "pure_parametric")
REGIONS = ("unobserved", "near")
CKPT = "results/standoff_ckpt.json"


def _jammer_xy(is_s2):
    j = (SCENARIO_2.jammers[1] if is_s2 else SCENARIO_1.jammers[0])
    return j.position[0], j.position[1]


def compute_seed(name, seed):
    """One seed -> {str(B): {region: {est: rmse}}}."""
    is_s2 = name == "S2"
    jx, jy = _jammer_xy(is_s2)
    rng = np.random.default_rng(4000 + seed)
    if is_s2:
        sc, A, pos, ts, obs, truth, X, Y, pts = RC.simulate_s2(seed, rng, df=None)
    else:
        sc, A = SCENARIO_1, None
        shadow = ShadowField(sc.grid_xlim, sc.grid_ylim, sigma=4.0, L=50.0, seed=seed)
        pos, ts, obs, _ = generate_uav_observations(sc, seed=seed, shadow_field=shadow)
        X, Y, pts = C._grid_points(sc, GRID)
        _, _, truth = compute_sinr_map(sc, grid_shape=GRID, path_loss_exponent=2.5,
                                       noise_floor_dBm=C.NOISE_FLOOR_DBM, shadow_field=shadow)
    gx = pts[:, 0].reshape(X.shape)
    gy = pts[:, 1].reshape(X.shape)
    near = (np.abs(gx - jx) <= NEAR_WIN) & (np.abs(gy - jy) <= NEAR_WIN)

    out = {}
    for B in SURVEY_EDGE:
        rec = {reg: {} for reg in REGIONS}
        obs_mask = pos[:, 0] <= B
        if obs_mask.sum() < 30:
            out[str(B)] = rec
            continue
        m_c, m_g, m_p = RC.reconstruct(sc, is_s2, pos[obs_mask], obs[obs_mask],
                                       ts[obs_mask], pts, jammer_a=A, pf_seed=1000 + seed)
        g_unobs = gx > B
        for est, m in [("combined", m_c), ("pure_gp", m_g), ("pure_parametric", m_p)]:
            rec["unobserved"][est] = RC.subset_rmse(truth, m, g_unobs)
            rec["near"][est] = RC.subset_rmse(truth, m, near)
        out[str(B)] = rec
    return out


def aggregate(store):
    res = {}
    for name in ("S1", "S2"):
        res[name] = {B: {reg: {e: [] for e in ESTS} for reg in REGIONS} for B in SURVEY_EDGE}
        for seed in SEEDS:
            sd = store.get(f"{name}|{seed}")
            if sd is None:
                continue
            for B in SURVEY_EDGE:
                rec = sd.get(str(B), {})
                for reg in REGIONS:
                    for e in ESTS:
                        v = rec.get(reg, {}).get(e)
                        if v is not None and not (isinstance(v, float) and math.isnan(v)):
                            res[name][B][reg][e].append(v)
    return res


def main():
    t0 = time.time()
    os.makedirs("results", exist_ok=True)
    store = ckpt.load(CKPT)
    for name in ("S1", "S2"):
        for seed in SEEDS:
            key = f"{name}|{seed}"
            if key in store:
                continue
            store[key] = compute_seed(name, seed)
            ckpt.save(CKPT, store)
            print(f"  [standoff {name}] seed {seed} done", flush=True)
    res = aggregate(store)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, name in enumerate(("S1", "S2")):
        jx, _ = _jammer_xy(name == "S2")
        standoffs = [jx - B for B in SURVEY_EDGE]
        for row, region in enumerate(("unobserved", "near")):
            ax = axes[row, col]
            for est, cstyle in [("combined", "C0-o"), ("pure_gp", "C1-s"),
                                ("pure_parametric", "C2-^")]:
                means = [np.mean(res[name][B][region][est]) if res[name][B][region][est] else np.nan
                         for B in SURVEY_EDGE]
                stds = [np.std(res[name][B][region][est]) if res[name][B][region][est] else 0.0
                        for B in SURVEY_EDGE]
                ax.errorbar(standoffs, means, yerr=stds, fmt=cstyle, capsize=3, label=est, linewidth=1.7)
            ax.set_title(f"{name} — {region} region RMSE (jammer outside band)")
            ax.set_xlabel("standoff distance jammer→survey edge (m)")
            ax.set_ylabel("SINR RMSE (dB)")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Standoff coverage: extrapolating toward an unobserved emitter", fontsize=13)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/standoff_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/standoff_sweep.png", flush=True)

    print("\n=== paired (combined - pure_gp) RMSE by standoff, NEAR-JAMMER window ===")
    for name in ("S1", "S2"):
        jx, _ = _jammer_xy(name == "S2")
        for B in SURVEY_EDGE:
            c = res[name][B]["near"]["combined"]; g = res[name][B]["near"]["pure_gp"]
            if not c or len(c) != len(g):
                continue
            d = np.array(c) - np.array(g)
            m, lo, hi = paired_bootstrap_ci(d)
            print(f"  {name} standoff={jx-B:.0f}m: {m:+.2f} dB [{lo:+.2f},{hi:+.2f}] "
                  f"d={cohens_d_paired(d):+.2f}  (combined {'better' if m < 0 else 'worse'}, n={len(c)})",
                  flush=True)
    print(f"\nDone ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
