"""Money-shot maps: reconstructions under partial coverage (f=0.5, seed 0).

Top row S1, bottom row S2. Columns: ground truth, pure-GP, combined. The
observed band is marked; outside it, pure-GP flattens to its mean while combined
extrapolates via physics — helpfully for S1, harmfully for S2.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contested_rf.estimators import combined as C
from contested_rf.simulation.ground_truth import compute_sinr_map
from contested_rf.simulation.scenario import SCENARIO_1
from contested_rf.simulation.shadow_field import ShadowField
from contested_rf.simulation.uav import generate_uav_observations
import run_coverage as RC

F = 0.5
SEED = 0


def s1_case():
    sc = SCENARIO_1
    shadow = ShadowField(sc.grid_xlim, sc.grid_ylim, sigma=4.0, L=50.0, seed=SEED)
    pos, ts, obs, _ = generate_uav_observations(sc, seed=SEED, shadow_field=shadow)
    X, Y, pts = C._grid_points(sc, RC.GRID)
    _, _, truth = compute_sinr_map(sc, grid_shape=RC.GRID, path_loss_exponent=2.5,
                                   noise_floor_dBm=C.NOISE_FLOOR_DBM, shadow_field=shadow)
    lo, hi = RC.band(sc.jammers[0].position[0], F)
    mask = (pos[:, 0] >= lo) & (pos[:, 0] <= hi)
    m_c, m_g, m_p = RC.reconstruct(sc, False, pos[mask], obs[mask], ts[mask], pts)
    return X, Y, truth, m_g, m_c, (lo, hi)


def s2_case():
    rng = np.random.default_rng(3000 + SEED)
    sc, A, pos, ts, obs, truth, X, Y, pts = RC.simulate_s2(SEED, rng)
    lo, hi = RC.band(sc.jammers[1].position[0], F)
    mask = (pos[:, 0] >= lo) & (pos[:, 0] <= hi)
    m_c, m_g, m_p = RC.reconstruct(sc, True, pos[mask], obs[mask], ts[mask], pts, jammer_a=A)
    return X, Y, truth, m_g, m_c, (lo, hi)


def main():
    rows = [("Scenario 1 (well-identified jammer)", s1_case()),
            ("Scenario 2 (mis-identified beam)", s2_case())]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for r, (label, (X, Y, truth, m_g, m_c, (lo, hi))) in enumerate(rows):
        vmin, vmax = np.percentile(truth, 2), np.percentile(truth, 98)
        for c, (title, M) in enumerate([("Ground truth", truth),
                                        ("Pure GP", m_g), ("Combined (SMC+GP)", m_c)]):
            ax = axes[r, c]
            im = ax.pcolormesh(X, Y, M, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.axvline(lo, color="w", lw=1.5, ls="--"); ax.axvline(hi, color="w", lw=1.5, ls="--")
            ax.set_title(f"{label if c==0 else ''}\n{title}", fontsize=10)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.text(0.5*(lo+hi), 1900, "surveyed", color="w", ha="center", fontsize=8)
        plt.colorbar(im, ax=axes[r, :], shrink=0.7, label="SINR (dB)")
    fig.suptitle("Reconstruction under 50% coverage: outside the surveyed band (dashed), "
                 "pure GP flattens; combined extrapolates via physics", fontsize=12)
    plt.savefig("figures/coverage_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/coverage_maps.png")


if __name__ == "__main__":
    main()
