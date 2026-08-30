"""Build paired-difference figures (4.2, 4.3, 4.4) from raw per-seed checkpoints.

Difference = combined - pure_gp, per seed, in the scored region.
CI = 95% paired bootstrap (10,000 resamples) of the mean difference.
Every series is verified against the published .log numbers before plotting.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "results"
OUT = "figures"
rng = np.random.default_rng(20240901)

os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})
C_S1 = "#1f77b4"; C_S2 = "#d62728"


def boot_ci(diffs, n=10000):
    diffs = np.asarray(diffs, float)
    m = diffs.mean()
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    bmeans = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(bmeans, [2.5, 97.5])
    sd = diffs.std(ddof=1)
    d = m / sd if sd > 0 else np.nan
    return m, lo, hi, d


def coverage_series(cov, terrain, scen, fracs):
    """Return dict frac -> (mean, lo, hi, d) for unobserved-region paired diff."""
    out = {}
    for f in fracs:
        fk = ("%s" % f).rstrip("0").rstrip(".") if f != 1.0 else "1.0"
        # keys in file: '1.0','0.66','0.5','0.33','0.25'
        cand = [fk, "%g" % f]
        diffs = []
        seed = 0
        while True:
            key = "%s|%s|%d" % (terrain, scen, seed)
            if key not in cov:
                break
            rec = cov[key]
            fkey = next((c for c in cand if c in rec), None)
            if fkey is None:
                fkey = next((k for k in rec if abs(float(k) - f) < 1e-6), None)
            u = rec[fkey]["unobserved"]
            if u["combined"] is not None and u["pure_gp"] is not None:
                diffs.append(u["combined"] - u["pure_gp"])
            seed += 1
        out[f] = boot_ci(diffs) + (len(diffs),)
    return out


def standoff_series(sto, scen, jammer_x):
    """near-window paired diff; ckpt key is survey far-edge B, standoff = jammer_x - B."""
    out = {}
    edges = sorted({float(dk) for k, v in sto.items() if k.startswith(scen + "|") for dk in v})
    for B in edges:
        diffs = []
        seed = 0
        while True:
            key = "%s|%d" % (scen, seed)
            if key not in sto:
                break
            rec = sto[key]
            dkey = next((k for k in rec if abs(float(k) - B) < 1e-6), None)
            nw = rec[dkey]["near"]
            if nw["combined"] is not None and nw["pure_gp"] is not None:
                diffs.append(nw["combined"] - nw["pure_gp"])
            seed += 1
        out[jammer_x - B] = boot_ci(diffs) + (len(diffs),)
    return out


cov = json.load(open(f"{RES}/coverage_ckpt.json"))
sto = json.load(open(f"{RES}/standoff_ckpt.json"))
FRACS = [0.66, 0.5, 0.33, 0.25]

series = {
    ("flat", "S1"): coverage_series(cov, "flat", "S1", FRACS),
    ("flat", "S2"): coverage_series(cov, "flat", "S2", FRACS),
    ("terrain", "S1"): coverage_series(cov, "terrain", "S1", FRACS),
    ("terrain", "S2"): coverage_series(cov, "terrain", "S2", FRACS),
}
sto_series = {
    "S1": standoff_series(sto, "S1", 1200.0),
    "S2": standoff_series(sto, "S2", 1400.0),
}

# ---------- VERIFICATION against published logs ----------
print("=== VERIFICATION (computed vs published) ===")
published = {
    ("flat","S1"): {0.66:-1.30,0.5:-1.07,0.33:-1.22,0.25:-1.21},
    ("flat","S2"): {0.66:+4.49,0.5:+3.67,0.33:+4.30,0.25:+4.87},
    ("terrain","S1"): {0.66:-0.69,0.5:-0.41,0.33:-0.48,0.25:+0.45},
    ("terrain","S2"): {0.66:+4.69,0.5:+3.53,0.33:+4.02,0.25:+4.65},
}
maxerr = 0.0
for cond, pub in published.items():
    for f, pv in pub.items():
        m = series[cond][f][0]
        err = abs(m - pv); maxerr = max(maxerr, err)
        flag = "OK" if err < 0.05 else "**CHECK**"
        print(f"  {cond} f={f}: computed {m:+.2f}  published {pv:+.2f}  |d|={err:.3f} {flag} (n={series[cond][f][4]})")
pub_sto = {"S1":{200:-3.84,400:-5.32,600:-5.87}, "S2":{400:+3.68,600:+4.53,800:+5.25}}
for sc, pub in pub_sto.items():
    for x, pv in pub.items():
        m = sto_series[sc][x][0]
        err = abs(m - pv); maxerr = max(maxerr, err)
        flag = "OK" if err < 0.05 else "**CHECK**"
        print(f"  standoff {sc} {x}m: computed {m:+.2f}  published {pv:+.2f}  |d|={err:.3f} {flag} (n={sto_series[sc][x][4]})")
print(f"  MAX |mean error| = {maxerr:.3f} dB")
print()

# ---------- PLOTTING ----------
def plot_coverage(terrain, fname, title, nseed):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for scen, col, lab in [("S1", C_S1, "Scenario 1 (well-identified)"),
                           ("S2", C_S2, "Scenario 2 (mis-identified)")]:
        s = series[(terrain, scen)]
        xs = FRACS
        m = np.array([s[f][0] for f in xs])
        lo = np.array([s[f][1] for f in xs])
        hi = np.array([s[f][2] for f in xs])
        ax.plot(xs, m, "-o", color=col, lw=2, ms=6, label=lab, zorder=3)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.18, zorder=1)
    ax.axhline(0, color="0.35", lw=1.2, ls="--", zorder=2)
    ax.set_xlabel("surveyed fraction  $f$")
    ax.set_ylabel("RMSE difference (combined $-$ pure-GP), dB")
    ax.set_title(title, fontsize=11.5)
    ax.invert_xaxis()  # more coverage on the left, matching the absolute-plot orientation
    ax.text(0.985, 0.03, "below 0 dB: decomposition better", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color="0.4", style="italic")
    ax.legend(fontsize=9, framealpha=0.92, loc="center left")
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


def plot_standoff(fname):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for scen, col, lab in [("S1", C_S1, "Scenario 1 (well-identified)"),
                           ("S2", C_S2, "Scenario 2 (mis-identified)")]:
        s = sto_series[scen]
        xs = sorted(s.keys())
        m = np.array([s[x][0] for x in xs])
        lo = np.array([s[x][1] for x in xs])
        hi = np.array([s[x][2] for x in xs])
        ax.plot(xs, m, "-o", color=col, lw=2, ms=6, label=lab, zorder=3)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.18, zorder=1)
    ax.axhline(0, color="0.35", lw=1.2, ls="--", zorder=2)
    ax.set_xlabel("standoff distance, jammer $-$ survey edge (m)")
    ax.set_ylabel("RMSE difference (combined $-$ pure-GP), dB")
    ax.set_title("Standoff geometry: near-jammer window (12 seeds)", fontsize=11.5)
    ax.text(0.985, 0.03, "below 0 dB: decomposition better", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color="0.4", style="italic")
    ax.legend(fontsize=9, framealpha=0.92, loc="center left")
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


plot_coverage("flat", "fig_4_2_diff.png",
              "Partial coverage, flat terrain: does the decomposition beat the GP? (20 seeds)", 20)
plot_coverage("terrain", "fig_4_3_diff.png",
              "Partial coverage, procedural terrain (12 seeds)", 12)
plot_standoff("fig_4_4_diff.png")
print("DONE")
