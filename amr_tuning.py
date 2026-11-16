"""Refinement criteria and step-size sensitivity for AMR register grids.

The walks are classical simulations of the register-grid algorithm. The study
varies the re-grid interval, the edge-occupancy threshold for forced widening,
the maximum width change per update, and the fraction of the measured mean
shift used to pan the window.
"""

import itertools
import pickle
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects

import adaptive_grid as A
import paper_style

ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "quantum_walk"
DATA = ROOT / "data" / "quantum_walk"
REGRID_BLOCKS = (1, 2, 4, 8)
EDGE_THRESHOLDS = (0.05, 0.20, 0.40)
ZOOM_LIMITS = (1.5, 2.0, 4.0)
PAN_FRACTIONS = (0.25, 0.5, 1.0)
SEEDS = range(6)


def score(case, samples):
    if case == "narrow":
        mu, sd = A.truth("narrow")
        x = samples[len(samples) // 2:]
        return float(np.max(np.abs((x - mu).mean(0) / sd)))
    x = samples[len(samples) // 2:]
    return float((np.linalg.norm(x - A.DECOY.comps[0][1], axis=1) < 0.06).mean())


def parameter_settings():
    settings = {(period, 0.20, 2.0, 1.0) for period in REGRID_BLOCKS}
    settings.update((1, edge, 2.0, 1.0) for edge in EDGE_THRESHOLDS)
    settings.update((1, 0.20, zoom, pan)
                    for zoom, pan in itertools.product(ZOOM_LIMITS, PAN_FRACTIONS))
    return sorted(settings)


def run():
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in ("narrow", "decoy"):
        cfg = A.CONFIGS[case]
        x0 = np.asarray(cfg["x0"])
        problem = A.make_problem(case)
        for seed in SEEDS:
            samples, _, _, _ = A.walk(problem, "fixed", cfg["n_bits"], cfg["n_steps"],
                                       cfg["block"], cfg["K"], seed, x0)
            rows.append(dict(case=case, strategy="fixed", seed=seed, score=score(case, samples),
                             regrid=1, edge=0.20, zoom=1.0, pan=0.0, cpu_per_step=np.nan))
        for regrid, edge, zoom, pan in parameter_settings():
            for seed in SEEDS:
                t0 = time.process_time()
                samples, _, _, _ = A.walk(
                    problem, "restart", cfg["n_bits"], cfg["n_steps"], cfg["block"],
                    cfg["K"], seed, x0, zoom_factor=zoom, pan_fraction=pan,
                    regrid_blocks=regrid, edge_threshold=edge)
                elapsed = time.process_time() - t0
                rows.append(dict(case=case, strategy="restart", seed=seed, score=score(case, samples),
                                 regrid=regrid, edge=edge, zoom=zoom, pan=pan,
                                 cpu_per_step=elapsed / cfg["n_steps"]))
    with open(DATA / "amr_tuning.pkl", "wb") as f:
        pickle.dump(rows, f)
    plot(rows)


def values(rows, case, **params):
    return np.asarray([r["score"] for r in rows if r["case"] == case and r["strategy"] == "restart"
                       and all(r[k] == v for k, v in params.items())])


def add_interval_plot(rows, case):
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    med, lo, hi = [], [], []
    for n in REGRID_BLOCKS:
        v = values(rows, case, regrid=n, edge=0.20, zoom=2.0, pan=1.0)
        med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
    x = np.array(REGRID_BLOCKS) * A.CONFIGS[case]["block"]
    ax.plot(x, med, "o-", color=paper_style.COLORS["QCLT_AMR"], lw=1.25, ms=4, label="AMR")
    ax.fill_between(x, lo, hi, color=paper_style.COLORS["QCLT_AMR"], alpha=.15, lw=0)
    base = [r["score"] for r in rows if r["case"] == case and r["strategy"] == "fixed"]
    ax.axhline(np.median(base), color=paper_style.MUTED, ls="--", lw=.85, label="Fixed grid")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(v) for v in x])
    ax.set_xlabel("Re-grid interval (walk steps)")
    ax.set_ylabel("Time in target peak" if case == "decoy" else "Mean error (true spreads)")
    ax.grid(color=paper_style.GRID, lw=.6); ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"amr_regrid_interval_{case}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_edge_plot(rows, case):
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    med, lo, hi = [], [], []
    for edge in EDGE_THRESHOLDS:
        v = values(rows, case, regrid=1, edge=edge, zoom=2.0, pan=1.0)
        med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
    ax.plot(np.array(EDGE_THRESHOLDS) * 100, med, "o-", color=paper_style.COLORS["QCLT_AMR"],
            lw=1.25, ms=4)
    ax.fill_between(np.array(EDGE_THRESHOLDS) * 100, lo, hi,
                    color=paper_style.COLORS["QCLT_AMR"], alpha=.15, lw=0)
    ax.set_xlabel(r"Edge occupancy threshold (\%)")
    ax.set_ylabel("Time in target peak" if case == "decoy" else "Mean error (true spreads)")
    ax.grid(color=paper_style.GRID, lw=.6); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"amr_edge_threshold_{case}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_zoom_pan_plot(rows, case):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    matrix = np.array([[np.median(values(rows, case, regrid=1, edge=0.20, zoom=z, pan=p))
                        for p in PAN_FRACTIONS] for z in ZOOM_LIMITS])
    im = ax.imshow(matrix, aspect="auto", cmap="Greens")
    ax.set_xticks(range(len(PAN_FRACTIONS)), [f"{p:g}" for p in PAN_FRACTIONS])
    ax.set_yticks(range(len(ZOOM_LIMITS)), [f"{z:g}x" for z in ZOOM_LIMITS])
    ax.set_xlabel("Pan fraction")
    ax.set_ylabel("Maximum width change per re-grid")
    for i in range(len(ZOOM_LIMITS)):
        for j in range(len(PAN_FRACTIONS)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white",
                    path_effects=[path_effects.withStroke(linewidth=1.2, foreground="black")])
    cb = fig.colorbar(im, ax=ax, fraction=.055, pad=.04)
    cb.set_label("Target peak occupancy" if case == "decoy" else "Mean error (true spreads)")
    fig.tight_layout()
    fig.savefig(OUT / f"amr_zoom_pan_{case}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot(rows):
    paper_style.apply()
    for case in ("decoy", "narrow"):
        add_interval_plot(rows, case)
        add_edge_plot(rows, case)
        add_zoom_pan_plot(rows, case)
    print("saved six AMR refinement-parameter figures")
    times = [r["cpu_per_step"] for r in rows if r["strategy"] == "restart"]
    print(f"AMR simulator CPU time: median {1e6*np.median(times):.2f} microseconds per proposal")


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        with open(DATA / "amr_tuning.pkl", "rb") as f:
            plot(pickle.load(f))
    else:
        run()
