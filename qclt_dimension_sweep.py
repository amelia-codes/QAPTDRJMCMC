"""Dimension sensitivity for stored-history QCLT proposals.

This is a small controlled extension of qclt_proposal_comparison.py. It holds
the history size, register precision, and target condition number fixed while
varying the parameter dimension.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qclt_proposal_comparison import (
    N_HIST, REG_BITS, chain_metric, de_proposal, gaussian_proposal,
    pair_proposal, posterior, setup,
)
import paper_style

ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "quantum_walk"
DATA = ROOT / "data" / "quantum_walk"
DIMS = (2, 3, 5, 8, 12, 20)
K = 8
N_CHAINS = 120
T_STEPS = 250
REPEATS = 5


def run():
    rng = np.random.default_rng(20261116)
    methods = ("QCLT full + AMR", "QCLT lazy + AMR", "AM, K states", "DE-MCz, K states")
    values = np.empty((len(DIMS), REPEATS, len(methods)))
    for i, d in enumerate(DIMS):
        for rep in range(REPEATS):
            post, Hf, amr, _ = setup(d, rng, n=N_CHAINS)
            sd2 = 2.38**2 / d
            Hs = np.array([h[rng.choice(N_HIST, K, replace=False)] for h in Hf])
            props = [
                pair_proposal(Hs, amr, True, sd2),
                pair_proposal(Hs, amr, False, sd2),
                gaussian_proposal(sd2 * np.array([np.cov(h.T) for h in Hs])),
                de_proposal(Hs, d),
            ]
            optimum = chain_metric(
                gaussian_proposal(np.tile(sd2 * post["Sig"], (N_CHAINS, 1, 1))),
                post, rng, n=N_CHAINS, T=T_STEPS)["worst"]
            for j, proposal in enumerate(props):
                score = chain_metric(proposal, post, rng, n=N_CHAINS, T=T_STEPS)["worst"]
                values[i, rep, j] = score / max(optimum, 1e-12)
        print(f"d={d}: " + ", ".join(
            f"{name} {np.median(values[i, :, j]):.3f}"
            for j, name in enumerate(methods)), flush=True)

    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA / "dimension_sweep.npz", dims=DIMS, methods=methods,
                        values=values, K=K, n_chains=N_CHAINS,
                        steps=T_STEPS, repeats=REPEATS, register_bits=REG_BITS)
    plot()


def plot():
    z = np.load(DATA / "dimension_sweep.npz")
    paper_style.apply()
    dims, values, methods = z["dims"], z["values"], z["methods"].tolist()
    colors = [paper_style.COLORS["QCLT_AMR"], paper_style.COLORS["QCLT_AMR"],
              paper_style.COLORS["CLASSICAL"], paper_style.COLORS["APPROX"]]
    styles = ["-", "--", "-", "-."]
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    for j, name in enumerate(methods):
        med = np.median(values[:, :, j], axis=1)
        lo, hi = np.quantile(values[:, :, j], [.1, .9], axis=1)
        ax.fill_between(dims, lo, hi, color=colors[j], alpha=.12, lw=0)
        ax.plot(dims, med, styles[j], marker=markers[j], ms=3.5,
                color=colors[j], lw=1.2, label=name)
    ax.axhline(1, color=paper_style.INK, lw=.7)
    ax.set(xscale="log", xticks=dims, xticklabels=[str(d) for d in dims],
           xlabel="Parameter dimension, $d$",
           ylabel="Worst-direction jump / optimum (1 = optimal)")
    ax.grid(color=paper_style.GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, fontsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "qclt_dimension_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        plot()
    else:
        run()
