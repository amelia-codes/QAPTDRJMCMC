"""Compare direct QCLT steps with the five selected readout pipelines.

All routes use the same rounded histories and continuous-history target.
Covariance preparation has finite-shot tomography added to supply the same
ideal Gaussian loader as the entry estimators. AMR is a reference with a
separate, fixed history-measurement budget, as in the method bar chart.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from covariance_benchmarks import (
    READOUT_KINDS, READOUT_LABELS, project_psd, readout_estimate,
)

from quantum_walk_study import D, SCALE2, ISO_VAR, MODES, history_branches

ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "quovar"
DATA = ROOT / "data" / "quantum_walk"
BUDGETS = np.array([12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144])
REPS = 100


def generate():
    from qclt_proposal_comparison import load_covariance_errors

    DATA.mkdir(parents=True, exist_ok=True)
    saved, _ = load_covariance_errors()
    rng, extra_rng = np.random.default_rng(91002), np.random.default_rng(91003)
    errors = np.empty((len(READOUT_KINDS), len(BUDGETS), 64 * REPS))
    qclt_errors = np.empty(64)
    spacing = .08 / 256
    for j, (signs, physical_history) in enumerate(history_branches()):
        history = np.cumsum(signs @ np.rint(MODES / spacing).astype(int), axis=0).astype(float)
        target = SCALE2 * np.cov(physical_history.T)
        qclt_cov = spacing**2 * (SCALE2 * np.cov(history.T) + ISO_VAR * np.eye(D))
        qclt_errors[j] = np.linalg.norm(qclt_cov - target) / np.linalg.norm(target)
        for b, budget in enumerate(BUDGETS):
            for kind in ("pairwise", "swap", "hadamard", "qae", "qpca"):
                stream = rng if kind in ("pairwise", "swap") else extra_rng
                estimated = project_psd(readout_estimate(history, kind, int(budget), stream, REPS))
                step_cov = spacing**2 * (SCALE2 * estimated + ISO_VAR * np.eye(D))
                errors[READOUT_KINDS.index(kind), b, j * REPS:(j + 1) * REPS] = (
                    np.linalg.norm(step_cov - target, axis=(1, 2)) / np.linalg.norm(target))
    if not np.all(np.isfinite(errors)):
        raise AssertionError("Readout comparison contains nonfinite values")
    np.savez_compressed(DATA / "step_estimator_comparison.npz", methods=np.array(READOUT_KINDS), budgets=BUDGETS, errors=errors,
                        reps=REPS, branches=64, qclt_errors=qclt_errors,
                        amr_errors=saved["qclt_amr"])
    for m, kind in enumerate(READOUT_KINDS):
        values = errors[m, 5]
        print(f"{READOUT_LABELS[kind]} at budget 384: mean {values.mean():.3%}, "
              f"SD {values.std(ddof=1):.3%}", flush=True)
    plot()


def plot():
    import paper_style

    with np.load(DATA / "step_estimator_comparison.npz") as data:
        errors, budgets = data["errors"], data["budgets"]
        references = (("QCLT, fixed local grid", data["qclt_errors"], "QCLT"),
                      ("QCLT + AMR", data["amr_errors"], "QCLT_AMR"))
    paper_style.apply()
    fig, ax = plt.subplots(figsize=(7., 4.6))
    styles = (("o", "-"), ("s", "--"), ("^", ":"), ("D", "-."), ("v", "-"))
    for m, (kind, (marker, line)) in enumerate(zip(READOUT_KINDS, styles)):
        values = 100 * errors[m]
        mean, sd = values.mean(axis=1), values.std(axis=1, ddof=1)
        color = paper_style.COLORS["BASELINE" if kind == "pairwise" else "MEASURED"]
        ax.fill_between(budgets, mean - sd, mean + sd, color=color, alpha=.05, lw=0)
        ax.plot(budgets, mean, marker=marker, linestyle=line, ms=4, lw=1.3,
                color=color, label=READOUT_LABELS[kind])
    for label, values, color_key in references:
        mean, sd = 100 * values.mean(), 100 * values.std(ddof=1)
        color = paper_style.COLORS[color_key]
        ax.axhspan(mean - sd, mean + sd, color=color, alpha=.09, lw=0)
        ax.axhline(mean, color=color, lw=1.5, label=label)
    ax.set(xscale="log", xlabel="Readout shots or QAE oracle calls",
           ylabel=r"Next-step covariance error (\%)")
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=paper_style.MUTED, alpha=.18))
    labels.append(r"Mean $\pm$ 1 SD")
    paper_style.legend_below(ax, handles, labels, ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "qclt_step_generation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    plot() if "--plot-only" in sys.argv else generate()
