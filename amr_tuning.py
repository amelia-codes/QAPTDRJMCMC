"""Compare AMR window widths and grid axes across QCLT walk distributions.

Every setting receives the same 1,000 measured histories in each repetition.
The final coherent walk has two 8-bit position registers (33 working qubits).
Errors compare its exact next-step covariance with the continuous target;
error bars show one sample standard deviation across 80 window selections.

Run once with: python3.12 amr_tuning.py
Redraw saved results with: python3.12 amr_tuning.py --plot-only
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

import amr_battle_test as study
import paper_style

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "quantum_walk"
DATA = ROOT / "data" / "quantum_walk" / "amr_tuning.npz"
HISTORY_SAMPLES = 1000
REPS = 80
WIDTH_FACTORS = np.array([1., 1.5, 2.])
METHOD_LABELS = ("Axis-aligned", "Rotated square", "Rotated rectangular")


def refinement_frames(histories):
    """Return three widths and three axis choices for a common history sample.

    The default window is 16 measured standard deviations wide along each
    principal axis. Width factors give 16, 24, and 32 standard deviations.
    The rotated square uses the largest principal spread for both widths.
    """
    center, axes, spacing = study.window_from_pilot(histories)
    width_frames = [(center, axes, factor * spacing) for factor in WIDTH_FACTORS]
    pooled = histories.reshape(-1, study.D)
    axis_spacing = np.maximum(pooled.std(axis=0, ddof=1) / 4, 1e-6)
    axis_spacing /= 1 << (study.BITS - study.PILOT_BITS)
    method_frames = [
        (center, np.eye(study.D), axis_spacing),
        (center, axes, np.full(study.D, spacing.max())),
        (center, axes, spacing),
    ]
    return width_frames, method_frames


def _score_frames(modes, histories, frames):
    scores = []
    for center, axes, spacing in frames:
        if not study.window_safe(modes, center, axes, spacing, histories):
            raise ValueError("AMR setting does not cover the preparatory walk; no trials were discarded")
        scores.append(study.relative_error(modes, center, axes, spacing))
    return scores


def run():
    from quantum_walk_study import sample_walk

    rng = np.random.default_rng(20261114)
    width_errors = np.empty((len(study.CASES), len(WIDTH_FACTORS), REPS))
    method_errors = np.empty((len(study.CASES), len(METHOD_LABELS), REPS))
    heldout_mismatch = np.empty((len(study.CASES), 2))
    for case_index, (name, modes, pilot_width) in enumerate(study.CASES):
        pilot = sample_walk(
            study.PILOT_BITS, study.X0, np.eye(study.D),
            np.full(study.D, pilot_width / (1 << study.PILOT_BITS)),
            8, study.PILOT_POOL, 300 + case_index, modes_physical=modes)
        pool = pilot["physical_hist"]
        histories = study.all_true_histories(modes)
        # Replay the stress test's draws so its AMR bar and this study's
        # default refinement use precisely the same 80 measurement subsets.
        subsets = []
        for budget in study.BUDGETS:
            for _ in range(REPS):
                indices = rng.choice(len(pool), int(budget), replace=False)
                if budget == HISTORY_SAMPLES:
                    subsets.append(pool[indices])
        if len(subsets) != REPS:
            raise ValueError("History sample count is missing from the AMR stress test")
        for replicate, subset in enumerate(subsets):
            widths, methods = refinement_frames(subset)
            width_errors[case_index, :, replicate] = _score_frames(modes, histories, widths)
            method_errors[case_index, :, replicate] = _score_frames(modes, histories, methods)

        # Check the widest window and the alternative square window against
        # measured coherent circuits, separately from the analytic scoring.
        widths, methods = refinement_frames(pool[:HISTORY_SAMPLES])
        for j, (center, axes, spacing) in enumerate((widths[-1], methods[1])):
            measured = sample_walk(
                study.BITS, center, axes, spacing, 14, 30_000,
                1100 + 2 * case_index + j, modes_physical=modes)
            heldout_mismatch[case_index, j] = measured["mismatch"]
            if measured["operational_qubits"] != 33:
                raise AssertionError("AMR settings changed the final walk's qubit count")
        print(f"{name}: mean errors for widths 16/24/32 SD: "
              + ", ".join(f"{value:.2%}" for value in width_errors[case_index].mean(axis=1)),
              flush=True)

    # The 16-SD width and rotated rectangular method are the same algorithm.
    np.testing.assert_array_equal(width_errors[:, 0], method_errors[:, 2])
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA, names=np.array([case[0] for case in study.CASES]),
        width_factors=WIDTH_FACTORS, widths_in_sd=16 * WIDTH_FACTORS,
        method_labels=np.array(METHOD_LABELS), width_errors=width_errors,
        method_errors=method_errors, history_samples=HISTORY_SAMPLES,
        reps=REPS, register_bits=study.BITS, qubits=33,
        heldout_mismatch=heldout_mismatch)
    plot()


def _grouped_errors(names, errors, labels, filename, *, colors=None):
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    x = np.arange(len(names))
    width = .24
    if colors is None:
        colors = paper_style.COMPARISON_COLORS
    mean = 100 * errors.mean(axis=2)
    sd = 100 * errors.std(axis=2, ddof=1)
    for j, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(x + (j - 1) * width, mean[:, j], width * .92,
               yerr=sd[:, j], capsize=2.5, color=color, edgecolor=color,
               linewidth=.7, alpha=paper_style.BAR_ALPHA, label=label,
               error_kw={"ecolor": paper_style.INK, "elinewidth": 1.0})
    ax.set_xticks(x, [name.replace("-", "\n") for name in names])
    ax.set_xlabel("Walk distribution")
    ax.set_ylabel(r"Covariance error (\%; mean $\pm$ 1 SD)")
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    paper_style.legend_below(ax, ncol=3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def plot():
    paper_style.apply()
    with np.load(DATA, allow_pickle=False) as saved:
        names = saved["names"].tolist()
        _grouped_errors(
            names, saved["width_errors"],
            [rf"Window width $= {width:g}\sigma_j$" for width in saved["widths_in_sd"]],
            "amr_zoom_limit.pdf")
        _grouped_errors(names, saved["method_errors"], saved["method_labels"].tolist(),
                        "amr_methods.pdf", colors=(paper_style.COLORS["QCLT"],
                                                   paper_style.COLORS["MEASURED"],
                                                   paper_style.COLORS["QCLT_AMR"]))


if __name__ == "__main__":
    plot() if "--plot-only" in sys.argv else run()
