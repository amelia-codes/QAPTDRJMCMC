"""Shared typography and method colors for paper figures."""

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


INK = "#171717"
MUTED = "#4A4A4A"
COLORS = {
    "QCLT_AMR": "#7A1D44",
    "QCLT": "#C65D5D",
    "MEASURED": "#9B3604",
    "BASELINE": "#4F5D5C",
}

# Match the covariance-method error bar plot across all comparisons.
BAR_ALPHA = .42
HEATMAP_COLOR_STRENGTH = .80
COMPARISON_COLORS = (
    COLORS["QCLT_AMR"], COLORS["QCLT"], COLORS["MEASURED"], COLORS["BASELINE"],
)
AMR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "quovar_amr",
    ["white", tuple(1 - HEATMAP_COLOR_STRENGTH + HEATMAP_COLOR_STRENGTH * channel
                    for channel in mpl.colors.to_rgb(COLORS["QCLT_AMR"]))],
)


def apply(font_scale=1.):
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12 * font_scale,
        "axes.labelsize": 13 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "xtick.labelsize": 12 * font_scale,
        "ytick.labelsize": 12 * font_scale,
        "legend.fontsize": 12 * font_scale,
        "axes.prop_cycle": mpl.cycler(color=COMPARISON_COLORS),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.linewidth": 0.9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def legend_below(ax, handles=None, labels=None, *, ncol=1):
    """Place the legend below the figure; box legends with multiple entries.

    Save with bbox_inches="tight" to include the legend without shrinking the
    axes or letting long labels overlap the data.
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    elif labels is None:
        labels = [handle.get_label() for handle in handles]
    legend = ax.figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(.5, 0),
        ncol=ncol, frameon=len(labels) > 1, fancybox=False,
        framealpha=1, facecolor="white", edgecolor=INK,
        borderpad=.65, columnspacing=1.3, handletextpad=.65,
    )
    legend.get_frame().set_linewidth(.8)
    return legend


def grouped_errors(names, errors, labels, filename, *, colors=None):
    # Wide figures are reduced more when placed in the paper.
    apply(font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    x = np.arange(len(names))
    width = .24
    if colors is None:
        colors = COMPARISON_COLORS
    mean = 100 * errors.mean(axis=2)
    sd = 100 * errors.std(axis=2, ddof=1)
    for j, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(x + (j - 1) * width, mean[:, j], width * .92,
               yerr=sd[:, j], capsize=2.5, color=color, edgecolor=color,
               linewidth=.7, alpha=BAR_ALPHA, label=label,
               error_kw={"ecolor": INK, "elinewidth": 1.0})
    ax.set_xticks(x, [name.replace("-", "\n") for name in names])
    ax.set_xlabel("Walk distribution")
    ax.set_ylabel(r"Covariance error (\%; mean $\pm$ 1 SD)")
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    legend_below(ax, ncol=3)
    fig.tight_layout()
    from pathlib import Path
    output = Path(__file__).parent / "figures" / "quovar"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / filename, bbox_inches="tight")
    plt.close(fig)
