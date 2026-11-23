# The shared look of every figure, where figures are saved, and which grids they show

import sys
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


FIGURES = Path(__file__).parent / "figures"

# Classical AMR, and so hybrid AMR, appears in the figures, tables and method drawings only with
# --classical-amr on the command line; otherwise they show the fixed local grid and quantum AMR
# alone. The experiments always compute every grid, so --plot-only works either way.
CLASSICAL_AMR = "--classical-amr" in sys.argv

# Text colours
INK = "#171717"
MUTED = "#4A4A4A"

# The project palette. The coin sum is indigo wherever it appears among the covariance methods.
# Where its grid setups are told apart by colour: full-range grid sage, local grid salmon,
# classical AMR burgundy, quantum AMR indigo, hybrid AMR gold.
COLORS = {
    "INDIGO": "#6040B0",
    "BURGUNDY": "#8E2450",
    "GOLD": "#C19A2B",
    "SALMON": "#D0605E",
    "RUST": "#9B3604",
    "SAGE": "#4F5D5C",
}
BAR_ALPHA = .42


# Computer Modern text; wide figures use font_scale 1.1 because the paper shrinks them
def apply(font_scale=1.):
    mpl.rcParams.update({
        # Fonts
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12 * font_scale,
        "axes.labelsize": 13 * font_scale,
        "xtick.labelsize": 12 * font_scale,
        "ytick.labelsize": 12 * font_scale,
        "legend.fontsize": 12 * font_scale,
        "pdf.fonttype": 42,

        # Axes: no top or right spines
        "axes.linewidth": 0.9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    })


# Legend centred below the axes, boxed when it has more than one entry, with an optional title
def legend_below(ax, handles=None, labels=None, ncol=1, title=None):
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    elif labels is None:
        labels = [handle.get_label() for handle in handles]

    legend = ax.figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(.5, 0), ncol=ncol, title=title,
        frameon=len(labels) > 1, fancybox=False, framealpha=1, edgecolor=INK,
        borderpad=.65, columnspacing=1.3, handletextpad=.65)
    legend.get_frame().set_linewidth(.8)


# One group of bars per walk geometry. `series` is a list of (label, mean, standard deviation
# or None, colour), each with one value per geometry, in per cent.
def grouped_bars(names, series, ylabel, filename, height=4.6, log=False):
    apply(font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9.8, height))
    x = np.arange(len(names))
    width = .72 / len(series)

    for j, (label, mean, sd, color) in enumerate(series):
        error = {} if sd is None else dict(yerr=sd, capsize=2.5, error_kw={"ecolor": INK, "elinewidth": 1.0})
        ax.bar(x + (j - (len(series) - 1) / 2) * width, mean, width * .92, color=color, edgecolor=color,
               linewidth=.7, alpha=BAR_ALPHA, label=label, **error)

    # Axes and legend
    ax.set_xticks(x, [name.replace("-", "\n") for name in names])
    ax.set_xlabel("Walk distribution")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    legend_below(ax, ncol=min(len(series), 5))

    save(fig, filename)


def save(fig, filename):
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close(fig)
