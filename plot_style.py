"""The shared look of every figure, and where figures are saved."""

from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

FIGURES = Path(__file__).parent / "figures"
INK = "#171717"
MUTED = "#4A4A4A"
COLORS = {
    "COIN_SUM_AMR": "#8E2450",
    "COIN_SUM": "#D0605E",
    "MEASURED": "#9B3604",
    "BASELINE": "#4F5D5C",
}
BAR_ALPHA = .42


def apply(font_scale=1.):
    """Computer Modern text; wide figures use font_scale 1.1 because the paper shrinks them."""
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12 * font_scale,
        "axes.labelsize": 13 * font_scale,
        "xtick.labelsize": 12 * font_scale,
        "ytick.labelsize": 12 * font_scale,
        "legend.fontsize": 12 * font_scale,
        "axes.linewidth": 0.9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "pdf.fonttype": 42,
    })


def legend_below(ax, handles=None, labels=None, ncol=1):
    """Legend centred below the axes, boxed when it has more than one entry."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    elif labels is None:
        labels = [handle.get_label() for handle in handles]
    legend = ax.figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(.5, 0), ncol=ncol,
        frameon=len(labels) > 1, fancybox=False, framealpha=1, edgecolor=INK,
        borderpad=.65, columnspacing=1.3, handletextpad=.65)
    legend.get_frame().set_linewidth(.8)


def grouped_bars(names, series, ylabel, filename, height=4.6, log=False):
    """One group of bars per walk geometry.

    series: a list of (label, mean, standard deviation or None, colour), each
    with one value per geometry, in per cent.
    """
    apply(font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9.8, height))
    x = np.arange(len(names))
    for j, (label, mean, sd, color) in enumerate(series):
        error = {} if sd is None else dict(yerr=sd, capsize=2.5, error_kw={"ecolor": INK, "elinewidth": 1.0})
        ax.bar(x + (j - 1) * .24, mean, .24 * .92, color=color, edgecolor=color, linewidth=.7,
               alpha=BAR_ALPHA, label=label, **error)
    ax.set_xticks(x, [name.replace("-", "\n") for name in names])
    ax.set_xlabel("Walk distribution")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    legend_below(ax, ncol=3)
    save(fig, filename)


def save(fig, filename):
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close(fig)
