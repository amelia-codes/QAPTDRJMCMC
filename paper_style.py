"""Shared typography and method colors for paper figures."""

import matplotlib as mpl


INK = "#171717"
MUTED = "#4A4A4A"
COLORS = {
    "SWAP": "#9B3604",
    "PAIRWISE": "#9B3604",
    "QCLT": "#C65D5D",
    "QCLT_AMR": "#7A1D44",
    "CLASSICAL": "#4F5D5C",
    "APPROX": "#4F5D5C",

    "MEASURED": "#9B3604",
    "BASELINE": "#4F5D5C",
    "FIXED": "#4F5D5C",
    "ZOOM_IN": "#C65D5D",
    "ZOOM_INOUT": "#7A1D44",
    "RESTART": "#7A1D44",
    "MULTIWIN": "#9B3604",
}

# Match the covariance-method error bar plot across all comparisons.
BAR_ALPHA = .42
COMPARISON_COLORS = (
    COLORS["QCLT_AMR"], COLORS["QCLT"], COLORS["MEASURED"], COLORS["BASELINE"],
)
AMR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "quovar_amr",
    ["white", tuple(1 - BAR_ALPHA + BAR_ALPHA * channel
                    for channel in mpl.colors.to_rgb(COLORS["QCLT_AMR"]))],
)


def apply():
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
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
