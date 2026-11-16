"""Shared typography and method colors for paper figures."""

import matplotlib as mpl

INK = "#1f1f1f"
MUTED = "#555555"
GRID = "#dedede"
COLORS = {
    "QCLT": "#1f77b4",
    "QCLT_AMR": "#2ca02c",
    "CLASSICAL": "#ff7f0e",
    "MEASURED": "#d62728",
    "APPROX": "#9467bd",
    "BASELINE": "#1f77b4",
    "FIXED": "#1f77b4",
    "ZOOM_IN": "#ff7f0e",
    "ZOOM_INOUT": "#2ca02c",
    "RESTART": "#9467bd",
    "MULTIWIN": "#8c564b",
}


def apply():
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
