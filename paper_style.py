"""Shared typography and method colors for paper figures."""

import matplotlib as mpl


#INK = "#1f1f1f"
#MUTED = "#555555"
#GRID = "#dedede"
#COLORS = {
#    "QCLT": "#1f77b4",
#    "QCLT_AMR": "#2ca02c",
#    "CLASSICAL": "#ff7f0e",
#    "MEASURED": "#d62728",
#    "APPROX": "#9467bd",
#    "BASELINE": "#1f77b4",
#    "FIXED": "#1f77b4",
#    "ZOOM_IN": "#ff7f0e",
#    "ZOOM_INOUT": "#2ca02c",
#    "RESTART": "#9467bd",
#    "MULTIWIN": "#8c564b",
#}
INK = "#303030"
MUTED = "#666666"
GRID = "#e5e5e5"

INK = "#252333"
MUTED = "#625F70"
GRID = "#E5E1EA"
#"SWAP": "#3973AC",
COLORS = {
    "SWAP": "#9B3604",
    "PAIRWISE": "#9B3604",
    "QCLT": "#C65D5D",
    "QCLT_AMR": "#7A1D44",
    "CLASSICAL": "#5A6E6D",
    "APPROX": "#381058",

    "MEASURED": "#9B3604",
    "BASELINE": "#4F5D5C",
    "FIXED": "#547F9E",
    "ZOOM_IN": "#3973AC",
    "ZOOM_INOUT": "#607CB9",
    "RESTART": "#913F61",
    "MULTIWIN": "#A95656",
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
