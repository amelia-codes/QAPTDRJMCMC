# When does the AMR grid improve the coin-sum step? Three experiments on seven geometries:
#   1. Fixed grids versus AMR windows chosen from 8 to 1,000 pilot histories, and versus
#      quantum AMR, which refines each branch's grid inside the circuit
#   2. AMR window widths and axis choices, from the same 1,000-history samples
#   3. Thresholds for rebuilding a window when the walk narrows or broadens
#
#   python3.12 amr.py              # recompute (about 15 s) and plot
#   python3.12 amr.py --plot-only  # redraw from data/amr.npz

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

import plot_style as style
from walk import (BITS, D, ISO_VAR, K, MODES, SCALE2, X0, all_histories, amr_window, pilot_histories,
                  quantum_amr_step_covariance)


DATA = Path(__file__).parent / "data" / "amr.npz"

# (name, preparatory modes, physical width of the pilot grid and of the fixed local grid)
CASES = [
    ("tilted", MODES, 0.08),
    ("axis-aligned", np.diag([0.0020, 0.0015]), 0.08),
    ("anti-tilted", MODES * np.array([1, -1]), 0.08),
    ("isotropic", np.diag([0.0015, 0.0015]), 0.08),
    ("broader", 4 * MODES, 0.32),

    # The fixed local grid already represents these two almost exactly
    ("grid-aligned", np.diag([0.0025, 0.00125]), 0.08),
    ("isotropic-exact", np.diag([0.0025, 0.0025]), 0.08),
]

# Each repetition draws a new sample of pilot histories from a pool of 20,000
PILOT_POOL = 20_000
PILOT_SAMPLES = [8, 16, 32, 64, 128, 512, 1000]
REPS = 80

# Window widths (in standard deviations), axis choices, and the relative width changes
# that trigger a rebuild
WIDTHS = [16, 24, 32]
AXES_LABELS = ["Axis-aligned", "Rotated square", "Rotated rectangular"]
THRESHOLDS = np.array([.10, .25, .50])


# ---------------------------------------------------------------------------
# Exact step covariance on a grid. Every error here is exact rather than estimated from
# samples; test_walk.py checks that sampled circuits agree with it.
# ---------------------------------------------------------------------------

# Covariance of one coin-sum step on a grid, averaged over all histories. A +-1 walk has
# expected history covariance (K + 1) / 6 * modes^T modes; the grid rounds the modes first,
# and its centre does not matter.
def step_covariance(modes, grid):
    center, axes, spacing = grid
    grid_modes = np.rint(modes @ axes / spacing).astype(int)
    basis = axes @ np.diag(spacing)

    lattice = SCALE2 * ((K + 1) / 6) * (grid_modes.T @ grid_modes)
    lattice += ISO_VAR * np.eye(D)
    return basis @ lattice @ basis.T


# Relative error against the same step without any grid rounding
def covariance_error(modes, grid):
    target = SCALE2 * ((K + 1) / 6) * (modes.T @ modes)
    return np.linalg.norm(step_covariance(modes, grid) - target) / np.linalg.norm(target)


# Can the 8-bit walk run on this grid? The start and every possible history must lie on it,
# and no pair difference may reach half its width.
def fits(modes, grid, histories):
    center, axes, spacing = grid
    n = 1 << BITS

    start = (X0 - center) @ axes / spacing + n / 2
    points = (histories - center) @ axes / spacing + n / 2
    grid_modes = np.rint(modes @ axes / spacing).astype(int)

    return bool(np.all((start >= 0) & (start < n)) and np.all((points >= 0) & (points < n))
                and np.all(K * np.abs(grid_modes).sum(axis=0) < n / 2))


# Quantum AMR's error: each branch's step on its own refinement of the local grid, averaged over
# the 64 equally likely branches, against the same target
def quantum_amr_error(modes, spacing):
    signs, _ = all_histories(modes)
    covariance = np.mean([quantum_amr_step_covariance(s, modes, spacing) for s in signs], axis=0)

    target = SCALE2 * ((K + 1) / 6) * (modes.T @ modes)
    return np.linalg.norm(covariance - target) / np.linalg.norm(target)


# The error of a window that must fit the walk
def checked_error(modes, grid, histories):
    if not fits(modes, grid, histories):
        raise ValueError("AMR window does not fit the walk")
    return covariance_error(modes, grid)


# ---------------------------------------------------------------------------
# The three experiments, for one geometry
# ---------------------------------------------------------------------------

# Experiments 1 and 2: AMR windows from growing numbers of pilot histories, and, for the
# 1,000-history samples, three window widths and three axis choices
def pilot_sample_errors(modes, pool, histories, rng):
    amr = np.full((len(PILOT_SAMPLES), REPS), np.nan)   # NaN: the window did not fit
    widths = np.empty((len(WIDTHS), REPS))
    axes_choices = np.empty((len(AXES_LABELS), REPS))

    for b, samples in enumerate(PILOT_SAMPLES):
        for r in range(REPS):
            sample = pool[rng.choice(PILOT_POOL, samples, replace=False)]

            # 1: the default AMR window
            window = amr_window(sample)
            if fits(modes, window, histories):
                amr[b, r] = covariance_error(modes, window)

            if samples == 1000:
                center, axes, spacing = window
                spreads = spacing * (1 << BITS) / 16
                # 2: wider windows
                for w, sds in enumerate(WIDTHS):
                    grid = (center, axes, spreads * sds / (1 << BITS))
                    widths[w, r] = checked_error(modes, grid, histories)

                # 2: axis-aligned, rotated with square cells, and the default rotated rectangular cells
                axis_aligned = sample.reshape(-1, D).std(axis=0, ddof=1) * 16 / (1 << BITS)
                for a, grid in enumerate([(center, np.eye(D), axis_aligned),
                                          (center, axes, np.full(D, spacing.max())),
                                          window]):
                    axes_choices[a, r] = checked_error(modes, grid, histories)

    return amr, widths, axes_choices


# Experiment 3: the walk narrows (scales[0]) or broadens (scales[1]). Rebuild the window only if
# some width changes by more than the threshold, then widen it until the walk fits.
def rebuild_errors(modes, pool, scales, rng):
    rebuilt = np.empty((2, len(THRESHOLDS), REPS))

    for s in range(2):
        for r, scale in enumerate(scales[s]):
            # Histories before and after the change; scaling about X0 gives the measured law
            # of the rescaled walk
            before = pool[rng.choice(PILOT_POOL, 1000, replace=False)]
            after = pool[rng.choice(PILOT_POOL, 1000, replace=False)]
            after = X0 + scale * (after - X0)
            new_modes = scale * modes
            new_histories = all_histories(new_modes)[1]

            # How much each window width would change
            current, proposed = amr_window(before), amr_window(after)
            ratio = proposed[2] / current[2]

            for t, threshold in enumerate(THRESHOLDS):
                rebuild = np.any(ratio < 1 - threshold) if s == 0 else np.any(ratio > 1 + threshold)
                center, axes, spacing = proposed if rebuild else current
                while not fits(new_modes, (center, axes, spacing), new_histories):
                    spacing = 2 * spacing
                rebuilt[s, t, r] = covariance_error(new_modes, (center, axes, spacing))

    return rebuilt


def run():
    # Separate random streams for experiments 1 and 2 and for experiment 3
    rng = np.random.default_rng(20261114)
    rng_change = np.random.default_rng(20261122)
    scales = np.stack((rng_change.uniform(.45, .95, REPS), rng_change.uniform(1.05, 1.85, REPS)))

    fixed = np.empty((len(CASES), 2))
    quantum = np.empty(len(CASES))
    amr = np.empty((len(CASES), len(PILOT_SAMPLES), REPS))
    widths = np.empty((len(CASES), len(WIDTHS), REPS))
    axes_choices = np.empty((len(CASES), len(AXES_LABELS), REPS))
    rebuilt = np.empty((len(CASES), 2, len(THRESHOLDS), REPS))
    full_grid = (np.full(D, .5), np.eye(D), np.full(D, 1 / (1 << BITS)))

    # Each geometry's pilot walk is sampled once and shared by all three experiments
    for c, (name, modes, width) in enumerate(CASES):
        pool = pilot_histories(modes, width, PILOT_POOL, seed=300 + c)
        histories = all_histories(modes)[1]

        local_grid = (X0, np.eye(D), np.full(D, width / (1 << BITS)))
        fixed[c] = [covariance_error(modes, full_grid), covariance_error(modes, local_grid)]
        quantum[c] = quantum_amr_error(modes, local_grid[2])
        amr[c], widths[c], axes_choices[c] = pilot_sample_errors(modes, pool, histories, rng)
        rebuilt[c] = rebuild_errors(modes, pool, scales, rng_change)
        print(f"{name}: AMR mean error {np.mean(amr[c, -1]):.3%}", flush=True)

    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, fixed=fixed, quantum=quantum, amr=amr, widths=widths, axes_choices=axes_choices,
                        rebuilt=rebuilt)


# ---------------------------------------------------------------------------
# Figures. Error bars are one sample standard deviation across the 80 pilot samples.
# ---------------------------------------------------------------------------

NAMES = [case[0] for case in CASES]


# Experiment 1: fixed grids versus 1,000-history AMR windows and quantum AMR
def plot_fixed_vs_amr(fixed, amr, quantum):
    if np.isnan(amr[:, -1]).any():
        raise ValueError("a 1,000-sample AMR window did not fit the walk")

    style.grouped_bars(NAMES, [
        ("Full-range grid", fixed[:, 0], None, style.COLORS["BASELINE"]),
        ("Informed local grid", fixed[:, 1], None, style.COLORS["COIN_SUM"]),
        ("Classical AMR grid", amr[:, -1].mean(axis=1), amr[:, -1].std(axis=-1, ddof=1), style.COLORS["COIN_SUM_AMR"]),
        ("Quantum AMR grids", quantum, None, style.COLORS["COIN_SUM_QAMR"])],
        r"Covariance error (\%; mean $\pm$ 1 SD, $n=80$)", "coin_sum_amr_covariance_error.pdf",
        height=4.8, log=True)


# Experiment 1: how often an AMR window beats the fixed local grid, by pilot sample count
def plot_pilot_success(fixed, amr):
    style.apply()
    win_rate = 100 * (amr < fixed[:, 1, None, None]).mean(axis=2)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.spines[:].set_visible(True)
    image = ax.imshow(win_rate, aspect="auto", origin="lower", cmap="Greens", vmin=0, vmax=100)

    # Print each cell's value, and outline each geometry's best sample count
    for row, values in enumerate(win_rate):
        for column, value in enumerate(values):
            ax.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=10,
                    color="white" if value > 50 else style.INK)
        if values.max() > 0:
            ax.add_patch(Rectangle((np.argmax(values) - .5, row - .5), 1, 1, fill=False,
                                   edgecolor=style.COLORS["MEASURED"], linewidth=2.0))

    # Axes and colour bar
    ax.set_xticks(np.arange(len(PILOT_SAMPLES)), PILOT_SAMPLES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(NAMES)), NAMES)
    ax.set_xlabel("Initial circuit samples")
    fig.colorbar(image, ax=ax, shrink=0.8, label=r"Windows improving on local grid (\%)")

    style.save(fig, "coin_sum_amr_pilot_success.pdf")


# Experiment 2: window widths and axis choices
def plot_window_choices(widths, axes_choices):
    ylabel = r"Covariance error (\%; mean $\pm$ 1 SD)"
    colors = style.COLORS

    width_colors = [colors["COIN_SUM_AMR"], colors["COIN_SUM"], colors["MEASURED"]]
    style.grouped_bars(NAMES, [
        (rf"Window width $= {w}\sigma_j$", widths[:, j].mean(axis=1), widths[:, j].std(axis=-1, ddof=1), color)
        for j, (w, color) in enumerate(zip(WIDTHS, width_colors))], ylabel, "amr_zoom_limit.pdf")

    axes_colors = [colors["COIN_SUM"], colors["MEASURED"], colors["COIN_SUM_AMR"]]
    style.grouped_bars(NAMES, [
        (label, axes_choices[:, j].mean(axis=1), axes_choices[:, j].std(axis=-1, ddof=1), color)
        for j, (label, color) in enumerate(zip(AXES_LABELS, axes_colors))], ylabel, "amr_methods.pdf")


# Experiment 3: rebuild thresholds when the walk narrows and when it broadens
def plot_rebuild_thresholds(rebuilt):
    ylabel = r"Covariance error (\%; mean $\pm$ 1 SD)"
    colors = [style.COLORS["COIN_SUM_AMR"], style.COLORS["COIN_SUM"], style.COLORS["MEASURED"]]

    for s, filename in enumerate(["amr_refinement_criterion.pdf", "amr_unrefinement_criterion.pdf"]):
        style.grouped_bars(NAMES, [
            (rf"Width-change threshold $= {100 * threshold:g}\%$",
             rebuilt[:, s, j].mean(axis=1), rebuilt[:, s, j].std(axis=-1, ddof=1), color)
            for j, (threshold, color) in enumerate(zip(THRESHOLDS, colors))], ylabel, filename)


# A sketch of "tilted": the same covariance eigenvalues as "axis-aligned", rotated
def plot_tilted_geometry():
    tilted = MODES.T @ MODES
    tilted = tilted / np.max(np.linalg.eigvalsh(tilted))

    style.apply()
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    for covariance, color, label, line in (
            (tilted, style.COLORS["COIN_SUM_AMR"], "Tilted modes", "-"),
            (np.diag(np.linalg.eigvalsh(tilted)), style.COLORS["BASELINE"], "Axis-aligned modes", "--")):
        values, vectors = np.linalg.eigh(covariance)
        angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))

        # Two contours per covariance
        for scale, alpha in ((1.0, .95), (1.7, .72)):
            ax.add_patch(Ellipse((0, 0), 2 * scale * np.sqrt(values[1]), 2 * scale * np.sqrt(values[0]),
                                 angle=angle, fill=False, edgecolor=color, linewidth=1.25,
                                 linestyle=line, alpha=alpha, label=label if scale == 1.0 else None))

    ax.set(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), xlabel="Parameter 1", ylabel="Parameter 2", aspect="equal")
    style.legend_below(ax, ncol=2)
    style.save(fig, "coin_sum_amr_tilted_geometry.pdf")


def plot():
    with np.load(DATA) as saved:
        fixed, quantum, amr, widths, axes_choices, rebuilt = (
            100 * saved[key] for key in ("fixed", "quantum", "amr", "widths", "axes_choices", "rebuilt"))

    plot_fixed_vs_amr(fixed, amr, quantum)
    plot_window_choices(widths, axes_choices)
    plot_rebuild_thresholds(rebuilt)
    plot_pilot_success(fixed, amr)
    plot_tilted_geometry()


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
