"""When does the AMR grid improve the coin-sum step? Three experiments on seven geometries.

1. Fixed grids versus AMR windows chosen from 8 to 1,000 pilot histories
   (coin_sum_amr_covariance_error.pdf, coin_sum_amr_pilot_success.pdf, and the
   sketch coin_sum_amr_tilted_geometry.pdf).
2. AMR window widths and axis choices, from the same 1,000-history samples
   (amr_zoom_limit.pdf, amr_methods.pdf).
3. Thresholds for rebuilding a window when the walk narrows or broadens
   (amr_refinement_criterion.pdf, amr_unrefinement_criterion.pdf).

The step covariance on a grid is known exactly (step_covariance), so every
error here is exact rather than estimated from samples. test_walk.py checks
that sampled circuits agree with it. Each repetition draws a new sample of
pilot histories; error bars are one sample standard deviation across them.

    python3.12 amr.py              # recompute (about 15 s) and plot
    python3.12 amr.py --plot-only  # redraw from data/amr.npz
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

import plot_style as style
from walk import BITS, D, ISO_VAR, K, MODES, SCALE2, X0, all_histories, amr_window, pilot_histories

DATA = Path(__file__).parent / "data" / "amr.npz"
# (name, preparatory modes, physical width of the pilot grid and the fixed local grid)
CASES = [
    ("tilted", MODES, 0.08),
    ("axis-aligned", np.diag([0.0020, 0.0015]), 0.08),
    ("anti-tilted", MODES * np.array([1, -1]), 0.08),
    ("isotropic", np.diag([0.0015, 0.0015]), 0.08),
    ("broader", 4 * MODES, 0.32),
    # The fixed local grid already represents these two almost exactly.
    ("grid-aligned", np.diag([0.0025, 0.00125]), 0.08),
    ("isotropic-exact", np.diag([0.0025, 0.0025]), 0.08),
]
PILOT_POOL = 20_000
PILOT_SAMPLES = [8, 16, 32, 64, 128, 512, 1000]
REPS = 80
WIDTHS = [16, 24, 32]                  # window widths, in standard deviations
AXES_LABELS = ["Axis-aligned", "Rotated square", "Rotated rectangular"]
THRESHOLDS = np.array([.10, .25, .50])  # relative width change that triggers a rebuild


def step_covariance(modes, grid):
    """Exact covariance of one coin-sum step on a grid, averaged over all histories.

    A +-1 walk has expected history covariance (K + 1) / 6 * modes^T modes;
    the grid rounds the modes first. The grid centre does not matter.
    """
    center, axes, spacing = grid
    grid_modes = np.rint(modes @ axes / spacing).astype(int)
    basis = axes @ np.diag(spacing)
    lattice = SCALE2 * ((K + 1) / 6) * (grid_modes.T @ grid_modes)
    lattice += ISO_VAR * np.eye(D)
    return basis @ lattice @ basis.T


def covariance_error(modes, grid):
    """Relative error against the same step without any grid rounding."""
    target = SCALE2 * ((K + 1) / 6) * (modes.T @ modes)
    return np.linalg.norm(step_covariance(modes, grid) - target) / np.linalg.norm(target)


def fits(modes, grid, histories):
    """Can the 8-bit walk run on this grid? The start and every possible
    history must lie on it, and no pair difference may reach half its width."""
    center, axes, spacing = grid
    n = 1 << BITS
    start = (X0 - center) @ axes / spacing + n / 2
    points = (histories - center) @ axes / spacing + n / 2
    grid_modes = np.rint(modes @ axes / spacing).astype(int)
    return bool(np.all((start >= 0) & (start < n)) and np.all((points >= 0) & (points < n))
                and np.all(K * np.abs(grid_modes).sum(axis=0) < n / 2))


def checked_error(modes, grid, histories):
    if not fits(modes, grid, histories):
        raise ValueError("AMR window does not fit the walk")
    return covariance_error(modes, grid)


def run():
    rng = np.random.default_rng(20261114)          # pilot samples, experiments 1 and 2
    rng_change = np.random.default_rng(20261122)   # experiment 3
    scales = np.stack((rng_change.uniform(.45, .95, REPS), rng_change.uniform(1.05, 1.85, REPS)))
    fixed = np.empty((len(CASES), 2))
    amr = np.full((len(CASES), len(PILOT_SAMPLES), REPS), np.nan)   # NaN: window did not fit
    widths = np.empty((len(CASES), len(WIDTHS), REPS))
    axes_choices = np.empty((len(CASES), len(AXES_LABELS), REPS))
    rebuilt = np.empty((len(CASES), 2, len(THRESHOLDS), REPS))

    for c, (name, modes, width) in enumerate(CASES):
        pool = pilot_histories(modes, width, PILOT_POOL, seed=300 + c)
        histories = all_histories(modes)[1]
        fixed[c] = [covariance_error(modes, (np.full(D, .5), np.eye(D), np.full(D, 1 / (1 << BITS)))),
                    covariance_error(modes, (X0, np.eye(D), np.full(D, width / (1 << BITS))))]

        # 1 and 2: AMR windows from growing numbers of pilot histories.
        for b, samples in enumerate(PILOT_SAMPLES):
            for r in range(REPS):
                sample = pool[rng.choice(PILOT_POOL, samples, replace=False)]
                window = amr_window(sample)
                if fits(modes, window, histories):
                    amr[c, b, r] = covariance_error(modes, window)
                if samples == 1000:
                    for w, sds in enumerate(WIDTHS):
                        widths[c, w, r] = checked_error(modes, amr_window(sample, sds), histories)
                    center, axes, spacing = window
                    axis_aligned = sample.reshape(-1, D).std(axis=0, ddof=1) * 16 / (1 << BITS)
                    for a, grid in enumerate([(center, np.eye(D), axis_aligned),
                                              (center, axes, np.full(D, spacing.max())),
                                              window]):
                        axes_choices[c, a, r] = checked_error(modes, grid, histories)

        # 3: the walk changes scale. Rebuild the window only if some width
        # changes by more than the threshold; widen it until the walk fits.
        for s in range(2):                            # 0 narrowing, 1 broadening
            for r, scale in enumerate(scales[s]):
                before = pool[rng_change.choice(PILOT_POOL, 1000, replace=False)]
                after = pool[rng_change.choice(PILOT_POOL, 1000, replace=False)]
                after = X0 + scale * (after - X0)    # the measured law of the rescaled walk
                new_modes = scale * modes
                new_histories = all_histories(new_modes)[1]
                current, proposed = amr_window(before), amr_window(after)
                ratio = proposed[2] / current[2]
                for t, threshold in enumerate(THRESHOLDS):
                    rebuild = np.any(ratio < 1 - threshold) if s == 0 else np.any(ratio > 1 + threshold)
                    center, axes, spacing = proposed if rebuild else current
                    while not fits(new_modes, (center, axes, spacing), new_histories):
                        spacing = 2 * spacing
                    rebuilt[c, s, t, r] = covariance_error(new_modes, (center, axes, spacing))
        print(f"{name}: AMR mean error {np.mean(amr[c, -1]):.3%}", flush=True)

    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, fixed=fixed, amr=amr, widths=widths,
                        axes_choices=axes_choices, rebuilt=rebuilt)


def plot():
    names = [case[0] for case in CASES]
    with np.load(DATA) as saved:
        fixed, amr, widths, axes_choices, rebuilt = (
            100 * saved[key] for key in ("fixed", "amr", "widths", "axes_choices", "rebuilt"))
    colors = style.COLORS
    sd = lambda errors: errors.std(axis=-1, ddof=1)

    if np.isnan(amr[:, -1]).any():
        raise ValueError("a 1,000-sample AMR window did not fit the walk")
    style.grouped_bars(names, [
        ("Full-range grid", fixed[:, 0], None, colors["BASELINE"]),
        ("Informed local grid", fixed[:, 1], None, colors["COIN_SUM"]),
        ("AMR grid", amr[:, -1].mean(axis=1), sd(amr[:, -1]), colors["COIN_SUM_AMR"])],
        r"Covariance error (\%; mean $\pm$ 1 SD, $n=80$)", "coin_sum_amr_covariance_error.pdf",
        height=4.8, log=True)

    ylabel = r"Covariance error (\%; mean $\pm$ 1 SD)"
    three_colors = [colors["COIN_SUM_AMR"], colors["COIN_SUM"], colors["MEASURED"]]
    style.grouped_bars(names, [
        (rf"Window width $= {w}\sigma_j$", widths[:, j].mean(axis=1), sd(widths[:, j]), color)
        for j, (w, color) in enumerate(zip(WIDTHS, three_colors))], ylabel, "amr_zoom_limit.pdf")
    style.grouped_bars(names, [
        (label, axes_choices[:, j].mean(axis=1), sd(axes_choices[:, j]), color)
        for j, (label, color) in enumerate(zip(
            AXES_LABELS, [colors["COIN_SUM"], colors["MEASURED"], colors["COIN_SUM_AMR"]]))],
        ylabel, "amr_methods.pdf")
    for s, filename in enumerate(["amr_refinement_criterion.pdf", "amr_unrefinement_criterion.pdf"]):
        style.grouped_bars(names, [
            (rf"Width-change threshold $= {100 * threshold:g}\%$",
             rebuilt[:, s, j].mean(axis=1), sd(rebuilt[:, s, j]), color)
            for j, (threshold, color) in enumerate(zip(THRESHOLDS, three_colors))], ylabel, filename)

    # How often does an AMR window beat the fixed local grid?
    style.apply()
    win_rate = 100 * (amr < fixed[:, 1, None, None]).mean(axis=2)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.spines[:].set_visible(True)
    image = ax.imshow(win_rate, aspect="auto", origin="lower", cmap="Greens", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(PILOT_SAMPLES)), PILOT_SAMPLES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_xlabel("Initial circuit samples")
    fig.colorbar(image, ax=ax, shrink=0.8, label=r"Windows improving on local grid (\%)")
    for row, values in enumerate(win_rate):
        for column, value in enumerate(values):
            ax.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=10,
                    color="white" if value > 50 else style.INK)
        if values.max() > 0:                       # outline each geometry's best sample count
            ax.add_patch(Rectangle((np.argmax(values) - .5, row - .5), 1, 1, fill=False,
                                   edgecolor=style.COLORS["MEASURED"], linewidth=2.0))
    style.save(fig, "coin_sum_amr_pilot_success.pdf")

    # "Tilted": the same covariance eigenvalues as "axis-aligned", rotated.
    tilted = MODES.T @ MODES
    tilted = tilted / np.max(np.linalg.eigvalsh(tilted))
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    for covariance, color, label, line in (
            (tilted, colors["COIN_SUM_AMR"], "Tilted modes", "-"),
            (np.diag(np.linalg.eigvalsh(tilted)), colors["BASELINE"], "Axis-aligned modes", "--")):
        values, vectors = np.linalg.eigh(covariance)
        angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
        for scale, alpha in ((1.0, .95), (1.7, .72)):
            ax.add_patch(Ellipse((0, 0), 2 * scale * np.sqrt(values[1]), 2 * scale * np.sqrt(values[0]),
                                 angle=angle, fill=False, edgecolor=color, linewidth=1.25,
                                 linestyle=line, alpha=alpha, label=label if scale == 1.0 else None))
    ax.set(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), xlabel="Parameter 1", ylabel="Parameter 2", aspect="equal")
    style.legend_below(ax, ncol=2)
    style.save(fig, "coin_sum_amr_tilted_geometry.pdf")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
