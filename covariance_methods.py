# Eleven ways to produce the next step's covariance, scored on the same 64 histories.
#
# Every method is judged by the covariance of the step it would produce, against
# SCALE2 * cov(history) for the continuous, unrounded history.
#
# Every method is a quantum circuit. The coin sums and the single pair output a step directly.
# The seven readouts measure the covariance with finite shots, including its scale (no variance,
# norm or trace is supplied), and then hand it to an ideal Gaussian step loader.
#
#   python3.12 covariance_methods.py              # recompute (about 20 s) and plot
#   python3.12 covariance_methods.py --plot-only  # redraw from data/covariance_methods.npz

import math
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_style as style
from walk import (D, ISO_VAR, MODES, PAIRS, SCALE2, X0, all_histories, amr_window, pilot_histories,
                  quantum_amr_step_covariance)


# Outputs
DATA = Path(__file__).parent / "data" / "covariance_methods.npz"
TABLE = style.FIGURES / "covariance_method_errors.tex"

# Method key: (label, bar colour). The methods proposed in this work come first, then the
# readout methods from the literature; this is also the table order. Bar colours extend the
# muted palette of amr.py's bar charts.
METHODS = {
    "coin_sum_qamr": ("Covariance-scaled coin sum + quantum AMR", style.COLORS["COIN_SUM_QAMR"]),
    "coin_sum_amr": ("Covariance-scaled coin sum + classical AMR", style.COLORS["COIN_SUM_AMR"]),
    "coin_sum": ("Covariance-scaled coin sum, fixed local grid", style.COLORS["COIN_SUM"]),
    "cs_upper": ("Cauchy--Schwarz upper bound", style.COLORS["MEASURED"]),
    "cs_average": ("Cauchy--Schwarz upper/lower average", style.COLORS["BASELINE"]),
    "single_pair": ("Single history-pair difference", "#381058"),
    "pairwise": ("Pairwise moments", "#2F7F7A"),
    "swap": ("SWAP test", "#C19A2B"),
    "hadamard": ("Hadamard test", "#547F9E"),
    "qae": ("Amplitude estimation", "#6E7F2D"),
    "qpca": ("Covariance matrix preparation + readout", "#A0617E"),
}

# The methods in the figures and tables. The Cauchy-Schwarz readouts are still computed, saved
# and tested, but left out of them.
SHOWN = [key for key in METHODS if key not in ("cs_upper", "cs_average")]

# The readouts, in table order. Markers distinguish them; their colours come from METHODS in every plot.
READOUTS = {
    "cs_upper": "P",
    "cs_average": "*",
    "pairwise": "v",
    "swap": "o",
    "hadamard": "s",
    "qae": "^",
    "qpca": "D",
}

# Separate measurement settings (circuits) each readout splits its budget over, for a 2 x 2
# covariance. Each readout measures the scale of its covariance as well:
#   Cauchy-Schwarz: the two variances
#   pairwise moments and amplitude estimation: the three entries
#   SWAP and Hadamard tests: the two variances and the off-diagonal entry
#   covariance preparation: the diagonal and off-diagonal of S / tr(S), and the trace
SETTINGS = {"cs_upper": 2, "cs_average": 2, "pairwise": 3, "swap": 3, "hadamard": 3, "qae": 3, "qpca": 3}

# Readout budgets: 384 shots in the method comparison, 12 to 6,144 in the budget sweep
SHOTS = 384
BUDGETS = np.array([12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144])

# The fixed local grid: 256 cells across 0.08 around the start
LOCAL_GRID = (X0, np.eye(D), np.full(D, .08 / 256))


# ---------------------------------------------------------------------------
# Finite-measurement readout laws. These are the exact output distributions of the circuits
# in circuit_costs.py (test_covariance_methods.py checks this). The circuits are compiled from
# the history, but every variance, norm and trace an estimate uses is measured, not supplied.
# ---------------------------------------------------------------------------

# The products (h_i - h_j)_p (h_i - h_j)_q / 2 for every ordered pair of history positions.
# Their mean is (k - 1) / k times the covariance entry S_pq, so no mean has to be known.
def pair_products(history, p, q):
    differences = history[:, None] - history[None, :]
    return differences[..., p] * differences[..., q] / 2


# `reps` estimates of the mean of `values` from `shots` runs of the pairwise-moment circuit, whose
# ancilla reads 1 with probability (mean - lo) / (hi - lo). With `qae`, the shots are instead
# oracle calls spent on one amplitude estimation.
def pairwise_mean(values, shots, rng, reps, qae=False):
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.full(reps, lo)
    probability = (values.mean() - lo) / (hi - lo)

    if qae:
        frequency = amplitude_estimates(probability, phase_states(shots), rng, reps)
    else:
        frequency = rng.binomial(shots, probability, reps) / shots
    return lo + (hi - lo) * frequency


# `reps` covariance estimates of shape (D, D) from `budget` shots (or, for amplitude
# estimation, oracle calls) split evenly over the method's settings
def readout(history, kind, budget, rng, reps):
    k = len(history)
    shots = budget // SETTINGS[kind]
    if shots < 1:
        raise ValueError("The budget must cover at least one shot per setting")

    # A constant history has nothing to load into the circuits; its covariance is zero
    estimates = np.zeros((reps, D, D))
    centered = history - history.mean(axis=0)
    if not np.any(centered):
        return estimates

    # Pairwise moments and amplitude estimation: each entry from its own pairwise-moment circuit
    if kind in ("pairwise", "qae"):
        for p, q in [(0, 0), (0, 1), (1, 1)]:
            mean = pairwise_mean(pair_products(history, p, q), shots, rng, reps, qae=kind == "qae")
            estimates[:, p, q] = estimates[:, q, p] = mean * k / (k - 1)
        return estimates

    # Covariance matrix preparation (Gordon et al. 2022): the coordinate qubits hold rho = S / tr(S)
    if kind == "qpca":
        scatter = centered.T @ centered
        rho = scatter / np.trace(scatter)

        # The diagonal of rho from the coordinate basis, and its off-diagonal from the basis
        # (|0> +- |1>) / sqrt(2), whose + outcome has probability 1/2 + rho_01
        estimates[:, [0, 1], [0, 1]] = rng.multinomial(shots, np.diag(rho), size=reps) / shots
        plus = rng.binomial(shots, np.clip(.5 + rho[0, 1], 0, 1), reps) / shots
        estimates[:, 0, 1] = estimates[:, 1, 0] = plus - .5

        # The trace, from a pairwise-moment circuit on the squared distances |h_i - h_j|^2 / 2
        distances = pair_products(history, 0, 0) + pair_products(history, 1, 1)
        trace = pairwise_mean(distances, shots, rng, reps) * k / (k - 1)
        return estimates * trace[:, None, None]

    # The other readouts measure the two variances with pairwise-moment circuits
    for p in range(D):
        estimates[:, p, p] = pairwise_mean(pair_products(history, p, p), shots, rng, reps) * k / (k - 1)
    variances = estimates[:, [0, 1], [0, 1]]

    # Cauchy-Schwarz upper bound: every correlation set to +1. Averaging the upper and
    # lower (-1) bounds sets the correlations to 0.
    if kind == "cs_upper":
        covariance = np.sqrt(variances[:, 0] * variances[:, 1])
    elif kind == "cs_average":
        covariance = 0.

    # SWAP test (Lloyd et al. 2013) on the norm-weighted centred columns x and y. It reads 0
    # with probability 1/2 + |x - y|^2 / (4 (|x|^2 + |y|^2)) = 3/4 - S_01 / (2 tr S).
    elif kind == "swap":
        probability = .5 + np.sum((centered[:, 0] - centered[:, 1])**2) / (4 * np.sum(centered**2))
        frequency = rng.binomial(shots, probability, reps) / shots
        covariance = variances.sum(axis=1) * (3 - 4 * frequency) / 2

    # Hadamard test on x/|x| and y/|y|. It reads 0 with probability (1 + correlation) / 2.
    else:
        norms = np.linalg.norm(centered, axis=0)
        correlation = centered[:, 0] @ centered[:, 1] / (norms[0] * norms[1]) if norms.all() else 0.
        frequency = rng.binomial(shots, np.clip((1 + correlation) / 2, 0, 1), reps) / shots
        covariance = np.sqrt(variances[:, 0] * variances[:, 1]) * (2 * frequency - 1)

    estimates[:, 0, 1] = estimates[:, 1, 0] = covariance
    return estimates


# Largest power of two M whose amplitude estimation, 2 (M - 1) + 1 calls to the
# preparation oracle or its inverse, fits the budget
def phase_states(oracle_calls):
    return 1 << (((oracle_calls + 1) // 2).bit_length() - 1)


# Outcomes of canonical amplitude estimation (Brassard et al. 2002) with M phase states
def amplitude_estimates(probability, phase_states, rng, size):
    # The phase register reads y with the standard probability
    theta = math.asin(math.sqrt(probability))
    y = np.arange(phase_states)
    weights = np.zeros(phase_states)
    for sign in (1, -1):
        delta = np.pi * y / phase_states - sign * theta
        sine = np.sin(delta)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights += .5 * np.where(np.abs(sine) < 1e-12, 1.,
                                     np.sin(phase_states * delta)**2 / (phase_states**2 * sine**2))
    weights /= weights.sum()

    # ...and the estimate is sin^2(pi y / M)
    return np.sin(np.pi * rng.choice(phase_states, size=size, p=weights) / phase_states) ** 2


# Clip the negative eigenvalues that finite-shot noise can produce
def nearest_psd(covariances):
    values, vectors = np.linalg.eigh(.5 * (covariances + np.swapaxes(covariances, -1, -2)))
    return (vectors * np.maximum(values, 0)[..., None, :]) @ np.swapaxes(vectors, -1, -2)


# The history in grid units; the start and the modes are rounded to the grid first, as in the circuit
def grid_history(signs, grid):
    center, axes, spacing = grid
    start = np.rint((X0 - center) @ axes / spacing)
    return start + np.cumsum(signs @ np.rint(MODES @ axes / spacing), axis=0)


# ---------------------------------------------------------------------------
# Experiment 1: every method at 384 readout shots, on the fixed grid and with AMR
# ---------------------------------------------------------------------------

# The step covariances, in grid units, that each method produces from the grid history h:
# one for each step circuit and `reps` for each readout
def step_covariances(h, rng, reps=50):
    iso = ISO_VAR * np.eye(D)

    # The coin sum's step is sum_{a<b} r_ab (h_a - h_b) plus the +-1 step, with Var(r_ab) = q,
    # so its covariance is exactly SCALE2 * S_h + iso (test_walk.py samples the circuit to check)
    covariances = {"coin_sum": [SCALE2 * np.cov(h.T) + iso]}

    # The single-pair step is one uniformly chosen signed pair difference, scaled by
    # sqrt(SCALE2 / 2) because a pair difference has covariance 2 S on average, and rounded
    pair_steps = np.rint(np.sqrt(SCALE2 / 2) * np.array([h[a] - h[b] for a, b in PAIRS]))
    covariances["single_pair"] = [pair_steps.T @ pair_steps / len(PAIRS) + iso]

    # The readouts, each followed by the ideal Gaussian loader
    for kind in READOUTS:
        covariances[kind] = SCALE2 * nearest_psd(readout(h, kind, SHOTS, rng, reps)) + iso

    return covariances


# Relative step-covariance error of every method on every history: on the fixed local
# grid (key) and on each of 40 AMR windows (key + "_amr")
def compare_methods():
    rng = np.random.default_rng(20261115)

    # 40 AMR windows, each chosen from 1,000 pilot histories
    pilot = pilot_histories()
    windows = [amr_window(pilot[rng.choice(len(pilot), 1000, replace=False)]) for _ in range(40)]
    grids = [(LOCAL_GRID, "")] + [(window, "_amr") for window in windows]

    errors = {}
    for signs, history in zip(*all_histories()):
        target = SCALE2 * np.cov(history.T)
        norm = np.linalg.norm(target)

        # Round the history onto each grid, let every method produce its step there, and map
        # the step covariance back to physical coordinates
        for grid, suffix in grids:
            center, axes, spacing = grid
            basis = axes @ np.diag(spacing)
            for key, covariances in step_covariances(grid_history(signs, grid), rng).items():
                physical = basis @ np.asarray(covariances) @ basis.T
                errors.setdefault(key + suffix, []).extend(
                    np.linalg.norm(physical - target, axis=(1, 2)) / norm)

        # Quantum AMR: the coin sum on the branch's own refinement of the fixed local grid,
        # chosen inside the circuit
        physical = quantum_amr_step_covariance(signs, MODES, LOCAL_GRID[2])
        errors.setdefault("coin_sum_qamr", []).append(np.linalg.norm(physical - target) / norm)

    return {key: np.array(values) for key, values in errors.items()}


# ---------------------------------------------------------------------------
# Experiment 2: the readouts at growing budgets, on the fixed local grid
# ---------------------------------------------------------------------------

def budget_sweep(reps=100):
    rng = np.random.default_rng(91002)
    errors = np.empty((len(READOUTS), len(BUDGETS), 64 * reps))
    spacing = LOCAL_GRID[2][0]

    for j, (signs, history) in enumerate(zip(*all_histories())):
        target = SCALE2 * np.cov(history.T)
        h = grid_history(signs, LOCAL_GRID)

        for b, budget in enumerate(BUDGETS):
            for i, kind in enumerate(READOUTS):
                estimates = nearest_psd(readout(h, kind, int(budget), rng, reps))
                covariances = spacing**2 * (SCALE2 * estimates + ISO_VAR * np.eye(D))
                errors[i, b, j * reps:(j + 1) * reps] = (
                    np.linalg.norm(covariances - target, axis=(1, 2)) / np.linalg.norm(target))

    return errors


def run():
    errors = compare_methods()
    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, budget_errors=budget_sweep(), **errors)


# ---------------------------------------------------------------------------
# Table and figures
# ---------------------------------------------------------------------------

# Mean and SD of every method's error, as a LaTeX table
def write_table(errors):
    lines = [
        r"% Generated by covariance_methods.py; requires booktabs.",
        r"\begin{tabular}{@{}p{0.48\textwidth}rrr@{}}",
        r"\toprule",
        r"Method & $N$ & Mean error (\%) & SD (\%) \\",
        r"\midrule",
    ]

    for key in SHOWN:
        label, values = METHODS[key][0], 100 * errors[key]
        lines.append(f"{label} & {len(values):,} & {values.mean():.2f} & {values.std(ddof=1):.2f} " + r"\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\footnotesize All final step covariances are scored against the same continuous-history target. "
        rf"Readout methods use a budget of {SHOTS} shots, or calls to $A$ and $A^{{-1}}$ for amplitude estimation, "
        r"split over their circuits, and the same ideal Gaussian loader. "
        r"No variance, norm or trace is supplied: SWAP and Hadamard measure the variances "
        r"with pairwise-moment circuits, and covariance preparation measures its trace with one. "
        r"Covariance preparation includes finite-shot state readout added for this comparison. "
        r"Quantum AMR refines the fixed local grid by a power of two per coordinate inside the circuit, "
        r"from the number of preparatory steps whose two coins disagree; it uses no pilot and no measurement. "
        r"SD is one sample standard deviation, not an uncertainty on the mean.\par",
    ]

    style.FIGURES.mkdir(exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n")


# Horizontal bars of every method's mean error, relative to the coin sum + classical AMR
def plot_methods(errors):
    style.apply(font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    reference = errors["coin_sum_amr"].mean()
    left, right = .1, 0.

    # One bar per method, with its mean +- SD marker
    for y, key in enumerate(SHOWN):
        label, color = METHODS[key]
        values = errors[key] / reference
        mean, sd = values.mean(), values.std(ddof=1)
        right = max(right, mean + sd)

        ax.barh(y, mean, height=.78, color=color, alpha=style.BAR_ALPHA, edgecolor=color, linewidth=.8, zorder=2)
        ax.errorbar(mean, y + .29, xerr=sd, fmt="o", markersize=3., markerfacecolor="white",
                    markeredgecolor=style.INK, markeredgewidth=.8, ecolor=style.INK,
                    elinewidth=.9, capsize=2.5, zorder=4)

        # The label inside the bar, or after the error bar if the bar is shorter than the reference
        if mean >= 1:
            ax.text(math.sqrt(left * mean), y, label, ha="center", va="center", color=style.INK, zorder=5)
        else:
            ax.text(1.15 * (mean + sd), y, label, ha="left", va="center", color=style.INK, zorder=5)

    # Axes and legend
    ax.set_yticks([])
    ax.set_ylim(len(SHOWN) - .5, -.5)
    ax.set_xscale("log")
    ax.set_xlim(left, max(10., 5 * np.ceil(1.15 * right / 5)))
    ax.tick_params(axis="x", direction="out", length=3, color=style.INK)
    ax.set_xlabel("Covariance error relative to covariance-scaled coin sum + classical AMR")
    style.legend_below(ax, handles=[Line2D([0], [0], color=style.INK, marker="o", markerfacecolor="white",
                                           markersize=4, label=r"Mean $\pm$ 1 SD")])

    style.save(fig, "coin_sum_vs_covariance_methods.pdf")


# Every method with and without classical AMR: means and one sample SD on both grids.
def plot_methods_amr(errors):
    style.apply(font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    keys = [key for key in SHOWN if key not in ("coin_sum_amr", "coin_sum_qamr")]

    for y, key in enumerate(keys):
        label, color = METHODS[key]
        if key == "coin_sum":
            color = style.COLORS["COIN_SUM_AMR"]

        # Fixed grid above, classical AMR below
        for offset, suffix, alpha in ((-.2, "", .18), (.2, "_amr", .6)):
            values = 100 * errors[key + suffix]
            mean, sd = values.mean(), values.std(ddof=1)
            ax.barh(y + offset, mean, height=.38, color=color, alpha=alpha, edgecolor=color, linewidth=.8)
            ax.errorbar(mean, y + offset, xerr=sd, fmt="o", markersize=3.,
                        markerfacecolor="white", markeredgecolor=style.INK, markeredgewidth=.8,
                        ecolor=style.INK, elinewidth=.9, capsize=2.5)

    # Axes and legend
    ax.set_yticks(range(len(keys)), [METHODS[key][0].replace(", fixed local grid", "") for key in keys])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(keys) - .5, -.5)
    ax.set_xscale("log")
    ax.set_xlim(.5, None)
    ax.set_xlabel(r"Next-step covariance error (\%)")
    style.legend_below(ax, [Patch(facecolor=style.MUTED, alpha=.18, edgecolor=style.MUTED),
                            Patch(facecolor=style.MUTED, alpha=.6, edgecolor=style.MUTED),
                            Line2D([0], [0], color=style.INK, marker="o", markerfacecolor="white", markersize=4)],
                       ["Fixed local grid", "Classical AMR grid", r"Mean $\pm$ 1 SD"], ncol=3)

    style.save(fig, "covariance_methods_amr.pdf")


# Readout error against budget, with mean +/- one sample SD throughout.
def plot_budgets(errors, budget_errors):
    style.apply()
    fig, ax = plt.subplots(figsize=(7., 4.6))

    # The readouts, spread slightly sideways so their bars do not overlap
    readouts = [(kind, values) for kind, values in zip(READOUTS, 100 * budget_errors) if kind in SHOWN]
    middle = (len(readouts) - 1) / 2
    for i, (kind, values) in enumerate(readouts):
        label, color = METHODS[kind]
        mean, sd = values.mean(axis=1), values.std(axis=1, ddof=1)
        ax.errorbar(BUDGETS * 1.06 ** (i - middle), mean, yerr=sd,
                    color=color, marker=READOUTS[kind], ms=5, lw=1.4, capsize=2.5, label=label)

    # The three coin sums
    for key in ("coin_sum", "coin_sum_amr", "coin_sum_qamr"):
        label, color = METHODS[key]
        values = 100 * errors[key]
        mean, sd = values.mean(), values.std(ddof=1)
        ax.axhspan(mean - sd, mean + sd, color=color, alpha=.09, lw=0)
        ax.axhline(mean, color=color, lw=1.8, label=label)

    # Axes and legend
    ax.set(xscale="log", yscale="log", xlabel="Readout shots or amplitude-estimation oracle calls",
           ylabel=r"Next-step covariance error (\%; mean $\pm$ 1 SD)")
    ax.set_yticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    style.legend_below(ax, ncol=2)

    style.save(fig, "coin_sum_step_generation.pdf")


def plot():
    with np.load(DATA) as saved:
        errors = {key: saved[key] for key in saved.files if key != "budget_errors"}
        budget_errors = saved["budget_errors"]

    write_table(errors)
    plot_methods(errors)
    plot_methods_amr(errors)
    plot_budgets(errors, budget_errors)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
