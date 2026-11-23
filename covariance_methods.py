# The covariance methods, each on four grids, scored on the same 64 histories.
#
# Every method is judged by the covariance of the step it would produce, against gamma^2 Sigma_N
# for the continuous, unrounded history. Each method runs on four grids, although the figures and
# tables show classical and hybrid AMR only with --classical-amr (plot_style.CLASSICAL_AMR):
#   fixed local grid   256 cells across 0.08 around the start
#   classical AMR      a window chosen from 1,000 pilot histories, between circuits
#   quantum AMR        each branch's own refinement of the local grid, inside the circuit
#   hybrid AMR         each branch's own refinement of a classical AMR window
#
# Every method is a quantum circuit. The two proposed steps, the coin sum and the single random
# pair, output a step directly and read the history straight from the walk's own coins, so they
# need no measurement or classical computation between steps. The six readouts measure the
# covariance with finite shots, including its scale (no variance, norm or trace is supplied), and
# then hand it to an ideal Gaussian step loader. With quantum or hybrid AMR, a readout reads the
# history from the walk's refined register.
#
# Comments quote each method's step in the paper's LaTeX (see walk.py for the notation):
# step_covariances has every step, and readout has each readout's estimate \widetilde\Sigma.
# Coordinate indices k, l count from 0 in the code and from 1 in the paper.
#
#   python3.12 covariance_methods.py             # recompute (about 15 s) and plot
#   python3.12 covariance_methods.py --plot-only  # redraw from data/covariance_methods.npz

import math
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import plot_style as style
from walk import (D, GAMMA2, ISO_VAR, LOCAL_GRID, MODES, SINGLE_RANDOM_PAIR, XI0, all_histories, amr_window,
                  pilot_histories, quantum_amr_grid, random_pair_covariance)


# Outputs
DATA = Path(__file__).parent / "data" / "covariance_methods.npz"
TABLE = style.FIGURES / "covariance_method_errors.tex"

# Method key: (label, bar colour). The coin sum comes first, then the single pair and the
# readout methods from the literature; this is also the table order. The shown methods share
# plots, so each colour is a different hue, at least 30 degrees from the others: indigo, soft blue,
# mint, soft orange, olive, mauve.
METHODS = {
    "coin_sum": ("Covariance-scaled coin sum", style.COLORS["INDIGO"]),
    "cs_upper": ("Cauchy--Schwarz upper bound", style.COLORS["RUST"]),
    "cs_average": ("Cauchy--Schwarz upper/lower average", style.COLORS["SAGE"]),
    "single_pair": ("Single random pair", "#78A5EF"),
    "swap": ("SWAP test", "#6EBB90"),
    "hadamard": ("Hadamard test", "#F49567"),
    "qae": ("Amplitude estimation", "#6E7F2D"),
    "qpca": ("Covariance matrix preparation + readout", "#A0617E"),
}

# The methods in the figures and tables. The Cauchy-Schwarz readouts are still computed, saved
# and tested, but left out of them.
SHOWN = [key for key in METHODS if key not in ("cs_upper", "cs_average")]

# The grids every method runs on: the suffix of its results' key, and the grid's name
GRIDS = {"": "fixed local grid", "_classical": "classical AMR", "_quantum": "quantum AMR", "_hybrid": "hybrid AMR"}


# The grids the figures and tables show: all four with --classical-amr, otherwise the fixed local
# grid and quantum AMR
def shown_grids():
    return {suffix: name for suffix, name in GRIDS.items() if style.CLASSICAL_AMR or suffix in ("", "_quantum")}

# The readouts, in table order. Markers distinguish them; their colours come from METHODS in every plot.
READOUTS = {
    "cs_upper": "P",
    "cs_average": "*",
    "swap": "o",
    "hadamard": "s",
    "qae": "^",
    "qpca": "D",
}

# Separate measurement settings (circuits) each readout splits its budget over, for a 2 x 2
# covariance. Each readout measures the scale of its covariance as well:
#   Cauchy-Schwarz: the two variances
#   amplitude estimation: the three entries
#   SWAP and Hadamard tests: the two variances and the off-diagonal entry
#   covariance preparation: the diagonal and off-diagonal of Sigma_N / tr(Sigma_N), and the trace
SETTINGS = {"cs_upper": 2, "cs_average": 2, "swap": 3, "hadamard": 3, "qae": 3, "qpca": 3}

# Readout budgets: 384 shots in the method comparison, 12 to 6,144 in the budget sweep
SHOTS = 384
BUDGETS = np.array([12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144])


# ---------------------------------------------------------------------------
# Finite-measurement readout laws. These are the exact output distributions of the circuits
# in circuit_costs.py (test_covariance_methods.py checks this). The circuits are compiled from
# the history, but every variance, norm and trace an estimate uses is measured, not supplied.
# ---------------------------------------------------------------------------

# The products v_{ij} for every ordered pair (i, j) of history samples. Their mean is a multiple of
# the covariance entry, so no mean has to be known:
#   v_{ij} = \tfrac12 (\bm\xi^{(i)} - \bm\xi^{(j)})_k (\bm\xi^{(i)} - \bm\xi^{(j)})_l, \quad
#   \bar v = \frac{1}{N^2} \sum_{i,j=1}^{N} v_{ij} = \frac{N-1}{N} (\widehat\Sigma_N)_{kl}
def pair_products(history, k, l):
    differences = history[:, None] - history[None, :]
    return differences[..., k] * differences[..., l] / 2


# `reps` estimates of the mean \bar v of `values` from `shots` runs of the pair-product circuit.
# Its ancilla reads 1 with probability P_1 = (\bar v - \mathrm{lo}) / (\mathrm{hi} - \mathrm{lo}), with
# lo and hi the smallest and largest v_{ij}, and the estimate is
#   \mathrm{lo} + (\mathrm{hi} - \mathrm{lo}) f,
# with f the frequency of 1. With `qae`, the shots are instead oracle calls spent on one amplitude
# estimation with M phase states, and f = \sin^2(\pi y / M) from its outcome y.
def pair_product_mean(values, shots, rng, reps, qae=False):
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
    N = len(history)
    shots = budget // SETTINGS[kind]
    if shots < 1:
        raise ValueError("The budget must cover at least one shot per setting")

    # A constant history has nothing to load into the circuits; its covariance is zero
    estimates = np.zeros((reps, D, D))
    centered = history - history.mean(axis=0)
    if not np.any(centered):
        return estimates

    # Amplitude estimation: each entry from amplitude estimation of its own pair-product circuit,
    #   \widetilde\Sigma_{kl} = \frac{N}{N-1} \bigl[\mathrm{lo} + (\mathrm{hi} - \mathrm{lo}) f\bigr]
    # with f = \sin^2(\pi y / M) (pair_product_mean)
    if kind == "qae":
        for k, l in [(0, 0), (0, 1), (1, 1)]:
            mean = pair_product_mean(pair_products(history, k, l), shots, rng, reps, qae=True)
            estimates[:, k, l] = estimates[:, l, k] = mean * N / (N - 1)
        return estimates

    # Covariance matrix preparation (Gordon et al. 2022): the coordinate qubits hold
    #   \rho = \widehat\Sigma_N / \operatorname{tr} \widehat\Sigma_N,
    # and the estimate is \widetilde\Sigma = \tilde\tau \widetilde\rho
    if kind == "qpca":
        scatter = centered.T @ centered
        rho = scatter / np.trace(scatter)

        # The diagonal of rho from the coordinate basis, \widetilde\rho_{kk} = f_k, and its
        # off-diagonal from the basis (|0> +- |1>) / sqrt(2), whose + outcome has probability
        # \tfrac12 + \rho_{12}, so \widetilde\rho_{12} = f_+ - \tfrac12
        estimates[:, [0, 1], [0, 1]] = rng.multinomial(shots, np.diag(rho), size=reps) / shots
        plus = rng.binomial(shots, np.clip(.5 + rho[0, 1], 0, 1), reps) / shots
        estimates[:, 0, 1] = estimates[:, 1, 0] = plus - .5

        # The trace \tilde\tau, from a pair-product circuit on the squared distances
        #   v_{ij} = \tfrac12 \lvert \bm\xi^{(i)} - \bm\xi^{(j)} \rvert^2, \quad
        #   \tilde\tau = \frac{N}{N-1} \bigl[\mathrm{lo} + (\mathrm{hi} - \mathrm{lo}) f\bigr]
        distances = pair_products(history, 0, 0) + pair_products(history, 1, 1)
        trace = pair_product_mean(distances, shots, rng, reps) * N / (N - 1)
        return estimates * trace[:, None, None]

    # The other readouts measure the two variances \widetilde\Sigma_{kk} with pair-product circuits,
    #   \widetilde\Sigma_{kk} = \frac{N}{N-1} \bigl[\mathrm{lo} + (\mathrm{hi} - \mathrm{lo}) f\bigr]
    for k in range(D):
        estimates[:, k, k] = pair_product_mean(pair_products(history, k, k), shots, rng, reps) * N / (N - 1)
    variances = estimates[:, [0, 1], [0, 1]]

    # Cauchy-Schwarz upper bound, every correlation set to +1:
    #   \widetilde\Sigma_{12} = \sqrt{\widetilde\Sigma_{11} \widetilde\Sigma_{22}}
    # Averaging the upper and lower (-1) bounds sets the correlations to 0: \widetilde\Sigma_{12} = 0.
    if kind == "cs_upper":
        covariance = np.sqrt(variances[:, 0] * variances[:, 1])
    elif kind == "cs_average":
        covariance = 0.

    # SWAP test (Lloyd et al. 2013) on the norm-weighted centred columns \bm x and \bm y,
    #   x_i = (\bm\xi^{(i)} - \widehat{\bm\mu}_N)_1, \quad y_i = (\bm\xi^{(i)} - \widehat{\bm\mu}_N)_2.
    # It reads 0 with probability
    #   P_0 = \frac12 + \frac{\lvert \bm x - \bm y \rvert^2}{4 (\lvert \bm x \rvert^2 + \lvert \bm y \rvert^2)}
    #       = \frac34 - \frac{(\widehat\Sigma_N)_{12}}{2 \operatorname{tr} \widehat\Sigma_N},
    # so from the frequency f_0 of 0,
    #   \widetilde\Sigma_{12} = \tfrac12 (\widetilde\Sigma_{11} + \widetilde\Sigma_{22}) (3 - 4 f_0)
    elif kind == "swap":
        probability = .5 + np.sum((centered[:, 0] - centered[:, 1])**2) / (4 * np.sum(centered**2))
        frequency = rng.binomial(shots, probability, reps) / shots
        covariance = variances.sum(axis=1) * (3 - 4 * frequency) / 2

    # Hadamard test on \bm x / \lvert \bm x \rvert and \bm y / \lvert \bm y \rvert. It reads 0 with probability
    #   P_0 = \tfrac12 \Bigl(1 + \frac{\bm x \cdot \bm y}{\lvert \bm x \rvert \lvert \bm y \rvert}\Bigr),
    # so from the frequency f_0 of 0,
    #   \widetilde\Sigma_{12} = \sqrt{\widetilde\Sigma_{11} \widetilde\Sigma_{22}} \, (2 f_0 - 1)
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


# \mathrm{PSD}(\widetilde\Sigma): clip the negative eigenvalues that finite-shot noise can produce
def nearest_psd(covariances):
    values, vectors = np.linalg.eigh(.5 * (covariances + np.swapaxes(covariances, -1, -2)))
    return (vectors * np.maximum(values, 0)[..., None, :]) @ np.swapaxes(vectors, -1, -2)


# The history in grid units; the start and the modes are rounded to the grid first, as in the circuit
def grid_history(signs, grid):
    center, axes, spacing = grid
    start = np.rint((XI0 - center) @ axes / spacing)
    return start + np.cumsum(signs @ np.rint(MODES @ axes / spacing), axis=0)


# ---------------------------------------------------------------------------
# Experiment 1: every method at 384 readout shots, on every grid
# ---------------------------------------------------------------------------

# The step covariances, in grid units, that each method produces on a grid, for the branch with
# these preparatory signs: one for each step circuit and `reps` for each readout
def step_covariances(signs, grid, rng, reps=50):
    center, axes, spacing = grid
    h = grid_history(signs, grid)
    iso = ISO_VAR * np.eye(D)

    # Covariance-scaled coin sum (walk.make_walk), whose covariance is exact (test_walk.py samples
    # the circuit to check):
    #   \bm\Delta^{(N+1)} = \sum_{1 \le a < b \le N} R_{ab} (\bm\xi^{(a)} - \bm\xi^{(b)}) + \bm e,
    #   \operatorname{Cov}(\bm\Delta^{(N+1)} \mid X_N) = \gamma^2 \widehat\Sigma_N + \nu I_d
    covariances = {"coin_sum": [GAMMA2 * np.cov(h.T) + iso]}

    # Single random pair (walk.make_walk with SINGLE_RANDOM_PAIR): one uniformly chosen pair (A, B)
    # with a Rademacher sign R. A pair difference has covariance 2 \widehat\Sigma_N on average, so the
    # circuit adds the difference of the history \tilde{\bm\xi} built from the modes scaled by
    # \gamma / \sqrt2 and rounded to the grid, straight from the walk's coins:
    #   \bm\Delta^{(N+1)} = R (\tilde{\bm\xi}^{(A)} - \tilde{\bm\xi}^{(B)}) + \bm e, \quad
    #   \operatorname{Cov}(\bm\Delta^{(N+1)} \mid X_N) = 2 \widehat\Sigma(\tilde{\bm\xi}) + \nu I_d,
    # which is \gamma^2 \widehat\Sigma_N + \nu I_d without the rounding (test_walk.py samples the
    # circuit to check)
    step_modes = np.rint(SINGLE_RANDOM_PAIR[2] * MODES @ axes / spacing).astype(int)
    covariances["single_pair"] = [random_pair_covariance(signs, step_modes, SINGLE_RANDOM_PAIR)]

    # The readouts: a finite-shot estimate \widetilde\Sigma (readout), made positive semidefinite
    # (nearest_psd) and handed to the ideal Gaussian loader:
    #   \bm\Delta^{(N+1)} \sim \Normal{\bm 0, \gamma^2 \, \mathrm{PSD}(\widetilde\Sigma) + \nu I_d}
    for kind in READOUTS:
        covariances[kind] = GAMMA2 * nearest_psd(readout(h, kind, SHOTS, rng, reps)) + iso

    return covariances


# Relative step-covariance error of every method on every history and every grid, keyed by
# method + grid suffix
def compare_methods():
    rng = np.random.default_rng(20261115)
    rng_refined = np.random.default_rng(20261123)

    # 40 classical AMR windows, each chosen from 1,000 pilot histories
    pilot = pilot_histories()
    windows = [amr_window(pilot[rng.choice(len(pilot), 1000, replace=False)]) for _ in range(40)]

    # Each step covariance C, in a grid's units, is mapped to physical coordinates and scored against
    # the unrounded physical history's \widehat\Sigma_N:
    #   \frac{\lVert B C B^\top - \gamma^2 \widehat\Sigma_N \rVert_F}{\lVert \gamma^2 \widehat\Sigma_N \rVert_F},
    #   \quad B = V \operatorname{diag}(\bm\delta)
    errors = {}
    for signs, history in zip(*all_histories()):
        target = GAMMA2 * np.cov(history.T)
        norm = np.linalg.norm(target)

        # This branch's grids, each with the random stream its readouts draw from: the fixed and
        # classical AMR grids share one, and the quantum and hybrid AMR refinements of them another
        grids = ([(LOCAL_GRID, "", rng)] + [(window, "_classical", rng) for window in windows]
                 + [(quantum_amr_grid(signs, LOCAL_GRID), "_quantum", rng_refined)]
                 + [(quantum_amr_grid(signs, window), "_hybrid", rng_refined) for window in windows])

        # Round the history onto each grid, let every method produce its step there, and map
        # the step covariance back to physical coordinates
        for grid, suffix, stream in grids:
            center, axes, spacing = grid
            basis = axes @ np.diag(spacing)
            for key, covariances in step_covariances(signs, grid, stream).items():
                physical = basis @ np.asarray(covariances) @ basis.T
                errors.setdefault(key + suffix, []).extend(
                    np.linalg.norm(physical - target, axis=(1, 2)) / norm)

    return {key: np.array(values) for key, values in errors.items()}


# ---------------------------------------------------------------------------
# Experiment 2: the readouts at growing budgets, each history on its quantum AMR grid
# ---------------------------------------------------------------------------

def budget_sweep(reps=100):
    rng = np.random.default_rng(91002)
    errors = np.empty((len(READOUTS), len(BUDGETS), 64 * reps))

    for j, (signs, history) in enumerate(zip(*all_histories())):
        target = GAMMA2 * np.cov(history.T)
        grid = quantum_amr_grid(signs, LOCAL_GRID)
        basis = grid[1] @ np.diag(grid[2])
        h = grid_history(signs, grid)

        for b, budget in enumerate(BUDGETS):
            for i, kind in enumerate(READOUTS):
                estimates = nearest_psd(readout(h, kind, int(budget), rng, reps))
                covariances = basis @ (GAMMA2 * estimates + ISO_VAR * np.eye(D)) @ basis.T
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

# The grid of the main chart and table. Hybrid AMR is at most slightly better than quantum AMR for
# any method but needs a pilot run and a new circuit per window, so quantum AMR is the default
# wherever one grid is shown; only the plots that compare grids show the others.
MAIN_GRID = "_quantum"


# Mean and SD of each method's error with quantum AMR, as a LaTeX table
def write_table(errors):
    lines = [
        r"% Generated by covariance_methods.py; requires booktabs.",
        r"\begin{tabular}{@{}p{0.48\textwidth}rrr@{}}",
        r"\toprule",
        r"Method, with quantum AMR & Samples & Mean error (\%) & SD (\%) \\",
        r"\midrule",
    ]

    for key in SHOWN:
        values = 100 * errors[key + MAIN_GRID]
        lines.append(f"{METHODS[key][0]} & {len(values):,} & {values.mean():.2f} & {values.std(ddof=1):.2f} "
                     + r"\\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    style.FIGURES.mkdir(exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n")


# Horizontal bars of each method's mean error with quantum AMR, relative to the coin sum's
def plot_methods(errors):
    style.apply(font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    reference = errors["coin_sum" + MAIN_GRID].mean()
    right = 0.

    # One bar per method, with its mean +- SD marker. A lower whisker below zero is clipped there.
    for y, key in enumerate(SHOWN):
        color = METHODS[key][1]
        values = errors[key + MAIN_GRID] / reference
        mean, sd = values.mean(), values.std(ddof=1)
        right = max(right, mean + sd)

        ax.barh(y, mean, height=.7, color=color, alpha=style.BAR_ALPHA, edgecolor=color, linewidth=.8, zorder=2)
        ax.errorbar(mean, y, xerr=sd, fmt="o", markersize=3., markerfacecolor="white",
                    markeredgecolor=style.INK, markeredgewidth=.8, ecolor=style.INK,
                    elinewidth=.9, capsize=2.5, zorder=4)

    # Axes, with the method names beside them so that every name fits, and legend
    ax.set_yticks(range(len(SHOWN)), [METHODS[key][0] for key in SHOWN])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(SHOWN) - .5, -.5)
    ax.set_xlim(0, 1.04 * right)
    ax.tick_params(axis="x", direction="out", length=3, color=style.INK)
    ax.set_xlabel("Covariance error relative to the coin sum")
    style.legend_below(ax, title="All with quantum AMR", handles=[
        Line2D([0], [0], color=style.INK, marker="o", markerfacecolor="white", markersize=4,
               label=r"Mean $\pm$ 1 SD")])

    style.save(fig, "coin_sum_vs_covariance_methods.pdf")


# Every method on every shown grid: means and one sample SD, each bar named in small text. A lower
# whisker below zero is clipped there.
def plot_all_grids(errors):
    grids = shown_grids()
    style.apply(font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8.0, 3. + 1.5 * len(grids)))
    alphas = {"": .15, "_classical": .35, "_quantum": .6, "_hybrid": .85}
    y, ticks, right = 0., [], 0.

    for key in SHOWN:
        color = METHODS[key][1]
        ticks.append(y + .36 * (len(grids) - 1) / 2)

        # Each grid's bar, its mean +- SD marker, and the grid's name after the marker
        for suffix, name in grids.items():
            values = 100 * errors[key + suffix]
            mean, sd = values.mean(), values.std(ddof=1)
            right = max(right, mean + sd)
            ax.barh(y, mean, height=.3, color=color, alpha=alphas[suffix], edgecolor=color, linewidth=.8)
            ax.errorbar(mean, y, xerr=sd, fmt="o", markersize=3.,
                        markerfacecolor="white", markeredgecolor=style.INK, markeredgewidth=.8,
                        ecolor=style.INK, elinewidth=.9, capsize=2.5)
            ax.annotate(name[0].upper() + name[1:], (mean + sd, y), xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8.5, color=style.MUTED)
            y += .36

        # A gap before the next method
        y += .4

    # Axes and legend
    ax.set_yticks(ticks, [METHODS[key][0] for key in SHOWN])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - .4 - .36 + .3, -.3)
    ax.set_xlim(0, 1.3 * right)
    ax.set_xlabel(r"Next-step covariance error (\%)")
    style.legend_below(ax, handles=[Line2D([0], [0], color=style.INK, marker="o", markerfacecolor="white",
                                           markersize=4, label=r"Mean $\pm$ 1 SD")])

    style.save(fig, "covariance_methods_amr.pdf")


# Readout error against budget, and the coin sum and the single pair, all on quantum AMR grids;
# mean +/- one sample SD throughout
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

    # The coin sum and the single random pair output a step, scored by its exact covariance, so
    # their error does not depend on the readout budget. A band below the axis is clipped at 1%.
    for key in ("coin_sum", "single_pair"):
        label, color = METHODS[key]
        values = 100 * errors[key + MAIN_GRID]
        mean, sd = values.mean(), values.std(ddof=1)
        ax.axhspan(mean - sd, mean + sd, color=color, alpha=.09, lw=0)
        ax.axhline(mean, color=color, lw=1.8, label=label)

    # Axes and legend
    ax.set(xscale="log", yscale="log", xlabel="Readout shots or amplitude-estimation oracle calls",
           ylabel=r"Next-step covariance error (\%; mean $\pm$ 1 SD)")
    ax.set_yticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    ax.set_ylim(bottom=1)
    style.legend_below(ax, ncol=2, title="All with quantum AMR")

    style.save(fig, "coin_sum_step_generation.pdf")


def plot():
    with np.load(DATA) as saved:
        errors = {key: saved[key] for key in saved.files if key != "budget_errors"}
        budget_errors = saved["budget_errors"]

    write_table(errors)
    plot_methods(errors)
    plot_all_grids(errors)
    plot_budgets(errors, budget_errors)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
