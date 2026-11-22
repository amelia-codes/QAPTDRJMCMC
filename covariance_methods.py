"""Ten ways to produce the next step's covariance, scored on the same 64 histories.

Every method is judged by the covariance of the step it would produce,
compared with SCALE2 * cov(history) for the continuous, unrounded history.

The covariance-scaled coin sum produces the step directly. The readout
methods instead estimate the history covariance from finitely many
measurements and are then handed an ideal Gaussian step loader. Their
measurement laws below are the exact output distributions of the circuits in
circuit_costs.py (test_covariance_methods.py checks this). The Cauchy-Schwarz
and single-pair methods are classical approximations.

Two experiments:
1. All ten methods, readouts with 384 shots (coin_sum_vs_covariance_methods.pdf
   and the table covariance_method_errors.tex).
2. The five readout methods at budgets from 12 to 6,144 shots
   (coin_sum_step_generation.pdf).

    python3.12 covariance_methods.py              # recompute (about 20 s) and plot
    python3.12 covariance_methods.py --plot-only  # redraw from data/covariance_methods.npz
"""

import math
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_style as style
from walk import D, ISO_VAR, MODES, PAIRS, SCALE2, X0, all_histories, amr_window, pilot_histories

DATA = Path(__file__).parent / "data" / "covariance_methods.npz"
TABLE = style.FIGURES / "covariance_method_errors.tex"
# Method key: (label, colour). The methods proposed in this work come first,
# then the readout methods from the literature. This is also the table order.
METHODS = {
    "coin_sum_amr": ("Covariance-scaled coin sum + AMR", style.COLORS["COIN_SUM_AMR"]),
    "coin_sum_fixed": ("Covariance-scaled coin sum, fixed local grid", style.COLORS["COIN_SUM"]),
    "cs_upper": ("Cauchy--Schwarz upper bound", "#6250D6"),
    "cs_average": ("Cauchy--Schwarz upper/lower average", "#E87BA4"),
    "single_pair": ("Single history-pair difference", "#B07A00"),
    "pairwise": ("Pairwise moments", "#C2477F"),
    "swap": ("SWAP test", "#1C5CAB"),
    "hadamard": ("Hadamard test", "#8A6FE0"),
    "qae": ("Amplitude estimation", "#008300"),
    "qpca": ("Covariance matrix preparation + readout", "#1BAF7A"),
}
READOUTS = ["swap", "hadamard", "qae", "qpca", "pairwise"]
# Separate measurement settings (circuits) each readout needs for a 2 x 2
# covariance. SWAP and Hadamard read the one off-diagonal entry (variances are
# supplied); covariance preparation reads the diagonal and one interference
# basis; pairwise moments and amplitude estimation read all three entries.
SETTINGS = {"swap": 1, "hadamard": 1, "qpca": 2, "pairwise": 3, "qae": 3}
SHOTS = 384                     # readout budget in experiment 1
BUDGETS = np.array([12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144])
LOCAL_GRID = (X0, np.eye(D), np.full(D, .08 / 256))


# Finite-measurement readout laws

def readout(history, kind, budget, rng, reps):
    """`reps` covariance estimates of shape (D, D) from `budget` shots (or, for
    amplitude estimation, oracle calls) split evenly over the settings."""
    centered = history - history.mean(axis=0)
    k = len(history)
    shots = budget // SETTINGS[kind]
    estimates = np.zeros((reps, D, D))
    if kind in ("swap", "hadamard"):
        norms2 = np.sum(centered**2, axis=0)
        estimates[:, [0, 1], [0, 1]] = norms2
        if kind == "swap":
            # SWAP test on (x, -y) / |(x, y)|: P(0) = 1/2 + |x - y|^2 / (4 (|x|^2 + |y|^2)).
            z = norms2.sum()
            probability = .5 + np.sum((centered[:, 0] - centered[:, 1])**2) / (4 * z)
            frequency = rng.binomial(shots, probability, reps) / shots
            value = (z - 4 * z * (frequency - .5)) / 2
        else:
            # Hadamard test on x/|x| and y/|y|: P(0) = (1 + x.y / (|x| |y|)) / 2.
            norm_product = np.sqrt(norms2[0] * norms2[1])
            probability = .5 * (1 + centered[:, 0] @ centered[:, 1] / norm_product)
            frequency = rng.binomial(shots, probability, reps) / shots
            value = norm_product * (2 * frequency - 1)
        estimates[:, 0, 1] = estimates[:, 1, 0] = value
        return estimates / (k - 1)
    if kind == "qpca":
        # The prepared state is rho = S / tr(S); the trace is supplied. Measure
        # the coordinate basis, then (|0> +- |1>)/sqrt(2), whose outcomes have
        # probabilities 1/2 +- rho_01. (The third outcome, "neither", has
        # probability zero in two dimensions; keeping it keeps the saved draws.)
        scatter = centered.T @ centered
        trace = np.trace(scatter)
        rho = scatter / trace
        estimates[:, [0, 1], [0, 1]] = rng.multinomial(shots, np.diag(rho), size=reps) / shots
        mass = rho[0, 0] + rho[1, 1]
        probabilities = np.clip([(mass + 2 * rho[0, 1]) / 2, (mass - 2 * rho[0, 1]) / 2, 1 - mass], 0, 1)
        counts = rng.multinomial(shots, probabilities / probabilities.sum(), size=reps)
        estimates[:, 0, 1] = estimates[:, 1, 0] = (counts[:, 0] - counts[:, 1]) / (2 * shots)
        return estimates * trace / (k - 1)
    # Pairwise moments: an ancilla is 1 with probability equal to the mean of
    # the rescaled products (x_i - x_j)(y_i - y_j) / 2 over all index pairs.
    for p, q in [(0, 0), (0, 1), (1, 1)]:
        values = (history[:, None, p] - history[None, :, p]) * (history[:, None, q] - history[None, :, q]) / 2
        values = values.ravel()
        lo, hi = values.min(), values.max()
        probability = (values.mean() - lo) / (hi - lo)
        if kind == "pairwise":
            frequency = rng.binomial(shots, probability, reps) / shots
        else:
            frequency = amplitude_estimates(probability, phase_states(shots), rng, reps)
        estimates[:, p, q] = estimates[:, q, p] = lo + (hi - lo) * frequency
    return estimates * k / (k - 1)


def phase_states(oracle_calls):
    """Largest power of two M whose amplitude estimation, 2 (M - 1) + 1 calls to
    the preparation oracle or its inverse, fits the budget."""
    return 1 << (((oracle_calls + 1) // 2).bit_length() - 1)


def amplitude_estimates(probability, phase_states, rng, size):
    """Outcomes of canonical amplitude estimation (Brassard et al. 2002) with M
    phase states: the phase register reads y with the standard probability and
    the estimate is sin^2(pi y / M)."""
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
    return np.sin(np.pi * rng.choice(phase_states, size=size, p=weights) / phase_states) ** 2


def nearest_psd(covariances):
    """Clip the negative eigenvalues that finite-shot noise can produce."""
    values, vectors = np.linalg.eigh(.5 * (covariances + np.swapaxes(covariances, -1, -2)))
    return (vectors * np.maximum(values, 0)[..., None, :]) @ np.swapaxes(vectors, -1, -2)


def grid_history(signs, grid):
    """The history in grid units. The start and the modes are rounded to the
    grid first, exactly as in the circuit."""
    center, axes, spacing = grid
    start = np.rint((X0 - center) @ axes / spacing)
    return start + np.cumsum(signs @ np.rint(MODES @ axes / spacing), axis=0)


# Experiment 1: every method at 384 readout shots

def compare_methods():
    """Relative step-covariance error of every method on every history."""
    rng = np.random.default_rng(20261115)
    # Hadamard and covariance preparation were added later; their own random
    # stream leaves the other methods' draws, and so the saved results, unchanged.
    rng_later = np.random.default_rng(20261122)
    pilot = pilot_histories()
    windows = [amr_window(pilot[rng.choice(len(pilot), 1000, replace=False)]) for _ in range(40)]
    errors = {key: [] for key in METHODS}
    identity = np.eye(D)
    basis = np.diag(LOCAL_GRID[2])
    for signs, history in zip(*all_histories()):
        target = SCALE2 * np.cov(history.T)
        norm = np.linalg.norm(target)
        h = grid_history(signs, LOCAL_GRID)
        S = np.cov(h.T)
        # Single pair: one random signed pair difference, scaled by
        # sqrt(SCALE2 / 2) because a pair difference has covariance 2 S on average.
        pair_steps = np.rint(np.sqrt(SCALE2 / 2) * np.array([h[a] - h[b] for a, b in PAIRS]))
        # Cauchy-Schwarz: every correlation set to +1 (upper bound). The
        # average of the upper and lower (-1) bounds sets correlations to 0.
        upper = np.outer(np.sqrt(np.diag(S)), np.sqrt(np.diag(S)))
        np.fill_diagonal(upper, np.diag(S))
        grid_covariances = {
            "coin_sum_fixed": SCALE2 * S + ISO_VAR * identity,
            "single_pair": pair_steps.T @ pair_steps / len(PAIRS) + ISO_VAR * identity,
            "cs_upper": SCALE2 * upper + ISO_VAR * identity,
            "cs_average": SCALE2 * np.diag(np.diag(S)) + ISO_VAR * identity,
        }
        for key, covariance in grid_covariances.items():
            errors[key].append(np.linalg.norm(basis @ covariance @ basis.T - target) / norm)

        for window in windows:
            center, axes, spacing = window
            amr_basis = axes @ np.diag(spacing)
            covariance = SCALE2 * np.cov(grid_history(signs, window).T) + ISO_VAR * identity
            errors["coin_sum_amr"].append(np.linalg.norm(amr_basis @ covariance @ amr_basis.T - target) / norm)

        for kind in ("pairwise", "swap", "qae", "hadamard", "qpca"):
            stream = rng_later if kind in ("hadamard", "qpca") else rng
            for estimate in nearest_psd(readout(h, kind, SHOTS, stream, 50)):
                covariance = SCALE2 * estimate + ISO_VAR * identity
                errors[kind].append(np.linalg.norm(basis @ covariance @ basis.T - target) / norm)
    return {key: np.array(values) for key, values in errors.items()}


# Experiment 2: readout methods at growing budgets

def budget_sweep(reps=100):
    rng, rng_later = np.random.default_rng(91002), np.random.default_rng(91003)
    errors = np.empty((len(READOUTS), len(BUDGETS), 64 * reps))
    spacing = LOCAL_GRID[2][0]
    for j, (signs, history) in enumerate(zip(*all_histories())):
        target = SCALE2 * np.cov(history.T)
        h = grid_history(signs, LOCAL_GRID)
        for b, budget in enumerate(BUDGETS):
            for kind in ("pairwise", "swap", "hadamard", "qae", "qpca"):
                stream = rng if kind in ("pairwise", "swap") else rng_later
                estimates = nearest_psd(readout(h, kind, int(budget), stream, reps))
                covariances = spacing**2 * (SCALE2 * estimates + ISO_VAR * np.eye(D))
                errors[READOUTS.index(kind), b, j * reps:(j + 1) * reps] = (
                    np.linalg.norm(covariances - target, axis=(1, 2)) / np.linalg.norm(target))
    return errors


def run():
    errors = compare_methods()
    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, budget_errors=budget_sweep(), **errors)


# Table and figures

def write_table(errors):
    lines = [
        r"% Generated by covariance_methods.py; requires booktabs.",
        r"\begin{tabular}{@{}p{0.48\textwidth}rrr@{}}",
        r"\toprule",
        r"Method & $N$ & Mean error (\%) & SD (\%) \\",
        r"\midrule",
    ]
    for key, (label, _) in METHODS.items():
        values = 100 * errors[key]
        lines.append(f"{label} & {len(values):,} & {values.mean():.2f} & {values.std(ddof=1):.2f} " + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\footnotesize All final step covariances are scored against the same continuous-history target. "
        rf"Readout methods use a budget of {SHOTS} shots, or calls to $A$ and $A^{{-1}}$ for amplitude estimation, "
        r"and the same ideal Gaussian loader. "
        r"Covariance preparation includes finite-shot state readout added for this comparison. "
        r"SWAP and Hadamard receive exact column norms; covariance preparation receives the exact trace. "
        r"Both Cauchy--Schwarz approximations receive exact variances. "
        r"Averaging the upper and lower bounds gives diagonal covariance. "
        r"SD is one sample standard deviation, not an uncertainty on the mean.\par",
    ]
    style.FIGURES.mkdir(exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n")


def plot():
    with np.load(DATA) as saved:
        errors = {key: saved[key] for key in METHODS}
        budget_errors = saved["budget_errors"]
    write_table(errors)

    # Horizontal bars, relative to the coin sum + AMR, with a gap between the
    # proposed methods and the literature readouts.
    style.apply(font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    reference = errors["coin_sum_amr"].mean()
    left, right = .1, 0.
    for i, (key, (label, color)) in enumerate(METHODS.items()):
        y = i + (.6 if key in READOUTS[:4] else 0)
        values = errors[key] / reference
        mean, sd = values.mean(), values.std(ddof=1)
        right = max(right, mean + sd)
        ax.barh(y, mean, height=.78, color=color, alpha=style.BAR_ALPHA, edgecolor=color, linewidth=.8, zorder=2)
        ax.errorbar(mean, y + .29, xerr=sd, fmt="o", markersize=3., markerfacecolor="white",
                    markeredgecolor=style.INK, markeredgewidth=.8, ecolor=style.INK,
                    elinewidth=.9, capsize=2.5, zorder=4)
        ax.text(math.sqrt(left * mean), y, label, ha="center", va="center",
                color=style.INK, zorder=5)
    ax.set_yticks([])
    ax.set_ylim(len(METHODS) + .1, -.5)
    ax.set_xscale("log")
    ax.set_xlim(left, max(10., 5 * np.ceil(1.15 * right / 5)))
    ax.tick_params(axis="x", direction="out", length=3, color=style.INK)
    ax.set_xlabel("Covariance error relative to covariance-scaled coin sum + AMR")
    style.legend_below(ax, handles=[Line2D([0], [0], color=style.INK, marker="o", markerfacecolor="white",
                                           markersize=4, label=r"Mean $\pm$ 1 SD")])
    style.save(fig, "coin_sum_vs_covariance_methods.pdf")

    # Error against readout budget, with the two coin sums as reference bands.
    style.apply()
    fig, ax = plt.subplots(figsize=(7., 4.6))
    for kind, values in zip(READOUTS, 100 * budget_errors):
        label, color = METHODS[kind]
        mean, sd = values.mean(axis=1), values.std(axis=1, ddof=1)
        ax.fill_between(BUDGETS, mean - sd, mean + sd, color=color, alpha=.05, lw=0)
        ax.plot(BUDGETS, mean, marker="o", ms=4, lw=1.4, color=color, label=label)
    for key in ("coin_sum_fixed", "coin_sum_amr"):
        label, color = METHODS[key]
        mean, sd = 100 * errors[key].mean(), 100 * errors[key].std(ddof=1)
        ax.axhspan(mean - sd, mean + sd, color=color, alpha=.09, lw=0)
        ax.axhline(mean, color=color, lw=1.8, label=label)
    ax.set(xscale="log", xlabel="Readout shots or amplitude-estimation oracle calls",
           ylabel=r"Next-step covariance error (\%)")
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    style.legend_below(ax, handles + [Patch(facecolor=style.MUTED, alpha=.18)],
                       labels + [r"Mean $\pm$ 1 SD"], ncol=2)
    style.save(fig, "coin_sum_step_generation.pdf")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
