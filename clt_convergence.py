"""Do repeated coin-sum steps become Gaussian? (coin_sum_clt_convergence.pdf)

A pilot walk chooses an AMR grid. On that grid, walks with N = 1 to 16
coin-sum steps are sampled, with fresh coins for every step and the same
history. For each history, the summed step is whitened by its exact
conditional covariance, N (SCALE2 S_h + ISO_VAR I), and compared with a
standard Gaussian. By the central limit theorem the distance should fall
roughly as N^(-1/2), until it reaches the estimator's own floor: the distance
it reports for exact Gaussian samples.

The 8-bit position register wraps around for N >= 2, so the summed step is
read from the 14-bit record registers, which are not counted in the walk's
quoted 33 qubits.

    python3.12 clt_convergence.py              # recompute (about 1 minute) and plot
    python3.12 clt_convergence.py --plot-only  # redraw from data/clt_convergence.npz
"""

import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
from matplotlib import pyplot as plt

import plot_style as style
from walk import BITS, D, ISO_VAR, SCALE2, amr_window, pilot_histories, run_walk

DATA = Path(__file__).parent / "data" / "clt_convergence.npz"
STEPS = list(range(1, 17))
WALK_SEEDS = [15] + [20 + n for n in STEPS[1:]]


def whitened_steps(history, step, grid, steps):
    """Each physical step multiplied by the inverse square root of its
    history's conditional covariance, so each should be close to N(0, I)."""
    center, axes, spacing = grid
    basis = axes @ np.diag(spacing)
    physical = (step * spacing) @ axes.T
    branches, branch = np.unique(history.reshape(len(history), -1), axis=0, return_inverse=True)
    whitened = np.empty_like(physical)
    for b, h in enumerate(branches):
        covariance = basis @ (steps * (SCALE2 * np.cov(h.reshape(-1, D).T) + ISO_VAR * np.eye(D))) @ basis.T
        values, vectors = np.linalg.eigh(covariance)
        whitened[branch == b] = physical[branch == b] @ (vectors / np.sqrt(values))
    return whitened


def sliced_wasserstein(samples, seed, n_bootstrap=48, sample_size=8192, n_directions=48):
    """Bootstrap estimates of the sliced Wasserstein-1 distance to N(0, I):
    the mean over 48 directions of the distance between the projected sample's
    sorted values and the exact Gaussian quantiles."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, np.pi, n_directions, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    quantiles = np.array([NormalDist().inv_cdf((i + .5) / sample_size) for i in range(sample_size)])
    distances = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = samples[rng.integers(len(samples), size=sample_size)]
        projected = np.sort(resample @ directions.T, axis=0)
        distances[i] = np.abs(projected - quantiles[:, None]).mean(axis=0).mean()
    return distances


def run(shots=100_000):
    grid = amr_window(pilot_histories(shots=shots, seed=11))
    distances = []
    for i, (steps, seed) in enumerate(zip(STEPS, WALK_SEEDS)):
        history, step = run_walk(BITS, grid, shots, seed, steps=steps, record_bits=14)
        distances.append(sliced_wasserstein(whitened_steps(history, step, grid, steps), 20261116 + i))
    gaussian = np.random.default_rng(20261122).standard_normal((shots, D))
    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, distances=np.array(distances), floor=sliced_wasserstein(gaussian, 20261116))


def plot():
    with np.load(DATA) as saved:
        distances, floor = saved["distances"], saved["floor"].mean()
    steps = np.array(STEPS)
    mean, sd = distances.mean(axis=1), distances.std(axis=1, ddof=1)
    style.apply()
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    ax.errorbar(steps, mean, yerr=sd, color=style.COLORS["COIN_SUM_AMR"], marker="o", ms=5,
                capsize=3, lw=1.6, label=r"Covariance-scaled coin sum (mean $\pm$ 1 bootstrap SD)")
    ax.plot(steps, mean[0] / np.sqrt(steps), ls="--", color=style.MUTED, lw=1.0, label=r"$N^{-1/2}$ reference")
    ax.axhline(floor, ls=":", color=style.COLORS["BASELINE"], lw=1.2, label="Exact Gaussian samples (estimator floor)")
    ax.set_xticks(steps)
    ax.set_yscale("log")
    ax.set_yticks([.02, .05, .1], ["0.02", "0.05", "0.1"])
    ax.set_xlabel("Number of coin-sum steps, $N$")
    ax.set_ylabel("Sliced Wasserstein distance to Gaussian")
    style.legend_below(ax)
    style.save(fig, "coin_sum_clt_convergence.pdf")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
