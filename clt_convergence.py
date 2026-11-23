# Do repeated steps of the two proposed methods, the coin sum and the single random pair, become
# Gaussian? (coin_sum_clt_convergence.pdf)
#
#   python3.12 clt_convergence.py              # recompute (about 5 minutes) and plot
#   python3.12 clt_convergence.py --plot-only  # redraw from data/clt_convergence.npz

import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
from matplotlib import pyplot as plt

import plot_style as style
from covariance_methods import METHODS
from walk import (BITS, D, GAMMA2, ISO_VAR, LOCAL_GRID, MODES, N, SINGLE_RANDOM_PAIR, random_pair_covariance,
                  run_walk, scaled_modes)


DATA = Path(__file__).parent / "data" / "clt_convergence.npz"

# Walks with L = 1 to 16 steps of each method, with its random-pair setting (None for the coin sum),
# walk seeds and bootstrap seeds. The coin sum's L = 1 keeps its original seed.
STEPS = list(range(1, 17))
WALKS = {
    "coin_sum": (None, [15] + [20 + n for n in STEPS[1:]], 20261116),
    "single_pair": (SINGLE_RANDOM_PAIR, [60 + n for n in STEPS], 20261216),
}


# The exact lattice covariance C of one step on a branch, from its history in grid units and the
# spacing of its quantum AMR refinement of the local grid: the coin sum's (random_pairs None) or
# that of random pairs (walk.random_pair_draws), whose history built from their scaled modes comes
# from the branch's preparatory signs
def step_covariance(random_pairs, history, spacing):
    if random_pairs is None:
        return GAMMA2 * np.cov(history.T) + ISO_VAR * np.eye(D)

    base = LOCAL_GRID[2]
    levels = np.rint(np.log2(base / spacing)).astype(int)[None]
    grid_modes, step_modes = (scaled_modes(MODES, base, levels, scale)[0] for scale in (1., random_pairs[2]))
    increments = np.diff(history, axis=0, prepend=np.full((1, D), 1 << (BITS - 1)))
    signs = increments @ np.linalg.inv(grid_modes)
    if not np.allclose(np.abs(signs), 1):
        raise ValueError("the history does not come from +-1 coins on this grid")
    return random_pair_covariance(np.rint(signs).astype(int), step_modes, random_pairs)


# L steps with fresh coins from the same history are i.i.d., so by the Lindeberg-Levy CLT
#   \frac{1}{\sqrt L} \sum_{k=1}^{L} \bm\Delta^{(N+1)}_k \xrightarrow{d} \Normal{\bm 0, C} \quad (L \to \infty),
# with C = \gamma^2 \widehat\Sigma_N + \nu I_d for the coin sum. Whiten each summed physical step with its
# history's conditional covariance,
#   L B C B^\top = U \Lambda U^\top, \quad
#   \bm w = \Lambda^{-1/2} U^\top B \sum_{k=1}^{L} \bm\Delta^{(N+1)}_k,
# so each \bm w should be close to \Normal{\bm 0, I_d}. Each shot is in the units of its own quantum
# AMR grid, whose spacing along the coordinate axes gives B = \operatorname{diag}(\bm\delta).
def whitened_steps(random_pairs, history, step, spacing, steps):
    physical = step * spacing

    # Whiten the shots of each distinct history (branch) together
    shots = np.hstack((history.reshape(len(history), -1), spacing))
    branches, branch = np.unique(shots, axis=0, return_inverse=True)
    whitened = np.empty_like(physical)
    for b, row in enumerate(branches):
        h, s = row[:N * D].reshape(N, D), row[N * D:]
        covariance = np.diag(s) @ (steps * step_covariance(random_pairs, h, s)) @ np.diag(s)
        values, vectors = np.linalg.eigh(covariance)
        whitened[branch == b] = physical[branch == b] @ (vectors / np.sqrt(values))

    return whitened


# Bootstrap estimates of the sliced Wasserstein-1 distance to Normal(0, I): the mean over 48
# directions of the distance between the projected sample's sorted values and the exact
# Gaussian quantiles
def sliced_wasserstein(samples, seed, n_bootstrap=48, sample_size=8192, n_directions=48):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, np.pi, n_directions, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    normal = NormalDist()
    quantiles = np.array([normal.inv_cdf((i + .5) / sample_size) for i in range(sample_size)])

    distances = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = samples[rng.integers(len(samples), size=sample_size)]
        projected = np.sort(resample @ directions.T, axis=0)
        distances[i] = np.abs(projected - quantiles[:, None]).mean(axis=0).mean()

    return distances


def run(shots=100_000):
    # Walks with L steps use fresh coins for every step and the same history, each branch on its
    # quantum AMR refinement of the local grid. The 8-bit position register wraps around for
    # L >= 2, so 32 record qubits hold the summed step. With the 2-qubit counter, the working width
    # is 25 + 10L for the coin sum and 26 + 7L for the single random pair; at L = 16 they use 217
    # and 170 qubits with the records.
    distances = {}
    for key, (random_pairs, seeds, bootstrap_seed) in WALKS.items():
        distances[key] = []
        for i, (steps, seed) in enumerate(zip(STEPS, seeds)):
            history, step, spacing = run_walk(BITS, LOCAL_GRID, shots, seed, steps=steps, record_bits=16,
                                              quantum_amr=True, random_pairs=random_pairs)
            distances[key].append(sliced_wasserstein(whitened_steps(random_pairs, history, step, spacing, steps),
                                                     bootstrap_seed + i))

    # The estimator's own floor: the distance it reports for exact Gaussian samples
    gaussian = np.random.default_rng(20261122).standard_normal((shots, D))
    floor = sliced_wasserstein(gaussian, 20261116)

    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, floor=floor, **{key: np.array(values) for key, values in distances.items()})


def plot():
    with np.load(DATA) as saved:
        distances, floor = {key: saved[key] for key in WALKS}, saved["floor"].mean()
    steps = np.array(STEPS)

    style.apply()
    fig, ax = plt.subplots(figsize=(5.8, 4.1))

    # Each method's distance, which by the central limit theorem should fall roughly as L^(-1/2)
    # until it reaches the estimator's floor
    for (key, values), marker in zip(distances.items(), "os"):
        label, color = METHODS[key]
        ax.errorbar(steps, values.mean(axis=1), yerr=values.std(axis=1, ddof=1), color=color, marker=marker,
                    ms=5, capsize=3, lw=1.6, label=label)
    first = distances["coin_sum"].mean(axis=1)[0]
    ax.plot(steps, first / np.sqrt(steps), ls="--", color=style.MUTED, lw=1.0, label=r"$L^{-1/2}$ reference")
    ax.axhline(floor, ls=":", color=style.COLORS["SAGE"], lw=1.2, label="Exact Gaussian samples (estimator floor)")

    # Axes
    ax.set_xticks(steps)
    ax.set_yscale("log")
    ax.set_yticks([.02, .05, .1, .2], ["0.02", "0.05", "0.1", "0.2"])
    ax.set_xlabel("Number of summed steps, $L$")
    ax.set_ylabel("Sliced Wasserstein distance to Gaussian")

    # Legend with the methods first, then the reference and the floor
    handles, labels = ax.get_legend_handles_labels()
    names = [METHODS[key][0] for key in WALKS]
    order = [labels.index(name) for name in names] + [i for i, label in enumerate(labels) if label not in names]
    style.legend_below(ax, [handles[i] for i in order], [labels[i] for i in order],
                       title=r"All with quantum AMR; mean $\pm$ 1 bootstrap SD")

    style.save(fig, "coin_sum_clt_convergence.pdf")

if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
