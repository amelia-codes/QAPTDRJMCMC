# How many random pairs should a step draw? (random_pair_draws.pdf)
#
# A random-pair step makes K draws, each of a uniformly chosen pair difference with a Rademacher
# sign (walk.random_pair_draws). One draw, the single random pair, must scale its modes by
# gamma / sqrt(2) and round them; two or more draws each move with probability gamma^2 / (2K) and
# have the coin sum's covariance exactly. Each K is compared with the coin sum, all with quantum AMR:
#   1. next-step covariance error, exact, against the same target as covariance_methods.py
#   2. how far one step is from Gaussian, sampled from the circuit as in clt_convergence.py
#   3. qubits and CX gates per step
#
#   python3.12 random_pairs.py              # recompute (about 1 minute) and plot
#   python3.12 random_pairs.py --plot-only  # redraw from data/random_pairs.npz

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate, StatePreparation

import plot_style as style
from clt_convergence import sliced_wasserstein, whitened_steps
from covariance_methods import METHODS, grid_history
from walk import (BITS, D, GAMMA2, ISO_VAR, LOCAL_GRID, MODES, UNIFORM_PAIRS, all_histories, make_walk,
                  quantum_amr_grid, quantum_amr_modes, random_pair_covariance, random_pair_draws, run_walk,
                  scaled_modes)


DATA = Path(__file__).parent / "data" / "random_pairs.npz"
DRAWS = list(range(1, 7))


# ---------------------------------------------------------------------------
# 1. Exact next-step covariance error with quantum AMR, on each of the 64 histories
# ---------------------------------------------------------------------------

def covariance_errors():
    errors = np.empty((len(DRAWS) + 1, 64))
    for b, (signs, history) in enumerate(zip(*all_histories())):
        target = GAMMA2 * np.cov(history.T)
        center, axes, spacing = grid = quantum_amr_grid(signs, LOCAL_GRID)
        basis = axes @ np.diag(spacing)

        # The coin sum first, then each number of draws
        lattice = [GAMMA2 * np.cov(grid_history(signs, grid).T) + ISO_VAR * np.eye(D)]
        for draws in DRAWS:
            random_pairs = random_pair_draws(draws)
            step_modes = np.rint(random_pairs[2] * MODES @ axes / spacing).astype(int)
            lattice.append(random_pair_covariance(signs, step_modes, random_pairs))
        for k, covariance in enumerate(lattice):
            errors[k, b] = np.linalg.norm(basis @ covariance @ basis.T - target) / np.linalg.norm(target)

    return errors


# ---------------------------------------------------------------------------
# 2. Distance of one step to Gaussian: 100,000 shots of each circuit, whitened with its own
#    exact covariance, so this measures shape, not covariance error
# ---------------------------------------------------------------------------

def gaussianity(shots=100_000):
    distances = []
    for k, random_pairs in enumerate([None] + [random_pair_draws(draws) for draws in DRAWS]):
        history, step, spacing = run_walk(BITS, LOCAL_GRID, shots, 80 + k, record_bits=16, quantum_amr=True,
                                          random_pairs=random_pairs)
        distances.append(sliced_wasserstein(whitened_steps(random_pairs, history, step, spacing, 1), 20261125 + k))
    return np.array(distances)


# ---------------------------------------------------------------------------
# 3. Qubits and CX gates per step: the walk with two steps minus the walk with one, so the history
#    preparation cancels. CX counts use Qiskit's default synthesis of each multi-controlled X gate
#    and of the pair-index preparation.
# ---------------------------------------------------------------------------

# Multi-controlled X gates by number of controls, and pair-index preparations
def gate_profile(qc):
    profile = Counter()
    for instruction in qc.data:
        gate = instruction.operation
        if gate.name == "state_preparation":
            profile["prep"] += 1
        elif gate.name.split("_o")[0] in ("cx", "ccx", "mcx", "mcx_gray"):
            profile[gate.num_ctrl_qubits] += 1
    return profile


# CX gates in Qiskit's synthesis of a gate on `width` qubits
def cx_count(gate, width):
    qc = QuantumCircuit(width)
    qc.append(gate, range(width))
    return transpile(qc, basis_gates=["cx", "u"], optimization_level=0).count_ops().get("cx", 0)


def costs():
    spacing = LOCAL_GRID[2]
    levels, grid_modes = quantum_amr_modes(MODES, spacing)
    start = np.full(D, 1 << (BITS - 1))
    synthesis = {"prep": cx_count(StatePreparation(UNIFORM_PAIRS), 2)}

    qubits, cx = [], []
    for random_pairs in [None] + [random_pair_draws(draws) for draws in DRAWS]:
        step_modes = None if random_pairs is None else scaled_modes(MODES, spacing, levels, random_pairs[2])
        one, two = (make_walk(BITS, start, grid_modes, steps, random_pairs=random_pairs and random_pairs[:2],
                              step_modes=step_modes)[0] for steps in (1, 2))
        profile = gate_profile(two)
        profile.subtract(gate_profile(one))
        for kind in profile:
            if kind not in synthesis:
                synthesis[kind] = cx_count(MCXGate(kind), kind + 1)
        qubits.append(two.num_qubits - one.num_qubits)
        cx.append(sum(count * synthesis[kind] for kind, count in profile.items()))

    return np.array(qubits), np.array(cx)


def run():
    qubits, cx = costs()
    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, errors=covariance_errors(), distances=gaussianity(), qubits=qubits, cx=cx)


# ---------------------------------------------------------------------------
# Figure: four panels against the number of draws, each with the coin sum as a reference line.
# Error bars are one sample SD across the 64 histories (error) or the bootstrap (distance).
# ---------------------------------------------------------------------------

def plot():
    with np.load(DATA) as saved:
        errors, distances, qubits, cx = (saved[key] for key in ("errors", "distances", "qubits", "cx"))

    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    coin_label, coin_color = METHODS["coin_sum"]
    color = METHODS["single_pair"][1]
    panels = [
        (r"Next-step covariance error (\%)", 100 * errors.mean(axis=1), 100 * errors.std(axis=1, ddof=1)),
        ("Distance of one step to Gaussian", distances.mean(axis=1), distances.std(axis=1, ddof=1)),
        ("Qubits per step", qubits, None),
        ("CX gates per step (thousands)", cx / 1000, None),
    ]

    for ax, (title, values, sd) in zip(axes.flat, panels):
        # The coin sum, drawn on top because two or more draws have its covariance exactly, then the
        # random pairs by number of draws
        ax.axhline(values[0], color=coin_color, lw=1.6, zorder=3)
        if sd is not None:
            ax.axhspan(values[0] - sd[0], values[0] + sd[0], color=coin_color, alpha=.09, lw=0)
        ax.errorbar(DRAWS, values[1:], yerr=None if sd is None else sd[1:], color=color, marker="o", ms=5,
                    lw=1.4, capsize=2.5)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim(bottom=0)
        ax.set_xticks(DRAWS)
    for ax in axes[1]:
        ax.set_xlabel("Random pairs drawn per step, $K$")

    style.legend_below(axes[1, 0], ncol=2, title="All with quantum AMR; mean $\\pm$ 1 SD", handles=[
        Line2D([0], [0], color=color, marker="o", ms=5, lw=1.4, label="Random pairs ($K=1$: single random pair)"),
        Line2D([0], [0], color=coin_color, lw=1.6, label=coin_label)])

    style.save(fig, "random_pair_draws.pdf")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
