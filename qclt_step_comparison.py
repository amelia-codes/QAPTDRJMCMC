"""Compare a direct coherent QCLT step with measured-covariance pipelines.

The measurement pipelines are deliberately optimistic: each history branch is
provided classically, and an ideal Gaussian step loader is assumed after
covariance estimation. They are not coherent alternatives to the QCLT walk.
Covariances are scored only to validate final step laws.
"""

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, StatePreparation
ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "quantum_walk"
DATA = ROOT / "data" / "quantum_walk"
BUDGETS = np.array([12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144])
REPS = 100
D, K, SCALE2, ISO_VAR, X0, MODES = 2, 3, 2.38**2 / 2, 0.1, np.array([0.69, 0.37]), np.array([[0.0020, 0.0012], [0.0007, 0.0015]])


def all_lattice_histories():
    for lattice, _ in all_history_pairs():
        yield lattice


def all_history_pairs():
    modes = np.rint(MODES / (0.08 / 256)).astype(int)
    for bits in itertools.product((0, 1), repeat=K * D):
        signs = 1 - 2 * np.array(bits).reshape(K, D)
        yield (np.cumsum(signs @ modes, axis=0).astype(float),
               np.cumsum(signs @ MODES, axis=0))


def project_psd(matrices):
    lam, vec = np.linalg.eigh(matrices)
    return (vec * np.maximum(lam, 0)[:, None, :]) @ np.swapaxes(vec, -1, -2)


def pairwise_estimates(history, budget, rng):
    """Three centered pair-moment ancilla estimates, with shots divided evenly."""
    shots = max(budget // 3, 1)
    estimates = np.zeros((REPS, D, D))
    for p, q in itertools.combinations_with_replacement(range(D), 2):
        values = ((history[:, p, None] - history[None, :, p])
                  * (history[:, q, None] - history[None, :, q]) / 2)
        lo, hi = values.min(), values.max()
        if hi == lo:
            estimate = np.full(REPS, lo)
        else:
            probability = (values.mean() - lo) / (hi - lo)
            estimate = lo + (hi - lo) * rng.binomial(shots, probability, REPS) / shots
        estimates[:, p, q] = estimates[:, q, p] = K / (K - 1) * estimate
    return estimates


def swap_estimates(history, budget, rng):
    """Optimistic SWAP overlap: centered columns and both variances are known."""
    centered = history - history.mean(axis=0)
    norms2 = np.sum(centered ** 2, axis=0)
    z = norms2.sum()
    probability0 = 0.5 + np.sum((centered[:, 0] - centered[:, 1]) ** 2) / (4 * z)
    p0 = rng.binomial(budget, probability0, REPS) / budget
    dot = (z - 4 * z * (p0 - 0.5)) / 2
    estimates = np.zeros((REPS, D, D))
    estimates[:, 0, 0] = norms2[0] / (K - 1)
    estimates[:, 1, 1] = norms2[1] / (K - 1)
    estimates[:, 0, 1] = estimates[:, 1, 0] = dot / (K - 1)
    return estimates


def pairwise_readout_circuit(history, p, q):
    """One actual 5-qubit pairwise covariance-entry circuit for K=3."""
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import RYGate, StatePreparation

    index = QuantumRegister(2, "i")
    other = QuantumRegister(2, "j")
    ancilla = QuantumRegister(1, "value")
    circuit = QuantumCircuit(index, other, ancilla)
    uniform_three = np.array([1 / np.sqrt(K)] * K + [0.0])
    circuit.append(StatePreparation(uniform_three), index)
    circuit.append(StatePreparation(uniform_three), other)
    values = ((history[:, p, None] - history[None, :, p])
              * (history[:, q, None] - history[None, :, q]) / 2)
    lo, hi = values.min(), values.max()
    controls = [*index, *other]
    if hi > lo:
        for i in range(K):
            for j in range(K):
                probability = float((values[i, j] - lo) / (hi - lo))
                if np.isclose(probability, 0):
                    continue
                state = i + (j << 2)
                zero_controls = [bit for bit in range(4) if not (state >> bit) & 1]
                for bit in zero_controls:
                    circuit.x(controls[bit])
                circuit.append(RYGate(2 * np.arcsin(np.sqrt(probability))).control(4),
                               [*controls, ancilla[0]])
                for bit in zero_controls:
                    circuit.x(controls[bit])
    return circuit, float((values.mean() - lo) / (hi - lo)) if hi > lo else 0.0


def swap_readout_circuit(history):
    """One actual 5-qubit centered-column SWAP overlap circuit for K=3."""
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import StatePreparation

    centered = history - history.mean(axis=0)
    first = np.pad(centered[:, 0], (0, 1))
    second = np.pad(centered[:, 1], (0, 1))
    n_first, n_second = np.linalg.norm(first), np.linalg.norm(second)
    z = n_first ** 2 + n_second ** 2
    data = QuantumRegister(2, "data")
    flag = QuantumRegister(1, "flag")
    phi = QuantumRegister(1, "phi")
    ancilla = QuantumRegister(1, "swap")
    circuit = QuantumCircuit(data, flag, phi, ancilla)
    circuit.append(StatePreparation(np.concatenate((first / n_first,
                                                    second / n_second)) / np.sqrt(2)),
                   [*data, flag[0]])
    circuit.append(StatePreparation(np.array([n_first, -n_second]) / np.sqrt(z)),
                   [phi[0]])
    circuit.h(ancilla[0])
    circuit.cswap(ancilla[0], flag[0], phi[0])
    circuit.h(ancilla[0])
    expected_p0 = .5 + np.sum((first - second) ** 2) / (4 * z)
    return circuit, expected_p0


def verify_resource_circuits():
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import RYGate, StatePreparation
    from qiskit.quantum_info import Statevector
    from quantum_walk_study import make_walk

    history = next(all_lattice_histories())
    pairwise, pair_probability = pairwise_readout_circuit(history, 0, 1)
    swap, swap_probability = swap_readout_circuit(history)
    actual_pair = Statevector.from_instruction(pairwise).probabilities([4])[1]
    actual_swap = Statevector.from_instruction(swap).probabilities([4])[0]
    if not np.isclose(actual_pair, pair_probability, atol=1e-10):
        raise AssertionError("pairwise readout circuit has wrong ancilla probability")
    if not np.isclose(actual_swap, swap_probability, atol=1e-10):
        raise AssertionError("SWAP readout circuit has wrong ancilla probability")
    direct, _ = make_walk(8, X0, np.eye(D), np.full(D, .08 / 256), 14,
                          track_displacement=False)
    return np.array([direct.num_qubits, pairwise.num_qubits, swap.num_qubits, 31])


def generate():
    global D, K, SCALE2, ISO_VAR, X0, MODES
    from quantum_walk_study import D, K, MODES, SCALE2, ISO_VAR, X0
    DATA.mkdir(parents=True, exist_ok=True)
    qubits = verify_resource_circuits()
    rng = np.random.default_rng(91002)
    errors = np.empty((2, len(BUDGETS), 64 * REPS))
    qclt_errors = np.empty(64)
    spacing = .08 / 256
    for j, (history, physical_history) in enumerate(all_history_pairs()):
        sample_cov = np.cov(history.T)
        for p, q in itertools.combinations_with_replacement(range(D), 2):
            pair_mean = np.mean((history[:, p, None] - history[None, :, p])
                                * (history[:, q, None] - history[None, :, q]) / 2)
            if not np.isclose(K / (K - 1) * pair_mean, sample_cov[p, q]):
                raise AssertionError("pairwise ancilla expectation is incorrect")
        # All routes are scored against the same unrounded physical history.
        # The small lattice regularizer is omitted from the physical target,
        # matching the convention of the AMR comparison.
        target = SCALE2 * np.cov(physical_history.T)
        qclt_cov = spacing ** 2 * (SCALE2 * sample_cov + ISO_VAR * np.eye(D))
        qclt_errors[j] = np.linalg.norm(qclt_cov - target) / np.linalg.norm(target)
        for b, budget in enumerate(BUDGETS):
            for m, estimator in enumerate((pairwise_estimates, swap_estimates)):
                estimated = project_psd(estimator(history, int(budget), rng))
                step_cov = spacing ** 2 * (SCALE2 * estimated + ISO_VAR * np.eye(D))
                errors[m, b, j * REPS:(j + 1) * REPS] = (
                    np.linalg.norm(step_cov - target, axis=(1, 2)) / np.linalg.norm(target))
    if not np.all(np.isfinite(errors)):
        raise AssertionError("measured-entry comparison contains nonfinite values")
    quantiles = np.quantile(errors, [.1, .5, .9], axis=2).transpose(1, 2, 0)
    qclt_quantiles = np.quantile(qclt_errors, [.1, .5, .9])
    np.savez_compressed(DATA / "step_estimator_comparison.npz",
                        budgets=BUDGETS, quantiles=quantiles, reps=REPS,
                        branches=64, qubits=qubits, qclt_quantiles=qclt_quantiles)
    print("Pairwise median error at 384 shots:", quantiles[0, 5, 1], flush=True)
    print("SWAP median error at 384 shots:", quantiles[1, 5, 1], flush=True)
    print("QCLT median physical error:", qclt_quantiles[1], flush=True)
    plot()


def plot():
    data = np.load(DATA / "step_estimator_comparison.npz")
    import paper_style
    paper_style.apply()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m, (label, color) in enumerate((("Pairwise moments + loader", paper_style.COLORS["MEASURED"]),
                                        ("SWAP overlap + loader", paper_style.COLORS["CLASSICAL"]))):
        q = 100 * data["quantiles"][m]
        ax.fill_between(data["budgets"], q[:, 0], q[:, 2], color=color, alpha=.12)
        ax.plot(data["budgets"], q[:, 1], "o-", ms=3.5, color=color, label=label)
    qclt = 100 * data["qclt_quantiles"]
    ax.axhspan(qclt[0], qclt[2], color=paper_style.COLORS["QCLT"], alpha=.12)
    ax.axhline(qclt[1], color=paper_style.COLORS["QCLT"], lw=1.5,
               label="QCLT direct step")
    ax.set(xscale="log", xlabel="Intermediate covariance-measurement shots",
           ylabel=r"Next-step covariance error (\%)")
    ax.grid(color=paper_style.GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "qclt_step_generation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    plot() if "--plot-only" in sys.argv else generate()
