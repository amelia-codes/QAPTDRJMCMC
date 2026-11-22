"""Shared method list and finite-shot laws for the focused covariance study.

Readout baselines receive classical centered histories and exact normalization
data. Their returned entries feed the same ideal Gaussian loader. Covariance
matrix preparation follows Gordon et al. (2022), with tomography added here
to turn its density-matrix output into a comparable next-step pipeline.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import GroverOperator, QFT, RYGate, StatePreparation
from qcov import add_const, iso_step, lazy_coin

# Method ID: (paper label, shared colour). This order is also the table order.
METHODS = {
    "qclt_amr": ("QCLT + AMR", "QCLT_AMR"),
    "qclt_fixed": ("QCLT, fixed local grid", "QCLT"),
    "swap": ("SWAP test", "MEASURED"),
    "hadamard": ("Hadamard test", "MEASURED"),
    "qae": ("Amplitude estimation", "MEASURED"),
    "qpca": ("Covariance matrix preparation + readout", "MEASURED"),
    "cs_upper": ("Cauchy--Schwarz upper bound", "BASELINE"),
    "cs_average": ("Cauchy--Schwarz upper/lower average", "BASELINE"),
    "single_pair": ("Single history-pair difference", "BASELINE"),
    "pairwise": ("Pairwise moments", "BASELINE"),
}
READOUT_KINDS = ("swap", "hadamard", "qae", "qpca", "pairwise")
READOUT_LABELS = {key: METHODS[key][0] for key in READOUT_KINDS}
READOUT_LABELS["qpca"] = "Covariance preparation + readout"


def centered_history(history):
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or len(history) < 2 or history.shape[1] < 1:
        raise ValueError("A history needs at least two samples and one coordinate")
    if not np.all(np.isfinite(history)):
        raise ValueError("History values must be finite")
    return history - history.mean(axis=0)


def pairwise_exact(history):
    """Ancilla probabilities and value ranges for centered pairwise moments."""
    k, d = history.shape
    out = []
    for p, q in itertools.combinations_with_replacement(range(d), 2):
        values = ((history[:, None, p] - history[None, :, p])
                  * (history[:, None, q] - history[None, :, q]) / 2).ravel()
        lo, hi = values.min(), values.max()
        out.append((p, q, (values.mean() - lo) / (hi - lo) if hi > lo else 0., lo, hi))
    return out


def qae_sample(probability, phase_states, rng, size):
    """Exact phase-register law of canonical amplitude estimation."""
    theta = math.asin(math.sqrt(float(np.clip(probability, 0., 1.))))
    y = np.arange(phase_states)
    probabilities = np.zeros(phase_states)
    for sign in (1, -1):
        delta = np.pi * y / phase_states - sign * theta
        sine = np.sin(delta)
        with np.errstate(divide="ignore", invalid="ignore"):
            probabilities += .5 * np.where(
                np.abs(sine) < 1e-12, 1.,
                np.sin(phase_states * delta) ** 2 / (phase_states ** 2 * sine ** 2))
    probabilities /= probabilities.sum()
    return np.sin(np.pi * rng.choice(phase_states, size=size, p=probabilities)
                  / phase_states) ** 2


def qae_phase_states(oracle_budget):
    """Largest power-of-two M with 2(M-1)+1 calls to A or A^{-1} in budget."""
    if oracle_budget < 1:
        raise ValueError("Amplitude estimation needs at least one oracle call")
    max_states = (int(oracle_budget) + 1) // 2
    phase_bits = max_states.bit_length() - 1
    return 1 << phase_bits


def readout_settings(kind, dimension):
    if kind in ("pairwise", "qae"):
        return dimension * (dimension + 1) // 2
    if kind == "qpca":
        return 1 + dimension * (dimension - 1) // 2
    if kind in ("swap", "hadamard"):
        return max(1, dimension * (dimension - 1) // 2)
    raise ValueError(f"Unknown readout method: {kind}")


def project_psd(covariance):
    """Clip negative eigenvalues introduced by finite-shot readout."""
    values, vectors = np.linalg.eigh(.5 * (covariance + np.swapaxes(covariance, -1, -2)))
    return (vectors * np.maximum(values, 0)[..., None, :]) @ np.swapaxes(vectors, -1, -2)


def readout_estimate(history, kind, budget, rng, reps):
    """Finite-shot covariance entries, before the common PSD projection.

    SWAP and Hadamard get exact column norms. QPCA gets the exact total
    centered squared norm and measures its prepared state in the coordinate
    basis and each real two-coordinate interference basis. QAE budgets count
    both A and A^{-1}, including the initial preparation.
    """
    centered = centered_history(history)
    k, d = centered.shape
    settings = readout_settings(kind, d)
    if budget < settings:
        raise ValueError(f"{kind} needs at least {settings} shots/oracle calls")
    shots = int(budget) // settings
    estimates = np.zeros((reps, d, d))
    if kind in ("swap", "hadamard"):
        norms2 = np.sum(centered ** 2, axis=0)
        estimates[:, np.arange(d), np.arange(d)] = norms2
        for p, q in itertools.combinations(range(d), 2):
            dot = centered[:, p] @ centered[:, q]
            if kind == "swap":
                z = norms2[p] + norms2[q]
                if z == 0:
                    continue
                probability = .5 + np.sum((centered[:, p] - centered[:, q]) ** 2) / (4 * z)
                frequency = rng.binomial(shots, np.clip(probability, 0, 1), reps) / shots
                value = (z - 4 * z * (frequency - .5)) / 2
            else:
                norm_product = np.sqrt(norms2[p] * norms2[q])
                if norm_product == 0:
                    continue
                probability = .5 * (1 + dot / norm_product)
                frequency = rng.binomial(shots, np.clip(probability, 0, 1), reps) / shots
                value = norm_product * (2 * frequency - 1)
            estimates[:, p, q] = estimates[:, q, p] = value
        return estimates / (k - 1)
    if kind == "qpca":
        scatter = centered.T @ centered
        trace = np.trace(scatter)
        if trace == 0:
            return estimates
        rho = scatter / trace
        diagonal = rng.multinomial(shots, np.diag(rho), size=reps) / shots
        estimates[:, np.arange(d), np.arange(d)] = diagonal
        for p, q in itertools.combinations(range(d), 2):
            # Measurement in (|p> +/- |q>)/sqrt(2), plus the other states.
            mass = rho[p, p] + rho[q, q]
            probabilities = np.clip([(mass + 2 * rho[p, q]) / 2,
                                     (mass - 2 * rho[p, q]) / 2, 1 - mass], 0, 1)
            probabilities /= np.sum(probabilities)
            counts = rng.multinomial(shots, probabilities, size=reps)
            estimates[:, p, q] = estimates[:, q, p] = (counts[:, 0] - counts[:, 1]) / (2 * shots)
        return estimates * trace / (k - 1)
    for p, q, probability, lo, hi in pairwise_exact(np.asarray(history)):
        frequency = (rng.binomial(shots, probability, reps) / shots if kind == "pairwise"
                     else qae_sample(probability, qae_phase_states(shots), rng, reps))
        estimates[:, p, q] = estimates[:, q, p] = lo + (hi - lo) * frequency
    return estimates * k / (k - 1)


def hadamard_readout_circuit(history, p=0, q=1):
    """Signed real overlap of centered columns; one ancilla, exact loaders."""

    centered = centered_history(history)
    bits = (len(history) - 1).bit_length()
    vectors = []
    for column in (p, q):
        vector = np.pad(centered[:, column], (0, (1 << bits) - len(history)))
        norm = np.linalg.norm(vector)
        vectors.append(vector / norm if norm else np.eye(1 << bits)[0])
    data, ancilla = QuantumRegister(bits, "data"), QuantumRegister(1, "hadamard")
    circuit = QuantumCircuit(data, ancilla)
    circuit.h(ancilla)
    circuit.append(StatePreparation(vectors[0]).control(1, ctrl_state=0), [ancilla[0], *data])
    circuit.append(StatePreparation(vectors[1]).control(1), [ancilla[0], *data])
    circuit.h(ancilla)
    return circuit, float((1 + vectors[0] @ vectors[1]) / 2)


def covariance_preparation_circuit(history):
    """Purify the norm-weighted centered ensemble; rho_data = S / tr(S).

    Coordinate qubits come first (least significant); history-index qubits
    purify the ensemble. This is a tiny exact-loader replica, with classical
    centering and norm data supplied, rather than a coherent centering oracle.
    """

    centered = centered_history(history)
    k, d = centered.shape
    data_bits, index_bits = max(1, (d - 1).bit_length()), (k - 1).bit_length()
    padded = np.zeros((1 << index_bits, 1 << data_bits))
    padded[:k, :d] = centered
    norm = np.linalg.norm(padded)
    if not norm:
        padded[0, 0] = 1
        norm = 1
    data, index = QuantumRegister(data_bits, "coordinate"), QuantumRegister(index_bits, "history")
    circuit = QuantumCircuit(data, index)
    circuit.append(StatePreparation((padded / norm).ravel()), [*data, *index])
    trace = float(np.sum(centered ** 2) / (k - 1))
    return circuit, trace


def pairwise_readout_circuit(history, p, q):
    """Uniform history-pair value oracle, with padding excluded from the sum."""

    history = np.asarray(history)
    centered_history(history)
    k = len(history)
    bits = (k - 1).bit_length()
    first, second = QuantumRegister(bits, "i"), QuantumRegister(bits, "j")
    ancilla = QuantumRegister(1, "value")
    circuit = QuantumCircuit(first, second, ancilla)
    amplitudes = np.pad(np.full(k, 1 / np.sqrt(k)), (0, (1 << bits) - k))
    for register in (first, second):
        circuit.append(StatePreparation(amplitudes), register)
    values = ((history[:, p, None] - history[None, :, p])
              * (history[:, q, None] - history[None, :, q]) / 2)
    lo, hi = values.min(), values.max()
    if hi > lo:
        for i, j in itertools.product(range(k), repeat=2):
            probability = float((values[i, j] - lo) / (hi - lo))
            if probability:
                gate = RYGate(2 * np.arcsin(np.sqrt(probability))).control(
                    2 * bits, ctrl_state=i + (j << bits))
                circuit.append(gate, [*first, *second, ancilla[0]])
    return circuit, float((values.mean() - lo) / (hi - lo)) if hi > lo else 0.


def swap_readout_circuit(history, p=0, q=1):
    """Signed-distance SWAP test, with centered-column norms supplied."""

    centered = centered_history(history)
    bits = (len(history) - 1).bit_length()
    first, second = [np.pad(centered[:, column], (0, (1 << bits) - len(history)))
                     for column in (p, q)]
    norms = np.array([np.linalg.norm(first), np.linalg.norm(second)])
    z = norms @ norms
    vectors = [vector / norm if norm else np.eye(1 << bits)[0]
               for vector, norm in zip((first, second), norms)]
    phi_state = norms * [1, -1] / np.sqrt(z) if z else np.array([1, -1]) / np.sqrt(2)
    data = QuantumRegister(bits, "data")
    flag, phi, ancilla = [QuantumRegister(1, name) for name in ("flag", "phi", "swap")]
    circuit = QuantumCircuit(data, flag, phi, ancilla)
    circuit.append(StatePreparation(np.concatenate(vectors) / np.sqrt(2)), [*data, flag[0]])
    circuit.append(StatePreparation(phi_state), phi)
    circuit.h(ancilla)
    circuit.cswap(ancilla[0], flag[0], phi[0])
    circuit.h(ancilla)
    probability = .5 + np.sum((first - second) ** 2) / (4 * z) if z else .5
    return circuit, float(probability)


def amplitude_readout_circuit(history, p, q, phase_bits):
    """Canonical QAE circuit; phase bits are the final register's qubits."""

    preparation, probability = pairwise_readout_circuit(history, p, q)
    oracle = QuantumCircuit(preparation.num_qubits)
    oracle.z(preparation.num_qubits - 1)
    grover = GroverOperator(oracle, state_preparation=preparation)
    phase = QuantumRegister(phase_bits, "phase")
    circuit = QuantumCircuit(*preparation.qregs, phase)
    circuit.compose(preparation, qubits=list(range(preparation.num_qubits)), inplace=True)
    circuit.h(phase)
    for bit in range(phase_bits):
        circuit.append(grover.power(1 << bit).control(), [phase[bit], *preparation.qubits])
    circuit.append(QFT(phase_bits, inverse=True), phase)
    return circuit, probability


def single_pair_step_circuit(history, bits=8, scale2=2.38 ** 2 / 2, iso_var=.1):
    """One uniformly selected signed pair, compiled from classical history."""

    history = np.asarray(history)
    centered_history(history)
    k, d = history.shape
    pairs = list(itertools.combinations(range(k), 2))
    deltas = np.rint(np.sqrt(scale2 / 2) * np.array([history[a] - history[b] for a, b in pairs])).astype(int)
    if np.any(np.abs(deltas) + 1 >= 1 << (bits - 1)):
        raise ValueError("Single-pair output could overflow the signed register")
    index_bits = max(1, (len(pairs) - 1).bit_length())
    position = [QuantumRegister(bits, f"step_{p}") for p in range(d)]
    index, sign = QuantumRegister(index_bits, "pair"), QuantumRegister(1, "sign")
    iso_move, iso_sign = QuantumRegister(d, "iso_move"), QuantumRegister(d, "iso_sign")
    circuit = QuantumCircuit(*position, index, sign, iso_move, iso_sign)
    weights = np.pad(np.full(len(pairs), 1 / np.sqrt(len(pairs))), (0, (1 << index_bits) - len(pairs)))
    circuit.append(StatePreparation(weights), index)
    circuit.h(sign)
    for i, delta in enumerate(deltas):
        zeros = [qubit for bit, qubit in enumerate(index) if not (i >> bit) & 1]
        for qubit in zeros:
            circuit.x(qubit)
        for register, value in zip(position, delta):
            if value:
                add_const(circuit, register, int(value), list(index))
                add_const(circuit, register, -2 * int(value), [*index, sign[0]])
        for qubit in zeros:
            circuit.x(qubit)
    for p in range(d):
        lazy_coin(circuit, iso_var, iso_move[p], iso_sign[p])
    iso_step(circuit, [position], [1] * d, iso_sign, iso_move)
    return circuit, position
