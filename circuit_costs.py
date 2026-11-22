"""Explicit circuits for the comparison methods, and their qubit counts.

covariance_methods.py samples each readout's measurement law directly. The
circuits here are what those laws come from: test_covariance_methods.py
checks each circuit's outcome probability against the law, and write_table
counts their qubits for figures/method_costs.tex. Classical data (centring,
norms, the trace) is supplied to the circuits, as in the comparison.

    python3.12 circuit_costs.py    # write figures/method_costs.tex
"""

import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, GroverOperator, RYGate, StatePreparation

import plot_style as style
from covariance_methods import METHODS, READOUTS, SETTINGS, SHOTS, phase_states
from walk import BITS, D, ISO_VAR, K, MODES, PAIRS, SCALE2, add_signed, lazy_coin, make_walk


def unit_columns(history, p, q):
    """Centred columns p and q, zero-padded to a power of two, and their norms."""
    centered = history - history.mean(axis=0)
    size = 1 << (len(history) - 1).bit_length()
    columns = [np.pad(centered[:, c], (0, size - len(history))) for c in (p, q)]
    norms = np.array([np.linalg.norm(c) for c in columns])
    return [c / n for c, n in zip(columns, norms)], norms


def hadamard_test(history, p=0, q=1):
    """Ancilla reads 0 with probability (1 + x.y / (|x| |y|)) / 2."""
    (x, y), _ = unit_columns(history, p, q)
    data, ancilla = QuantumRegister(int(np.log2(len(x))), "data"), QuantumRegister(1, "ancilla")
    qc = QuantumCircuit(data, ancilla)
    qc.h(ancilla)
    qc.append(StatePreparation(x).control(1, ctrl_state=0), [ancilla[0], *data])
    qc.append(StatePreparation(y).control(1), [ancilla[0], *data])
    qc.h(ancilla)
    return qc, float((1 + x @ y) / 2)


def swap_test(history, p=0, q=1):
    """SWAP test between (x, -y)-weighted flag states; the ancilla reads 0 with
    probability 1/2 + |x - y|^2 / (4 (|x|^2 + |y|^2))."""
    (x, y), norms = unit_columns(history, p, q)
    data = QuantumRegister(int(np.log2(len(x))), "data")
    flag, phi, ancilla = (QuantumRegister(1, name) for name in ("flag", "phi", "ancilla"))
    qc = QuantumCircuit(data, flag, phi, ancilla)
    qc.append(StatePreparation(np.concatenate([x, y]) / np.sqrt(2)), [*data, flag[0]])
    qc.append(StatePreparation(norms * [1, -1] / np.linalg.norm(norms)), phi)
    qc.h(ancilla)
    qc.cswap(ancilla[0], flag[0], phi[0])
    qc.h(ancilla)
    difference = norms[0] * x - norms[1] * y
    return qc, float(.5 + difference @ difference / (4 * norms @ norms))


def covariance_preparation(history):
    """Prepare rows of the centred history with weights |row|^2, so the
    coordinate qubits hold rho = S / tr(S) (Gordon et al. 2022). Returns the
    circuit and tr(S)."""
    centered = history - history.mean(axis=0)
    k, d = centered.shape
    data_bits, index_bits = max(1, (d - 1).bit_length()), (k - 1).bit_length()
    amplitudes = np.zeros((1 << index_bits, 1 << data_bits))
    amplitudes[:k, :d] = centered
    data, index = QuantumRegister(data_bits, "coordinate"), QuantumRegister(index_bits, "history")
    qc = QuantumCircuit(data, index)
    qc.append(StatePreparation((amplitudes / np.linalg.norm(amplitudes)).ravel()), [*data, *index])
    return qc, float(np.sum(centered**2) / (k - 1))


def pairwise_moment(history, p, q):
    """Uniform superposition over index pairs (i, j); an ancilla rotation makes
    P(1) the mean of the rescaled products (x_i - x_j)(y_i - y_j) / 2."""
    k = len(history)
    bits = (k - 1).bit_length()
    first, second, ancilla = QuantumRegister(bits, "i"), QuantumRegister(bits, "j"), QuantumRegister(1, "value")
    qc = QuantumCircuit(first, second, ancilla)
    uniform = np.pad(np.full(k, 1 / np.sqrt(k)), (0, (1 << bits) - k))
    for register in (first, second):
        qc.append(StatePreparation(uniform), register)
    values = (history[:, p, None] - history[None, :, p]) * (history[:, q, None] - history[None, :, q]) / 2
    lo, hi = values.min(), values.max()
    for i, j in itertools.product(range(k), repeat=2):
        probability = (values[i, j] - lo) / (hi - lo)
        if probability:
            rotation = RYGate(2 * np.arcsin(np.sqrt(probability))).control(2 * bits, ctrl_state=i + (j << bits))
            qc.append(rotation, [*first, *second, ancilla[0]])
    return qc, float((values.mean() - lo) / (hi - lo))


def amplitude_estimation(history, p, q, phase_bits):
    """Canonical amplitude estimation of the pairwise-moment probability; the
    phase register is the last `phase_bits` qubits."""
    preparation, probability = pairwise_moment(history, p, q)
    oracle = QuantumCircuit(preparation.num_qubits)
    oracle.z(preparation.num_qubits - 1)
    grover = GroverOperator(oracle, state_preparation=preparation)
    phase = QuantumRegister(phase_bits, "phase")
    qc = QuantumCircuit(*preparation.qregs, phase)
    qc.compose(preparation, qubits=list(range(preparation.num_qubits)), inplace=True)
    qc.h(phase)
    for bit in range(phase_bits):
        qc.append(grover.power(1 << bit).control(), [phase[bit], *preparation.qubits])
    qc.append(QFT(phase_bits, inverse=True), phase)
    return qc, probability


def single_pair_step(history, bits=BITS):
    """Pick one history pair uniformly, add its difference with a random sign,
    scaled by sqrt(SCALE2 / 2), plus the +-1 lattice step. The differences are
    compiled in as constants. Returns the circuit and its step registers."""
    k, d = history.shape
    deltas = np.rint(np.sqrt(SCALE2 / 2) * np.array([history[a] - history[b] for a, b in PAIRS])).astype(int)
    index_bits = max(1, (len(PAIRS) - 1).bit_length())
    step = [QuantumRegister(bits, f"step_{p}") for p in range(d)]
    index, sign = QuantumRegister(index_bits, "pair"), QuantumRegister(1, "sign")
    iso_move, iso_sign = QuantumRegister(d, "iso_move"), QuantumRegister(d, "iso_sign")
    qc = QuantumCircuit(*step, index, sign, iso_move, iso_sign)
    qc.append(StatePreparation(np.pad(np.full(len(PAIRS), 1 / np.sqrt(len(PAIRS))),
                                      (0, (1 << index_bits) - len(PAIRS)))), index)
    qc.h(sign)
    for i, delta in enumerate(deltas):
        zeros = [qubit for bit, qubit in enumerate(index) if not (i >> bit) & 1]
        qc.x(zeros)                                   # control on index == i
        for register, value in zip(step, delta):
            if value:
                add_signed(qc, register, int(value), sign[0], list(index))
        qc.x(zeros)
    for p in range(d):
        lazy_coin(qc, ISO_VAR, iso_move[p], iso_sign[p])
        add_signed(qc, step[p], 1, iso_sign[p], [iso_move[p]])
    return qc, step


def write_table():
    spacing = .08 / 256
    history = np.cumsum(np.ones((K, D)) @ np.rint(MODES / spacing), axis=0)
    phase_bits = phase_states(SHOTS // SETTINGS["qae"]).bit_length() - 1
    walk_qubits = make_walk(BITS, np.zeros(D, int), np.eye(D, dtype=int))[0].num_qubits
    qubits = {
        "coin_sum_amr": walk_qubits, "coin_sum_fixed": walk_qubits,
        "cs_upper": 0, "cs_average": 0,
        "single_pair": single_pair_step(history)[0].num_qubits,
        "pairwise": pairwise_moment(history, 0, 1)[0].num_qubits,
        "swap": swap_test(history)[0].num_qubits,
        "hadamard": hadamard_test(history)[0].num_qubits,
        "qae": amplitude_estimation(history, 0, 1, phase_bits)[0].num_qubits,
        "qpca": covariance_preparation(history)[0].num_qubits,
    }
    lines = [
        r"% Generated by circuit_costs.py; requires booktabs.",
        r"\begin{tabular}{@{}p{0.35\textwidth}rrp{0.19\textwidth}r@{}}",
        r"\toprule",
        r"Method & Stage qubits & Settings & Output & Step time \\",
        r"\midrule",
    ]
    for key, (label, _) in METHODS.items():
        if key in READOUTS:
            settings, output = SETTINGS[key], r"Entries$^*$"
        elif key.startswith("cs_"):
            settings, output = 0, r"Classical bound$^*$"
        else:
            settings, output = 1, "Step"
        lines.append(f"{label} & {qubits[key]} & {settings} & {output} & -- " + r"\\")
    qae_calls = SETTINGS["qae"] * (2 * ((1 << phase_bits) - 1) + 1)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\footnotesize Counts use the same $d=2$, $K=3$ history and two 8-bit step coordinates. "
        r"Coin-sum widths include the complete walk; single-pair width includes a compiled step with history supplied classically. "
        r"Other quantum widths count the covariance stage only. "
        r"$^*$These outputs require a Gaussian loader: at least 16 step-register qubits, plus implementation-dependent workspace. "
        r"Readout-stage and step registers may be reused between circuits. "
        r"Cauchy--Schwarz rows are classical arithmetic with exact variances supplied; zero counts only that stage. "
        r"The covariance-preparation replica has one coordinate qubit and two history-index qubits; its two settings read real matrix entries. "
        rf"Amplitude estimation uses {phase_bits} phase qubits and {qae_calls} total calls to $A$ or $A^{{-1}}$ "
        rf"within the {SHOTS}-call budget. "
        r"Classical loader construction, coherent data-access oracles, AMR history measurements, and hardware execution times are not included in these stage widths. "
        r"Dashes indicate that hardware step latency has not been measured.\par",
    ]
    style.FIGURES.mkdir(exist_ok=True)
    (style.FIGURES / "method_costs.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_table()
