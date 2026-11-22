# Explicit circuits for the comparison methods, and their qubit counts.
#
# covariance_methods.py samples each readout's measurement law directly; the circuits here are
# where those laws come from. test_covariance_methods.py checks each circuit's outcome
# probability against its law, and write_table counts their qubits for figures/method_costs.tex.
#
# Each circuit is compiled from the history it reads (its data loading). The estimates use
# measured outcomes only: every variance, norm or trace a readout needs comes from a
# pairwise-moment circuit, not from the classical data.
#
#   python3.12 circuit_costs.py    # write figures/method_costs.tex

import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, GroverOperator, RYGate, StatePreparation

import plot_style as style
from covariance_methods import METHODS, READOUTS, SETTINGS, SHOTS, SHOWN, pair_products, phase_states
from walk import BITS, D, ISO_VAR, K, MODES, SCALE2, add_signed, lazy_coin, make_walk, quantum_amr_modes


# Centred columns p and q, zero-padded to a power of two, normalised; and their norms
def unit_columns(history, p, q):
    centered = history - history.mean(axis=0)
    size = 1 << (len(history) - 1).bit_length()
    columns = [np.pad(centered[:, c], (0, size - len(history))) for c in (p, q)]
    norms = np.array([np.linalg.norm(c) for c in columns])
    # A zero column has zero covariance scale; any unit state can represent it.
    return [c / n if n else np.eye(1, len(c))[0] for c, n in zip(columns, norms)], norms


# ---------------------------------------------------------------------------
# Readout circuits. Each returns the circuit and the outcome probability its law uses.
# ---------------------------------------------------------------------------

# Hadamard test: the ancilla reads 0 with probability (1 + x.y / (|x| |y|)) / 2
def hadamard_test(history, p=0, q=1):
    (x, y), _ = unit_columns(history, p, q)
    data, ancilla = QuantumRegister(int(np.log2(len(x))), "data"), QuantumRegister(1, "ancilla")
    qc = QuantumCircuit(data, ancilla)

    # Load x if the ancilla is 0 and y if it is 1, then interfere the two branches
    qc.h(ancilla)
    qc.append(StatePreparation(x).control(1, ctrl_state=0), [ancilla[0], *data])
    qc.append(StatePreparation(y).control(1), [ancilla[0], *data])
    qc.h(ancilla)

    return qc, float((1 + x @ y) / 2)


# SWAP test between (x, -y)-weighted flag states: the ancilla reads 0 with probability
# 1/2 + |x - y|^2 / (4 (|x|^2 + |y|^2))
def swap_test(history, p=0, q=1):
    (x, y), norms = unit_columns(history, p, q)
    data = QuantumRegister(int(np.log2(len(x))), "data")
    flag, phi, ancilla = (QuantumRegister(1, name) for name in ("flag", "phi", "ancilla"))
    qc = QuantumCircuit(data, flag, phi, ancilla)

    # The two states to compare
    qc.append(StatePreparation(np.concatenate([x, y]) / np.sqrt(2)), [*data, flag[0]])
    length = np.linalg.norm(norms)
    phi_state = norms * [1, -1] / length if length else np.array([1, -1]) / np.sqrt(2)
    qc.append(StatePreparation(phi_state), phi)

    # The SWAP test itself
    qc.h(ancilla)
    qc.cswap(ancilla[0], flag[0], phi[0])
    qc.h(ancilla)

    difference = norms[0] * x - norms[1] * y
    return qc, float(.5 + difference @ difference / (4 * norms @ norms)) if length else .5


# Covariance matrix preparation (Gordon et al. 2022): prepare the rows of the centred history
# with weights |row|^2, so the coordinate qubits hold rho = S / tr(S). Returns the circuit and rho.
# The normalisation discards tr(S); the readout measures it with a pairwise-moment circuit.
def covariance_preparation(history):
    centered = history - history.mean(axis=0)
    k, d = centered.shape
    data_bits, index_bits = max(1, (d - 1).bit_length()), (k - 1).bit_length()

    amplitudes = np.zeros((1 << index_bits, 1 << data_bits))
    amplitudes[:k, :d] = centered

    data, index = QuantumRegister(data_bits, "coordinate"), QuantumRegister(index_bits, "history")
    qc = QuantumCircuit(data, index)
    norm = np.linalg.norm(amplitudes)
    if not norm:
        return qc, np.zeros((d, d))
    qc.append(StatePreparation((amplitudes / norm).ravel()), [*data, *index])

    return qc, centered.T @ centered / norm**2


# Pairwise moment: a uniform superposition over ordered pairs (i, j) of history positions, then
# an ancilla rotation that makes P(1) the mean of values[i, j], rescaled into [0, 1]. The readouts
# load products of pair differences (covariance_methods.pair_products), whose mean needs no centring.
def pairwise_moment(values):
    k = len(values)
    bits = (k - 1).bit_length()
    first, second, ancilla = QuantumRegister(bits, "i"), QuantumRegister(bits, "j"), QuantumRegister(1, "value")
    qc = QuantumCircuit(first, second, ancilla)

    # Uniform superposition over the k real indices in each register
    uniform = np.pad(np.full(k, 1 / np.sqrt(k)), (0, (1 << bits) - k))
    for register in (first, second):
        qc.append(StatePreparation(uniform), register)

    # One controlled rotation per index pair, scaled into [0, 1]
    lo, hi = values.min(), values.max()
    if hi == lo:
        return qc, 0.
    for i, j in itertools.product(range(k), repeat=2):
        probability = (values[i, j] - lo) / (hi - lo)
        if probability:
            rotation = RYGate(2 * np.arcsin(np.sqrt(probability))).control(2 * bits, ctrl_state=i + (j << bits))
            qc.append(rotation, [*first, *second, ancilla[0]])

    return qc, float((values.mean() - lo) / (hi - lo))


# Canonical amplitude estimation of the pairwise-moment probability; the phase register is
# the last `phase_bits` qubits
def amplitude_estimation(values, phase_bits):
    preparation, probability = pairwise_moment(values)

    # Grover operator that marks the ancilla's 1 outcome
    oracle = QuantumCircuit(preparation.num_qubits)
    oracle.z(preparation.num_qubits - 1)
    grover = GroverOperator(oracle, state_preparation=preparation)

    # Phase estimation of the Grover operator
    phase = QuantumRegister(phase_bits, "phase")
    qc = QuantumCircuit(*preparation.qregs, phase)
    qc.compose(preparation, qubits=list(range(preparation.num_qubits)), inplace=True)
    qc.h(phase)
    for bit in range(phase_bits):
        qc.append(grover.power(1 << bit).control(), [phase[bit], *preparation.qubits])
    qc.append(QFT(phase_bits, inverse=True), phase)

    return qc, probability


# ---------------------------------------------------------------------------
# The single-pair step, which outputs a step rather than covariance entries
# ---------------------------------------------------------------------------

# Pick one history pair uniformly, add its difference with a random sign, scaled by
# sqrt(SCALE2 / 2), plus the +-1 lattice step. The differences are compiled in as constants.
# Returns the circuit and its step registers.
def single_pair_step(history, bits=BITS):
    k, d = history.shape
    pairs = list(itertools.combinations(range(k), 2))
    deltas = np.rint(np.sqrt(SCALE2 / 2) * np.array([history[a] - history[b] for a, b in pairs])).astype(int)
    if np.any(np.abs(deltas) + 1 >= 1 << (bits - 1)):
        raise ValueError("the single-pair step could overflow its signed register")
    index_bits = max(1, (len(pairs) - 1).bit_length())

    # Registers
    step = [QuantumRegister(bits, f"step_{p}") for p in range(d)]
    index, sign = QuantumRegister(index_bits, "pair"), QuantumRegister(1, "sign")
    iso_move, iso_sign = QuantumRegister(d, "iso_move"), QuantumRegister(d, "iso_sign")
    qc = QuantumCircuit(*step, index, sign, iso_move, iso_sign)

    # A uniformly chosen pair and a random sign
    qc.append(StatePreparation(np.pad(np.full(len(pairs), 1 / np.sqrt(len(pairs))),
                                      (0, (1 << index_bits) - len(pairs)))), index)
    qc.h(sign)

    # Add pair i's difference when the index register equals i
    for i, delta in enumerate(deltas):
        zeros = [qubit for bit, qubit in enumerate(index) if not (i >> bit) & 1]
        qc.x(zeros)
        for register, value in zip(step, delta):
            if value:
                add_signed(qc, register, int(value), sign[0], list(index))
        qc.x(zeros)

    # The extra +-1 step in each coordinate
    for p in range(d):
        lazy_coin(qc, ISO_VAR, iso_move[p], iso_sign[p])
        add_signed(qc, step[p], 1, iso_sign[p], [iso_move[p]])

    return qc, step


# ---------------------------------------------------------------------------
# The qubit-count table
# ---------------------------------------------------------------------------

def write_table():
    # Every circuit is built for the same history on the fixed local grid
    spacing = .08 / 256
    history = np.cumsum(np.ones((K, D)) @ np.rint(MODES / spacing), axis=0)
    phase_bits = phase_states(SHOTS // SETTINGS["qae"]).bit_length() - 1
    walk_qubits = make_walk(BITS, np.zeros(D, int), np.eye(D, dtype=int))[0].num_qubits
    quantum_modes = quantum_amr_modes(MODES, np.full(D, spacing))[1]
    quantum_amr_qubits = make_walk(BITS, np.zeros(D, int), quantum_modes)[0].num_qubits

    # The pairwise-moment circuits the readouts use to measure a variance or the trace
    variance = pairwise_moment(pair_products(history, 0, 0))[0].num_qubits
    trace = pairwise_moment(pair_products(history, 0, 0) + pair_products(history, 1, 1))[0].num_qubits

    # A method's width is that of its widest circuit; its circuits run one after another
    qubits = {
        "coin_sum_qamr": quantum_amr_qubits, "coin_sum_amr": walk_qubits, "coin_sum": walk_qubits,
        "single_pair": single_pair_step(history)[0].num_qubits,
        "pairwise": pairwise_moment(pair_products(history, 0, 1))[0].num_qubits,
        "swap": max(swap_test(history)[0].num_qubits, variance),
        "hadamard": max(hadamard_test(history)[0].num_qubits, variance),
        "qae": amplitude_estimation(pair_products(history, 0, 1), phase_bits)[0].num_qubits,
        "qpca": max(covariance_preparation(history)[0].num_qubits, trace),
    }

    # Header
    lines = [
        r"% Generated by circuit_costs.py; requires booktabs.",
        r"\begin{tabular}{@{}p{0.35\textwidth}rrp{0.19\textwidth}r@{}}",
        r"\toprule",
        r"Method & Stage qubits & Settings & Output & Step time \\",
        r"\midrule",
    ]

    # One row per method
    for key in SHOWN:
        label = METHODS[key][0]
        if key in READOUTS:
            settings, output = SETTINGS[key], r"Entries$^*$"
        else:
            settings, output = 1, "Step"
        lines.append(f"{label} & {qubits[key]} & {settings} & {output} & -- " + r"\\")

    # Footnote
    qae_calls = SETTINGS["qae"] * (2 * ((1 << phase_bits) - 1) + 1)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\footnotesize Counts use the same $d=2$, $K=3$ history and two 8-bit step coordinates. "
        r"Coin-sum widths include the complete walk; single-pair width includes a compiled step with history supplied classically. "
        r"Quantum AMR adds a 2-qubit counter of preparatory steps whose coins disagree, which sets each branch's grid inside the circuit. "
        r"Readout widths count the covariance stage only, and are those of a method's widest circuit; its settings run one after another. "
        r"$^*$These outputs require a Gaussian loader: at least 16 step-register qubits, plus implementation-dependent workspace. "
        r"Readout-stage and step registers may be reused between circuits. "
        r"Readouts measure every variance or trace they use: SWAP and Hadamard run a pairwise-moment circuit for each variance, "
        r"and covariance preparation runs one for its trace. "
        r"The covariance-preparation replica has one coordinate qubit and two history-index qubits; its settings read the diagonal, the off-diagonal and the trace. "
        rf"Amplitude estimation uses {phase_bits} phase qubits and {qae_calls} total calls to $A$ or $A^{{-1}}$ "
        rf"within the {SHOTS}-call budget. "
        r"Classical loader construction, coherent data-access oracles, classical-AMR history measurements, and hardware execution times are not included in these stage widths. "
        r"Dashes indicate that hardware step latency has not been measured.\par",
    ]

    style.FIGURES.mkdir(exist_ok=True)
    (style.FIGURES / "method_costs.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_table()
