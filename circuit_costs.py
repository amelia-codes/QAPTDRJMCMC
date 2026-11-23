# Explicit circuits for the comparison methods, and their qubit counts.
#
# covariance_methods.py samples each readout's measurement law directly; the circuits here are
# where those laws come from. test_covariance_methods.py checks each circuit's outcome
# probability against its law, and write_table counts their qubits for figures/method_costs.tex.
#
# Each readout circuit is compiled from the history it reads (its data loading). The estimates use
# measured outcomes only: every variance, norm or trace a readout needs comes from a
# pair-product circuit, not from the classical data. The two proposed steps, the coin sum and the
# single random pair, are walk circuits (walk.make_walk) and need no data loading.
#
#   python3.12 circuit_costs.py    # write figures/method_costs.tex

import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, GroverOperator, RYGate, StatePreparation

import plot_style as style
from covariance_methods import METHODS, READOUTS, SETTINGS, SHOTS, SHOWN, pair_products, phase_states
from walk import BITS, D, LOCAL_GRID, MODES, N, SINGLE_RANDOM_PAIR, make_walk, quantum_amr_modes, scaled_modes


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
# with weights |row|^2, so the coordinate qubits hold rho = Sigma_N / tr(Sigma_N). Returns the
# circuit and rho. The normalisation discards tr(Sigma_N); the readout measures it with a
# pair-product circuit.
def covariance_preparation(history):
    centered = history - history.mean(axis=0)
    N, d = centered.shape
    data_bits, index_bits = max(1, (d - 1).bit_length()), (N - 1).bit_length()

    amplitudes = np.zeros((1 << index_bits, 1 << data_bits))
    amplitudes[:N, :d] = centered

    data, index = QuantumRegister(data_bits, "coordinate"), QuantumRegister(index_bits, "history")
    qc = QuantumCircuit(data, index)
    norm = np.linalg.norm(amplitudes)
    if not norm:
        return qc, np.zeros((d, d))
    qc.append(StatePreparation((amplitudes / norm).ravel()), [*data, *index])

    return qc, centered.T @ centered / norm**2


# Pair-product circuit: a uniform superposition over ordered pairs (i, j) of history samples, then
# an ancilla rotation that makes P(1) the mean of values[i, j], rescaled into [0, 1]. The readouts
# load products of pair differences (covariance_methods.pair_products), whose mean needs no centring.
def pair_product_circuit(values):
    N = len(values)
    bits = (N - 1).bit_length()
    first, second, ancilla = QuantumRegister(bits, "i"), QuantumRegister(bits, "j"), QuantumRegister(1, "value")
    qc = QuantumCircuit(first, second, ancilla)

    # Uniform superposition over the N real indices in each register
    uniform = np.pad(np.full(N, 1 / np.sqrt(N)), (0, (1 << bits) - N))
    for register in (first, second):
        qc.append(StatePreparation(uniform), register)

    # One controlled rotation per index pair, scaled into [0, 1]
    lo, hi = values.min(), values.max()
    if hi == lo:
        return qc, 0.
    for i, j in itertools.product(range(N), repeat=2):
        probability = (values[i, j] - lo) / (hi - lo)
        if probability:
            rotation = RYGate(2 * np.arcsin(np.sqrt(probability))).control(2 * bits, ctrl_state=i + (j << bits))
            qc.append(rotation, [*first, *second, ancilla[0]])

    return qc, float((values.mean() - lo) / (hi - lo))


# Canonical amplitude estimation of the pair-product probability; the phase register is
# the last `phase_bits` qubits
def amplitude_estimation(values, phase_bits):
    preparation, probability = pair_product_circuit(values)

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
# The qubit-count table
# ---------------------------------------------------------------------------

def write_table():
    # Every circuit is built for the same history on the fixed local grid
    spacing = LOCAL_GRID[2]
    history = np.cumsum(np.ones((N, D)) @ np.rint(MODES / spacing), axis=0)
    phase_bits = phase_states(SHOTS // SETTINGS["qae"]).bit_length() - 1
    # The two proposed steps' walks, on a fixed grid and with quantum AMR's counter
    levels, quantum_modes = quantum_amr_modes(MODES, spacing)
    walk_qubits = {}
    for key, random_pairs in (("coin_sum", None), ("single_pair", SINGLE_RANDOM_PAIR)):
        for amr, grid_modes, grid_levels in (("fixed", np.eye(D, dtype=int), np.zeros((1, D), int)),
                                             ("quantum", quantum_modes, levels)):
            step_modes = None if random_pairs is None else scaled_modes(MODES, spacing, grid_levels,
                                                                         random_pairs[2])
            qc = make_walk(BITS, np.zeros(D, int), grid_modes, random_pairs=random_pairs and random_pairs[:2],
                           step_modes=step_modes)[0]
            walk_qubits[key, amr] = qc.num_qubits

    # The pair-product circuits the readouts use to measure a variance or the trace
    variance = pair_product_circuit(pair_products(history, 0, 0))[0].num_qubits
    trace = pair_product_circuit(pair_products(history, 0, 0) + pair_products(history, 1, 1))[0].num_qubits

    # A method's width is that of its widest circuit; its circuits run one after another
    qubits = {
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

    # The two proposed steps on each kind of grid: quantum AMR (and hybrid AMR, shown with
    # --classical-amr) adds the 2-qubit counter
    fixed, refined = ((", fixed local grid or classical AMR", " + quantum or hybrid AMR") if style.CLASSICAL_AMR
                      else (", fixed local grid", " + quantum AMR"))
    for key in ("coin_sum", "single_pair"):
        name = METHODS[key][0]
        lines.append(f"{name}{fixed} & {walk_qubits[key, 'fixed']} & 1 & Step & -- " + r"\\")
        lines.append(f"{name}{refined} & {walk_qubits[key, 'quantum']} & 1 & Step & -- " + r"\\")

    # The readouts, whose widths do not depend on the grid
    for key in [key for key in SHOWN if key in READOUTS]:
        lines.append(f"{METHODS[key][0]} & {qubits[key]} & {SETTINGS[key]} & Entries & -- " + r"\\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    style.FIGURES.mkdir(exist_ok=True)
    (style.FIGURES / "method_costs.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_table()
