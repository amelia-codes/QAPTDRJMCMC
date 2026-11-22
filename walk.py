# The covariance-scaled coin sum: its circuit, an exact sampler for it, and the AMR grid,
# chosen either classically from a pilot walk or inside the circuit (quantum AMR).
#
# The walker takes K preparatory steps; its K positions h_1, ..., h_K are the history.
# A coin-sum step then adds r_ab (h_a - h_b) for every pair a < b, with each coin
# r_ab = -1, 0, +1 (probabilities q/2, 1 - q, q/2). The outer products of the pair
# differences sum to K (K - 1) S_h, so q = SCALE2 / (K (K - 1)) gives covariance SCALE2 * S_h.

import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


# Problem setup
D = 2                           # coordinates
K = 3                           # history positions
SCALE2 = 2.38**2 / D            # step covariance = SCALE2 * history covariance
ISO_VAR = .1                    # variance of the extra +-1 step, in grid cells
X0 = np.array([0.69, 0.37])     # starting point

# The two preparatory displacement modes; the first couples x and y
MODES = np.array([[0.0020, 0.0012],
                  [0.0007, 0.0015]])
PAIRS = list(itertools.combinations(range(K), 2))

# Register widths of the final walk and of the coarse pilot walk
BITS = 8
PILOT_BITS = 6


# ---------------------------------------------------------------------------
# Integer arithmetic with controlled X gates
# ---------------------------------------------------------------------------

# Add an integer to a register, modulo 2^len(register); qubits are least significant first
def add_const(qc, register, value, controls=()):
    value %= 1 << len(register)

    for bit in range(len(register)):
        if (value >> bit) & 1:
            # Adding 2^bit flips that bit, plus every higher bit whose lower bits are all 1 (the carry)
            for target in reversed(range(bit, len(register))):
                carry = list(controls) + list(register[bit:target])
                if carry:
                    qc.mcx(carry, register[target])
                else:
                    qc.x(register[target])


# Add +value if the sign qubit is 0, or -value if it is 1
def add_signed(qc, register, value, sign, controls=()):
    add_const(qc, register, value, controls)
    add_const(qc, register, -2 * value, [*controls, sign])


# add_signed with a value that depends on a counter register: values[n] when the counter
# holds n. The value is split over subsets of the counter's bits, values[n] = sum of a_S over
# the subsets S of n's 1 bits, so each non-zero a_S is one adder controlled on the bits in S.
# Without a counter, values has one entry and this is a plain add_signed.
def add_counted(qc, register, values, sign, controls=(), counter=()):
    for subset in range(len(values)):
        # a_S by inclusion-exclusion over the subsets of S
        term = sum((-1) ** bin(subset ^ part).count("1") * values[part]
                   for part in range(len(values)) if part & subset == part)
        if term:
            bits = [counter[bit] for bit in range(len(counter)) if (subset >> bit) & 1]
            add_signed(qc, register, int(term), sign, [*controls, *bits])


# Coins for r = move * (1 - 2 sign), which is -1, 0, +1 with probabilities q/2, 1 - q, q/2
def lazy_coin(qc, q, move, sign):
    qc.ry(2 * np.arcsin(np.sqrt(q)), move)
    qc.h(sign)


# ---------------------------------------------------------------------------
# The walk circuit
# ---------------------------------------------------------------------------

# K preparatory steps from `start`, then `steps` coin-sum steps, on integer registers.
# grid_modes holds the modes in grid units, shape (D, D). For quantum AMR it holds them on
# each branch's own grid instead, shape (K + 1, D, D), indexed by the branch's number of
# disagreeing coin pairs (quantum_amr_modes).
# Returns the circuit, the record registers, the preparatory coins and the quantum-AMR counter.
def make_walk(bits, start, grid_modes, steps=1, record_bits=0):
    grid_modes = np.reshape(grid_modes, (-1, D, D))
    quantum_amr = len(grid_modes) > 1

    # The optional record registers hold the total step, so it can be read even when the
    # position register wraps around; they must be wide enough for the largest possible step
    largest_step = steps * (len(PAIRS) * (K - 1) * np.abs(grid_modes).sum(axis=1).max(axis=0) + 1)
    if record_bits and np.any(largest_step >= 1 << (record_bits - 1)):
        raise ValueError("the record register could overflow")

    # Registers
    position = [QuantumRegister(bits, f"position_{p}") for p in range(D)]
    record = [QuantumRegister(record_bits, f"record_{p}") for p in range(D)] if record_bits else []
    prep = QuantumRegister(K * D, "preparation_signs")
    move = QuantumRegister(len(PAIRS) * steps, "pair_move")
    sign = QuantumRegister(len(PAIRS) * steps, "pair_sign")
    iso_move = QuantumRegister(D * steps, "iso_move")
    iso_sign = QuantumRegister(D * steps, "iso_sign")
    parity_register = QuantumRegister(1, "parity")
    parity = parity_register[0]
    qc = QuantumCircuit(*position, *record, prep, move, sign, iso_move, iso_sign, parity_register)

    # Quantum AMR adds a counter of disagreeing coin pairs, 0 to K
    counter = QuantumRegister(K.bit_length(), "disagreements") if quantum_amr else ()
    if quantum_amr:
        qc.add_register(counter)

    # Start at X0
    for p in range(D):
        for bit in range(bits):
            if (start[p] >> bit) & 1:
                qc.x(position[p][bit])

    # The preparatory coins
    for coin in prep:
        qc.h(coin)

    # Quantum AMR: count the preparatory steps whose two coins disagree. The count fixes the
    # branch's path length, and so the finest grid the branch fits on.
    if quantum_amr:
        for t in range(K):
            qc.cx(prep[t * D], parity)
            qc.cx(prep[t * D + 1], parity)
            add_const(qc, counter, 1, [parity])

            # Uncompute the parity
            qc.cx(prep[t * D + 1], parity)
            qc.cx(prep[t * D], parity)

    # Preparatory steps: h_t = h_{t-1} + sum_m (1 - 2 prep[t, m]) mode_m, on the branch's grid
    for t in range(K):
        for m in range(D):
            for p in range(D):
                add_counted(qc, position[p], grid_modes[:, m, p], prep[t * D + m], counter=counter)

    # Coin-sum steps. The preparatory coins fix the whole history, so each pair difference
    # is added straight from them and the history is never copied into registers.
    q = SCALE2 / (K * (K - 1))
    for s in range(steps):
        for j, (a, b) in enumerate(PAIRS):
            pair_move, pair_sign = move[s * len(PAIRS) + j], sign[s * len(PAIRS) + j]
            lazy_coin(qc, q, pair_move, pair_sign)

            # h_a - h_b = -sum_{t=a+1..b} sum_m (1 - 2 prep[t, m]) mode_m, and the pair sign
            # flips it, so add -mode_m with sign = pair_sign XOR prep[t, m]
            for t in range(a + 1, b + 1):
                for m in range(D):
                    qc.cx(prep[t * D + m], parity)
                    qc.cx(pair_sign, parity)

                    for registers in (position, record):
                        for p, register in enumerate(registers):
                            add_counted(qc, register, -grid_modes[:, m, p], parity, [pair_move], counter)

                    # Uncompute the parity
                    qc.cx(pair_sign, parity)
                    qc.cx(prep[t * D + m], parity)

        # The extra +-1 step in each coordinate keeps the step from collapsing with the history
        for p in range(D):
            lazy_coin(qc, ISO_VAR, iso_move[s * D + p], iso_sign[s * D + p])
        for registers in (position, record):
            for p, register in enumerate(registers):
                add_signed(qc, register, 1, iso_sign[s * D + p], [iso_move[s * D + p]])

    return qc, record, prep, counter


# ---------------------------------------------------------------------------
# Exact sampling of coin-plus-controlled-X circuits
# ---------------------------------------------------------------------------

# Measure every qubit `shots` times; returns booleans of shape (qubits, shots).
# A coin is drawn when H or RY acts on a fresh |0> qubit, and after that the qubit may only
# be a control. Controlled X gates only permute basis states, so no two paths ever interfere
# and drawing the coins classically gives exactly the circuit's measurement distribution.
def sample_circuit(qc, shots, seed):
    rng = np.random.default_rng(seed)
    bits = np.zeros((qc.num_qubits, shots), dtype=bool)
    used = np.zeros(qc.num_qubits, dtype=bool)
    coin = np.zeros(qc.num_qubits, dtype=bool)

    for instruction in qc.data:
        gate = instruction.operation
        *controls, target = [qc.find_bit(q).index for q in instruction.qubits]

        # A fresh coin
        if gate.name in ("h", "ry"):
            if used[target]:
                raise ValueError("a coin must be prepared on a fresh qubit")
            p_one = .5 if gate.name == "h" else np.sin(float(gate.params[0]) / 2) ** 2
            bits[target] = rng.random(shots) < p_one
            coin[target] = True

        # A (multi-)controlled X
        elif gate.name in ("x", "cx", "ccx", "mcx", "mcx_gray"):
            if coin[target]:
                raise ValueError("a coin may only be used as a control")
            bits[target] ^= np.all(bits[controls], axis=0)

        else:
            raise ValueError(f"unsupported gate: {gate.name}")
        used[target] = True

    return bits


# Integer value of each register in each shot, shape (shots, len(registers))
def read(bits, qc, registers, signed=False):
    values = []
    for register in registers:
        index = [qc.find_bit(q).index for q in register]
        value = (bits[index] * (1 << np.arange(len(register)))[:, None]).sum(axis=0)

        # Two's complement for signed registers
        if signed:
            value = np.where(value >= 1 << (len(register) - 1), value - (1 << len(register)), value)
        values.append(value)

    return np.stack(values, axis=1)


# ---------------------------------------------------------------------------
# Running the walk on a grid. A grid (centre, axes, spacing) maps physical points to integers.
# ---------------------------------------------------------------------------

# Nearest grid point to physical point x; the grid centre maps to 2^(bits-1)
def to_grid(x, grid, bits):
    center, axes, spacing = grid
    return np.rint((x - center) @ axes / spacing + (1 << (bits - 1))).astype(int)


# Grid points back to physical coordinates
def to_physical(points, grid, bits):
    center, axes, spacing = grid
    return (points - (1 << (bits - 1))) * spacing @ axes.T + center


# Sample the walk on a grid. Returns the history in grid units, shape (shots, K, D), and the
# recorded total coin-sum step in grid units, shape (shots, D), or None if nothing is recorded.
def run_walk(bits, grid, shots, seed, modes=MODES, steps=1, record_bits=0):
    # Round the start and the modes to the grid, then build and sample the circuit
    center, axes, spacing = grid
    start = to_grid(X0, grid, bits)
    grid_modes = np.rint(modes @ axes / spacing).astype(int)
    qc, record, prep, _ = make_walk(bits, start, grid_modes, steps, record_bits)
    measured = sample_circuit(qc, shots, seed)

    # Decode the history from the measured preparatory coins
    coins = measured[[qc.find_bit(q).index for q in prep]].T.reshape(-1, K, D)
    history = start + np.cumsum((1 - 2 * coins.astype(int)) @ grid_modes, axis=1)
    if np.any((history < 0) | (history >= 1 << bits)):
        raise ValueError("a history left the grid")

    return history, read(measured, qc, record, signed=True) if record_bits else None


# All 2^(K D) = 64 possible preparatory sign patterns, and their physical histories
def all_histories(modes=MODES):
    signs = 1 - 2 * np.array(list(itertools.product((0, 1), repeat=K * D))).reshape(-1, K, D)
    return signs, X0 + np.cumsum(signs @ modes, axis=1)


# ---------------------------------------------------------------------------
# AMR: choose the next circuit's grid from histories measured in a coarse pilot walk
# ---------------------------------------------------------------------------

# Physical histories measured in a pilot walk: a 6-bit grid of the given physical width,
# centred on X0 and aligned with the coordinate axes
def pilot_histories(modes=MODES, width=.08, shots=20_000, seed=11):
    grid = (X0, np.eye(D), np.full(D, width / (1 << PILOT_BITS)))
    history, _ = run_walk(PILOT_BITS, grid, shots, seed, modes)
    return to_physical(history, grid, PILOT_BITS)


# The AMR grid: centred on the histories' mean, aligned with their principal axes, with the
# 2^BITS cells spanning `width` standard deviations along each axis
def amr_window(histories, width=16):
    points = histories.reshape(-1, D)
    variances, axes = np.linalg.eigh(np.cov(points.T))
    return points.mean(axis=0), axes, np.sqrt(variances) * width / (1 << BITS)


# ---------------------------------------------------------------------------
# Quantum AMR: inside the circuit, each branch refines its own grid. No pilot, no measurement.
# ---------------------------------------------------------------------------

# With two coins per step (D = 2), a branch's path length in coordinate p is
#   sum_t |h_t - h_{t-1}|_p = (K - n) |mode_0p + mode_1p| + n |mode_0p - mode_1p|,
# where n is the number of steps whose two coins disagree. For each n, the branch uses the
# finest power-of-two refinement of the grid (X0, I, spacing) at which that path length stays
# below half the register; its history and every pair difference then fit around the start.
# Returns the refinement levels, shape (K + 1, D), and the modes in each branch's grid units,
# shape (K + 1, D, D), for make_walk.
def quantum_amr_modes(modes, spacing, bits=BITS, finest=12):
    levels = np.zeros((K + 1, D), dtype=int)
    grid_modes = np.zeros((K + 1, D, D), dtype=int)

    for n, p in itertools.product(range(K + 1), range(D)):
        # Every level whose path length fits; the branch uses the finest
        fitting = []
        for level in range(finest + 1):
            column = np.rint(modes[:, p] * 2**level / spacing[p]).astype(int)
            if (K - n) * abs(column[0] + column[1]) + n * abs(column[0] - column[1]) < 1 << (bits - 1):
                fitting.append((level, column))
        if not fitting:
            raise ValueError("the walk does not fit the base grid")
        levels[n, p], grid_modes[n, :, p] = fitting[-1]

    return levels, grid_modes


# Exact physical covariance of the coin-sum step for the branch with these preparatory signs,
# shape (K, D), on its own quantum-AMR grid
def quantum_amr_step_covariance(signs, modes, spacing):
    levels, grid_modes = quantum_amr_modes(modes, spacing)
    n = np.sum(signs[:, 0] != signs[:, 1])

    history = np.cumsum(signs @ grid_modes[n], axis=0)
    basis = np.diag(spacing / 2.0 ** levels[n])
    return basis @ (SCALE2 * np.cov(history.T) + ISO_VAR * np.eye(D)) @ basis
