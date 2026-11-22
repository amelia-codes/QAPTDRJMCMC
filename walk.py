"""The covariance-scaled coin sum: its circuit, an exact sampler, and the AMR grid.

The walker moves in D = 2 dimensions. It first takes K = 3 preparatory steps,
each a sum of two fixed displacement modes (the rows of MODES) with random
signs. Its K positions h_1, ..., h_K are the history. A covariance-scaled
coin-sum step ("coin sum" for short) then adds

    sum over pairs a < b of r_ab (h_a - h_b),

where each r_ab is -1, 0 or +1 with probabilities q/2, 1 - q and q/2. Since
sum_{a<b} (h_a - h_b)(h_a - h_b)^T = K (K - 1) S_h, with S_h the sample
covariance of the history, choosing q = SCALE2 / (K (K - 1)) gives the step
covariance SCALE2 S_h. A small extra +-1 step in each coordinate (variance
ISO_VAR) keeps the step from collapsing when the history does.

Positions are integers in quantum registers. A grid (centre, axes, spacing)
maps physical points to those integers. AMR chooses the grid for the next
circuit from the histories measured in a coarse first circuit (the pilot).

Every gate is either a coin (H or RY on a fresh |0> qubit) or a controlled X.
Controlled X gates only permute basis states, so measuring the circuit gives
the same distribution as drawing the coins at random and applying the gates
to ordinary bits. sample_circuit does exactly that and refuses any circuit
outside this class.
"""

import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

D = 2                           # coordinates
K = 3                           # history positions
SCALE2 = 2.38**2 / D            # step covariance = SCALE2 * history covariance
ISO_VAR = .1                    # variance of the extra +-1 step, in grid cells
X0 = np.array([0.69, 0.37])     # starting point
MODES = np.array([[0.0020, 0.0012],    # the two preparatory displacement modes;
                  [0.0007, 0.0015]])   # the first couples x and y
PAIRS = list(itertools.combinations(range(K), 2))
BITS = 8                        # register width of the final walk
PILOT_BITS = 6                  # register width of the coarse pilot walk


# Integer arithmetic with controlled X gates

def add_const(qc, register, value, controls=()):
    """Add an integer to a register, modulo 2^len(register).

    Qubits are least significant first. Adding 2^bit flips that bit and every
    higher bit whose lower bits (from `bit` up) are all 1, i.e. the carry.
    """
    value %= 1 << len(register)
    for bit in range(len(register)):
        if (value >> bit) & 1:
            for target in reversed(range(bit, len(register))):
                carry = list(controls) + list(register[bit:target])
                if carry:
                    qc.mcx(carry, register[target])
                else:
                    qc.x(register[target])


def add_signed(qc, register, value, sign, controls=()):
    """Add +value if the sign qubit is 0, or -value if it is 1."""
    add_const(qc, register, value, controls)
    add_const(qc, register, -2 * value, [*controls, sign])


def lazy_coin(qc, q, move, sign):
    """Coins for r = move * (1 - 2 sign): -1, 0, +1 with probabilities q/2, 1-q, q/2."""
    qc.ry(2 * np.arcsin(np.sqrt(q)), move)
    qc.h(sign)


# The walk circuit

def make_walk(bits, start, grid_modes, steps=1, record_bits=0):
    """K preparatory steps from `start`, then `steps` coin-sum steps, on integer registers.

    The preparatory sign coins determine the whole history, so each pair
    difference h_a - h_b is added straight from those coins and the history
    is never copied into registers. If record_bits > 0, every coin-sum step is also
    added to wide signed "record" registers, so the total step can be read
    even when the position register wraps around.
    Returns the circuit, the record registers and the preparatory coins.
    """
    largest_step = steps * (len(PAIRS) * (K - 1) * np.abs(grid_modes).sum(axis=0) + 1)
    if record_bits and np.any(largest_step >= 1 << (record_bits - 1)):
        raise ValueError("the record register could overflow")
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

    for p in range(D):
        for bit in range(bits):
            if (start[p] >> bit) & 1:
                qc.x(position[p][bit])

    # Preparatory steps: h_t = h_{t-1} + sum_m (1 - 2 prep[t, m]) mode_m.
    for t in range(K):
        for m in range(D):
            coin = prep[t * D + m]
            qc.h(coin)
            for p in range(D):
                if grid_modes[m, p]:
                    add_signed(qc, position[p], int(grid_modes[m, p]), coin)

    q = SCALE2 / (K * (K - 1))
    for s in range(steps):
        for j, (a, b) in enumerate(PAIRS):
            pair_move, pair_sign = move[s * len(PAIRS) + j], sign[s * len(PAIRS) + j]
            lazy_coin(qc, q, pair_move, pair_sign)
            # h_a - h_b = -sum_{t=a+1..b} sum_m (1 - 2 prep[t, m]) mode_m, and the
            # pair sign flips it, so add -mode_m with sign = pair_sign XOR prep[t, m].
            for t in range(a + 1, b + 1):
                for m in range(D):
                    qc.cx(prep[t * D + m], parity)
                    qc.cx(pair_sign, parity)
                    for registers in (position, record):
                        for p, register in enumerate(registers):
                            if grid_modes[m, p]:
                                add_signed(qc, register, -int(grid_modes[m, p]), parity, [pair_move])
                    qc.cx(pair_sign, parity)
                    qc.cx(prep[t * D + m], parity)
        for p in range(D):
            lazy_coin(qc, ISO_VAR, iso_move[s * D + p], iso_sign[s * D + p])
        for registers in (position, record):
            for p, register in enumerate(registers):
                add_signed(qc, register, 1, iso_sign[s * D + p], [iso_move[s * D + p]])
    return qc, record, prep


# Exact sampling of coin-plus-controlled-X circuits

def sample_circuit(qc, shots, seed):
    """Measure every qubit `shots` times; returns booleans of shape (qubits, shots).

    A coin is drawn when H or RY acts on a fresh |0> qubit. After that the
    qubit may only be a control. Under these rules no two computational paths
    ever meet again, so there is no interference and drawing the coins
    classically gives exactly the circuit's measurement distribution.
    """
    rng = np.random.default_rng(seed)
    bits = np.zeros((qc.num_qubits, shots), dtype=bool)
    used = np.zeros(qc.num_qubits, dtype=bool)
    coin = np.zeros(qc.num_qubits, dtype=bool)
    for instruction in qc.data:
        gate = instruction.operation
        *controls, target = [qc.find_bit(q).index for q in instruction.qubits]
        if gate.name in ("h", "ry"):
            if used[target]:
                raise ValueError("a coin must be prepared on a fresh qubit")
            p_one = .5 if gate.name == "h" else np.sin(float(gate.params[0]) / 2) ** 2
            bits[target] = rng.random(shots) < p_one
            coin[target] = True
        elif gate.name in ("x", "cx", "ccx", "mcx", "mcx_gray"):
            if coin[target]:
                raise ValueError("a coin may only be used as a control")
            bits[target] ^= np.all(bits[controls], axis=0)
        else:
            raise ValueError(f"unsupported gate: {gate.name}")
        used[target] = True
    return bits


def read(bits, qc, registers, signed=False):
    """Integer value of each register in each shot, shape (shots, len(registers))."""
    values = []
    for register in registers:
        index = [qc.find_bit(q).index for q in register]
        value = (bits[index] * (1 << np.arange(len(register)))[:, None]).sum(axis=0)
        if signed:
            value = np.where(value >= 1 << (len(register) - 1), value - (1 << len(register)), value)
        values.append(value)
    return np.stack(values, axis=1)


# Running the walk on a grid

def to_grid(x, grid, bits):
    """Nearest grid point to physical point x; the grid centre maps to 2^(bits-1)."""
    center, axes, spacing = grid
    return np.rint((x - center) @ axes / spacing + (1 << (bits - 1))).astype(int)


def to_physical(points, grid, bits):
    center, axes, spacing = grid
    return (points - (1 << (bits - 1))) * spacing @ axes.T + center


def run_walk(bits, grid, shots, seed, modes=MODES, steps=1, record_bits=0):
    """Sample the walk on a grid.

    Returns the history in grid units, shape (shots, K, D), decoded from the
    measured preparatory coins, and the recorded total coin-sum step in grid units,
    shape (shots, D), or None when nothing is recorded.
    """
    center, axes, spacing = grid
    start = to_grid(X0, grid, bits)
    grid_modes = np.rint(modes @ axes / spacing).astype(int)
    qc, record, prep = make_walk(bits, start, grid_modes, steps, record_bits)
    measured = sample_circuit(qc, shots, seed)
    coins = measured[[qc.find_bit(q).index for q in prep]].T.reshape(-1, K, D)
    history = start + np.cumsum((1 - 2 * coins.astype(int)) @ grid_modes, axis=1)
    if np.any((history < 0) | (history >= 1 << bits)):
        raise ValueError("a history left the grid")
    return history, read(measured, qc, record, signed=True) if record_bits else None


def all_histories(modes=MODES):
    """All 2^(K D) = 64 possible preparatory sign patterns and physical histories."""
    signs = 1 - 2 * np.array(list(itertools.product((0, 1), repeat=K * D))).reshape(-1, K, D)
    return signs, X0 + np.cumsum(signs @ modes, axis=1)


# AMR: choose the grid from a coarse pilot walk

def pilot_histories(modes=MODES, width=.08, shots=20_000, seed=11):
    """Physical histories measured in a pilot walk: a 6-bit grid of the given
    physical width, centred on X0 and aligned with the coordinate axes."""
    grid = (X0, np.eye(D), np.full(D, width / (1 << PILOT_BITS)))
    history, _ = run_walk(PILOT_BITS, grid, shots, seed, modes)
    return to_physical(history, grid, PILOT_BITS)


def amr_window(histories, width=16):
    """The AMR grid for measured histories: centred on their mean, aligned with
    their principal axes, with the 2^BITS cells spanning `width` standard
    deviations along each axis."""
    points = histories.reshape(-1, D)
    variances, axes = np.linalg.eigh(np.cov(points.T))
    return points.mean(axis=0), axes, np.sqrt(variances) * width / (1 << BITS)
