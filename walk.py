# The walk and its two proposed steps, the covariance-scaled coin sum and the single random pair:
# their circuits, an exact sampler for them, and their grids. Classical AMR chooses a window from a
# pilot walk; quantum AMR refines each branch's grid inside the circuit; hybrid AMR does both,
# refining inside a classical AMR window. Both steps read the history straight from the walk's own
# coins, so neither needs a measurement or classical computation between steps.
#
# Notation follows the paper, and comments quote its equations in LaTeX with its macros (\bm, \P,
# \Normal). The paper counts from 1 where the code counts from 0: history[i] is \bm\xi^{(i+1)}.
#
# The walker starts at \bm\xi^{(0)} and takes N preparatory steps; its positions
# \bm\xi^{(1)}, \dots, \bm\xi^{(N)} in d = D coordinates are the history X_N, with sample covariance
#   \widehat\Sigma_N = \frac{1}{N-1} \sum_{i=1}^{N} (\bm\xi^{(i)} - \widehat{\bm\mu}_N)
#                      (\bm\xi^{(i)} - \widehat{\bm\mu}_N)^\top.
# The pair differences satisfy
#   \sum_{1 \le a < b \le N} (\bm\xi^{(a)} - \bm\xi^{(b)}) (\bm\xi^{(a)} - \bm\xi^{(b)})^\top
#     = N(N-1) \widehat\Sigma_N,
# so the coin-sum step, with independent lazy coins R_{ab} and an isotropic step \bm e in grid cells,
#   \bm\Delta^{(N+1)} = \sum_{1 \le a < b \le N} R_{ab} (\bm\xi^{(a)} - \bm\xi^{(b)}) + \bm e,
#   \P[R_{ab} = \pm 1] = q/2, \quad \P[R_{ab} = 0] = 1 - q, \quad
#   q = \frac{\gamma^2}{N(N-1)}, \quad \gamma^2 = \frac{2.38^2}{d},
#   \P[e_k = \pm 1] = \nu/2, \quad \P[e_k = 0] = 1 - \nu,
# has covariance exactly
#   \operatorname{Cov}(\bm\Delta^{(N+1)} \mid X_N) = \gamma^2 \widehat\Sigma_N + \nu I_d.
# The single random pair adds one uniformly chosen pair difference with a Rademacher sign R,
#   \bm\Delta^{(N+1)} = R (\tilde{\bm\xi}^{(A)} - \tilde{\bm\xi}^{(B)}) + \bm e,
# where \tilde{\bm\xi} is the history built from the modes scaled by \gamma / \sqrt2 and rounded
# (random_pair_draws), so its covariance is \gamma^2 \widehat\Sigma_N + \nu I_d up to that rounding.

import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation


# Problem setup
D = 2                           # d, the number of coordinates
N = 3                           # history samples xi^(1), ..., xi^(N)
GAMMA2 = 2.38**2 / D            # gamma^2: the step covariance is gamma^2 Sigma_N
ISO_VAR = .1                    # nu, the variance of the extra +-1 step, in grid cells
XI0 = np.array([0.69, 0.37])    # xi^(0), the start

# The d preparatory displacement modes v_l, one per row; the first couples the two coordinates
MODES = np.array([[0.0020, 0.0012],
                  [0.0007, 0.0015]])
PAIRS = list(itertools.combinations(range(N), 2))

# Random pairs: each step makes `draws` draws of a pair (A, B), uniform over the pairs, with a fair
# sign R and a move coin u that is 1 with probability p, and adds u R (\bm\xi^{(A)} - \bm\xi^{(B)}),
# with the modes scaled by \beta. A uniformly chosen pair difference has covariance 2 \widehat\Sigma_N,
# so the step covariance is 2 K p \beta^2 \widehat\Sigma_N for K draws. Returns (K, p, \beta):
#   K = 1 (the single random pair): p = 1 and \beta = \gamma / \sqrt2, rounded with the modes,
#     because p = \gamma^2 / 2 would exceed 1 for d < 3
#   K >= 2: p = \gamma^2 / (2K) and \beta = 1, which is exact
def random_pair_draws(draws):
    return (1, 1., np.sqrt(GAMMA2 / 2)) if draws == 1 else (draws, GAMMA2 / (2 * draws), 1.)


SINGLE_RANDOM_PAIR = random_pair_draws(1)
INDEX_BITS = (len(PAIRS) - 1).bit_length()
UNIFORM_PAIRS = np.pad(np.full(len(PAIRS), len(PAIRS) ** -.5), (0, (1 << INDEX_BITS) - len(PAIRS)))

# Register widths of the final walk and of the coarse pilot walk
BITS = 8
PILOT_BITS = 6

# The fixed local grid (centre, axes, spacing): 256 cells across 0.08 around the start. Quantum
# AMR refines it.
LOCAL_GRID = (XI0, np.eye(D), np.full(D, .08 / (1 << BITS)))


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


# add_signed with a value that depends on a counter register: values[g] when the counter
# holds g. The value is split over subsets of the counter's bits, values[g] = sum of a_S over
# the subsets S of g's 1 bits, so each non-zero a_S is one adder controlled on the bits in S.
# Without a counter, values has one entry and this is a plain add_signed.
def add_counted(qc, register, values, sign, controls=(), counter=()):
    for subset in range(len(values)):
        # a_S by inclusion-exclusion over the subsets of S
        term = sum((-1) ** bin(subset ^ part).count("1") * values[part]
                   for part in range(len(values)) if part & subset == part)
        if term:
            bits = [counter[bit] for bit in range(len(counter)) if (subset >> bit) & 1]
            add_signed(qc, register, int(term), sign, [*controls, *bits])


# Coins for a lazy coin R = u (1 - 2 s) from the qubits move (u) and sign (s):
#   \P[R = \pm 1] = q/2, \quad \P[R = 0] = 1 - q
def lazy_coin(qc, q, move, sign):
    qc.ry(2 * np.arcsin(np.sqrt(q)), move)
    qc.h(sign)


# ---------------------------------------------------------------------------
# The walk circuit
# ---------------------------------------------------------------------------

# N preparatory steps from `start`, then `steps` coin-sum steps, on integer registers.
# grid_modes holds the modes in grid units, shape (D, D). For quantum AMR it holds them on
# each branch's own grid instead, shape (N + 1, D, D), indexed by the branch's number of
# disagreeing coin pairs (quantum_amr_modes).
# With random_pairs = (draws, move probability), each step is random pairs (random_pair_draws)
# instead of the coin sum, adding step_modes, the modes scaled and then rounded (default grid_modes).
# Returns the circuit, the record registers, the preparatory coins and the quantum-AMR counter.
def make_walk(bits, start, grid_modes, steps=1, record_bits=0, random_pairs=None, step_modes=None):
    grid_modes = np.reshape(grid_modes, (-1, D, D))
    step_modes = grid_modes if step_modes is None else np.reshape(step_modes, (-1, D, D))
    quantum_amr = len(grid_modes) > 1

    # The optional record registers hold the total step, so it can be read even when the
    # position register wraps around; they must be wide enough for the largest possible step.
    # A step adds at most every pair difference (coin sum) or one per draw.
    terms = len(PAIRS) if random_pairs is None else random_pairs[0]
    largest_step = steps * (terms * (N - 1) * np.abs(step_modes).sum(axis=1).max(axis=0) + 1)
    if record_bits and np.any(largest_step >= 1 << (record_bits - 1)):
        raise ValueError("the record register could overflow")

    # Registers: the coin sum's move and sign coins, or each draw's pair index, sign and (if
    # lazy) move coin, plus one flag qubit
    position = [QuantumRegister(bits, f"position_{p}") for p in range(D)]
    record = [QuantumRegister(record_bits, f"record_{p}") for p in range(D)] if record_bits else []
    prep = QuantumRegister(N * D, "preparation_signs")
    if random_pairs is None:
        move = QuantumRegister(len(PAIRS) * steps, "pair_move")
        sign = QuantumRegister(len(PAIRS) * steps, "pair_sign")
        step_registers = [move, sign]
    else:
        draws, probability = random_pairs
        index = [QuantumRegister(INDEX_BITS, f"pair_index_{k}") for k in range(draws * steps)]
        move = QuantumRegister(draws * steps, "pair_move") if probability < 1 else None
        sign = QuantumRegister(draws * steps, "pair_sign")
        flag = QuantumRegister(1, "pair_flag")
        step_registers = [*index, *([move] if move else []), sign, flag]
    iso_move = QuantumRegister(D * steps, "iso_move")
    iso_sign = QuantumRegister(D * steps, "iso_sign")
    parity_register = QuantumRegister(1, "parity")
    parity = parity_register[0]
    qc = QuantumCircuit(*position, *record, prep, *step_registers, iso_move, iso_sign, parity_register)

    # Quantum AMR adds a counter of disagreeing coin pairs, 0 to N
    counter = QuantumRegister(N.bit_length(), "disagreements") if quantum_amr else ()
    if quantum_amr:
        qc.add_register(counter)

    # Start at XI0
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
        for t in range(N):
            qc.cx(prep[t * D], parity)
            qc.cx(prep[t * D + 1], parity)
            add_const(qc, counter, 1, [parity])

            # Uncompute the parity
            qc.cx(prep[t * D + 1], parity)
            qc.cx(prep[t * D], parity)

    # Preparatory steps on the branch's grid, with the coins c_{i,\ell} = 1 - 2 prep[i-1, \ell-1]:
    #   \bm\xi^{(i)} = \bm\xi^{(i-1)} + \sum_{\ell=1}^{d} c_{i,\ell} \mathbf v_\ell
    for t in range(N):
        for m in range(D):
            for p in range(D):
                add_counted(qc, position[p], grid_modes[:, m, p], prep[t * D + m], counter=counter)

    # Coin-sum steps, each with fresh lazy coins R_{ab} and \bm e:
    #   \bm\Delta^{(N+1)} = \sum_{1 \le a < b \le N} R_{ab} (\bm\xi^{(a)} - \bm\xi^{(b)}) + \bm e
    # The preparatory coins fix the whole history,
    #   \bm\xi^{(a)} - \bm\xi^{(b)} = -\sum_{i=a+1}^{b} \sum_{\ell=1}^{d} c_{i,\ell} \mathbf v_\ell,
    # so each pair difference is added straight from them and the history is never copied into registers.
    q = GAMMA2 / (N * (N - 1))
    for s in range(steps):
        if random_pairs is not None:
            add_random_pairs(qc, position, record, prep, parity, counter, step_modes, random_pairs,
                             index[s * draws:(s + 1) * draws], move[s * draws:(s + 1) * draws] if move else None,
                             sign[s * draws:(s + 1) * draws], flag[0])

        for j, (a, b) in enumerate(PAIRS if random_pairs is None else []):
            pair_move, pair_sign = move[s * len(PAIRS) + j], sign[s * len(PAIRS) + j]
            lazy_coin(qc, q, pair_move, pair_sign)

            # In code indices, history[a] - history[b] = -sum_{t=a+1..b} sum_m (1 - 2 prep[t, m]) v_m.
            # R_{ab} = u_{ab} (1 - 2 s_{ab}) flips its sign, so add -v_m with sign = pair_sign XOR prep[t, m].
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

        # The isotropic step e_k = u_k (1 - 2 s_k), \P[e_k = \pm 1] = \nu/2, keeps the step from
        # collapsing with the history
        for p in range(D):
            lazy_coin(qc, ISO_VAR, iso_move[s * D + p], iso_sign[s * D + p])
        for registers in (position, record):
            for p, register in enumerate(registers):
                add_signed(qc, register, 1, iso_sign[s * D + p], [iso_move[s * D + p]])

    return qc, record, prep, counter


# One random-pair step. Each draw prepares a pair index (A, B), uniform over the pairs, a fair sign
# R and, if lazy, a move coin u, and adds, with the step modes \tilde{\mathbf v}_\ell = \beta \mathbf v_\ell,
#   u R (\tilde{\bm\xi}^{(A)} - \tilde{\bm\xi}^{(B)})
#     = -u R \sum_{i=A+1}^{B} \sum_{\ell=1}^{d} c_{i,\ell} \tilde{\mathbf v}_\ell
# straight from the preparatory coins, as the coin sum does: the history is never measured or
# compiled in. Each history increment is added once, controlled on a flag that is set when it lies
# inside the drawn pair.
def add_random_pairs(qc, position, record, prep, parity, counter, step_modes, random_pairs, index, move, sign,
                     flag):
    draws, probability = random_pairs
    for k in range(draws):
        qc.append(StatePreparation(UNIFORM_PAIRS), index[k])
        if move:
            lazy_coin(qc, probability, move[k], sign[k])
        else:
            qc.h(sign[k])
        controls = [move[k], flag] if move else [flag]

        # In code indices, history[a] - history[b] = -sum_{t=a+1..b} sum_m (1 - 2 prep[t, m]) v_m: set the
        # flag for increment t when the drawn pair (a, b) has a < t <= b
        for t in range(1, N):
            inside = [j for j, (a, b) in enumerate(PAIRS) if a < t <= b]
            for j in inside:
                qc.mcx(list(index[k]), flag, ctrl_state=j)

            # Add -v_m with sign = pair sign XOR prep[t, m], as in the coin sum
            for m in range(D):
                qc.cx(prep[t * D + m], parity)
                qc.cx(sign[k], parity)
                for registers in (position, record):
                    for p, register in enumerate(registers):
                        add_counted(qc, register, -step_modes[:, m, p], parity, controls, counter)
                qc.cx(sign[k], parity)
                qc.cx(prep[t * D + m], parity)

            # Uncompute the flag
            for j in inside:
                qc.mcx(list(index[k]), flag, ctrl_state=j)


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
        qubits = [qc.find_bit(q).index for q in instruction.qubits]
        *controls, target = qubits

        # A joint coin, such as a pair index: a fresh register prepared in a fixed state, read as one random
        # index with the squared amplitudes as probabilities, least significant qubit first
        if gate.name == "state_preparation":
            if used[qubits].any():
                raise ValueError("a coin must be prepared on a fresh qubit")
            probabilities = np.abs(np.asarray(gate.params, dtype=complex)) ** 2
            value = rng.choice(len(probabilities), size=shots, p=probabilities / probabilities.sum())
            bits[qubits] = (value >> np.arange(len(qubits))[:, None]) & 1
            coin[qubits] = used[qubits] = True
            continue

        # A fresh coin
        if gate.name in ("h", "ry"):
            if used[target]:
                raise ValueError("a coin must be prepared on a fresh qubit")
            p_one = .5 if gate.name == "h" else np.sin(float(gate.params[0]) / 2) ** 2
            bits[target] = rng.random(shots) < p_one
            coin[target] = True

        # A (multi-)controlled X, whose controls may be open (Qiskit names those *_o<ctrl_state>)
        elif gate.name.split("_o")[0] in ("x", "cx", "ccx", "mcx", "mcx_gray"):
            if coin[target]:
                raise ValueError("a coin may only be used as a control")
            state = getattr(gate, "ctrl_state", (1 << len(controls)) - 1)
            wanted = ((state >> np.arange(len(controls))) & 1).astype(bool)
            bits[target] ^= np.all(bits[controls] == wanted[:, None], axis=0)

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


# Sample the walk on a grid. Returns the history in grid units, shape (shots, N, D), the
# recorded total coin-sum step in grid units, shape (shots, D), or None if nothing is recorded,
# and each shot's grid spacing, shape (shots, D). With quantum_amr, each branch refines the
# grid inside the circuit, and its positions count from the start at the register's centre.
def run_walk(bits, grid, shots, seed, modes=MODES, steps=1, record_bits=0, quantum_amr=False, random_pairs=None):
    # Round the start and the modes to the grid (to each branch's grid with quantum AMR), then
    # build and sample the circuit
    center, axes, spacing = grid
    if quantum_amr:
        start = np.full(D, 1 << (bits - 1))
        levels, grid_modes = quantum_amr_modes(modes @ axes, spacing, bits)
    else:
        start = to_grid(XI0, grid, bits)
        levels, grid_modes = np.zeros((1, D), dtype=int), np.rint(modes @ axes / spacing).astype(int)[None]

    # Random pairs (draws, move probability, \beta) add their own modes, scaled by \beta
    step_modes = None if random_pairs is None else scaled_modes(modes @ axes, spacing, levels, random_pairs[2])
    qc, record, prep, _ = make_walk(bits, start, grid_modes, steps, record_bits,
                                    None if random_pairs is None else random_pairs[:2], step_modes)
    measured = sample_circuit(qc, shots, seed)

    # Decode the history from the measured preparatory coins, on each shot's grid
    signs = 1 - 2 * measured[[qc.find_bit(q).index for q in prep]].T.reshape(-1, N, D).astype(int)
    g = disagreements(signs) if quantum_amr else np.zeros(shots, dtype=int)
    history = start + np.cumsum(np.einsum("stm,smp->stp", signs, grid_modes[g]), axis=1)
    if np.any((history < 0) | (history >= 1 << bits)):
        raise ValueError("a history left the grid")

    step = read(measured, qc, record, signed=True) if record_bits else None
    return history, step, spacing / 2.0 ** levels[g]


# The modes along the grid's axes, scaled by \beta and then rounded on each refinement level,
# shape (levels, D, D); with \beta = 1 these are the grid modes themselves
def scaled_modes(modes, spacing, levels, scale):
    return np.rint(scale * modes[None] * 2.0 ** levels[:, None, :] / spacing).astype(int)


# Exact lattice covariance of a random-pair step with K draws on a branch with these preparatory
# signs (N, D) and step modes (D, D), whose history built from the step modes is \tilde{\bm\xi}:
#   K p \binom{N}{2}^{-1} \sum_{a<b} (\tilde{\bm\xi}^{(a)} - \tilde{\bm\xi}^{(b)})
#     (\tilde{\bm\xi}^{(a)} - \tilde{\bm\xi}^{(b)})^\top + \nu I_d = 2 K p \widehat\Sigma(\tilde{\bm\xi}) + \nu I_d
def random_pair_covariance(signs, step_modes, random_pairs):
    draws, probability = random_pairs[:2]
    history = np.cumsum(signs @ step_modes, axis=0)
    return 2 * draws * probability * np.cov(history.T) + ISO_VAR * np.eye(D)


# All 2^(N D) = 64 possible preparatory sign patterns, and their physical histories
def all_histories(modes=MODES):
    signs = 1 - 2 * np.array(list(itertools.product((0, 1), repeat=N * D))).reshape(-1, N, D)
    return signs, XI0 + np.cumsum(signs @ modes, axis=1)


# ---------------------------------------------------------------------------
# Classical AMR: choose the next circuit's grid from histories measured in a coarse pilot walk
# ---------------------------------------------------------------------------

# Physical histories measured in a pilot walk: a 6-bit grid of the given physical width,
# centred on XI0 and aligned with the coordinate axes
def pilot_histories(modes=MODES, width=.08, shots=20_000, seed=11):
    grid = (XI0, np.eye(D), np.full(D, width / (1 << PILOT_BITS)))
    history, _, _ = run_walk(PILOT_BITS, grid, shots, seed, modes)
    return to_physical(history, grid, PILOT_BITS)


# The classical AMR window: centred on the histories' mean, aligned with their principal axes,
# with the 2^BITS cells spanning `width` standard deviations along each axis
def amr_window(histories, width=16):
    points = histories.reshape(-1, D)
    variances, axes = np.linalg.eigh(np.cov(points.T))
    return points.mean(axis=0), axes, np.sqrt(variances) * width / (1 << BITS)


# ---------------------------------------------------------------------------
# Quantum AMR: inside the circuit, each branch refines its own grid. No pilot, no measurement.
# Refining the fixed local grid is quantum AMR; refining a classical AMR window is hybrid AMR.
# ---------------------------------------------------------------------------

# With two coins per step (d = 2), a branch's path length in coordinate k is
#   \sum_{i=1}^{N} \lvert \bm\xi^{(i)} - \bm\xi^{(i-1)} \rvert_k = (N - g) \lvert v_{1k} + v_{2k} \rvert
#                                                              + g \lvert v_{1k} - v_{2k} \rvert,
# where g is the number of steps whose two coins disagree and v_{\ell k} is coordinate k of \mathbf v_\ell.
# For each g, the branch uses the finest power-of-two refinement \delta_k / 2^{z_k(g)} of the base
# grid's spacing at which that path length stays below half the register, 2^{n-1} cells; its
# history and every pair difference then fit around the start.
# `modes` are given along the base grid's axes. Returns the refinement levels, shape (N + 1, D),
# and the modes in each branch's grid units, shape (N + 1, D, D), for make_walk.
def quantum_amr_modes(modes, spacing, bits=BITS, finest=12):
    levels = np.zeros((N + 1, D), dtype=int)
    grid_modes = np.zeros((N + 1, D, D), dtype=int)

    for g, p in itertools.product(range(N + 1), range(D)):
        # Every level whose path length fits; the branch uses the finest
        fitting = []
        for level in range(finest + 1):
            column = np.rint(modes[:, p] * 2**level / spacing[p]).astype(int)
            if (N - g) * abs(column[0] + column[1]) + g * abs(column[0] - column[1]) < 1 << (bits - 1):
                fitting.append((level, column))
        if not fitting:
            raise ValueError("the walk does not fit the base grid")
        levels[g, p], grid_modes[g, :, p] = fitting[-1]

    return levels, grid_modes


# g, the number of preparatory steps whose two coins disagree, for signs of shape (..., N, D)
def disagreements(signs):
    return np.sum(signs[..., 0] != signs[..., 1], axis=-1)


# The grid of the branch with these preparatory signs, shape (..., N, D), when it refines the
# base grid: the base grid's axes, a finer spacing, and centred on the start
def quantum_amr_grid(signs, grid, modes=MODES):
    center, axes, spacing = grid
    levels, _ = quantum_amr_modes(modes @ axes, spacing)
    return XI0, axes, spacing / 2.0 ** levels[disagreements(signs)]


# Exact physical covariance of the coin-sum step for each of the 64 branches, shape (64, D, D),
# each on its own refinement of the base grid (centre, V, \bm\delta):
#   B_g (\gamma^2 \widehat\Sigma_N + \nu I_d) B_g^\top, \quad B_g = V \operatorname{diag}(\delta_k / 2^{z_k(g)}),
# with \widehat\Sigma_N of the branch's history in its own grid units
def quantum_amr_step_covariances(grid, modes=MODES):
    center, axes, spacing = grid
    levels, grid_modes = quantum_amr_modes(modes @ axes, spacing)
    signs, _ = all_histories(modes)
    g = disagreements(signs)

    history = np.cumsum(np.einsum("btm,bmp->btp", signs, grid_modes[g]), axis=1)
    centered = history - history.mean(axis=1, keepdims=True)
    lattice = GAMMA2 * np.einsum("btp,btq->bpq", centered, centered) / (N - 1) + ISO_VAR * np.eye(D)
    basis = axes * (spacing / 2.0 ** levels[g])[:, None, :]
    return basis @ lattice @ np.swapaxes(basis, 1, 2)
