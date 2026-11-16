"""Covariance-adaptive (QCLT) kick circuits with no mid-circuit measurement or reset.

What these circuits are, precisely: every coin is a fresh |0> qubit rotated by H or RY and used only as a control
afterwards, and every other gate (X, MCX, CSWAP) permutes computational basis states. Branches therefore never
interfere: the output distribution is that of a classical randomised reversible computation driven by
independent coin bits, and sample_circuit() draws from it exactly and efficiently on a classical computer at any
size. These circuits are a coherent (measurement-free) implementation of a classical random step; by themselves
they offer no quantum speedup, and the coins cannot be uncomputed (many coin outcomes give the same step), so they
cannot serve as the garbage-free proposal of a Szegedy-type quantum walk either.

qcov.py is the maintained copy; the library cell of quantum-cov-rw.ipynb is kept in sync with it.
"""
import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.circuit import ControlledGate


# =============================================================================
# 1. Reversible integer arithmetic (X / MCX / CSWAP only -> exact, ancilla-free)
#    Registers are little-endian (reg[0] = LSB), values are taken mod 2^len(reg).
# =============================================================================

def _mcx(qc, controls, target):
    if controls:
        qc.mcx(list(controls), target)
    else:
        qc.x(target)


def increment(qc, reg, ctrls=()):
    """reg <- reg + 1 (mod 2^len(reg)), controlled on every qubit in `ctrls`."""
    for j in reversed(range(len(reg))):
        _mcx(qc, list(ctrls) + list(reg[:j]), reg[j])


def decrement(qc, reg, ctrls=()):
    """reg <- reg - 1 (mod 2^len(reg)); the exact inverse of increment()."""
    for j in range(len(reg)):
        _mcx(qc, list(ctrls) + list(reg[:j]), reg[j])


def add_const(qc, reg, k, ctrls=()):
    """reg <- reg + k (mod 2^len(reg)); k may be negative."""
    k %= 2 ** len(reg)
    for i in range(len(reg)):
        if (k >> i) & 1:
            increment(qc, reg[i:], ctrls)


def add_reg(qc, src, dst, ctrls=(), sign=+1):
    """dst <- dst + sign*src (mod 2^len(dst)); src is untouched (it is only a control)."""
    step = increment if sign > 0 else decrement
    for i in range(min(len(src), len(dst))):
        step(qc, dst[i:], list(ctrls) + [src[i]])


# =============================================================================
# 2. Coins and walk primitives
# =============================================================================

def lazy_coin(qc, q_move, move, sign):
    """Fresh three-point coin r in {-1, 0, +1} with P(r = +-1) = q_move / 2.

    move = 1 w.p. q_move (RY), sign uniform (H). Both qubits must be fresh |0>.
    """
    if np.isclose(q_move, 1.0):
        qc.x(move)
    else:
        qc.ry(2 * np.arcsin(np.sqrt(q_move)), move)
    qc.h(sign)


def iso_step(qc, targets, velocity_list, signs, moves=None):
    """Independent +-v_p shift on each parameter (identity-covariance proposal).

    sign = 0 -> +v, sign = 1 -> -v; only applied when move = 1 (if `moves` given).
    `targets` is a list of register-lists that all receive the same shift.
    """
    for regs in targets:
        for p, (reg, v) in enumerate(zip(regs, velocity_list)):
            base = [] if moves is None else [moves[p]]
            add_const(qc, reg, +v, base)
            add_const(qc, reg, -2 * v, base + [signs[p]])


def snapshot(qc, P, H):
    """H <- P by CNOT fan-out. H must be fresh |0...0> (no cloning issue in the basis)."""
    for reg, hreg in zip(P, H):
        for q, h in zip(reg, hreg):
            qc.cx(q, h)


def add_signed(qc, t, dst, ctrls=(), sign=+1):
    """dst <- dst + sign * signed(t), reading the n-bit register t as two's complement."""
    n = len(t)
    up, down = (increment, decrement) if sign > 0 else (decrement, increment)
    for i in range(n - 1):
        up(qc, dst[i:], list(ctrls) + [t[i]])
    down(qc, dst[n - 1:], list(ctrls) + [t[n - 1]])


def pair_kick(qc, targets, Ha, Hb, move, sign, periodic=False, tmp=None):
    """target <- target + move * (+-)(Ha - Hb); the history registers are restored.

    Targets as wide as the history registers (the state P) are updated mod 2^n, which is
    exact on the periodic grid. Wider targets (e.g. an unwrapped displacement tracker)
    receive either the raw integer difference (periodic=False) or the minimal-image
    difference signed((Ha - Hb) mod 2^n) (periodic=True). The latter is computed in the
    clean scratch register `tmp` and uncomputed afterwards (tmp ends in |0>: no reset).
    """
    wide_periodic = []
    for ra, rb in zip(Ha, Hb):
        for qa, qb in zip(ra, rb):
            qc.cswap(sign, qa, qb)          # sign = 1 swaps the orientation of the pair
    for regs in targets:
        for reg, ra, rb in zip(regs, Ha, Hb):
            if periodic and len(reg) > len(ra):
                wide_periodic.append((reg, ra, rb))
                continue
            add_reg(qc, ra, reg, [move], +1)
            add_reg(qc, rb, reg, [move], -1)
    for ra, rb in zip(Ha, Hb):
        for qa, qb in zip(ra, rb):
            qc.cswap(sign, qa, qb)          # uncompute the swap

    # minimal-image targets: the sign is applied arithmetically (a swap would break the
    # +-2^(n-1) tie symmetry, because signed(a-b) = signed(b-a) = -2^(n-1) there)
    for reg, ra, rb in wide_periodic:
        if tmp is None:
            raise ValueError("periodic=True with a wider target needs a scratch register `tmp`")
        t = tmp[: len(ra)]
        add_reg(qc, ra, t, (), +1)                      # t = (a - b) mod 2^n
        add_reg(qc, rb, t, (), -1)
        add_signed(qc, t, reg, [move], +1)              # reg += signed(t)
        add_signed(qc, t, reg[1:], [move, sign], -1)    # reg -= 2 signed(t) if sign = 1
        add_reg(qc, rb, t, (), +1)                      # uncompute t -> |0>
        add_reg(qc, ra, t, (), -1)


def history_pairs(K):
    return list(itertools.combinations(range(K), 2))


def kick_probability(K, scale2, n_rounds=1):
    q = scale2 / (K * (K - 1) * n_rounds)
    if not 0 < q <= 1:
        raise ValueError(
            f"q = scale2 / (K (K-1) n_rounds) = {q:.3f} must be in (0, 1]: "
            "lower scale2 or raise K / n_rounds"
        )
    return q


def covariance_kick(qc, targets, history, coins, scale2, n_rounds=1, periodic=False, tmp=None):
    """Coherent proposal  eps = sum_{rounds} sum_{a<b} r_ab (h_a - h_b).

    With r_ab in {-1, 0, +1} w.p. {q/2, 1-q, q/2} and q = scale2 / (K (K-1) n_rounds):
        E[eps | h] = 0,   Cov[eps | h] = scale2 * S_hist   (unbiased sample covariance)
    because  sum_{a<b} (h_a - h_b)(h_a - h_b)^T = K (K-1) S_hist.
    On the periodic grid the differences are minimal-image ones (see torus_cov()).
    `coins` yields one fresh (move, sign) qubit pair per (round, pair).
    """
    K = len(history)
    q = kick_probability(K, scale2, n_rounds)
    coins = iter(coins)
    for _ in range(n_rounds):
        for a, b in history_pairs(K):
            move, sign = next(coins)
            lazy_coin(qc, q, move, sign)
            pair_kick(qc, targets, history[a], history[b], move, sign, periodic, tmp)
    return q


# =============================================================================
# 3. Timestep sub-circuits (same layout as qc_timestep in qapthtdrjmcmc.ipynb)
# =============================================================================

def conventional_timestep(P, velocity_list, t, extra_targets=()):
    """Phase-1 step: independent +-v_p on every parameter, one FRESH coin per parameter."""
    coin = AncillaRegister(len(P), f"Walk{t}")
    extra = [r for regs in extra_targets for r in regs]
    step = QuantumCircuit(*P, *extra, coin, name=f"Walk{t}")
    step.h(coin)
    iso_step(step, [P, *extra_targets], velocity_list, coin)
    return step


def adaptive_timestep(P, H, t, scale2, iso_var, n_rounds=1, extra_targets=(), tmp=None):
    """Phase-2 step: covariance kick from the history registers + identity term.

    Cov[step | history] = scale2 * S_hist + iso_var * I   (grid units).
    """
    K, d = len(H), len(P)
    n_coins = n_rounds * K * (K - 1) // 2
    move = AncillaRegister(n_coins, f"Move{t}")
    sign = AncillaRegister(n_coins, f"Sign{t}")
    coin_regs = [move, sign]
    if iso_var > 0:
        imove = AncillaRegister(d, f"IsoMove{t}")
        isign = AncillaRegister(d, f"IsoSign{t}")
        coin_regs += [imove, isign]
    extra = [r for regs in extra_targets for r in regs]
    hist = [r for h in H for r in h]
    work = [tmp] if tmp is not None else []
    step = QuantumCircuit(*P, *extra, *hist, *work, *coin_regs, name=f"AdaptiveWalk{t}")
    targets = [P, *extra_targets]
    covariance_kick(step, targets, H, zip(move, sign), scale2, n_rounds, True, tmp)
    if iso_var > 0:
        for p in range(d):
            lazy_coin(step, iso_var, imove[p], isign[p])
        iso_step(step, targets, [1] * d, isign, imove)
    return step


def snapshot_step(P, H, k):
    step = QuantumCircuit(*P, *H, name=f"Snapshot{k}")
    snapshot(step, P, H)
    return step


def build_adaptive_walk(
    params,
    n_bits_list,
    velocity_list,
    t_1,
    t_2,
    scale2=None,
    iso_var=0.1,
    n_rounds=1,
    initial_values=None,
    include_initial=False,
    track_displacement=False,
    e_bits=8,
    record_path=False,
    measure=True,
):
    """Full fully-coherent adaptive walk: t_1 conventional steps (recording history),
    then t_2 adaptive steps whose covariance is the history covariance (+ iso_var * I).
    No measurement or reset anywhere before the final read-out.

    record_path=True also copies the state into a fresh record register after every
    adaptive step (for plotting the walk). Records are never read by the walk itself.
    """
    d = len(params)
    if scale2 is None:
        scale2 = 2.38 ** 2 / d                         # Haario / Gelman-Roberts-Gilks
    n_snap = t_1 + int(include_initial)
    q = kick_probability(n_snap, scale2, n_rounds)     # fail fast on an invalid scale
    for p, n, v in zip(params, n_bits_list, velocity_list):
        if (2 * v) % 2 ** n == 0:
            raise ValueError(f"{p}: +{v} and -{v} coincide mod {2**n}; the walk would be deterministic")
    P = [QuantumRegister(n, p) for p, n in zip(params, n_bits_list)]
    H = [[QuantumRegister(n, f"{p}_h{k}") for p, n in zip(params, n_bits_list)]
         for k in range(n_snap)]
    Dreg = [QuantumRegister(e_bits, f"{p}_disp") for p in params] if track_displacement else []
    extra = [Dreg] if track_displacement else []
    tmp = AncillaRegister(max(n_bits_list), "diff_tmp") if track_displacement else None

    qc = QuantumCircuit(*P, *Dreg, *[r for h in H for r in h], *([tmp] if tmp is not None else []))

    # initial basis state (X gates instead of initialize(), which contains a reset)
    for reg, val in zip(P, initial_values or [0] * d):
        for i in range(len(reg)):
            if (val >> i) & 1:
                qc.x(reg[i])
    qc.barrier()

    k = 0
    if include_initial:
        snap = snapshot_step(P, H[k], k)
        qc.append(snap, snap.qubits)
        k += 1

    # ---- phase 1: conventional walk, record the state history ---------------------
    for t in range(t_1):
        step = conventional_timestep(P, velocity_list, t)
        qc.add_register(*step.qregs[d:])
        qc.append(step, step.qubits)
        snap = snapshot_step(P, H[k], k)
        qc.append(snap, snap.qubits)
        k += 1
    qc.barrier()

    # ---- phase 2: adaptive walk, step covariance = scale2 * S_hist + iso_var * I ---
    R = []
    for t in range(t_1, t_1 + t_2):
        step = adaptive_timestep(P, H, t, scale2, iso_var, n_rounds, extra_targets=extra, tmp=tmp)
        n_own = d + len(Dreg) + len(H) * d + int(tmp is not None)
        qc.add_register(*step.qregs[n_own:])
        qc.append(step, step.qubits)
        if record_path:
            rec = [QuantumRegister(n, f"{p}_r{t}") for p, n in zip(params, n_bits_list)]
            qc.add_register(*rec)
            snap = snapshot_step(P, rec, t)
            snap.name = f"Record{t}"
            qc.append(snap, snap.qubits)
            R.append(rec)

    if measure:
        qc.barrier()
        creg_P = ClassicalRegister(sum(n_bits_list), "c_state")
        qc.add_register(creg_P)
        qc.measure(sum(map(list, P), []), creg_P)
        for kk, h in enumerate(H):
            c = ClassicalRegister(sum(n_bits_list), f"c_h{kk}")
            qc.add_register(c)
            qc.measure(sum(map(list, h), []), c)
        for t, rec in zip(range(t_1, t_1 + t_2), R):
            c = ClassicalRegister(sum(n_bits_list), f"c_r{t}")
            qc.add_register(c)
            qc.measure(sum(map(list, rec), []), c)
        if track_displacement:
            c = ClassicalRegister(d * e_bits, "c_disp")
            qc.add_register(c)
            qc.measure(sum(map(list, Dreg), []), c)

    regs = dict(P=P, H=H, D=Dreg, R=R, tmp=tmp, q=q, scale2=scale2, iso_var=iso_var)
    return qc, regs


# =============================================================================
# 4. Exact simulation for "fresh coins + reversible gates" circuits
# =============================================================================
#
# Every h / ry in these circuits hits a fresh |0> qubit that is only ever used as a
# control afterwards, so no branch ever interferes with another and (deferred
# measurement principle) the output distribution equals that of a classical
# reversible circuit driven by independent coin bits. This lets us evaluate the
# exact same QuantumCircuit object at sizes far beyond statevector memory. It is
# cross-checked against Aer's statevector simulator below.

_PERM_GATES = {"x", "cx", "ccx", "mcx", "cswap", "swap"}
_COIN_GATES = {"h", "ry"}
_SKIP = {"barrier", "measure"}


def _flat_ops(qc, qmap=None):
    """Yield (name, operation, [global qubit indices]) with composite blocks inlined."""
    if qmap is None:
        qmap = {q: i for i, q in enumerate(qc.qubits)}
    for inst in qc.data:
        op = inst.operation
        qs = [qmap[q] for q in inst.qubits]
        if getattr(op, "condition", None) is not None:
            raise ValueError(f"classically conditioned {op.name}: not supported by the sampler")
        if isinstance(op, ControlledGate) and op.base_gate.name == "x":
            if op.ctrl_state != 2 ** op.num_ctrl_qubits - 1:
                raise ValueError("open-controlled X not supported by the sampler")
            if len(qs) != op.num_ctrl_qubits + 1:
                raise ValueError(f"{op.name} with ancilla qubits (e.g. mode='v-chain') not supported by the sampler")
            yield "mcx", op, qs                 # cx, ccx, C3X, C4X, MCXGrayCode, ...
        elif op.name in _PERM_GATES | _COIN_GATES | _SKIP | {"reset", "initialize"}:
            yield op.name, op, qs
        elif op.definition is not None:
            inner = op.definition
            yield from _flat_ops(inner, {iq: qs[i] for i, iq in enumerate(inner.qubits)})
        else:
            raise ValueError(f"cannot inline gate {op.name}")


def check_coherent(qc):
    """Assert there is no reset and no mid-circuit measurement: after the first measurement only further
    measurements (and barriers) may follow. Returns the number of measurements."""
    seen_measure = False
    n_meas = 0
    for name, _, qs in _flat_ops(qc):
        if name in ("reset", "initialize"):
            raise AssertionError(f"found {name}: not allowed")
        if name == "measure":
            seen_measure = True
            n_meas += 1
        elif name != "barrier" and seen_measure:
            raise AssertionError(f"{name} after a measurement: mid-circuit measurement not allowed")
    return n_meas


def _coin_list(qc):
    coins = []
    for name, op, qs in _flat_ops(qc):
        if name in _COIN_GATES:
            p1 = 0.5 if name == "h" else float(np.sin(float(op.params[0]) / 2) ** 2)
            coins.append((qs[0], p1))
    return coins


def _run_reversible(qc, coin_columns, n_rows, init=None):
    """Propagate basis states; coin_columns[j] gives the bits of the j-th coin gate.
    `init` (num_qubits, n_rows) optionally sets the input basis states (default |0...0>)."""
    nq = qc.num_qubits
    bits = np.zeros((nq, n_rows), dtype=bool) if init is None else np.array(init, dtype=bool)
    fresh = ~bits.any(axis=1)                   # a coin must start in |0> in every row
    is_coin = np.zeros(nq, dtype=bool)
    j = 0
    for name, op, qs in _flat_ops(qc):
        if name in _SKIP:
            continue
        if name in _COIN_GATES:
            (t,) = qs
            if not fresh[t]:
                raise ValueError(f"{name} on non-fresh qubit {t}: interference possible")
            bits[t] = coin_columns[j]
            j += 1
            fresh[t] = False
            is_coin[t] = True
            continue
        if name in ("x", "cx", "ccx", "mcx"):
            *c, t = qs
            if is_coin[t]:
                raise ValueError("a coin is used as a target")
            if c:
                m = bits[c[0]].copy()
                for ci in c[1:]:
                    m &= bits[ci]
                bits[t] ^= m
            else:
                bits[t] ^= True
            fresh[t] = False
            continue
        if name in ("cswap", "swap"):
            if name == "cswap":
                c, a, b = qs
                diff = (bits[a] ^ bits[b]) & bits[c]
            else:
                a, b = qs
                diff = bits[a] ^ bits[b]
            if is_coin[a] or is_coin[b]:
                raise ValueError("a coin is swapped")
            bits[a] ^= diff
            bits[b] ^= diff
            fresh[a] = fresh[b] = False
            continue
        raise ValueError(f"unsupported gate {name}")
    return bits


def sample_circuit(qc, shots, seed=None):
    """Exact sampling of the circuit's final computational-basis state.
    Returns a bool array of shape (num_qubits, shots)."""
    rng = np.random.default_rng(seed)
    cols = [rng.random(shots) < p1 for _, p1 in _coin_list(qc)]
    return _run_reversible(qc, cols, shots)


def exact_distribution(qc, max_coins=22):
    """Exact output distribution by enumerating every coin outcome.
    Returns (bits of shape (num_qubits, rows), probabilities of shape (rows,))."""
    coins = _coin_list(qc)
    nc = len(coins)
    if nc > max_coins:
        raise ValueError(f"{nc} coins: too many to enumerate, use sample_circuit()")
    grid = ((np.arange(2 ** nc)[None, :] >> np.arange(nc)[:, None]) & 1).astype(bool)
    w = np.ones(2 ** nc)
    for j, (_, p1) in enumerate(coins):
        w *= np.where(grid[j], p1, 1 - p1)
    keep = w > 0
    return _run_reversible(qc, list(grid[:, keep]), int(keep.sum())), w[keep]


def reg_value(bits, qc, reg, signed=False):
    """Integer value of a register from a (num_qubits, rows) bit array."""
    cols = [qc.find_bit(q).index for q in reg]
    v = (bits[cols].astype(np.int64) * (1 << np.arange(len(cols)))[:, None]).sum(0)
    if signed:
        v = np.where(v >= 2 ** (len(cols) - 1), v - 2 ** len(cols), v)
    return v


def regs_values(bits, qc, regs, signed=False):
    """(rows, len(regs)) array of register values."""
    return np.stack([reg_value(bits, qc, r, signed) for r in regs], axis=1)


def weighted_cov(X, w=None):
    """Covariance of rows of X under probability weights w (population form)."""
    if w is None:
        w = np.full(len(X), 1 / len(X))
    w = w / w.sum()
    mu = w @ X
    Xc = X - mu
    return Xc.T @ (Xc * w[:, None]), mu


def torus_cov(hist, n_bits):
    """Pairwise (minimal-image) sample covariance of K snapshots on a periodic 2^n grid.

    hist: (..., K, d) integer array. Equals np.cov(..., ddof=1) whenever every pairwise difference
    |h_a - h_b| is below half the period, 2^(n-1), in every component (not merely when the snapshots avoid the
    wrap-around point); it is always PSD (a sum of outer products).
    """
    K = hist.shape[-2]
    S = 0
    for a, b in history_pairs(K):
        dlt = (hist[..., a, :] - hist[..., b, :]) % 2 ** n_bits
        dlt = np.where(dlt >= 2 ** (n_bits - 1), dlt - 2 ** n_bits, dlt)
        S = S + dlt[..., :, None] * dlt[..., None, :]
    return S / (K * (K - 1))


def cov_metrics(A, B, eps=1e-12):
    """The three metrics used in FINAL.ipynb (A = estimate, B = target)."""
    den = np.linalg.norm(A, "fro") * np.linalg.norm(B, "fro")
    return dict(
        hs_similarity=np.vdot(A, B).real / den if den > eps else np.nan,
        relative_error=np.linalg.norm(A - B, "fro") / max(np.linalg.norm(B, "fro"), eps),
        frobenius_distance=np.linalg.norm(A - B, "fro"),
    )
