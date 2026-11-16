"""QCLT + AMR as a general-purpose adaptive Metropolis sampler (any posterior with continuous parameters).

The same move as the wavelet sampler (tdrjmcmc_qclt.py, method "qamr"), without the gravitational-wave parts
(no birth/death of wavelets, no tempering, no time-frequency cells):

  * every parameter lives on a 16-bit grid over its allowed range (periodic parameters wrap);
  * after a warm-up of T0 iterations, 80% of moves are the adaptive move and 20% a tuned isotropic random walk;
  * the adaptive move is one of
      "am"   Adaptive Metropolis (Haario et al. 2001): Gaussian step, covariance = scaled running covariance of the
             whole history (classical);
      "qclt" QCLT kick on 16-bit registers spanning the full range: eps = sum_{a<b} r_ab (h_a - h_b) over K stored
             history states h, lazy three-point coins r_ab (quantum kick, simulated exactly);
      "qamr" QCLT + AMR (ours): the same kick on AMR_BITS-bit registers whose window follows the walk (adaptive
             gridlines, "zoom in & out" of adaptive_grid.py): the registers hold coordinates along the principal
             axes of the recent (exponentially weighted) history covariance, centred on the recent mean, with
             about AMR_RES grid points per spread; re-gridded every AMR_EVERY history states, zooming out x2
             while more than 20% of the AMR moves fall outside the window (quantum kick, simulated exactly);
      "qamr_axis" the same with one register per parameter (axis-aligned windows, power-of-2 spacing set by
             each parameter's own spread), as a check of why the principal axes are needed;
    and "rw" uses only the tuned isotropic random walk (classical baseline);
  * the step scale s_c is tuned to 23.4% acceptance (Andrieu & Thoms 2008), as in Tian et al.;
  * everything adaptive (scales, history, reservoir, windows) is frozen after n_adapt iterations, so the chain is
    an ordinary Metropolis chain afterwards.

The QCLT step lies in the span of the stored differences h_a - h_b, which has at most K - 1 dimensions, so K must
exceed the number of parameters d; here K = 2d. The wavelet sampler uses K = 6 for d = 5.

"Quantum (simulated)": the kick distribution is drawn by an exact classical sampler for the circuit's gate class;
validate_kick() checks it against the qcov circuit. No quantum hardware is involved.
"""
import itertools
import math

import numpy as np

NB = 16
G = 1 << NB
AMR_BITS = 6
AMR_RES = 4.0                      # grid points per recent spread along each register axis
AMR_MEM = 200                      # memory of the recent mean / covariance (history states)
AMR_EVERY = 50                     # re-grid every this many history states
AMR_OUT_HI, AMR_OUT_LO = 0.2, 0.05   # zoom out (x2) / back in (/2) when this share of AMR moves falls outside
TARGET_ACC = 0.234
T0 = 500
P_FALLBACK = 0.2
N_MIN_HIST = 20
LOG_MIN, LOG_MAX = math.log(1e-10), math.log(1e10)
SAMPLERS = ["rw", "am", "qclt", "qamr_axis", "qamr"]
LABEL = {"rw": "Random walk, tuned step (classical)",
         "am": "Adaptive Metropolis, Haario et al. (classical)",
         "qclt": "QCLT, 16-bit registers (quantum kick, simulated)",
         "qamr_axis": "QCLT + AMR, axis-aligned 6-bit windows (quantum kick, simulated)",
         "qamr": "QCLT + AMR, ours: principal-axis 6-bit windows (quantum kick, simulated)"}


def n_stored(d):
    return 2 * d


def out_bits(K, bits):
    """Width of each signed output register: |eps| <= n_pairs (2^bits - 1), so bits + ceil(log2 n_pairs) + 1."""
    return bits + math.ceil(math.log2(K * (K - 1) // 2)) + 1


def kick_qubits(d, bits, K=None):
    """Qubits of one QCLT kick circuit: K history registers of `bits` bits per parameter, one output register
    per parameter (out_bits), and two coin qubits per pair. Not counted: the register for the current state and
    any scratch for minimal-image (periodic) differences. With K = 6 and d = 5 (the wavelet sampler): 615 qubits
    at 16 bits, 265 at 6 bits."""
    K = n_stored(d) if K is None else K
    return K * d * bits + d * out_bits(K, bits) + K * (K - 1)


def wrapd(x, per, period=G):
    """Minimal-image difference for the periodic components (integer period: exact integer arithmetic)."""
    half = period // 2 if isinstance(period, (int, np.integer)) else period / 2
    return np.where(per, (x + half) % period - half, x)


def pair_diffs(buf):
    a, b = np.array(list(itertools.combinations(range(len(buf)), 2))).T
    return buf[a] - buf[b]


def qclt_kick(dm, q, p_iso, rng):
    """eps = sum_pairs r (h_a - h_b) + iso, r in {-1, 0, +1} w.p. {q/2, 1 - q, q/2}; iso in {-1, 0, +1} per axis."""
    r = np.where(rng.random(len(dm)) < q, np.where(rng.random(len(dm)) < 0.5, -1, 1), 0)
    iso = np.where(rng.random(dm.shape[1]) < p_iso, np.where(rng.random(dm.shape[1]) < 0.5, -1, 1), 0)
    return r @ dm + iso


def iact(x, c=5.0):
    """Integrated autocorrelation time (Sokal's adaptive window)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0] if acf[0] > 0 else 1.0
    tau = 2 * np.cumsum(acf) - 1
    m = np.arange(n)
    w = np.nonzero(m >= c * tau)[0]
    return float(max(tau[w[0]] if len(w) else tau[-1], 1.0))


def run(problem, sampler, seed, n_iter, n_adapt, thin=10):
    """One chain. Returns thinned samples (unit-cube coordinates), cumulative likelihood evaluations at the same
    points, acceptance rates, and the final AMR windows (unit coordinates)."""
    rng = np.random.default_rng(seed)
    d, per = problem.d, np.asarray(problem.periodic, bool)
    K = n_stored(d)
    sd_fac = 2.38**2 / d
    nreg = 1 << AMR_BITS
    u = np.minimum(np.floor(problem.start(rng) * G), G - 1).astype(np.int64)
    lp = problem.logp((u + 0.5) / G)
    evals = 1
    n, mean, M2 = 0, np.zeros(d), np.zeros((d, d))
    res = np.zeros((K, d), np.int64)
    half = nreg // 2
    win_c, win_V, win_s, win_next = np.full(d, G / 2), np.eye(d), np.full(d, float(G >> AMR_BITS)), N_MIN_HIST
    ew_mean, ew_cov, widen, n_try, n_out = np.zeros(d), np.zeros((d, d)), 1.0, 0, 0

    def regs(v):                                   # register values of 16-bit point(s) v in the current window
        return np.floor(wrapd(np.asarray(v) - win_c, per, float(G)) @ win_V / win_s).astype(np.int64) + half
    log_sc, i_ad = 0.0, 0
    log_si, i_iso = math.log(0.02), 0              # isotropic step: fraction of the range
    xs = np.empty((n_iter // thin, d), np.float32)
    ev = np.empty(n_iter // thin, np.int64)
    acc = {"iso": [0, 0], "ad": [0, 0]}
    for t in range(n_iter):
        adapting = t < n_adapt
        use_ad = sampler != "rw" and t >= T0 and n >= N_MIN_HIST and rng.random() >= P_FALLBACK
        held, delta = True, None
        if not use_ad:
            kind = "iso"
            delta = np.rint(rng.normal(size=d) * math.exp(log_si) * G).astype(np.int64)
        else:
            kind = "ad"
            sc = math.exp(log_sc)
            if sampler == "am":
                C = sc * sd_fac * (M2 / (n - 1) + np.eye(d))
                w, V = np.linalg.eigh(0.5 * (C + C.T))
                delta = np.rint(V @ (np.sqrt(np.clip(w, 0, None)) * rng.normal(size=d))).astype(np.int64)
            elif sampler == "qclt":
                dm = wrapd(pair_diffs(res), per)
                delta = qclt_kick(dm, min(1.0, sc * sd_fac / (K * (K - 1))), min(1.0, sc * sd_fac), rng)
            else:                                   # QCLT + AMR: registers inside the adaptive window
                k = regs(u)
                n_try += 1
                if np.any((k < 0) | (k >= nreg)):
                    held = False                   # the register cannot hold the current point: rejection
                    n_out += 1
                else:
                    hk = np.clip(regs(res), 0, nreg - 1)
                    eps = qclt_kick(pair_diffs(hk), min(1.0, sc * sd_fac / (K * (K - 1))), min(1.0, sc * sd_fac), rng)
                    # displacement: depends only on eps (symmetric); both ends must lie in the window
                    delta = np.rint(win_V @ (win_s * eps)).astype(np.int64)
                    k2 = regs(np.where(per, (u + delta) % G, u + delta))
                    if not np.all((k2 >= 0) & (k2 < nreg)):
                        delta = None
                        n_out += 1
        alpha = 0.0
        if delta is not None:
            cand = np.where(per, (u + delta) % G, u + delta)
            if delta.any() and np.all(per | ((cand >= 0) & (cand < G))):
                lpc = problem.logp((cand + 0.5) / G)
                evals += 1
                alpha = math.exp(min(0.0, lpc - lp))
                if rng.random() < alpha:
                    u, lp = cand, lpc
                    acc[kind][0] += 1
        acc[kind][1] += 1
        if adapting:
            if kind == "iso":
                i_iso += 1
                log_si = min(max(log_si + (i_iso + 1) ** -0.5 * (alpha - TARGET_ACC), LOG_MIN), 0.0)
            elif held:
                i_ad += 1
                log_sc = min(max(log_sc + (i_ad + 1) ** -0.5 * (alpha - TARGET_ACC), LOG_MIN), LOG_MAX)
            if sampler != "rw":                     # history: Welford (minimal image for periodic) + reservoir
                n += 1
                dev = wrapd(u - mean, per)
                mean = mean + dev / n
                mean = np.where(per, mean % G, mean)
                M2 += np.outer(dev, wrapd(u - mean, per))
                if n <= K:
                    res[n - 1] = u
                else:
                    j = rng.integers(n)
                    if j < K:
                        res[j] = u
                if sampler in ("qamr", "qamr_axis"):
                    a = max(1.0 / n, 1.0 / AMR_MEM)
                    dv = wrapd(u - ew_mean, per, float(G))
                    ew_mean = ew_mean + a * dv
                    ew_mean = np.where(per, ew_mean % G, ew_mean)
                    ew_cov = (1 - a) * (ew_cov + a * np.outer(dv, dv))
                if sampler in ("qamr", "qamr_axis") and n >= win_next:
                    if n_try:                      # zoom in & out
                        widen = widen * 2.0 if n_out / n_try > AMR_OUT_HI else (
                            max(1.0, widen / 2.0) if n_out / n_try < AMR_OUT_LO else widen)
                    n_try = n_out = 0
                    C, mean_w = ew_cov, ew_mean
                    if sampler == "qamr_axis":     # one register per parameter, power-of-2 spacing
                        V, sd = np.eye(d), widen * np.sqrt(np.maximum(np.diag(C), 1.0))
                        win_s = np.minimum(np.exp2(np.floor(np.log2(np.maximum(sd / AMR_RES, 1.0)))), G >> AMR_BITS)
                        width = win_s * nreg
                        lo = np.rint(mean_w) - width // 2
                        win_c = np.where(per, lo % G, np.clip(lo, 0, G - width)) + width / 2
                    else:                          # principal axes of the recent covariance
                        lam, V = np.linalg.eigh(0.5 * (C + C.T))
                        win_s = np.maximum(widen * np.sqrt(np.maximum(lam, 0.0)) / AMR_RES, 1.0)
                        win_c = mean_w.copy()
                    win_V = V
                    win_next = n + AMR_EVERY
        if t % thin == 0:
            xs[t // thin] = (u + 0.5) / G
            ev[t // thin] = evals
    return dict(x=xs, evals=ev, thin=thin, n_adapt=n_adapt,
                acc={k: a / m if m else np.nan for k, (a, m) in acc.items()},
                window=dict(c=win_c / G, V=win_V, s=win_s / G, nreg=nreg) if sampler.startswith("qamr") else None)


def validate_kick(K=4, d=3, bits=6, q=0.05, seed=0):
    """Total-variation distance between the exact output distribution of the qcov circuit and that of
    qclt_kick (without the isotropic term) for random register contents."""
    from qiskit import QuantumCircuit, QuantumRegister
    from qcov import covariance_kick, exact_distribution, regs_values
    rng = np.random.default_rng(seed)
    buf = rng.integers(0, 1 << bits, size=(K, d))
    H = [[QuantumRegister(bits, f"p{p}_h{k}") for p in range(d)] for k in range(K)]
    E = [QuantumRegister(out_bits(K, bits), f"p{p}_eps") for p in range(d)]
    npairs = K * (K - 1) // 2
    mv, sg = QuantumRegister(npairs, "move"), QuantumRegister(npairs, "sign")
    qc = QuantumCircuit(*E, *[r for h in H for r in h], mv, sg)
    for k in range(K):
        for p in range(d):
            for i in range(bits):
                if (buf[k, p] >> i) & 1:
                    qc.x(H[k][p][i])
    covariance_kick(qc, [E], H, zip(mv, sg), q * K * (K - 1), periodic=False)
    bits_out, w = exact_distribution(qc)
    circ = {}
    for e, p in zip(map(tuple, regs_values(bits_out, qc, E, signed=True)), w):
        circ[e] = circ.get(e, 0.0) + p
    dm = pair_diffs(buf)
    ref = {}
    for rs in itertools.product((-1, 0, 1), repeat=npairs):
        p = np.prod([q / 2 if r else 1 - q for r in rs])
        e = tuple(np.array(rs) @ dm)
        ref[e] = ref.get(e, 0.0) + p
    return 0.5 * sum(abs(circ.get(k, 0) - ref.get(k, 0)) for k in set(circ) | set(ref)), qc.num_qubits


def validate_kick_sampled(K=6, d=5, bits=6, q=0.2, shots=20_000, seed=0):
    """Production-size check (too many coins to enumerate): draw `shots` outputs of the qcov circuit with
    sample_circuit and the same number from qclt_kick (no isotropic term), and compare them per parameter with a
    two-sample Kolmogorov-Smirnov test. Returns (smallest p-value, qubits, relative Frobenius difference of the
    circuit's sample covariance from the exact q (h-pair) covariance)."""
    from qiskit import QuantumCircuit, QuantumRegister
    from scipy import stats
    from qcov import covariance_kick, regs_values, sample_circuit
    rng = np.random.default_rng(seed)
    buf = rng.integers(0, 1 << bits, size=(K, d))
    H = [[QuantumRegister(bits, f"p{p}_h{k}") for p in range(d)] for k in range(K)]
    E = [QuantumRegister(out_bits(K, bits), f"p{p}_eps") for p in range(d)]
    npairs = K * (K - 1) // 2
    mv, sg = QuantumRegister(npairs, "move"), QuantumRegister(npairs, "sign")
    qc = QuantumCircuit(*E, *[r for h in H for r in h], mv, sg)
    for k in range(K):
        for p in range(d):
            for i in range(bits):
                if (buf[k, p] >> i) & 1:
                    qc.x(H[k][p][i])
    covariance_kick(qc, [E], H, zip(mv, sg), q * K * (K - 1), periodic=False)
    circ = regs_values(sample_circuit(qc, shots, seed=seed + 1), qc, E, signed=True)
    dm = pair_diffs(buf)
    cls = np.array([qclt_kick(dm, q, 0.0, rng) for _ in range(shots)])
    p_min = min(stats.ks_2samp(circ[:, p], cls[:, p]).pvalue for p in range(d))
    exact = q * dm.T @ dm                              # Cov(eps) = q sum_pairs (h_a - h_b)(h_a - h_b)^T
    cov_err = np.linalg.norm(np.cov(circ.T) - exact) / np.linalg.norm(exact)
    return p_min, qc.num_qubits, cov_err
