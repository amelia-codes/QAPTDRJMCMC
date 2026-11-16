"""BayesWave-style trans-dimensional burst sampler with adaptive parallel tempering, comparing the
intrinsic ("covariance") step:

  bayeswave : Fisher-information-matrix Gaussian jumps (BayesWave's default for wavelet updates),
              with the full Hastings ratio because the Fisher matrix depends on the state.
  adaptive  : classical adaptive Gaussian, Cov = scale2 * S_hist + iso_var * I (Haario / Tian-Lee-Zhou),
              S_hist from a per-wavelet reservoir of K past states.
  qclt      : the quantum-CLT kick of quantum-cov-rw.ipynb on the same reservoir:
              eps = sum_{a<b} r_ab (h_a - h_b) + identity term, r_ab in {-1, 0, +1} w.p. {q/2, 1-q, q/2}.
              Same covariance as `adaptive`, but generated coherently on integer registers.

Everything else is shared: data, priors, birth/death (reversible jump), a data-driven
time-frequency birth proposal, time-frequency redraws, parallel tempering with an adaptive
temperature ladder (Vousden et al. 2016) during burn-in.

Model (as in BayesWave): white-noise whitened data d = n + h, signal = sum of K Morlet-Gabor wavelets
    psi(t) = A exp(-(t - t0)^2 / tau^2) cos(2 pi f0 (t - t0) + phi),   tau = Q / (2 pi f0),
K in {0..K_MAX} uniform. Each parameter lives on a 2^NB-point register grid (the representation the
quantum kernel acts on); all three kernels use the same grid, so they target the same posterior.

The qclt kernel is evaluated with a classical sampler of r (the circuit's coins), which is exactly
equivalent to running the circuit: `validate_qclt_against_circuit()` builds the actual qcov circuit for
a reservoir and checks that its exact output distribution equals this sampler's (TV distance ~1e-16).

Run:   python3.12 bayeswave_qclt.py            # full comparison (multiprocessing), writes figures/
       python3.12 bayeswave_qclt.py --plot-only
"""
import itertools
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================================ model / data
FS, N = 1024, 1024                       # 1 s of whitened data, sigma = 1
TT = np.arange(N) / FS
NB = 16                                  # bits per parameter register
G = 1 << NB
PARAMS = ["t0", "f0", "Q", "lnA", "phi"]
D = len(PARAMS)
LO = np.array([0.10, 30.0, 2.0, np.log(0.1), 0.0])
HI = np.array([0.90, 400.0, 30.0, np.log(20.0), 2 * np.pi])
DX = (HI - LO) / G
PERIODIC = np.array([False, False, False, False, True])
K_MAX = 10

# BayesWave-style amplitude prior: Gamma(2) in the wavelet SNR rho (vanishes as rho -> 0, which is
# what stops the sampler from piling up zero-amplitude wavelets). rho = A * c(tau) in white noise.
SNR_STAR = 5.0

# TF birth proposal: 64 x 64 cells aligned with the register grid of (t0, f0)
TF_BITS = 6
CELL_SHIFT = NB - TF_BITS

# sampler settings
N_TEMPS = 8
T_MAX = 200.0
P_RJ = 0.2               # fraction of moves that are birth/death
P_TF = 0.1               # fraction of intrinsic moves that are TF redraws (shared by all methods)
K_HIST = 6               # reservoir size = history registers in the QCLT circuit
THIN = 4                 # reservoir sees every THIN-th intrinsic update of a wavelet
SCALE2 = 2.38**2 / D     # Haario scaling of the history covariance
ISO_STEP, ISO_P = 2, 0.25            # identity term: +-2 grid units w.p. 0.25 per parameter
ISO_VAR = ISO_P * ISO_STEP**2
FISHER_SCALE = 2.38 / math.sqrt(D)
FD_STEP = 4.0            # finite-difference step (grid units) for the Fisher matrix

METHODS = ["bayeswave", "adaptive", "qclt"]
LABELS = {"bayeswave": "BayesWave (Fisher)", "adaptive": "Adaptive (classical)", "qclt": "QCLT (quantum)"}


def phys(u):
    return LO + (np.asarray(u, dtype=float) + 0.5) * DX


def snr_factor(th):
    tau = th[2] / (2 * np.pi * th[1])
    return math.sqrt(FS * tau * math.sqrt(math.pi / 2) / 2)


def log_prior_lnA(u):
    """log of the prior density of lnA given (f0, Q), times the lnA range (i.e. relative to uniform)."""
    th = phys(u)
    c = snr_factor(th)
    rho = math.exp(th[3]) * c
    x_lo, x_hi = math.exp(LO[3]) * c / SNR_STAR, math.exp(HI[3]) * c / SNR_STAR
    Z = (1 + x_lo) * math.exp(-x_lo) - (1 + x_hi) * math.exp(-x_hi)       # Gamma(2) mass in range
    x = rho / SNR_STAR
    return math.log(x * x * math.exp(-x) / Z * (HI[3] - LO[3]))           # p(lnA) = x^2 e^-x / Z


def wave_window(th):
    tau = th[2] / (2 * np.pi * th[1])
    half = 4.5 * tau
    i0 = max(0, int((th[0] - half) * FS))
    i1 = min(N, int((th[0] + half) * FS) + 2)
    return i0, i1


def wave_on(th, i0, i1):
    t0, f0, Q, lnA, phi = th
    tau = Q / (2 * np.pi * f0)
    x = TT[i0:i1] - t0
    return np.exp(lnA - (x / tau) ** 2) * np.cos(2 * np.pi * f0 * x + phi)


def make_data(seed=1, snr=20.0):
    """Chirping sine-Gaussian burst (needs several wavelets) in white Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = TT - 0.5
    h = np.exp(-(x / 0.03) ** 2) * np.cos(2 * np.pi * (120 * x + 0.5 * 3000 * x**2) + 0.7)
    h *= snr / np.linalg.norm(h)
    return h + rng.normal(size=N), h


def tf_proposal(d):
    """Cell weights from the data's quadrature matched-filter power on the (t0, f0) grid (Q = 8)."""
    n_cell = 1 << TF_BITS
    w = np.zeros((n_cell, n_cell))
    for i in range(n_cell):
        for j in range(n_cell):
            u = np.array([(i << CELL_SHIFT) + (1 << (CELL_SHIFT - 1)), (j << CELL_SHIFT) + (1 << (CELL_SHIFT - 1)), 0, 0, 0])
            th = phys(u)
            th[2], th[3] = 8.0, 0.0
            i0, i1 = wave_window(th)
            c = wave_on(th, i0, i1); th[4] = np.pi / 2; s = wave_on(th, i0, i1)
            w[i, j] = (d[i0:i1] @ c) ** 2 / (c @ c) + (d[i0:i1] @ s) ** 2 / (s @ s)
    w = np.maximum(w - 4.0, 0.0) + 0.5
    P = w / w.sum()
    return P, np.cumsum(P.ravel())


# ============================================================================ state
class Wavelet:
    __slots__ = ("u", "i0", "i1", "h", "buf", "nb", "seen", "upd")

    def __init__(self, u):
        self.set(u)
        self.buf = np.zeros((K_HIST, D), dtype=np.int64)
        self.nb = self.seen = self.upd = 0

    def set(self, u):
        self.u = np.asarray(u, dtype=np.int64)
        th = phys(self.u)
        self.i0, self.i1 = wave_window(th)
        self.h = wave_on(th, self.i0, self.i1)


class Chain:
    __slots__ = ("ws", "r", "logL")

    def __init__(self, d):
        self.ws, self.r = [], d.copy()
        self.logL = -0.5 * self.r @ self.r


def delta_logL(r, old, new):
    """Change of log L when wavelet `old` (or None) is replaced by `new` (or None)."""
    spans = [(w.i0, w.i1) for w in (old, new) if w is not None]
    lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
    seg0 = r[lo:hi]
    seg = seg0.copy()
    if old is not None:
        seg[old.i0 - lo:old.i1 - lo] += old.h
    if new is not None:
        seg[new.i0 - lo:new.i1 - lo] -= new.h
    return -0.5 * (seg @ seg - seg0 @ seg0), lo, hi, seg


# ============================================================================ proposals
def in_prior(u):
    return np.all((u >= 0) | PERIODIC) and np.all((u < G) | PERIODIC)


def wrap(u):
    u = u.copy()
    u[PERIODIC] %= G
    return u


def fisher(u):
    th = phys(u)
    i0, i1 = wave_window(th)
    J = np.empty((i1 - i0, D))
    for k in range(D):
        tp, tm = th.copy(), th.copy()
        tp[k] += FD_STEP * DX[k]
        tm[k] -= FD_STEP * DX[k]
        J[:, k] = (wave_on(tp, i0, i1) - wave_on(tm, i0, i1)) / (2 * FD_STEP)
    lam, V = np.linalg.eigh(J.T @ J)
    lam = np.maximum(lam, (8.0 / G) ** 2)                 # no single jump wider than ~G/8
    return lam, V


def gauss_logpdf(delta, lam, V, s):
    y = V.T @ delta
    return -0.5 * np.sum(lam * y**2) / s**2 + 0.5 * np.sum(np.log(lam)) - D * np.log(s)


def propose_fisher(u, rng):
    lam, V = fisher(u)
    delta = np.rint(V @ (rng.normal(size=D) / np.sqrt(lam)) * FISHER_SCALE).astype(np.int64)
    if not delta.any():
        return None, 0.0
    u_new = wrap(u + delta)
    if not in_prior(u_new):
        return None, 0.0
    lam2, V2 = fisher(u_new)
    logH = gauss_logpdf(-delta, lam2, V2, FISHER_SCALE) - gauss_logpdf(delta, lam, V, FISHER_SCALE)
    return u_new, logH


def pair_diffs(buf):
    """Raw pair differences h_a - h_b of the history registers (what the circuit adds)."""
    a, b = np.array(list(itertools.combinations(range(len(buf)), 2))).T
    return buf[a] - buf[b]


def min_image(dm):
    dm = dm.copy()
    dm[:, PERIODIC] = (dm[:, PERIODIC] + G // 2) % G - G // 2
    return dm


def propose_adaptive(w, rng):
    dm = min_image(pair_diffs(w.buf))
    S = dm.T @ dm / (K_HIST * (K_HIST - 1))
    C = SCALE2 * S + ISO_VAR * np.eye(D)
    delta = np.rint(np.linalg.cholesky(C) @ rng.normal(size=D)).astype(np.int64)
    return delta


def propose_qclt(w, rng):
    """QCLT kick: lazy three-point coins on every history pair + lattice identity term."""
    dm = pair_diffs(w.buf)
    q = SCALE2 / (K_HIST * (K_HIST - 1))
    move = rng.random(len(dm)) < q
    sign = rng.random(len(dm)) < 0.5
    r = np.where(move, np.where(sign, -1, 1), 0)
    iso = np.where(rng.random(D) < ISO_P, np.where(rng.random(D) < 0.5, -ISO_STEP, ISO_STEP), 0)
    return r @ dm + iso


def log_pi_over_g(u, P):
    """log(prior / proposal) of a TF draw: (t0, f0) from the TF cells, the rest uniform on the grid."""
    cell = P[u[0] >> CELL_SHIFT, u[1] >> CELL_SHIFT]
    return -math.log(cell * (1 << (2 * TF_BITS))) + log_prior_lnA(u)


def draw_tf(rng, cdf):
    k = int(np.searchsorted(cdf, rng.random() * cdf[-1]))
    i, j = divmod(k, 1 << TF_BITS)
    u = rng.integers(0, G, size=D)
    u[0] = (i << CELL_SHIFT) + rng.integers(0, 1 << CELL_SHIFT)
    u[1] = (j << CELL_SHIFT) + rng.integers(0, 1 << CELL_SHIFT)
    return u


# ============================================================================ sampler
def p_birth(K):
    return 1.0 if K == 0 else (0.0 if K == K_MAX else 0.5)


def run(method, seed, n_iter, n_adapt, data_seed=1, record_every=1, beta=1.0, n_temps=N_TEMPS):
    """beta scales the log-likelihood (beta = 0 samples the prior: used to test the RJ bookkeeping)."""
    d, h_inj = make_data(data_seed)
    hh = h_inj @ h_inj
    logL_inj = -0.5 * (d - h_inj) @ (d - h_inj)
    P, cdf = tf_proposal(d)
    rng = np.random.default_rng(seed)
    chains = [Chain(d) for _ in range(n_temps)]
    logT = np.linspace(0, np.log(T_MAX), n_temps)
    swap_avg = np.full(n_temps - 1, 0.5)
    n_rec = n_iter // record_every
    rec = dict(logL=np.empty(n_rec), K=np.empty(n_rec, np.int16), match=np.empty(n_rec), lnA=[])
    acc = {k: [0, 0] for k in ("cov", "fisher_fallback", "tf", "birth", "death")}
    t_start = time.process_time()

    def intrinsic(ch, T, cold):
        w = ch.ws[rng.integers(len(ch.ws))]
        u = w.u
        if rng.random() < P_TF:
            kind, u_new = "tf", draw_tf(rng, cdf)
            logH = log_pi_over_g(u_new, P) - log_pi_over_g(u, P)
        elif method == "bayeswave" or w.nb < K_HIST:
            kind = "cov" if method == "bayeswave" else "fisher_fallback"
            u_new, logH = propose_fisher(u, rng)
        else:
            kind = "cov"
            delta = (propose_qclt if method == "qclt" else propose_adaptive)(w, rng)
            u_new, logH = (wrap(u + delta), 0.0) if delta.any() else (None, 0.0)
            if u_new is not None and not in_prior(u_new):
                u_new = None
        accepted = False
        if u_new is not None and kind != "tf":
            logH += log_prior_lnA(u_new) - log_prior_lnA(u)
        if u_new is not None:
            new = Wavelet.__new__(Wavelet)
            new.set(u_new)
            dL, lo, hi, seg = delta_logL(ch.r, w, new)
            if np.log(rng.random()) < beta * dL / T + logH:
                ch.r[lo:hi] = seg
                ch.logL += dL
                w.u, w.i0, w.i1, w.h = new.u, new.i0, new.i1, new.h
                accepted = True
        if cold:
            acc[kind][0] += accepted
            acc[kind][1] += 1
        # per-wavelet reservoir of past states (the QCLT history registers); frozen after burn-in
        if adapting:
            w.upd += 1
            if w.upd % THIN == 0:
                w.seen += 1
                if w.nb < K_HIST:
                    w.buf[w.nb] = w.u
                    w.nb += 1
                else:
                    j = rng.integers(w.seen)
                    if j < K_HIST:
                        w.buf[j] = w.u

    def rj(ch, T, cold):
        K = len(ch.ws)
        if rng.random() < p_birth(K):
            new = Wavelet(draw_tf(rng, cdf))
            dL, lo, hi, seg = delta_logL(ch.r, None, new)
            log_a = beta * dL / T + log_pi_over_g(new.u, P) + np.log((1 - p_birth(K + 1)) / p_birth(K))
            kind = "birth"
            if np.log(rng.random()) < log_a:
                ch.r[lo:hi] = seg; ch.logL += dL; ch.ws.append(new)
                ok = True
            else:
                ok = False
        else:
            i = rng.integers(K)
            old = ch.ws[i]
            dL, lo, hi, seg = delta_logL(ch.r, old, None)
            log_a = beta * dL / T - log_pi_over_g(old.u, P) + np.log(p_birth(K - 1) / (1 - p_birth(K)))
            kind = "death"
            if np.log(rng.random()) < log_a:
                ch.r[lo:hi] = seg; ch.logL += dL; ch.ws.pop(i)
                ok = True
            else:
                ok = False
        if cold:
            acc[kind][0] += ok
            acc[kind][1] += 1

    for it in range(n_iter):
        adapting = it < n_adapt
        T = np.exp(logT)
        for c, ch in enumerate(chains):
            # the RJ move must be chosen with the same probability at every K (including K = 0),
            # otherwise p_birth / p_death in the acceptance ratio are wrong at the boundary
            if rng.random() < P_RJ:
                rj(ch, T[c], c == 0)
            elif ch.ws:
                intrinsic(ch, T[c], c == 0)
        # parallel-tempering swaps, hot to cold
        for c in range(n_temps - 2, -1, -1):
            a = beta * (1 / T[c] - 1 / T[c + 1]) * (chains[c + 1].logL - chains[c].logL)
            ok = np.log(rng.random()) < a
            if ok:
                chains[c], chains[c + 1] = chains[c + 1], chains[c]
            swap_avg[c] += 0.01 * (ok - swap_avg[c])
        # adaptive ladder (Vousden et al. 2016): equalise swap acceptance, T_1 = 1 and T_max fixed
        if adapting:
            kappa = 0.01 * 1000.0 / (it + 1000.0)                 # 1/nu * t0/(t + t0), nu=100, t0=1000
            gaps = np.diff(logT) * np.exp(kappa * 0.5 * (swap_avg - swap_avg.mean()))
            logT = np.concatenate([[0.0], np.cumsum(gaps / gaps.sum() * np.log(T_MAX))])
        if it % record_every == 0:
            k = it // record_every
            cold = chains[0]
            h = d - cold.r
            nh = h @ h
            rec["logL"][k] = cold.logL - logL_inj
            rec["K"][k] = len(cold.ws)
            rec["match"][k] = (h @ h_inj) / math.sqrt(nh * hh) if nh > 0 else 0.0
            if beta == 0 and cold.ws:
                th = phys(cold.ws[rng.integers(len(cold.ws))].u)          # uniformly random member
                rec["lnA"].append(th[3] + math.log(snr_factor(th)))
    rec["cpu"] = time.process_time() - t_start
    rec["acc"] = {k: (a / n if n else np.nan) for k, (a, n) in acc.items()}
    rec["temps"] = np.exp(logT)
    rec["swap"] = swap_avg
    return rec


# ============================================================================ circuit validation
def validate_qclt_against_circuit(buf, n_check_pairs_hist=4):
    """Exact output distribution of the real qcov circuit vs the qclt sampler, same reservoir."""
    from qiskit import QuantumCircuit, QuantumRegister
    from qcov import covariance_kick, exact_distribution, regs_values

    hist = buf[:n_check_pairs_hist]
    K = len(hist)
    q = SCALE2 / (K * (K - 1))
    H = [[QuantumRegister(NB, f"{p}_h{k}") for p in PARAMS] for k in range(K)]
    E = [QuantumRegister(NB + 4, f"{p}_eps") for p in PARAMS]
    npairs = K * (K - 1) // 2
    mv, sg = QuantumRegister(npairs, "move"), QuantumRegister(npairs, "sign")
    qc = QuantumCircuit(*E, *[r for h in H for r in h], mv, sg)
    for k in range(K):
        for p in range(D):
            for i in range(NB):
                if (hist[k, p] >> i) & 1:
                    qc.x(H[k][p][i])
    covariance_kick(qc, [E], H, zip(mv, sg), q * K * (K - 1), periodic=False)
    bits, w = exact_distribution(qc)
    eps = regs_values(bits, qc, E, signed=True)
    circ = {}
    for e, p in zip(map(tuple, eps), w):
        circ[e] = circ.get(e, 0.0) + p
    dm = pair_diffs(hist)
    ref = {}
    for rs in itertools.product((-1, 0, 1), repeat=npairs):
        p = np.prod([q / 2 if r else 1 - q for r in rs])
        e = tuple(np.array(rs) @ dm)
        ref[e] = ref.get(e, 0.0) + p
    keys = set(circ) | set(ref)
    tv = 0.5 * sum(abs(circ.get(k, 0) - ref.get(k, 0)) for k in keys)
    return tv, qc.num_qubits, len(qc.data)


def prior_rho_reference(n, seed=5):
    """Exact prior draws of the wavelet SNR: (f0, Q) uniform, rho ~ Gamma(2, rho*) truncated by the lnA range."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        u = rng.integers(0, G, size=D)
        c = snr_factor(phys(u))
        lo, hi = math.exp(LO[3]) * c, math.exp(HI[3]) * c
        while True:
            rho = rng.gamma(2.0, SNR_STAR)
            if lo <= rho <= hi:
                break
        out[i] = rho
    return out


def prior_test(n_iter=150000):
    """With the likelihood off every method must sample the prior: K ~ U{0..K_MAX}, SNR ~ Gamma(2, rho*)."""
    from scipy import stats
    out = {}
    from multiprocessing import Pool
    with Pool(len(METHODS)) as pool:
        runs = pool.starmap(run, [(m, 11, n_iter, n_iter // 2, 1, 1, 0.0, 1) for m in METHODS])
    for m, r in zip(METHODS, runs):
        K = r["K"][n_iter // 10:]
        hist = np.bincount(K, minlength=K_MAX + 1) / len(K)
        log_rho = np.array(r["lnA"])
        rho = np.exp(log_rho[len(log_rho) // 10::100])                  # thinned: draws are correlated
        ks_res = stats.ks_2samp(rho, prior_rho_reference(200_000))
        ks = (ks_res.statistic, ks_res.pvalue)
        out[m] = (hist, ks)
        print(f"{m:10s} P(K) = {np.round(hist, 3)}  (uniform = {1 / (K_MAX + 1):.3f});  "
              f"KS(wavelet SNR vs exact prior) D = {ks[0]:.3f}, p = {ks[1]:.2f}  [{len(rho)} thinned draws]")
    return out


if __name__ == "__main__" and "--prior-test" in sys.argv:
    prior_test()

if __name__ == "__main__" and "--smoke" in sys.argv:
    for m in METHODS:
        t0 = time.time()
        r = run(m, seed=0, n_iter=2000, n_adapt=1000, record_every=1)
        print(f"{m:10s} {time.time() - t0:5.1f}s  final match {r['match'][-1]:.3f}  dlogL {r['logL'][-1]:7.1f}  "
              f"K {r['K'][-1]}  acc {({k: round(v, 3) for k, v in r['acc'].items()})}  T {np.round(r['temps'], 1)}")


# ============================================================================ experiment
N_SEEDS = 16
N_ITER = 20000
N_ADAPT = N_ITER // 2
OUT = Path(__file__).parent / "figures"


def _run_job(args):
    method, seed = args
    return method, seed, run(method, seed, N_ITER, N_ADAPT)


def iact(x, c=5.0):
    """Integrated autocorrelation time with Sokal's automatic window."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] <= 0:
        return np.nan
    acf /= acf[0]
    taus = 2 * np.cumsum(acf) - 1
    for m in range(1, n):
        if m >= c * taus[m]:
            return taus[m]
    return taus[-1]


def split_rhat(chains):
    """Split-R-hat over an array (n_chains, n_samples)."""
    n = chains.shape[1] // 2
    x = np.concatenate([chains[:, :n], chains[:, n:2 * n]])
    W = x.var(axis=1, ddof=1).mean()
    B = n * x.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * W + B / n) / W) if W > 0 else np.nan


def experiment():
    from multiprocessing import Pool

    # --- 1. the QCLT kernel equals the qcov circuit, checked on a real reservoir -------------
    tv, nq, ng = validate_qclt_against_circuit(_example_reservoir())
    print(f"QCLT kernel vs qcov circuit (4 history registers x 5 params x {NB} bits, {nq} qubits, "
          f"{ng} gates): TV distance of exact distributions = {tv:.2e}")
    assert tv < 1e-9

    # --- 2. runs --------------------------------------------------------------------------
    jobs = [(m, s) for m in METHODS for s in range(N_SEEDS)]
    t0 = time.time()
    with Pool(min(len(jobs), os.cpu_count() or 4)) as pool:
        res = pool.map(_run_job, jobs, chunksize=1)
    print(f"{len(jobs)} runs x {N_ITER} iterations x {N_TEMPS} temperatures in {time.time() - t0:.0f}s")
    out = {}
    for m, s, r in res:
        out.setdefault(m, []).append(r)
    np.savez_compressed(
        OUT / "bayeswave_qclt_runs.npz",
        **{f"{m}_{k}": np.stack([r[k] for r in out[m]]) for m in METHODS for k in ("logL", "K", "match")},
        **{f"{m}_cpu": np.array([r["cpu"] for r in out[m]]) for m in METHODS},
        **{f"{m}_acc_{a}": np.array([r["acc"][a] for r in out[m]]) for m in METHODS
           for a in ("cov", "fisher_fallback", "tf", "birth", "death")},
    )
    summarize()
    plot()


def _example_reservoir():
    """A synthetic reservoir of 16-bit register values (spread like a wavelet's history near the
    injection). The circuit/sampler equivalence holds for any register contents."""
    rng = np.random.default_rng(3)
    centre = np.array([0.5, 120.0, 6.0, np.log(3.0), 1.0])
    u0 = np.rint((centre - LO) / DX - 0.5).astype(np.int64)
    spread = np.array([40, 300, 2000, 1500, 1200])
    return np.clip(u0 + rng.normal(size=(K_HIST, D)) * spread, 0, G - 1).astype(np.int64)


def load():
    z = np.load(OUT / "bayeswave_qclt_runs.npz")
    return {k: z[k] for k in z.files}


def summarize():
    z = load()
    half = N_ITER // 2
    post_logL = np.concatenate([z[f"{m}_logL"][:, half:].ravel() for m in METHODS])
    target = np.median(post_logL)
    print(f"\npooled posterior median of log L - log L(injection): {target:.2f}")
    print(f"{'method':22s} {'hit median logL':>16s} {'hit match>=0.9':>15s} {'R-hat logL':>11s} "
          f"{'ESS/1k (logL)':>14s} {'ESS/1k (K)':>11s} {'acc cov':>8s} {'ms/iter':>8s} {'post match':>11s} {'post <K>':>9s}")
    rows = {}
    for m in METHODS:
        L, Kc, M = z[f"{m}_logL"], z[f"{m}_K"], z[f"{m}_match"]
        hit = np.array([np.argmax(l >= target) if np.any(l >= target) else np.nan for l in L])
        hit_m = np.array([np.argmax(x >= 0.9) if np.any(x >= 0.9) else np.nan for x in M])
        ess_L = np.array([1000 / iact(l[half:]) for l in L])
        ess_K = np.array([1000 / iact(k[half:]) for k in Kc])
        rows[m] = dict(hit=hit, hit_m=hit_m, ess_L=ess_L, ess_K=ess_K)
        print(f"{LABELS[m]:22s} {np.nanmedian(hit):16.0f} {np.nanmedian(hit_m):15.0f} "
              f"{split_rhat(L[:, half:]):11.3f} {np.median(ess_L):14.1f} {np.nanmedian(ess_K):11.1f} "
              f"{np.nanmean(z[f'{m}_acc_cov']):8.3f} {1e3 * np.mean(z[f'{m}_cpu']) / N_ITER:8.2f} "
              f"{np.median(M[:, half:]):11.3f} {Kc[:, half:].mean():9.2f}")
    return rows, target


# ============================================================================ figure
COLORS = {"bayeswave": "#2a78d6", "adaptive": "#eda100", "qclt": "#e87ba4"}
LSTYLE = {"bayeswave": "-", "adaptive": "--", "qclt": "-"}
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"


def plot():
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    z = load()
    rows, target = _rows_quiet(z)
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    it = np.arange(1, N_ITER + 1)
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.9, 0.9]))

    for k, (key, ylab, title) in enumerate([("match", "cold-chain match with injection", "(a) Waveform match"),
                                             ("logL", r"$\log\mathcal{L} - \log\mathcal{L}_\mathrm{inj}$", r"(b) Log-likelihood")]):
        ax = axes[k]
        for m in METHODS:
            x = z[f"{m}_{key}"]
            lo, med, hi = np.percentile(x, [25, 50, 75], axis=0)
            ax.fill_between(it, lo, hi, color=COLORS[m], alpha=0.18, lw=0)
            ax.plot(it, med, LSTYLE[m], color=COLORS[m], lw=1.3, label=LABELS[m])
        ax.set_xscale("log")
        ax.set_xlim(10, N_ITER)
        ax.axvline(N_ADAPT, color=MUTED, lw=0.6, ls=":")
        ax.set_xlabel("iteration"); ax.set_ylabel(ylab); ax.set_title(title, loc="left")
    axes[0].set_ylim(0, 1)
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].axhline(target, color=MUTED, lw=0.6)
    axes[0].text(N_ADAPT * 0.92, 0.04, "adaptation off", color=MUTED, fontsize=6.5, ha="right", rotation=90, va="bottom")

    ax = axes[2]
    data = [rows[m]["hit"] for m in METHODS]
    _box(ax, data)
    ax.set_yscale("log")
    ax.set_ylabel("iterations to posterior median log L")
    ax.set_title("(c) Burn-in", loc="left")

    ax = axes[3]
    data = [rows[m]["ess_L"] for m in METHODS]
    _box(ax, data)
    ax.set_ylabel(r"ESS per $10^3$ iterations (log L)")
    ax.set_title("(d) Mixing after burn-in", loc="left")

    for ax in axes:
        ax.grid(color=GRIDC, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.8)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "bayeswave_qclt.pdf", bbox_inches="tight")
    fig.savefig(OUT / "bayeswave_qclt.png", dpi=300, bbox_inches="tight")
    print("saved", OUT / "bayeswave_qclt.pdf")


def _box(ax, data):
    bp = ax.boxplot([np.asarray(x)[np.isfinite(x)] for x in data], positions=range(len(METHODS)), widths=0.55,
                    patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.0),
                    whiskerprops=dict(color=MUTED, lw=0.8), capprops=dict(color=MUTED, lw=0.8))
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set(facecolor=COLORS[m], alpha=0.85, edgecolor="white", lw=0.8)
    rng = np.random.default_rng(0)
    for i, x in enumerate(data):
        x = np.asarray(x)[np.isfinite(x)]
        ax.plot(i + rng.uniform(-0.15, 0.15, len(x)), x, "o", ms=2.2, color=INK, alpha=0.5, mew=0)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(["Fisher", "Adapt.", "QCLT"])
    ax.tick_params(axis="x", length=0)


def _rows_quiet(z):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return summarize()


if __name__ == "__main__" and "--plot-only" in sys.argv:
    summarize()
    plot()
elif __name__ == "__main__" and not any(a.startswith("--") for a in sys.argv[1:]):
    OUT.mkdir(exist_ok=True)
    experiment()
