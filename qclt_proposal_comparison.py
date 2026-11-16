"""Compare QCLT covariance errors and stored-history proposal behavior.

The covariance comparison is written to paper/figures/covariance_method_errors.tex
as median and 10th--90th percentile errors. Separate proposal-efficiency plots
are written to figures/quantum_walk/ for dimensions two and five.

Methods
  classical, from the literature
    AM            Adaptive Metropolis, Gaussian with 2.38^2/d S (Haario, Saksman & Tamminen 2001)
    AM-LW         the same with Ledoit-Wolf shrinkage of S towards (tr S / d) I (Ledoit & Wolf 2004)
    DE-MCz        one random pair difference, gamma (h_a - h_b), gamma = 2.38/sqrt(2d) (ter Braak & Vrugt 2008)
  naive approximations of the covariance (Gaussian proposals: the best case for each approximation)
    isotropic     history-blind direction, overall scale given exactly
    diagonal      variances only
    zero-mean     uncentred second moment about the register origin (window centre), as a register circuit
                  computes it without centring: about the AMR window centre, and about a fixed box centre
    max / min     Cauchy-Schwarz bounds, every correlation set to +1 / -1
  hybrid quantum: covariance read out by measurement, then an ideal Gaussian step loader
    pairwise      centred pairwise-moment value-oracle circuits, B shots in total
    SWAP          SWAP-test distance estimation (Lloyd, Mohseni & Rebentrost 2013), exact norms, B shots
    QAE           the pairwise circuits read out by canonical amplitude estimation (Brassard, Hoyer, Mosca &
                  Tapp 2002); exact ideal output law of the phase register, B oracle queries in total
  QCLT family (quantum kick, simulated exactly; the measured law equals a classical pair-difference sampler)
    QCLT lazy     as built: each pair joins with probability q = (2.38^2/d)/(K(K-1)) and sign +-1
    QCLT full     every pair joins with a random sign (q = 1), output scaled by alpha = sqrt((2.38^2/d)/(K(K-1)));
                  same covariance, never a null step, sign coins only
    on a fixed full-range register, or on an AMR window (principal axes of a pilot, 4 grid points per spread)

Every chain also uses the Roberts & Rosenthal (2009) safety component: with probability 0.05 an isotropic step
of variance 0.1^2/d times the mean posterior variance. Nothing here runs on quantum hardware, and nothing here
is a quantum speedup: every QCLT law is sampled exactly by a classical pair-difference sampler.

Run:   python3.12 qclt_proposal_comparison.py             # ~5 min
       python3.12 qclt_proposal_comparison.py --plot-only
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "quantum_walk"
DATA = ROOT / "data" / "quantum_walk"
B_SHOTS = 384                       # toy: intermediate-measurement budget of the hybrid pipelines (as in the step study)
PER_CIRCUIT = 128                   # efficiency study: shots (or QAE queries) per readout circuit (= 384 / 3 in the toy)
RR_BETA = 0.05                      # Roberts & Rosenthal safety component
KS = [3, 4, 6, 8, 12, 16, 24]
N_HIST = 100                        # stored past states available to every method (a converged chain's history)
N_CHAIN, T_STEPS = 400, 400
REG_BITS = 6                        # register bits per coordinate for the QCLT family in the efficiency study
BOX = 6.0                           # full parameter range +-BOX (posterior sd <= 1) for the fixed register
D_TOY, K_TOY, SCALE2_TOY, ISO_VAR = 2, 3, 2.38**2 / 2, 0.1
X0 = np.array([0.69, 0.37])
MODES = np.array([[0.0020, 0.0012], [0.0007, 0.0015]])


# ============================================================================ shared estimators
def ordered_pairs(K):
    return np.array(list(itertools.combinations(range(K), 2)))


def pairwise_exact(H):
    """For each entry (p, q), p <= q: probability, lo, hi of the centred pairwise value-oracle circuit."""
    K, d = H.shape
    out = []
    for p, q in itertools.combinations_with_replacement(range(d), 2):
        v = ((H[:, None, p] - H[None, :, p]) * (H[:, None, q] - H[None, :, q]) / 2).ravel()
        lo, hi = v.min(), v.max()
        out.append((p, q, (v.mean() - lo) / (hi - lo) if hi > lo else 0.0, lo, hi))
    return out


def qae_sample(p, M, rng, size):
    """Canonical amplitude estimation (Brassard et al. 2002) with M phase states: exact ideal output law.
    theta_a = arcsin(sqrt p); outcome y has P(y) = sin^2(M delta) / (M^2 sin^2 delta), delta = pi y / M - theta_a,
    for the two eigenphases +-theta_a (each with weight 1/2); the estimate is sin^2(pi y / M)."""
    th = math.asin(math.sqrt(min(max(p, 0.0), 1.0)))
    y = np.arange(M)
    prob = np.zeros(M)
    for sgn in (1, -1):
        dl = np.pi * y / M - sgn * th
        s = np.sin(dl)
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = np.where(np.abs(s) < 1e-12, 1.0, np.sin(M * dl) ** 2 / (M**2 * s**2))
        prob += 0.5 * pr
    prob /= prob.sum()
    return np.sin(np.pi * rng.choice(M, size=size, p=prob) / M) ** 2


def readout_estimate(H, kind, budget, rng, reps):
    """Sample covariance read out by `kind` in {"pairwise", "swap", "qae"} with `budget` shots / queries in total
    (split evenly over the circuits), `reps` independent repetitions. Unbiased factor K/(K-1) included."""
    K, d = H.shape
    est = np.zeros((reps, d, d))
    if kind == "swap":                                  # exact norms -> variances; one SWAP circuit per pair
        A = H - H.mean(0)
        n2 = (A * A).sum(0)
        prs = list(itertools.combinations(range(d), 2))
        s = max(budget // len(prs), 1)
        for p in range(d):
            est[:, p, p] = n2[p]
        for p, q in prs:
            Z = n2[p] + n2[q]
            P0 = 0.5 + ((A[:, p] - A[:, q]) ** 2).sum() / (4 * Z)
            dot = (Z - 4 * Z * (rng.binomial(s, P0, reps) / s - 0.5)) / 2
            est[:, p, q] = est[:, q, p] = dot
        return est / (K - 1)
    ex = pairwise_exact(H)
    s = max(budget // len(ex), 1)
    for p, q, p1, lo, hi in ex:
        ph = rng.binomial(s, p1, reps) / s if kind == "pairwise" else qae_sample(p1, s, rng, reps)
        est[:, p, q] = est[:, q, p] = lo + (hi - lo) * ph
    return est * K / (K - 1)


def project_psd(C):
    w, V = np.linalg.eigh(0.5 * (C + np.swapaxes(C, -1, -2)))
    return (V * np.maximum(w, 0)[..., None, :]) @ np.swapaxes(V, -1, -2)


def ledoit_wolf(H):
    """Ledoit & Wolf (2004) shrinkage of the sample covariance towards (tr S / d) I."""
    K, d = H.shape
    X = H - H.mean(0)
    S = X.T @ X / K
    F = np.trace(S) / d * np.eye(d)
    b2 = sum(np.linalg.norm(np.outer(x, x) - S) ** 2 for x in X) / K**2
    d2 = np.linalg.norm(S - F) ** 2
    lam = min(1.0, b2 / d2) if d2 > 0 else 1.0
    return ((1 - lam) * S + lam * F) * K / (K - 1)


def cs_bound(S, sign):
    sd = np.sqrt(np.maximum(np.diag(S), 0))
    C = sign * np.outer(sd, sd)
    np.fill_diagonal(C, np.diag(S))
    return C


# ============================================================================ figure 1: next-step covariance in the toy
def toy_branches():
    """(signs, physical history) of the 64 preparatory branches; histories are positions after steps 1..K."""
    for bits in itertools.product((0, 1), repeat=K_TOY * D_TOY):
        signs = 1 - 2 * np.array(bits).reshape(K_TOY, D_TOY)
        yield signs, X0 + np.cumsum(signs @ MODES, axis=0)


def frame_history(signs, center, axes, spacing):
    """Lattice coordinates (relative to the window centre) of a branch, as the coherent walk holds them."""
    init = np.rint((X0 - center) @ axes / spacing)
    lmodes = np.rint(MODES @ axes / spacing)
    return init + np.cumsum(signs @ lmodes, axis=0)


def toy_errors(rng, n_pilots=40, pilot_size=1000):
    """Relative Frobenius error of each method's next-step covariance, per branch (and per pilot for AMR)."""
    from quantum_walk_study import sample_walk
    from amr_battle_test import window_from_pilot

    sd2 = SCALE2_TOY
    local = (X0, np.eye(D_TOY), np.full(D_TOY, 0.08 / 256))         # informed local 8-bit grid of the study
    pilot = sample_walk(6, X0, np.eye(D_TOY), np.full(D_TOY, 0.08 / 64), 8, 20_000, 11)["physical_hist"]
    subsets = [pilot[rng.choice(len(pilot), pilot_size, replace=False)] for _ in range(n_pilots)]
    amr_frames = [window_from_pilot(sub) for sub in subsets]
    # window-only kick: no history registers; covariance = pilot average of the within-history covariance
    pilot_S = [sd2 * np.mean([np.cov(h.T) for h in sub], axis=0) for sub in subsets]
    errs = {}

    def add(name, value):
        errs.setdefault(name, []).append(value)

    for signs, P in toy_branches():
        target = sd2 * np.cov(P.T)
        rel = lambda C: np.linalg.norm(C - target) / np.linalg.norm(target)
        c, A, s = local
        Bm = A @ np.diag(s)
        H = frame_history(signs, c, A, s)
        S = np.cov(H.T)
        phys = lambda L: Bm @ L @ Bm.T
        I = ISO_VAR * np.eye(D_TOY)
        add("QCLT, local 8-bit", rel(phys(sd2 * S + I)))
        for C_w in pilot_S:
            add("window-only kick, pilot covariance", rel(C_w))
        for c2, A2, s2 in amr_frames:
            H2 = frame_history(signs, c2, A2, s2)
            B2 = A2 @ np.diag(s2)
            add("QCLT + AMR, 8-bit", rel(B2 @ (sd2 * np.cov(H2.T) + I) @ B2.T))
            add("zero-mean kick, AMR window", rel(B2 @ (sd2 * H2.T @ H2 / (K_TOY - 1) + I) @ B2.T))
        g = 2.38 / math.sqrt(2 * D_TOY)
        dm = np.rint(g * (H[ordered_pairs(K_TOY)[:, 0]] - H[ordered_pairs(K_TOY)[:, 1]]))
        add("single-pair kick (DE-MCz), local 8-bit", rel(phys(dm.T @ dm / len(dm) + I)))
        add("zero-mean kick, local 8-bit", rel(phys(sd2 * H.T @ H / (K_TOY - 1) + I)))
        add("diagonal-only kick, local 8-bit", rel(phys(np.diag(np.diag(sd2 * S)) + I)))
        add("isotropic step (scale given)", rel(np.trace(target) / D_TOY * np.eye(D_TOY)))
        add("max bound (all rho = +1)", rel(phys(sd2 * cs_bound(S, +1) + I)))
        add("min bound (all rho = -1)", rel(phys(sd2 * cs_bound(S, -1) + I)))
        for kind, label in (("pairwise", "pairwise moments"), ("swap", "SWAP test"), ("qae", "amplitude estimation")):
            est = project_psd(readout_estimate(H, kind, B_SHOTS, rng, 50))
            for C in est:
                add(f"{label} ({B_SHOTS} {'queries' if kind == 'qae' else 'shots'}) + loader", rel(phys(sd2 * C + I)))
    return {k: np.array(v) for k, v in errs.items()}


# ============================================================================ figure 2: Metropolis efficiency
def posterior(d, seed=3):
    """Correlated Gaussian posterior: d = 2 uses the toy's step geometry; d = 5 a random rotation, sd 1 ... 0.1."""
    rng = np.random.default_rng(seed)
    if d == 2:
        w, V = np.linalg.eigh(MODES.T @ MODES)
        sd = np.sqrt(w / w.max())
    else:
        V, _ = np.linalg.qr(rng.normal(size=(d, d)))
        sd = np.geomspace(1.0, 0.1, d)
    Sig = V @ np.diag(sd**2) @ V.T
    mu = rng.uniform(-2, 2, d)                          # posterior mean away from the register-box centre
    return mu, Sig


class Proposal:
    """Vectorised proposal for n chains. step(x, rng) -> (y, null) with null = no move (no evaluation)."""


def gaussian_proposal(C):
    w, V = np.linalg.eigh(project_psd(C))
    Ach = V * np.sqrt(np.maximum(w, 0))[:, None, :]          # (n, d, d)
    p = Proposal()
    p.step = lambda x, rng: (x + np.einsum("nij,nj->ni", Ach, rng.normal(size=x.shape)), np.zeros(len(x), bool))
    return p


def pair_proposal(Hs, frame, full, sd2):
    """QCLT law (lazy or full strength) on a register frame (or continuous if frame is None)."""
    n, K, d = Hs.shape
    P = ordered_pairs(K)
    q = 1.0 if full else min(1.0, sd2 / (K * (K - 1)))
    alpha = math.sqrt(sd2 / (K * (K - 1))) if full else 1.0
    prop = Proposal()
    if frame is None:
        dm = Hs[:, P[:, 0]] - Hs[:, P[:, 1]]                  # (n, pairs, d)

        def step(x, rng):
            r = np.where(rng.random(dm.shape[:2]) < q, np.where(rng.random(dm.shape[:2]) < 0.5, -1, 1), 0)
            eps = alpha * np.einsum("np,npd->nd", r, dm)
            return x + eps, ~np.any(r != 0, axis=1)
        prop.step = step
        return prop
    c, A, s, bits = frame                                     # per-chain (n,d), (n,d,d), (n,d)
    nreg = 1 << bits
    lat = lambda v: np.floor(np.einsum("nd,nde->ne", v - c, A) / s).astype(int) + nreg // 2
    Hl = np.clip(np.floor(np.einsum("nkd,nde->nke", Hs - c[:, None], A) / s[:, None]).astype(int) + nreg // 2,
                 0, nreg - 1)
    dm = Hl[:, P[:, 0]] - Hl[:, P[:, 1]]
    p_iso = min(1.0, sd2 * 1.0)                                # QCLT's +-1 lattice floor, as in the samplers

    def step(x, rng):
        k = lat(x)
        held = np.all((k >= 0) & (k < nreg), axis=1)
        r = np.where(rng.random(dm.shape[:2]) < q, np.where(rng.random(dm.shape[:2]) < 0.5, -1, 1), 0)
        iso = np.where(rng.random(x.shape) < p_iso, np.where(rng.random(x.shape) < 0.5, -1, 1), 0)
        eps = alpha * np.einsum("np,npd->nd", r, dm) + iso
        y = x + np.einsum("nde,ne->nd", A, s * eps)
        k2 = lat(y)
        ok = held & np.all((k2 >= 0) & (k2 < nreg), axis=1)
        null = ~ok | ~np.any(eps != 0, axis=1)
        return np.where(null[:, None], x, y), null
    prop.step = step
    return prop


def window_kick_proposal(frame, sd2, n_coins=8):
    """No history registers: n_coins sign coins per AMR-window axis, output scaled classically so that the step
    covariance is sd2 x the window's (whole-history) covariance; both ends must lie in the window."""
    c, A, s, bits = frame
    nreg = 1 << bits
    scale = math.sqrt(16 * sd2 / n_coins)                       # window spacing s = sqrt(lambda) / 4
    lat = lambda v: np.floor(np.einsum("nd,nde->ne", v - c, A) / s).astype(int) + nreg // 2
    p = Proposal()

    def step(x, rng):
        eps = (2 * rng.binomial(n_coins, 0.5, size=x.shape) - n_coins) * scale
        y = x + np.einsum("nde,ne->nd", A, s * eps)
        ok = np.all((lat(x) >= 0) & (lat(x) < nreg), axis=1) & np.all((lat(y) >= 0) & (lat(y) < nreg), axis=1)
        null = ~ok | ~np.any(eps != 0, axis=1)
        return np.where(null[:, None], x, y), null
    p.step = step
    return p


def de_proposal(Hs, d):
    n, K, _ = Hs.shape
    g = 2.38 / math.sqrt(2 * d)
    p = Proposal()

    def step(x, rng):
        a = rng.integers(K, size=n)
        b = (a + rng.integers(1, K, size=n)) % K
        return x + g * (Hs[np.arange(n), a] - Hs[np.arange(n), b]), np.zeros(n, bool)
    p.step = step
    return p


def chain_metric(prop, post, rng, n=N_CHAIN, T=T_STEPS):
    """Metropolis on the Gaussian posterior with a fixed proposal per chain (chains start in the posterior).
    Returns the mean over chains of the worst-direction and total whitened squared jump per likelihood
    evaluation (null proposals cost no evaluation), and the acceptance rate."""
    mu, Si, Li, reg_sd, draw = post["mu"], post["Si"], post["Li"], post["reg_sd"], post["draw"]
    d = len(mu)
    x = draw(n)
    lp = -0.5 * np.einsum("ni,ij,nj->n", x - mu, Si, x - mu)
    M = np.zeros((n, d, d))
    ev = np.zeros(n)
    acc = np.zeros(n)
    for _ in range(T):
        safe = rng.random(n) < RR_BETA                       # Roberts & Rosenthal safety component
        y, null = prop.step(x, rng)
        y = np.where(safe[:, None], x + rng.normal(size=x.shape) * reg_sd, y)
        null = null & ~safe
        ly = -0.5 * np.einsum("ni,ij,nj->n", y - mu, Si, y - mu)
        a = ~null & (np.log(rng.random(n)) < ly - lp)
        ev += ~null
        dz = (y - x) @ Li.T * a[:, None]
        M += np.einsum("ni,nj->nij", dz, dz)
        acc += a
        x = np.where(a[:, None], y, x)
        lp = np.where(a, ly, lp)
    worst = np.linalg.eigvalsh(M)[:, 0] / np.maximum(ev, 1)
    total = np.trace(M, axis1=1, axis2=2) / np.maximum(ev, 1)
    return dict(worst=worst.mean(), worst_se=worst.std() / math.sqrt(n), total=total.mean(),
                acc=float(acc.sum() / max(ev.sum(), 1)))


def setup(d, rng, n=N_CHAIN):
    """Posterior, a past history of N_HIST states per chain (drawn from the posterior, as from a converged chain),
    and the register frames: the AMR window (principal axes of the history, 4 grid points per spread) and a fixed
    full-range register."""
    mu, Sig = posterior(d)
    L = np.linalg.cholesky(Sig)
    draw = lambda *shape: mu + rng.normal(size=(*shape, d)) @ L.T
    post = dict(mu=mu, Sig=Sig, Si=np.linalg.inv(Sig), Li=np.linalg.inv(L), draw=draw,
                reg_sd=math.sqrt(0.1**2 / d * np.trace(Sig) / d))
    Hf = draw(n, N_HIST)
    w, V = np.linalg.eigh(np.array([np.cov(h.T) for h in Hf]))
    amr = (Hf.mean(1), V, np.sqrt(w) / 4, REG_BITS)
    fixed = (np.zeros((n, d)), np.tile(np.eye(d), (n, 1, 1)), np.full((n, d), 2 * BOX / 2**REG_BITS), REG_BITS)
    return post, Hf, amr, fixed


def efficiency_full(d, post, Hf, rng, amr=None):
    """Methods that use the whole stored history (classical memory or logarithmic-width readout circuits)."""
    n = len(Hf)
    sd2 = 2.38**2 / d
    S = np.array([np.cov(h.T) for h in Hf])
    n_circ = {"pairwise": d * (d + 1) // 2, "qae": d * (d + 1) // 2, "swap": d * (d - 1) // 2}
    readout = lambda kind: gaussian_proposal(sd2 * np.array(
        [readout_estimate(h, kind, PER_CIRCUIT * n_circ[kind], rng, 1)[0] for h in Hf]))
    methods = {
        "optimal Gaussian (true covariance)": gaussian_proposal(np.tile(sd2 * post["Sig"], (n, 1, 1))),
        "AM (Haario et al. 2001)": gaussian_proposal(sd2 * S),
        "AM, Ledoit-Wolf shrinkage": gaussian_proposal(sd2 * np.array([ledoit_wolf(h) for h in Hf])),
        "DE-MCz (ter Braak & Vrugt 2008)": de_proposal(Hf, d),
        "diagonal only": gaussian_proposal(sd2 * np.array([np.diag(np.diag(s)) for s in S])),
        "isotropic (scale given)": gaussian_proposal(np.tile(sd2 * np.trace(post["Sig"]) / d * np.eye(d), (n, 1, 1))),
        "max bound (rho = +1)": gaussian_proposal(sd2 * np.array([cs_bound(s, 1) for s in S])),
        "min bound (rho = -1)": gaussian_proposal(sd2 * np.array([cs_bound(s, -1) for s in S])),
        "pairwise readout + loader": readout("pairwise"),
        "SWAP readout + loader": readout("swap"),
        "QAE readout + loader": readout("qae"),
        f"window-only kick, AMR {REG_BITS}-bit (no history registers)": window_kick_proposal(amr, sd2),
    }
    return {name: chain_metric(prop, post, rng) for name, prop in methods.items()}


def efficiency_register(d, K, post, Hf, amr, fixed, rng):
    """Methods whose step uses only K stored states (a uniform subset of the history, as a reservoir keeps), as a
    register circuit must; register windows still come from the whole history."""
    n = len(Hf)
    sd2 = 2.38**2 / d
    Hs = np.array([h[rng.choice(N_HIST, K, replace=False)] for h in Hf])
    um = lambda c: np.einsum("nkd,nke->nde", Hs - c[:, None], Hs - c[:, None]) / (K - 1)
    methods = {
        f"QCLT lazy, fixed {REG_BITS}-bit range": pair_proposal(Hs, fixed, False, sd2),
        f"QCLT lazy + AMR, {REG_BITS}-bit": pair_proposal(Hs, amr, False, sd2),
        f"QCLT full + AMR, {REG_BITS}-bit (ours)": pair_proposal(Hs, amr, True, sd2),
        "zero-mean, K states, about AMR centre": gaussian_proposal(sd2 * um(amr[0])),
        "zero-mean, K states, about box centre": gaussian_proposal(sd2 * um(np.zeros((n, d)))),
        "AM from the same K states": gaussian_proposal(sd2 * np.array([np.cov(h.T) for h in Hs])),
        "single pair from the K states": de_proposal(Hs, d),
    }
    return {name: chain_metric(prop, post, rng) for name, prop in methods.items()}


def resource_rows():
    """Qubits of one circuit execution for the 2-D toy (two 8-bit coordinates, K = 3), counted from circuits
    built here or in the existing study; 'loader' = a Gaussian step loader still needed after readout."""
    from qiskit import QuantumCircuit
    from compact_qclt import build_compiled_kick
    from qclt_step_comparison import pairwise_readout_circuit, swap_readout_circuit
    from quantum_walk_study import make_walk
    hist = next(toy_branches())[1]
    spacing = 0.08 / 256
    walk, _ = make_walk(8, X0, np.eye(D_TOY), np.full(D_TOY, spacing), 14, track_displacement=False)
    H = frame_history(next(toy_branches())[0], X0, np.eye(D_TOY), np.full(D_TOY, spacing)).astype(np.int64)
    dm = H[ordered_pairs(K_TOY)[:, 0]] - H[ordered_pairs(K_TOY)[:, 1]]
    q = SCALE2_TOY / (K_TOY * (K_TOY - 1))
    lazy_c, _ = build_compiled_kick(dm, q, ISO_VAR)
    full_c, _ = build_compiled_kick(dm, 1.0, ISO_VAR)
    pw, _ = pairwise_readout_circuit(np.rint(hist / spacing), 0, 1)
    sw, _ = swap_readout_circuit(np.rint(hist / spacing))
    m_qae = int(math.log2(PER_CIRCUIT))
    return [("QCLT walk, history in prep. coins", walk.num_qubits, 1, "step", "no"),
            ("QCLT kick, classical history, lazy", lazy_c.num_qubits, 1, "step", "no"),
            ("QCLT kick, classical history, full", full_c.num_qubits, 1, "step", "no"),
            ("pairwise-moment readout", pw.num_qubits, 3, "entries", "yes"),
            ("SWAP-test readout", sw.num_qubits, 1, "cross term", "yes"),
            ("amplitude-estimation readout", pw.num_qubits + m_qae, 3, "entries", "yes"),
            ("window-only kick, 8 coins/axis", D_TOY * 8 + D_TOY * ((8).bit_length() + 1), 1, "step", "no"),
            ("Gaussian loader (KW shear)", 31, 1, "step, C given", "-")]


def write_covariance_table(errs, b_shots):
    """Write all covariance-error summaries as a paper-ready LaTeX table."""
    rows = [
        ("QCLT + AMR, 8-bit grid", "QCLT + AMR, 8-bit", 2560),
        ("QCLT, informed local 8-bit grid", "QCLT, local 8-bit", 64),
        ("Single history-pair difference (DE-MCz)", "single-pair kick (DE-MCz), local 8-bit", 64),
        ("AMR-window covariance only", "window-only kick, pilot covariance", 2560),
        ("Diagonal covariance", "diagonal-only kick, local 8-bit", 64),
        ("Uncentred second moment, fixed grid", "zero-mean kick, local 8-bit", 64),
        ("Uncentred second moment, AMR grid", "zero-mean kick, AMR window", 2560),
        (f"Amplitude estimation ({b_shots} queries) + ideal loader",
         f"amplitude estimation ({b_shots} queries) + loader", 64 * 50),
        (f"SWAP overlap ({b_shots} shots) + ideal loader",
         f"SWAP test ({b_shots} shots) + loader", 64 * 50),
        (f"Pairwise moments ({b_shots} shots) + ideal loader",
         f"pairwise moments ({b_shots} shots) + loader", 64 * 50),
        ("Maximum-correlation bound", "max bound (all rho = +1)", 64),
        ("Minimum-correlation bound", "min bound (all rho = -1)", 64),
        ("Isotropic covariance", "isotropic step (scale given)", 64),
    ]
    lines = [
        r"\begin{tabular}{@{}p{0.48\textwidth}r r p{0.20\textwidth}@{}}",
        r"\toprule",
        r"Method & $N$ & Median error (\%) & 10th--90th percentile (\%) \\",
        r"\midrule",
    ]
    for label, key, n in rows:
        lo, median, hi = np.quantile(100 * errs[key], [0.1, 0.5, 0.9])
        lines.append(f"{label} & {n:,} & {median:.2f} & [{lo:.2f}, {hi:.2f}] " + r"\\")
    lines.extend([
        r"Classical AM / DE-MCz, exact history & 64 & 0 (exact) & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    path = ROOT / "paper" / "figures" / "covariance_method_errors.tex"
    path.write_text("\n".join(lines) + "\n")


def plot():
    """Write the full covariance-error table and stored-history plots."""
    import pickle
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import paper_style

    with open(DATA / "proposal_comparison.pkl", "rb") as f:
        z = pickle.load(f)
    paper_style.apply()
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
    })
    errs = z["errs"]
    write_covariance_table(errs, z["b_shots"])

    full, reg, ks = z["full"], z["reg"], z["ks"]
    colors = paper_style.COLORS
    styles = {
        f"QCLT full + AMR, {REG_BITS}-bit (ours)": (colors["QCLT_AMR"], "-", "o", "QCLT full + AMR"),
        f"QCLT lazy + AMR, {REG_BITS}-bit": (colors["QCLT_AMR"], "--", "o", "QCLT lazy + AMR"),
        f"QCLT lazy, fixed {REG_BITS}-bit range": (colors["QCLT"], ":", "o", "QCLT lazy, fixed grid"),
        "AM from the same K states": (colors["CLASSICAL"], "-", "s", "AM, same K states"),
        "single pair from the K states": (colors["CLASSICAL"], "-.", "^", "Single pair, same K states"),
        "zero-mean, K states, about AMR centre": (colors["APPROX"], "-", "v", "Uncentred, AMR centre"),
        "zero-mean, K states, about box centre": (colors["APPROX"], ":", "v", "Uncentred, box centre"),
    }
    for d in (2, 5):
        if (d, N_HIST) not in reg:
            raise RuntimeError(
                "This data file predates QCLT+AMR normalization. Rerun the script without --plot-only "
                "once to generate the full-history QCLT+AMR reference."
            )
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        full_endpoint = {
            "AM from the same K states": "AM (Haario et al. 2001)",
            "single pair from the K states": "DE-MCz (ter Braak & Vrugt 2008)",
        }
        qclt_amr_key = f"QCLT full + AMR, {REG_BITS}-bit (ours)"
        qclt_amr_reference = np.array([
            reg[(d, k)][qclt_amr_key]["worst"] for k in ks
        ])
        for key, (color, ls, marker, label) in styles.items():
            vals = np.array([reg[(d, k)][key]["worst"] for k in ks]) / qclt_amr_reference
            se = np.array([reg[(d, k)][key]["worst_se"] for k in ks]) / qclt_amr_reference
            xvals = list(ks)
            if key in full_endpoint:
                end = full[d][full_endpoint[key]]
                # Normalize whole-history classical endpoints against QCLT + AMR
                # evaluated with the same history size, K = N_HIST.
                reference_at_full_history = reg[(d, N_HIST)][qclt_amr_key]["worst"]
                vals = np.r_[vals, end["worst"] / reference_at_full_history]
                se = np.r_[se, end["worst_se"] / reference_at_full_history]
                xvals.append(N_HIST)
            ax.fill_between(xvals, vals - 2 * se, vals + 2 * se, color=color, alpha=.1, lw=0)
            ax.plot(xvals, vals, ls=ls, marker=marker, ms=3, color=color, lw=1.1, label=label)
        ax.axhline(1, color=paper_style.INK, lw=.7)
        tickvals = list(ks) + [N_HIST]
        ax.set_xscale("log", base=2); ax.set_xticks(tickvals, [str(k) for k in tickvals])
        ax.set_ylim(bottom=0)  # show the QCLT + AMR reference line at 1
        ax.set_xlabel("Stored history states, K")
        ax.set_ylabel(f"d={d}: worst-direction jump / QCLT + AMR (1 = QCLT + AMR)")
        ax.grid(color=paper_style.GRID, lw=.6); ax.set_axisbelow(True)
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(.5, 1.14), columnspacing=1.0)
        fig.tight_layout(rect=(0, 0, 1, .84))
        fig.savefig(OUT / f"qclt_proposal_efficiency_d{d}.pdf", bbox_inches="tight")
        plt.close(fig)

    print("wrote covariance-error table and two stored-history proposal-efficiency plots")


def run():
    global D_TOY, K_TOY, MODES, SCALE2_TOY, ISO_VAR, X0, sample_walk, window_from_pilot
    from quantum_walk_study import D as D_TOY, K as K_TOY, MODES, SCALE2 as SCALE2_TOY, ISO_VAR, X0, sample_walk
    from amr_battle_test import window_from_pilot
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261115)
    errs = toy_errors(rng)
    for k, v in errs.items():
        print(f"{k:48s} median {np.median(v):8.2%}  (10-90%: {np.quantile(v, .1):.2%} - {np.quantile(v, .9):.2%})",
              flush=True)
    full, reg = {}, {}
    for d in (2, 5):
        post, Hf, amr, fixed = setup(d, rng)
        full[d] = efficiency_full(d, post, Hf, rng, amr)
        o = full[d]["optimal Gaussian (true covariance)"]["worst"]
        print(f"d={d}, whole history: " + "; ".join(f"{m.split(' (')[0]} {v['worst'] / o:.2f}" for m, v in full[d].items()),
              flush=True)
        for K in KS:
            reg[(d, K)] = efficiency_register(d, K, post, Hf, amr, fixed, rng)
            print(f"d={d} K={K:2d}: " + "; ".join(f"{m.split(',')[0] if m.startswith('QCLT') else m} {v['worst'] / o:.2f}"
                                                for m, v in reg[(d, K)].items()), flush=True)
        # Extra full-history reference for normalizing the AM / DE-MCz endpoints.
        reg[(d, N_HIST)] = efficiency_register(d, N_HIST, post, Hf, amr, fixed, rng)
    import pickle
    with open(DATA / "proposal_comparison.pkl", "wb") as f:
        pickle.dump(dict(errs=errs, full=full, reg=reg, ks=KS, b_shots=B_SHOTS, n_hist=N_HIST,
                         per_circuit=PER_CIRCUIT, qubits=resource_rows()), f)
    plot()


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        write_covariance_table(toy_errors(np.random.default_rng(20261115)), B_SHOTS)
    elif "--plot-only" in sys.argv:
        plot()
    else:
        run()
