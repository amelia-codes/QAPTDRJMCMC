"""Wavelet TD-RJMCMC (trans-dimensional reversible-jump MCMC) with auxiliary tempering, comparing
three choices of the within-model ("covariance") move.

This is NOT LIGO's BayesWave. It is a small Python sampler with BayesWave's model structure
(a variable number N of Morlet-Gabor wavelets, SNR amplitude prior, data-driven time-frequency birth
proposal, Fisher-matrix jumps, parallel tempering) on synthetic white-noise data with an injected
chirp. No noise-PSD model, glitch/signal model selection, detector network or sky location.

Samplers (everything except the move step is shared, unless noted):

  fisher  conventional TD-RJMCMC: Fisher-information-matrix Gaussian moves (full Hastings ratio).

  ap      AP-PT-RJMCMC of Tian, Lee & Zhou (arXiv:2402.08844, Inverse Problems 2024), mapped to wavelets:
          * their "candidate points" -> the 64x64 time-frequency cells of (t0, f0). Every iteration every
            wavelet adds its parameter vector to its cell's history; Haario recursive mean/covariance
            (their Eqs. 10-11), one table per temperature ("each chain advancing with its own covariance").
          * move: N(u, C_m), C_m = s_c s_d C_cell + s_c s_d eps I (their step 3), s_d = 2.4^2/d,
            s_c adapted per temperature by Andrieu-Thoms towards acceptance 0.234 with
            gamma_i = (i+1)^-0.5, log s_c clipped to log[1e-10, 1e10] (their Eq. 12).
            Their move keeps the knot locations fixed, so the covariance index set is unchanged;
            here a move must stay inside its cell (proposals leaving it are rejected).
          * birth: (t0, f0) from the time-frequency proposal, (Q, lnA) ~ N(mean_cell, s_d C_cell + s_d eps I)
            when the cell has history (their sigma_b^2 = s_d C + s_d eps), else from the prior;
            death is its exact reverse (proposal density in the acceptance ratio).
          * first T0 = 1000 iterations: conventional (Fisher) moves only (their initial t0 period).
          Deviations from the paper:
          * Fisher moves remain as a fixed-probability mixture component (P_FISHER_AP), so a wavelet in a
            cell without enough history can still move.
          * All adaptation (history tables, s_c, temperatures) stops at the end of burn-in (N_ADAPT), after
            which every kernel is a fixed, exact MH kernel. Tian et al. adapt throughout (diminishing
            adaptation). With continuous adaptation the history-informed birth/death is self-referential
            here (a long-lived wavelet dominates its own cell's history, so its reverse-birth density is
            inflated): with the likelihood switched off it gave P(N=0) = 0.57 instead of the prior 1/11 (N = number of wavelets).

  qclt    identical to `ap`, except the move-step proposal is the quantum-CLT kick
              eps = sum_{a<b} r_ab (h_a - h_b) + lattice identity term,
          r_ab in {-1, 0, +1} w.p. {q/2, 1-q, q/2}, q = s_c s_d / (K (K-1)), over a per-cell reservoir of
          K register snapshots (uniform subsample of the cell's history). Cov = s_c s_d S_reservoir + ...,
          the same as `ap` up to reservoir noise. It is evaluated with a classical sampler of the coins,
          which is exactly the circuit's output distribution: validate_qclt_against_circuit() builds the real
          qcov circuit (432 qubits) and compares exact distributions. No quantum circuit is executed
          inside the MCMC.

Parallel tempering (all samplers): non-reversible deterministic even-odd swaps (Syed et al. 2021, as used
by Tian et al.), temperatures adapted during burn-in to equalise swap acceptance.

Convergence (Tian et al. Sec. 4): pairs of independent runs; R_c1 / R_c2 compare the posterior mean / std of
the reconstructed waveform on the sample grid between the two runs; converged when both < 0.2.
Formulas (14)-(15) did not extract cleanly from the PDF; implemented as
    R_c1 = mean_i |mu_i1 - mu_i2| / ((sigma_i1 + sigma_i2)/2),  R_c2 = mean_i |sigma_i1 - sigma_i2| / ((sigma_i1 + sigma_i2)/2).

Run:   python3.12 tdrjmcmc_qclt.py --prior-test    # likelihood off: every sampler must recover the prior
       python3.12 tdrjmcmc_qclt.py                 # full comparison -> figures/previous/tdrjmcmc_qclt.pdf
       python3.12 tdrjmcmc_qclt.py --plot-only
"""
import itertools
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================================ model / data
FS, N = 1024, 1024
TT = np.arange(N) / FS
NB = 16
G = 1 << NB
PARAMS = ["t0", "f0", "Q", "lnA", "phi"]
D = len(PARAMS)
LO = np.array([0.10, 30.0, 2.0, np.log(0.1), 0.0])
HI = np.array([0.90, 400.0, 30.0, np.log(20.0), 2 * np.pi])
DX = (HI - LO) / G
PERIODIC = np.array([False, False, False, False, True])
K_MAX = 10
SNR_STAR = 5.0

DATASET = "chirp"          # "chirp": SNR-20 chirp (default); "hard": see make_data()
BIRTH = "tf"               # "tf": data-driven time-frequency birth; "prior": births drawn from the prior
SNR_INJ = 20.0

TF_BITS = 6
CELL_SHIFT = NB - TF_BITS
N_CELLS = 1 << (2 * TF_BITS)

# ---------------------------------------------------------------------------- sampler settings
# Naming: in this file N is the number of data samples, so the number of wavelets is held in variables
# called K / K_MAX internally; every plot, table and printout calls it N.
N_TEMPS = 8
T_MAX = 200.0
P_RJ = 0.2
P_TF = 0.1
P_FISHER_AP = 0.2                 # ap/qclt: share of intrinsic moves that stay Fisher
T0 = 1000                         # Tian et al.: initial conventional period
S_D = 2.4**2 / D                  # Tian et al. tip 1
EPS = 1.0                         # epsilon (grid units^2)
TARGET_ACC = 0.234                # Tian et al. tip 2
LOG_SC_MIN, LOG_SC_MAX = math.log(1e-10), math.log(1e10)
N_MIN_HIST = 20                   # a cell needs this much history before its covariance is used
K_RES = 6                         # QCLT history registers per cell
AMR_BITS = 6                      # QCLT + AMR: register bits per parameter (QCLT-AP: NB = 16 over the full range)
AMR_RES = 4.0                     # QCLT + AMR: grid points per recent spread along each principal axis
AMR_MEM = 200                     # QCLT + AMR: memory (history states) of the recent mean / covariance per cell
AMR_EVERY = 50                    # QCLT + AMR: re-grid a cell's window every this many history states
AMR_OUT_HI, AMR_OUT_LO = 0.2, 0.05  # zoom out (x2) / back in (/2) when this share of moves falls outside
FISHER_SCALE = 2.38 / math.sqrt(D)
FD_STEP = 4.0

METHODS = ["fisher", "ap", "qclt"]
LABELS = {"fisher": "Fisher TD-RJMCMC (classical)", "ap": "AP, Tian et al. (classical)",
          "qclt": "QCLT-AP (quantum kick, simulated)", "rw": "Naive TD-RJMCMC (classical)",
          "ap_nb": "AP, TF birth (classical)", "qclt_nb": "QCLT-AP, TF birth (quantum kick, simulated)",
          "qamr": "QCLT + AMR, ours (quantum kick, simulated; 6-bit registers)",
          "qamr_nb": "QCLT + AMR, TF birth, ours (quantum kick, simulated; 6-bit registers)"}
QC_NOTE = ("classical: runs entirely on a classical computer. quantum kick, simulated: the adaptive step is drawn from "
           "the QCLT quantum circuit,\nsimulated exactly on a classical computer (no quantum hardware); the rest of the "
           "sampler is classical. BayesWave is classical.")
# "_nb" variants: Tian et al.'s adaptive move step only; birth/death use the same time-frequency birth
# proposal as the Fisher sampler (no history-informed birth).

# window of the waveform used for the Tian et al. convergence statistics (the burst region)
W0, W1 = int(0.40 * FS), int(0.60 * FS)
WAVE_EVERY = 10


def phys(u):
    return LO + (np.asarray(u, dtype=float) + 0.5) * DX


def snr_factor(th):
    tau = th[2] / (2 * np.pi * th[1])
    return math.sqrt(FS * tau * math.sqrt(math.pi / 2) / 2)


def log_prior_lnA(u):
    """log prior density of lnA given (f0, Q) relative to uniform on its range (Gamma(2) in SNR)."""
    th = phys(u)
    c = snr_factor(th)
    x_lo, x_hi = math.exp(LO[3]) * c / SNR_STAR, math.exp(HI[3]) * c / SNR_STAR
    Z = (1 + x_lo) * math.exp(-x_lo) - (1 + x_hi) * math.exp(-x_hi)
    x = math.exp(th[3]) * c / SNR_STAR
    return math.log(x * x * math.exp(-x) / Z * (HI[3] - LO[3]))


def wave_window(th):
    tau = th[2] / (2 * np.pi * th[1])
    half = 4.5 * tau
    return max(0, int((th[0] - half) * FS)), min(N, int((th[0] + half) * FS) + 2)


EVALS = [0]          # waveform evaluations (the expensive oracle call); reset at the start of every run()
FD_EVALS = [0]       # the subset spent on finite-difference derivatives for Fisher moves
ADAPT_TEMPS = True   # False: keep the geometric temperature ladder fixed (tempering ablation)
RW_FRAC = 0.01       # naive sampler: fixed step of 1% of each parameter's range
AP_FALLBACK = "fisher"   # move used by ap/qclt before T0 and as the fixed mixture component:
                         # "fisher" (default) or "rw" (Tian et al.'s fixed untuned proposal: derivative-free)


def wave_on(th, i0, i1):
    EVALS[0] += 1
    t0, f0, Q, lnA, phi = th
    tau = Q / (2 * np.pi * f0)
    x = TT[i0:i1] - t0
    return np.exp(lnA - (x / tau) ** 2) * np.cos(2 * np.pi * f0 * x + phi)


def make_data(seed=1, snr=None):
    """'chirp': short chirp (30-210 Hz over ~60 ms). 'hard': long, wide chirp (60 -> 340 Hz over a ~0.2 s
    envelope, needs many wavelets) plus a separate weak burst (t = 0.22 s, 280 Hz, Q = 5, 25% of the power)."""
    rng = np.random.default_rng(seed)
    snr = SNR_INJ if snr is None else snr
    if DATASET == "hard":
        x = TT - 0.5
        chirp = np.exp(-(x / 0.1) ** 2) * np.cos(2 * np.pi * (200 * x + 0.5 * 700 * x**2) + 0.3)
        tau = 5 / (2 * np.pi * 280)
        burst = np.exp(-((TT - 0.22) / tau) ** 2) * np.cos(2 * np.pi * 280 * (TT - 0.22) + 1.1)
        h = chirp / np.linalg.norm(chirp) * math.sqrt(0.75) + burst / np.linalg.norm(burst) * math.sqrt(0.25)
    else:
        x = TT - 0.5
        h = np.exp(-(x / 0.03) ** 2) * np.cos(2 * np.pi * (120 * x + 0.5 * 3000 * x**2) + 0.7)
    h *= snr / np.linalg.norm(h)
    return h + rng.normal(size=N), h


def tf_proposal(d):
    n_cell = 1 << TF_BITS
    if BIRTH == "prior":                         # births from the prior: no data map is computed (or counted)
        P = np.full((n_cell, n_cell), 1.0 / n_cell**2)
        return P, np.cumsum(P.ravel())
    w = np.zeros((n_cell, n_cell))
    for i in range(n_cell):
        for j in range(n_cell):
            u = np.array([(i << CELL_SHIFT) + (1 << (CELL_SHIFT - 1)), (j << CELL_SHIFT) + (1 << (CELL_SHIFT - 1)), 0, 0, 0])
            th = phys(u)
            th[2], th[3] = 8.0, 0.0
            i0, i1 = wave_window(th)
            c = wave_on(th, i0, i1)
            th[4] = np.pi / 2
            s = wave_on(th, i0, i1)
            w[i, j] = (d[i0:i1] @ c) ** 2 / (c @ c) + (d[i0:i1] @ s) ** 2 / (s @ s)
    w = np.maximum(w - 4.0, 0.0) + 0.5
    P = w / w.sum()
    return P, np.cumsum(P.ravel())


def cell_of(u):
    return int((u[0] >> CELL_SHIFT) << TF_BITS | (u[1] >> CELL_SHIFT))


# ============================================================================ state
class Wavelet:
    __slots__ = ("u", "i0", "i1", "h")

    def __init__(self, u):
        self.set(u)

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
    spans = [(w.i0, w.i1) for w in (old, new) if w is not None]
    lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
    seg0 = r[lo:hi]
    seg = seg0.copy()
    if old is not None:
        seg[old.i0 - lo:old.i1 - lo] += old.h
    if new is not None:
        seg[new.i0 - lo:new.i1 - lo] -= new.h
    return -0.5 * (seg @ seg - seg0 @ seg0), lo, hi, seg


class History:
    """Per-temperature adaptive state of Tian et al.: per-cell Haario statistics, per-cell QCLT
    reservoirs, and the Andrieu-Thoms scale s_c. Tables belong to the temperature slot, not to the
    state, so they stay put when states are swapped."""

    def __init__(self, with_reservoir, amr=False):
        self.amr = amr
        if amr:                                    # adaptive gridlines: per-cell window of the n-bit registers
            self.win_c = np.zeros((N_CELLS, D))           # centre (16-bit units)
            self.win_V = np.tile(np.eye(D), (N_CELLS, 1, 1))  # register axes = principal axes of the history
            self.win_s = np.ones((N_CELLS, D))            # grid spacing along each axis (16-bit units)
            self.win_next = np.full(N_CELLS, N_MIN_HIST, np.int64)
            self.ew_mean = np.zeros((N_CELLS, D))         # recent (exponentially weighted) mean and covariance
            self.ew_cov = np.zeros((N_CELLS, D, D))
            self.widen = np.ones(N_CELLS)                 # zoom factor (x2 out / back in)
            self.n_try = np.zeros(N_CELLS, np.int64)      # AMR moves in the cell since the last re-grid
            self.n_out = np.zeros(N_CELLS, np.int64)      # ... of which the point or the step fell outside
        self.n = np.zeros(N_CELLS, np.int64)
        self.mean = np.zeros((N_CELLS, D))
        self.M2 = np.zeros((N_CELLS, D, D))
        self.res = np.zeros((N_CELLS, K_RES, D), np.int64) if with_reservoir else None
        self.log_sc = 0.0
        self.i_move = 0

    def add(self, u, rng):
        c = cell_of(u)
        self.n[c] += 1
        n = self.n[c]
        x = u.astype(float)
        dev = x - self.mean[c]
        dev[PERIODIC] = (dev[PERIODIC] + G / 2) % G - G / 2
        self.mean[c] += dev / n
        self.mean[c, PERIODIC] %= G
        dev2 = x - self.mean[c]
        dev2[PERIODIC] = (dev2[PERIODIC] + G / 2) % G - G / 2
        self.M2[c] += np.outer(dev, dev2)                    # Welford form of Haario's recursion
        if self.res is not None:
            if n <= K_RES:
                self.res[c, n - 1] = u
            else:
                j = rng.integers(n)
                if j < K_RES:
                    self.res[c, j] = u
        if self.amr:
            a = max(1.0 / n, 1.0 / AMR_MEM)                # recent mean / covariance (exponential weights)
            dv = u - self.ew_mean[c]
            dv[PERIODIC] = (dv[PERIODIC] + G / 2) % G - G / 2
            self.ew_mean[c] += a * dv
            self.ew_mean[c, PERIODIC] %= G
            self.ew_cov[c] = (1 - a) * (self.ew_cov[c] + a * np.outer(dv, dv))
            if n >= self.win_next[c]:                      # re-grid every AMR_EVERY states; add() is only called
                self.regrid(c)                             # while adapting, so windows freeze with the rest
                self.win_next[c] = n + AMR_EVERY

    def regrid(self, c):
        """AMR ("zoom in & out" of adaptive_grid.py): the AMR_BITS-bit registers hold coordinates along the
        principal axes of the cell's recent covariance, centred on its recent mean, with about AMR_RES grid points
        per spread along each axis (each register spans 2^AMR_BITS / AMR_RES = 16 spreads), widened x2 while
        more than AMR_OUT_HI of the cell's AMR moves fell outside the window, narrowed back below AMR_OUT_LO."""
        if self.n_try[c]:                                  # zoom in & out, as in adaptive_grid.py
            out = self.n_out[c] / self.n_try[c]
            if out > AMR_OUT_HI:
                self.widen[c] *= 2.0
            elif out < AMR_OUT_LO:
                self.widen[c] = max(1.0, self.widen[c] / 2.0)
        self.n_try[c] = self.n_out[c] = 0
        lam, V = np.linalg.eigh(0.5 * (self.ew_cov[c] + self.ew_cov[c].T))
        self.win_c[c] = self.ew_mean[c]
        self.win_V[c] = V
        self.win_s[c] = np.maximum(self.widen[c] * np.sqrt(np.maximum(lam, 0.0)) / AMR_RES, 1.0)

    def cov(self, c):
        C = self.M2[c] / (self.n[c] - 1)
        return 0.5 * (C + C.T)

    def adapt_scale(self, alpha):
        self.i_move += 1
        gamma = (self.i_move + 1) ** -0.5
        self.log_sc = min(max(self.log_sc + gamma * (alpha - TARGET_ACC), LOG_SC_MIN), LOG_SC_MAX)


# ============================================================================ proposals
def in_prior(u):
    return bool(np.all((u >= 0) | PERIODIC) and np.all((u < G) | PERIODIC))


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
    FD_EVALS[0] += 2 * D
    lam, V = np.linalg.eigh(J.T @ J)
    return np.maximum(lam, (8.0 / G) ** 2), V


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
    return u_new, gauss_logpdf(-delta, lam2, V2, FISHER_SCALE) - gauss_logpdf(delta, lam, V, FISHER_SCALE)


def pair_diffs(buf):
    a, b = np.array(list(itertools.combinations(range(len(buf)), 2))).T
    return buf[a] - buf[b]


def propose_ap_gauss(hist, c, rng):
    sc = math.exp(hist.log_sc)
    C = sc * S_D * (hist.cov(c) + EPS * np.eye(D))
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(C)
        L = V * np.sqrt(np.clip(w, 0, None))
    return np.rint(L @ rng.normal(size=D)).astype(np.int64)


def propose_ap_qclt(hist, c, rng):
    """QCLT kick on the cell's history registers: Cov = s_c s_d (S_res + eps I) (q capped at 1)."""
    sc = math.exp(hist.log_sc)
    dm = pair_diffs(hist.res[c])
    q = min(1.0, sc * S_D / (K_RES * (K_RES - 1)))
    r = np.where(rng.random(len(dm)) < q, np.where(rng.random(len(dm)) < 0.5, -1, 1), 0)
    p_iso = min(1.0, sc * S_D * EPS)
    iso = np.where(rng.random(D) < p_iso, np.where(rng.random(D) < 0.5, -1, 1), 0)
    return r @ dm + iso


def amr_register(x, c, V, s):
    """Register value(s) of 16-bit point(s) x in a window (centre c, axes V, spacings s):
    floor(V^T (x - c) / s) + 2^(AMR_BITS - 1), with the minimal-image offset for the periodic phase."""
    off = np.asarray(x) - c
    off = np.where(PERIODIC, (off + G / 2) % G - G / 2, off)
    return np.floor(off @ V / s).astype(np.int64) + (1 << (AMR_BITS - 1))


def propose_ap_qclt_amr(hist, c, u, rng):
    """QCLT kick on n-bit registers inside the cell's adaptive window (QCLT + AMR).

    The registers hold the wavelet's coordinates along the principal axes V of the cell's history, in units of
    the spacings s, relative to the window centre: k = floor(V^T (u - c) / s) + 2^(n-1); the part below the
    spacing, and the map back to parameters (c, V, s), are classical. The history registers hold the reservoir
    states in the same coordinates. The kick eps is the QCLT pair kick on those registers, and the displacement
    is delta = round(V (s * eps)). delta has a symmetric distribution that does not depend on u, and the move is
    rejected unless both u and u + delta lie in the window, so it is a symmetric Metropolis move. Returns
    (delta, held): held is False if u lies outside the window (the register cannot hold it: a rejection)."""
    cc, V, s = hist.win_c[c], hist.win_V[c], hist.win_s[c]
    n = 1 << AMR_BITS
    hist.n_try[c] += 1
    k = amr_register(u, cc, V, s)
    if np.any((k < 0) | (k >= n)):
        hist.n_out[c] += 1
        return None, False
    hk = np.clip(amr_register(hist.res[c], cc, V, s), 0, n - 1)
    dm = pair_diffs(hk)
    sc = math.exp(hist.log_sc)
    q = min(1.0, sc * S_D / (K_RES * (K_RES - 1)))
    r = np.where(rng.random(len(dm)) < q, np.where(rng.random(len(dm)) < 0.5, -1, 1), 0)
    p_iso = min(1.0, sc * S_D * EPS)
    iso = np.where(rng.random(D) < p_iso, np.where(rng.random(D) < 0.5, -1, 1), 0)
    delta = np.rint(V @ (s * (r @ dm + iso))).astype(np.int64)
    k2 = amr_register(wrap(u + delta), cc, V, s)
    if np.any((k2 < 0) | (k2 >= n)):
        hist.n_out[c] += 1
        return None, True
    return delta, True


def log_tf(u, P):
    """log of the TF-cell proposal density of (t0, f0) per grid point."""
    return math.log(P[u[0] >> CELL_SHIFT, u[1] >> CELL_SHIFT]) - 2 * CELL_SHIFT * math.log(2)


def log_prior(u):
    """log prior per grid point (t0, f0, Q, phi uniform; lnA from the SNR prior)."""
    return -D * NB * math.log(2) + log_prior_lnA(u)


def birth_dist(hist, c):
    """(mean, cov) of the history-informed (Q, lnA) birth proposal of Tian et al., or None (-> prior)."""
    if hist is None or hist.n[c] < N_MIN_HIST:
        return None
    C = S_D * (hist.cov(c)[2:4, 2:4] + EPS * np.eye(2))
    return hist.mean[c, 2:4], C


def draw_birth(rng, cdf, P, hist):
    k = int(np.searchsorted(cdf, rng.random() * cdf[-1]))
    i, j = divmod(k, 1 << TF_BITS)
    u = rng.integers(0, G, size=D)
    u[0] = (i << CELL_SHIFT) + rng.integers(0, 1 << CELL_SHIFT)
    u[1] = (j << CELL_SHIFT) + rng.integers(0, 1 << CELL_SHIFT)
    bd = birth_dist(hist, cell_of(u))
    if bd is not None:
        m, C = bd
        u[2:4] = np.rint(rng.multivariate_normal(m, C)).astype(np.int64)
    return u


def log_birth_density(u, P, hist):
    """log proposal probability of drawing grid point u in a birth."""
    lg = log_tf(u, P) - NB * math.log(2)                       # phi uniform
    bd = birth_dist(hist, cell_of(u))
    if bd is None:
        return lg - 2 * NB * math.log(2)                       # Q, lnA uniform
    m, C = bd
    x = u[2:4] - m
    Ci = np.linalg.inv(C)
    return lg - 0.5 * x @ Ci @ x - 0.5 * math.log(np.linalg.det(C)) - math.log(2 * math.pi)


# ============================================================================ sampler
def p_birth(K):
    return 1.0 if K == 0 else (0.0 if K == K_MAX else 0.5)


def run(method, seed, n_iter, n_adapt, data_seed=1, beta=1.0, n_temps=N_TEMPS, record_waves=True,
        record_params_every=0):
    d, h_inj = make_data(data_seed)
    hh = h_inj @ h_inj
    logL_inj = -0.5 * (d - h_inj) @ (d - h_inj)
    EVALS[0] = FD_EVALS[0] = 0                         # count everything, including the TF birth map
    P, cdf = tf_proposal(d)
    evals_setup = EVALS[0]
    rng = np.random.default_rng(seed)
    base = method.replace("_nb", "")
    adaptive_birth = not method.endswith("_nb")
    chains = [Chain(d) for _ in range(n_temps)]
    hists = [History(base in ("qclt", "qamr"), amr=base == "qamr") if base not in ("fisher", "rw") else None
             for _ in range(n_temps)]
    logT = np.linspace(0, np.log(T_MAX), n_temps)
    swap_avg = np.full(n_temps - 1, 0.5)
    rec = dict(logL=np.empty(n_iter), K=np.empty(n_iter, np.int16), match=np.empty(n_iter), lnA=[],
               evals=np.empty(n_iter, np.int64), evals_fd=np.empty(n_iter, np.int64), evals_setup=evals_setup,
               waves=np.empty((n_iter // WAVE_EVERY, W1 - W0), np.float32) if record_waves else None,
               params=[])
    acc = {k: [0, 0] for k in ("fisher", "ap", "tf", "birth", "death")}
    t_start = time.process_time()

    def intrinsic(ch, T, slot, it):
        hist = hists[slot]
        w = ch.ws[rng.integers(len(ch.ws))]
        u = w.u
        x = rng.random()
        logH, u_new = 0.0, None
        if x < P_TF:
            kind = "tf"
            u_new = draw_birth(rng, cdf, P, None)              # TF redraw: prior-only (no history), all samplers
            logH = (log_prior(u_new) - log_prior(u)) - (log_birth_density(u_new, P, None) - log_birth_density(u, P, None))
        elif base == "rw" or (AP_FALLBACK == "rw" and base != "fisher" and (it < T0 or x < P_TF + P_FISHER_AP)):
            kind = "fisher"                                    # (counted with the conventional moves)
            delta = np.rint(rng.normal(size=D) * RW_FRAC * G).astype(np.int64)
            cand = wrap(u + delta)
            if delta.any() and in_prior(cand):
                u_new = cand
                logH = log_prior(u_new) - log_prior(u)
        elif base == "fisher" or it < T0 or x < P_TF + P_FISHER_AP:
            kind = "fisher"
            u_new, logH = propose_fisher(u, rng)
            if u_new is not None:
                logH += log_prior(u_new) - log_prior(u)
        else:
            kind = "ap"
            c = cell_of(u)
            ready = hist.n[c] >= N_MIN_HIST
            if ready:
                if base == "qamr":
                    delta, ready = propose_ap_qclt_amr(hist, c, u, rng)   # not held -> rejection, no adaptation
                    if delta is None:
                        delta = np.zeros(D, np.int64)
                else:
                    delta = (propose_ap_qclt if base == "qclt" else propose_ap_gauss)(hist, c, rng)
                cand = wrap(u + delta)
                # Tian et al.'s move never changes the occupied candidate points: stay in the cell
                if delta.any() and in_prior(cand) and cell_of(cand) == c:
                    u_new = cand
                    logH = log_prior(u_new) - log_prior(u)
        accepted, alpha = False, 0.0
        if u_new is not None:
            new = Wavelet(u_new)
            dL, lo, hi, seg = delta_logL(ch.r, w, new)
            log_a = beta * dL / T + logH
            alpha = math.exp(min(0.0, log_a))
            if rng.random() < alpha:
                ch.r[lo:hi] = seg
                ch.logL += dL
                w.u, w.i0, w.i1, w.h = new.u, new.i0, new.i1, new.h
                accepted = True
        if kind == "ap" and ready and adapting:
            hist.adapt_scale(alpha)                            # Andrieu-Thoms on the move step
        if slot == 0:
            acc[kind][0] += accepted
            acc[kind][1] += 1

    def rj(ch, T, slot):
        hist = hists[slot] if adaptive_birth else None
        K = len(ch.ws)
        if rng.random() < p_birth(K):
            new = Wavelet(draw_birth(rng, cdf, P, hist))
            if not in_prior(new.u):
                ok = False
            else:
                dL, lo, hi, seg = delta_logL(ch.r, None, new)
                log_a = (beta * dL / T + log_prior(new.u) - log_birth_density(new.u, P, hist)
                         + math.log((1 - p_birth(K + 1)) / p_birth(K)))
                ok = math.log(rng.random()) < log_a
                if ok:
                    ch.r[lo:hi] = seg; ch.logL += dL; ch.ws.append(new)
            kind = "birth"
        else:
            i = rng.integers(K)
            old = ch.ws[i]
            dL, lo, hi, seg = delta_logL(ch.r, old, None)
            log_a = (beta * dL / T - log_prior(old.u) + log_birth_density(old.u, P, hist)
                     + math.log(p_birth(K - 1) / (1 - p_birth(K))))
            ok = math.log(rng.random()) < log_a
            if ok:
                ch.r[lo:hi] = seg; ch.logL += dL; ch.ws.pop(i)
            kind = "death"
        if slot == 0:
            acc[kind][0] += ok
            acc[kind][1] += 1

    adapting = True
    for it in range(n_iter):
        adapting_T = adapting = it < n_adapt
        T = np.exp(logT)
        for c, ch in enumerate(chains):
            if rng.random() < P_RJ:
                rj(ch, T[c], c)
            elif ch.ws:
                intrinsic(ch, T[c], c, it)
            if hists[c] is not None and adapting:             # Tian et al.: history collected every step
                for w in ch.ws:
                    hists[c].add(w.u, rng)
        # non-reversible PT: deterministic even-odd swaps (Syed et al. 2021)
        for c in range(it % 2, n_temps - 1, 2):
            a = beta * (1 / T[c] - 1 / T[c + 1]) * (chains[c + 1].logL - chains[c].logL)
            ok = math.log(rng.random()) < a
            if ok:
                chains[c], chains[c + 1] = chains[c + 1], chains[c]
            swap_avg[c] += 0.02 * (ok - swap_avg[c])
        if adapting_T and ADAPT_TEMPS and n_temps > 1:
            kappa = 0.01 * 1000.0 / (it + 1000.0)
            gaps = np.diff(logT) * np.exp(kappa * 0.5 * (swap_avg - swap_avg.mean()))
            logT = np.concatenate([[0.0], np.cumsum(gaps / gaps.sum() * np.log(T_MAX))])
        cold = chains[0]
        h = d - cold.r
        nh = h @ h
        rec["logL"][it] = cold.logL - logL_inj
        rec["K"][it] = len(cold.ws)
        rec["evals"][it] = EVALS[0]
        rec["evals_fd"][it] = FD_EVALS[0]
        rec["match"][it] = (h @ h_inj) / math.sqrt(nh * hh) if nh > 0 else 0.0
        if record_params_every and it % record_params_every == 0:
            # cold-chain wavelets in physical units, plus each wavelet's SNR
            rec["params"].append((it, np.array([[*phys(w.u), math.sqrt(w.h @ w.h)] for w in cold.ws])))
        if record_waves and it % WAVE_EVERY == 0:
            rec["waves"][it // WAVE_EVERY] = h[W0:W1]
        if beta == 0 and cold.ws:
            th = phys(cold.ws[rng.integers(len(cold.ws))].u)
            rec["lnA"].append(th[3] + math.log(snr_factor(th)))
    rec["cpu"] = time.process_time() - t_start
    rec["acc"] = {k: (a / n if n else np.nan) for k, (a, n) in acc.items()}
    rec["temps"] = np.exp(logT)
    rec["swap"] = swap_avg
    rec["sc"] = np.array([math.exp(h.log_sc) for h in hists]) if base not in ("fisher", "rw") else np.ones(n_temps)
    return rec


# ============================================================================ checks
def prior_rho_reference(n, seed=5):
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


def _prior_job(m):
    return run(m, 11, 200_000, 50_000, beta=0.0, n_temps=1, record_waves=False, record_params_every=500)


def prior_test():
    """Likelihood off: every sampler must return N ~ U{0..N_max} (N = number of wavelets) and the exact SNR prior."""
    from multiprocessing import Pool
    from scipy import stats
    with Pool(len(METHODS)) as pool:
        runs = pool.map(_prior_job, METHODS)
    ref = prior_rho_reference(200_000)
    for m, r in zip(METHODS, runs):
        K = r["K"][60_000:]                                   # after the adaptation freeze (50k) + 10k
        hist = np.bincount(K, minlength=K_MAX + 1) / len(K)
        lnA = np.array(r["lnA"])
        rho = np.exp(lnA[int(len(lnA) * 0.3)::100])
        ks = stats.ks_2samp(rho, ref)
        print(f"{m:7s} P(N) = {np.round(hist, 3)} (uniform {1 / (K_MAX + 1):.3f});  "
              f"KS(SNR vs exact prior) D = {ks.statistic:.3f}, p = {ks.pvalue:.2f} [{len(rho)} thinned draws]")
        P = np.concatenate([w for it, w in r["params"] if it >= 60_000 and len(w)])   # every 500 iterations
        for k in (0, 1, 2, 4):                                  # t0, f0, Q, phase: uniform priors
            ks = stats.kstest((P[:, k] - LO[k]) / (HI[k] - LO[k]), "uniform")
            print(f"        {PARAMS[k]:4s} vs uniform prior: KS D = {ks.statistic:.3f}, p = {ks.pvalue:.2f} [{len(P)} wavelets]")


def validate_qclt_against_circuit(buf):
    """Exact output distribution of the real qcov circuit vs the qclt sampler for the same registers."""
    from qiskit import QuantumCircuit, QuantumRegister
    from qcov import covariance_kick, exact_distribution, regs_values

    K = len(buf)
    q = 0.05
    H = [[QuantumRegister(NB, f"{p}_h{k}") for p in PARAMS] for k in range(K)]
    E = [QuantumRegister(NB + 4, f"{p}_eps") for p in PARAMS]
    npairs = K * (K - 1) // 2
    mv, sg = QuantumRegister(npairs, "move"), QuantumRegister(npairs, "sign")
    qc = QuantumCircuit(*E, *[r for h in H for r in h], mv, sg)
    for k in range(K):
        for p in range(D):
            for i in range(NB):
                if (buf[k, p] >> i) & 1:
                    qc.x(H[k][p][i])
    covariance_kick(qc, [E], H, zip(mv, sg), q * K * (K - 1), periodic=False)
    bits, w = exact_distribution(qc)
    eps = regs_values(bits, qc, E, signed=True)
    circ = {}
    for e, p in zip(map(tuple, eps), w):
        circ[e] = circ.get(e, 0.0) + p
    dm = pair_diffs(buf)
    ref = {}
    for rs in itertools.product((-1, 0, 1), repeat=npairs):
        p = np.prod([q / 2 if r else 1 - q for r in rs])
        e = tuple(np.array(rs) @ dm)
        ref[e] = ref.get(e, 0.0) + p
    keys = set(circ) | set(ref)
    return 0.5 * sum(abs(circ.get(k, 0) - ref.get(k, 0)) for k in keys), qc.num_qubits, len(qc.data)


def example_registers(K=4):
    """Synthetic 16-bit register contents (the equivalence holds for any contents)."""
    rng = np.random.default_rng(3)
    centre = np.array([0.5, 120.0, 6.0, np.log(3.0), 1.0])
    u0 = np.rint((centre - LO) / DX - 0.5).astype(np.int64)
    spread = np.array([40, 300, 2000, 1500, 1200])
    return np.clip(u0 + rng.normal(size=(K, D)) * spread, 0, G - 1).astype(np.int64)


# ============================================================================ experiment
N_SEEDS = 16
N_ITER = 40_000
N_ADAPT = N_ITER // 2
MONITOR_EVERY = 500
OUT = Path(__file__).parent / "figures" / "previous"
DATA = Path(__file__).parent / "data"            # run data and logs (figures/ holds PDFs only)
TAG = ""                                  # set by --tag; suffix for output files


def _job(args):
    m, s = args
    return m, s, run(m, s, N_ITER, N_ADAPT, record_params_every=50 if s == 0 else 0)


def iact(x, c=5.0):
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


def tian_convergence_lengths(waves, thresh=0.2):
    """Tian et al. two-run diagnostic over all pairs of runs. waves: (runs, samples, n_points).
    At each monitoring step t, statistics use the second half of the samples collected so far."""
    n_runs, n_s, _ = waves.shape
    steps = np.arange(MONITOR_EVERY // WAVE_EVERY, n_s + 1, MONITOR_EVERY // WAVE_EVERY)
    mu = np.empty((n_runs, len(steps), waves.shape[2]))
    sd = np.empty_like(mu)
    for k, s in enumerate(steps):
        seg = waves[:, s // 2:s]
        mu[:, k] = seg.mean(1)
        sd[:, k] = seg.std(1)
    lengths = []
    for a, b in itertools.combinations(range(n_runs), 2):
        den = np.maximum((sd[a] + sd[b]) / 2, 1e-3)
        r1 = (np.abs(mu[a] - mu[b]) / den).mean(1)
        r2 = (np.abs(sd[a] - sd[b]) / den).mean(1)
        ok = np.nonzero((r1 < thresh) & (r2 < thresh))[0]
        lengths.append(steps[ok[0]] * WAVE_EVERY if len(ok) else np.nan)
    return np.array(lengths)


def experiment():
    from multiprocessing import Pool

    tv, nq, ng = validate_qclt_against_circuit(example_registers())
    print(f"QCLT move vs qcov circuit ({nq} qubits, {ng} gates): TV distance of exact distributions = {tv:.2e}")
    assert tv < 1e-9

    jobs = [(m, s) for m in METHODS for s in range(N_SEEDS)]
    t0 = time.time()
    with Pool(min(len(jobs), int(os.environ.get("N_WORKERS", os.cpu_count() or 4)))) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    print(f"{len(jobs)} runs x {N_ITER} iterations x {N_TEMPS} temperatures in {time.time() - t0:.0f}s")
    out = {}
    for m, s, r in res:
        out.setdefault(m, []).append(r)
    conv = {m: tian_convergence_lengths(np.stack([r["waves"] for r in out[m]])) for m in METHODS}
    np.savez_compressed(
        DATA / f"tdrjmcmc_qclt{TAG}_runs.npz",
        **{f"{m}_{k}": np.stack([r[k] for r in out[m]]) for m in METHODS for k in ("logL", "K", "match", "sc", "temps")},
        **{f"{m}_cpu": np.array([r["cpu"] for r in out[m]]) for m in METHODS},
        **{f"{m}_conv": conv[m] for m in METHODS},
        **{f"{m}_acc_{a}": np.array([r["acc"][a] for r in out[m]]) for m in METHODS
           for a in ("fisher", "ap", "tf", "birth", "death")},
    )
    import pickle
    seed0 = {m: [{k: out[m][0][k] for k in ("params", "K", "match")}] for m in METHODS}
    with open(DATA / f"tdrjmcmc_qclt{TAG}_seed0.pkl", "wb") as f:
        pickle.dump(seed0, f)
    summarize()
    if TAG in ("_ablation", "_amr"):            # partial sampler sets: merged into the main figure by load()
        return
    plot()
    burnin_plot()


def burnin_plot():
    import pickle
    import tdrjmcmc_figures as Fg
    with open(DATA / f"tdrjmcmc_qclt{TAG}_seed0.pkl", "rb") as f:
        seed0 = pickle.load(f)
    Fg.fig_burnin(seed0, methods=METHODS, n_iter=N_ITER, n_adapt=N_ADAPT, param_every=50,
                  name=f"tdrjmcmc_burnin{TAG}", t0=T0, chirp_marks=(DATASET == "chirp"))


ALL_METHODS = ["fisher", "ap", "qclt", "ap_nb", "qclt_nb", "qamr", "qamr_nb"]


def load():
    """Run data for the current tag. The default (main) figure merges the history-birth runs with the
    time-frequency-birth runs ("ablation", same seeds and settings) into one five-sampler comparison."""
    z = dict(np.load(DATA / f"tdrjmcmc_qclt{TAG}_runs.npz"))
    for extra in ("ablation", "amr"):               # TF-birth variants; QCLT + AMR (ours)
        f = DATA / f"tdrjmcmc_qclt_{extra}_runs.npz"
        if TAG == "" and f.exists():
            za = np.load(f)
            z.update({k: za[k] for k in za.files if not k.startswith("fisher_")})
    return z


def summarize(quiet=False):
    z = load()
    half = N_ITER // 2
    rows = {}
    lines = [f"{'sampler':18s} {'it->match0.9':>12s} {'conv. pairs':>11s} {'conv. length':>13s} {'ESS/1k logL':>12s} {'ESS/1k N':>9s} "
             f"{'acc move':>9s} {'s_c (cold)':>11s} {'ms/iter':>8s} {'post match':>11s} {'post <N>':>9s} {'R-hat logL':>10s}"]
    for m in METHODS:
        L, Kc, M = z[f"{m}_logL"], z[f"{m}_K"].astype(float), z[f"{m}_match"]
        conv = z[f"{m}_conv"]
        ess_L = np.array([1000 / iact(l[half:]) for l in L])
        ess_K = np.array([1000 / iact(k[half:]) for k in Kc])
        acc_move = z[f"{m}_acc_fisher"] if m == "fisher" else z[f"{m}_acc_ap"]
        n = L.shape[1] // 2
        x = np.concatenate([L[:, half:half + n // 2], L[:, half + n // 2:]])
        W, B = x.var(1, ddof=1).mean(), (n // 2) * x.mean(1).var(ddof=1)
        rhat = math.sqrt(((n // 2 - 1) / (n // 2) * W + B / (n // 2)) / W)
        hit = np.array([np.argmax(x >= 0.9) + 1 if np.any(x >= 0.9) else np.nan for x in M])
        rows[m] = dict(conv=conv, ess_L=ess_L, ess_K=ess_K, hit=hit)
        lines.append(f"{LABELS[m]:18s} {np.nanmedian(hit):12.0f} {np.isfinite(conv).sum():5d}/{len(conv):<5d} {np.nanmean(conv):13.0f} "
                     f"{np.median(ess_L):12.1f} {np.nanmedian(ess_K):9.1f} {np.nanmean(acc_move):9.3f} "
                     f"{np.median(z[f'{m}_sc'][:, 0]):11.3g} {1e3 * np.mean(z[f'{m}_cpu']) / N_ITER:8.2f} "
                     f"{np.median(M[:, half:]):11.3f} {Kc[:, half:].mean():9.2f} {rhat:10.3f}")
    if not quiet:
        print("\n".join(lines))
    return rows


# ============================================================================ figure
COLORS = {"fisher": "#2a78d6", "ap": "#eda100", "qclt": "#e87ba4", "ap_nb": "#1baf7a", "qclt_nb": "#6250d6",
          "qamr": "#d62f2f", "qamr_nb": "#8c1c13",
          "rw": "#52514e"}
LSTYLE = {"fisher": "-", "ap": "--", "qclt": "-", "ap_nb": "-.", "qclt_nb": ":", "rw": "-", "qamr": "-", "qamr_nb": "--"}
SHORT = {"fisher": "Fisher\n\nclassical", "ap": "AP\n\nclassical", "qclt": "QCLT-AP\n\nquantum",
         "ap_nb": "AP\nTF birth\nclassical", "qclt_nb": "QCLT-AP\nTF birth\nquantum", "rw": "Naive\n\nclassical",
         "qamr": "QCLT+AMR\n(ours)\nquantum", "qamr_nb": "QCLT+AMR\nTF birth\nquantum"}
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"


def plot():
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    z = load()
    rows = summarize(quiet=True)
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    it = np.arange(1, N_ITER + 1)
    try:                                               # real BayesWave on the same chirp and SNR, if available
        import bw_ref as bw
        bw_med = bw.median_curve(int(SNR_INJ)) if DATASET == "chirp" else None
        bw_counts = bw.wavelet_counts(int(SNR_INJ)) if DATASET == "chirp" else []
        bw_med_m = bw.median_curve(int(SNR_INJ), "matched") if DATASET == "chirp" else None
        bw_counts_m = bw.wavelet_counts(int(SNR_INJ), "matched") if DATASET == "chirp" else []
        if bw_med is None and not bw_counts:
            bw = None
    except Exception:
        bw, bw_med, bw_counts, bw_med_m, bw_counts_m = None, None, [], None, []
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4))
    axes = axes.ravel()
    for k, (key, ylab, title) in enumerate([
            ("match", "match with the injected signal", "(a) Waveform match"),
            ("K", r"wavelets in cold chain, $N$", "(b) Number of wavelets")]):
        ax = axes[k]
        for m in METHODS:
            x = z[f"{m}_{key}"].astype(float)
            xs = it
            if key == "K":                                  # trailing mean over 500 iterations
                ker = np.ones(500) / 500
                x = np.array([np.convolve(r, ker, mode="valid") for r in x])
                xs = it[499:]
            lo, med, hi = np.percentile(x, [25, 50, 75], axis=0)
            ax.fill_between(xs, lo, hi, color=COLORS[m], alpha=0.18, lw=0)
            ax.plot(xs, med, LSTYLE[m], color=COLORS[m], lw=1.3, label=LABELS[m])
        bw_hi = N_ITER
        for variant, med_c, counts, col, mfc in (("default", bw_med, bw_counts, "#0b0b0b", None),
                                                 ("matched", bw_med_m, bw_counts_m, "#9d9c97", "white")):
            if bw is None:
                break
            if key == "match" and med_c is not None:
                ax.plot(med_c["updates"], med_c["match"], "D-", ms=3, lw=1.1, color=col, mfc=mfc,
                        label=bw.LABELS[variant])
                bw_hi = max(bw_hi, med_c["updates"].max() * 1.5)
            if key == "K" and counts:
                n = min(len(c[1]) for c in counts)
                ker = np.ones(20) / 20                       # trailing mean over 20 saved samples
                Ds = np.array([np.convolve(np.nan_to_num(c[1][:n], nan=np.nanmedian(c[1])), ker, mode="valid")
                               for c in counts])
                xs_bw = counts[0][0][19:n]
                ax.plot(xs_bw, np.median(Ds, 0), "-", color=col, lw=1.1, label=bw.LABELS[variant])
                bw_hi = max(bw_hi, xs_bw.max() * 1.5)
        ax.set_xscale("log")
        ax.set_xlim(10 if key == "match" else 500, bw_hi)
        ax.axvline(T0, color=MUTED, lw=0.6, ls=":")
        ax.axvline(N_ADAPT, color=MUTED, lw=0.6, ls="--")
        ax.set_xlabel("model updates per chain\n(ours: 1 per iteration; BayesWave: 100 per iteration)")
        ax.set_ylabel(ylab); ax.set_title(title, loc="left")
    axes[0].set_ylim(0, 1)
    axes[0].text(T0 * 1.1, 0.04, r"$t_0$", color=MUTED, fontsize=7)
    axes[0].text(N_ADAPT * 0.9, 0.04, "adaptation\nfrozen", color=MUTED, fontsize=6.5, ha="right")

    ax = axes[2]
    _box(ax, [rows[m]["conv"] for m in METHODS])
    ax.set_ylabel("iterations until two runs agree\n(lower = faster)")
    ax.set_title("(c) When do independent runs agree? (ours)", loc="left")
    ax.set_ylim(0, N_ITER)
    ax.set_xticklabels([f"{SHORT[m]}\n{np.isfinite(rows[m]['conv']).sum()}/{len(rows[m]['conv'])}"
                        for m in METHODS], fontsize=5.6)
    ax.set_xlabel("run pairs that agreed (Tian et al. test)", color=MUTED, fontsize=7)
    ax.text(0.98, 0.03, "not computable for BayesWave: the test\ncompares per-sample waveforms, which\nBayesWave does not save",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color=MUTED)

    ax = axes[3]
    bw_ess = {}                                        # BayesWave: from its saved samples (one per 10^4 updates)
    if bw is not None:
        for variant in ("default", "matched"):
            cnt = bw.wavelet_counts(int(SNR_INJ), variant) if DATASET == "chirp" else []
            e = []
            for x, Dn in cnt:
                h = len(Dn) // 2
                d = Dn[h:][np.isfinite(Dn[h:])]
                e.append(len(d) / iact(d) / ((x[-1] - x[h]) / 1000))
            if e:
                bw_ess[variant] = np.array(e)
    _box(ax, [rows[m]["ess_K"] for m in METHODS])
    for k, (variant, e) in enumerate(bw_ess.items()):
        xk = len(METHODS) + k
        ax.plot(xk + np.linspace(-0.12, 0.12, len(e)), e, "D", ms=3.5, color=bw.COLORS[variant] if hasattr(bw, "COLORS")
                else ("#0b0b0b" if variant == "default" else "#9d9c97"), mec="white", mew=0.5)
        ax.plot([xk - 0.25, xk + 0.25], [np.median(e)] * 2, color=INK, lw=1.0)
    ax.set_xticks(range(len(METHODS) + len(bw_ess)))
    ax.set_xticklabels([SHORT[m] for m in METHODS] + [f"BW\n{v}\nclassical" for v in bw_ess], fontsize=5.6)
    ax.set_yscale("log")
    ax.set_ylabel("independent samples of N per 1000\nmodel updates per chain (higher = better)")
    ax.set_title("(d) How fast N is re-explored", loc="left")
    if bw_ess:
        ax.text(0.99, 0.5, "BW = BayesWave (real)\nper CPU second instead:\nours 0.24-0.34, BayesWave\n0.05-1.3 (compiled vs Python)",
                transform=ax.transAxes, ha="right", va="center", fontsize=6.0, color=MUTED)

    for ax in axes:
        ax.grid(color=GRIDC, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=6.6)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), w_pad=1.5, h_pad=1.5)
    fig.text(0.01, 0.0, QC_NOTE, fontsize=6.2, color=MUTED, va="bottom")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"tdrjmcmc_qclt{TAG}.pdf", bbox_inches="tight")
    if os.environ.get("FIG_PREVIEW_DIR"):           # PNG previews only outside the repo (PDF is the figure)
        fig.savefig(Path(os.environ["FIG_PREVIEW_DIR"]) / f"tdrjmcmc_qclt{TAG}.png", dpi=110, bbox_inches="tight")
    print("saved", OUT / f"tdrjmcmc_qclt{TAG}.pdf")


def _box(ax, data):
    bp = ax.boxplot([np.asarray(x)[np.isfinite(x)] for x in data], positions=range(len(data)), widths=0.55,
                    patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.0),
                    whiskerprops=dict(color=MUTED, lw=0.8), capprops=dict(color=MUTED, lw=0.8))
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set(facecolor=COLORS[m], alpha=0.85, edgecolor="white", lw=0.8)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([SHORT[m] for m in METHODS])
    ax.tick_params(axis="x", length=0)


if __name__ == "__main__":
    if "--methods" in sys.argv:
        METHODS[:] = sys.argv[sys.argv.index("--methods") + 1].split(",")
    if "--tag" in sys.argv:
        TAG = "_" + sys.argv[sys.argv.index("--tag") + 1]
    if "--t0" in sys.argv:                         # start of the adaptive move step (Tian et al.'s t0)
        T0 = int(sys.argv[sys.argv.index("--t0") + 1])
    if "--nmin" in sys.argv:                       # history entries a cell needs before its covariance is used
        N_MIN_HIST = int(sys.argv[sys.argv.index("--nmin") + 1])
    if "--data" in sys.argv:
        DATASET = sys.argv[sys.argv.index("--data") + 1]
        if DATASET == "hard":                      # the signal spans most of the segment
            W0, W1, WAVE_EVERY = int(0.10 * FS), int(0.90 * FS), 20
    if "--snr" in sys.argv:
        SNR_INJ = float(sys.argv[sys.argv.index("--snr") + 1])
    if "--birth" in sys.argv:
        BIRTH = sys.argv[sys.argv.index("--birth") + 1]
    if "--kmax" in sys.argv:
        K_MAX = int(sys.argv[sys.argv.index("--kmax") + 1])
    if "--tmax" in sys.argv:
        T_MAX = float(sys.argv[sys.argv.index("--tmax") + 1])
    if "--seeds" in sys.argv:
        N_SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1])
    print(f"T0 = {T0}, N_MIN_HIST = {N_MIN_HIST}, methods = {METHODS}, tag = '{TAG}', data = {DATASET}, "
          f"SNR = {SNR_INJ}, birth = {BIRTH}, N_max = {K_MAX}, T_MAX = {T_MAX}, seeds = {N_SEEDS}", flush=True)
    if "--prior-test" in sys.argv:
        prior_test()
    elif "--plot-only" in sys.argv:
        if TAG == "" and "--methods" not in sys.argv and (DATA / "tdrjmcmc_qclt_ablation_runs.npz").exists():
            METHODS[:] = ALL_METHODS                # main figure: all five samplers
        summarize()
        plot()
    elif "--smoke" in sys.argv:
        for m in METHODS:
            t0 = time.time()
            r = run(m, 0, 4000, 2000)
            print(f"{m:7s} {time.time() - t0:5.1f}s match {r['match'][-1]:.3f} K {r['K'][-1]} "
                  f"acc {({k: round(v, 3) for k, v in r['acc'].items()})} s_c {np.round(r['sc'], 3)}")
    else:
        OUT.mkdir(exist_ok=True)
        DATA.mkdir(exist_ok=True)
        experiment()
