"""Comparison of fully-quantum covariance estimators for the adaptive walk.

Methods (all estimate the sample covariance S of N history samples of d parameters):
  1. Max  (Cauchy-Schwarz upper end):  C_pq = +sigma_p sigma_q          (|rho| = 1)
  2. Min  (Cauchy-Schwarz lower end):  C_pp = sigma_p^2, C_pq = -sigma_p sigma_q (p != q)
  3. Zero-mean approximation:          C = sum_j x_j x_j^T / (N - 1)  (no centering)
  4. CLT walk
       classical: FINAL.ipynb (S computed classically, eps = L z, z = sum of Rademacher / sqrt(R))
       quantum:   quantum-cov-rw.ipynb (eps = sum_{a<b} r_ab (h_a - h_b), lazy 3-point coins)

  5. SWAP test (the standard quantum inner-product / distance estimator; Lloyd, Mohseni & Rebentrost 2013,
     arXiv:1307.0411): each centred column is amplitude-encoded; a SWAP test between
     (|0>|a^> + |1>|b^>)/sqrt2 and (|a||0> - |b||1>)/sqrt(Z), Z = |a|^2 + |b|^2, has P(0) = 1/2 + |a - b|^2/(4Z),
     which gives the signed inner product a.b = (Z - |a - b|^2)/2 and so C_pq. (A plain SWAP test between |a^>
     and |b^> only gives |<a^|b^>|^2, losing the sign of the covariance.) Like amplitude encoding in general, it
     needs the centred columns and their norms classically, so the variances come for free.

Methods 1-3 and 5 are run as coherent circuits that encode a value in an ancilla's |1> probability and
are measured only at the end (as in cov.ipynb / cross-cov.ipynb). Their ancilla probabilities are
checked against exact statevector simulation (qiskit.quantum_info.Statevector), then shot noise is drawn as
Binomial(shots, p).
The quantum CLT walk is sampled from the actual circuit (qcov.sample_circuit, validated against
Aer in quantum-cov-rw.ipynb).

Fixes relative to amelia.py:
  * error metric: amelia used trace(C_est^H C_true), an inner product, not an error;
    here: relative Frobenius error ||C_est - S||_F / ||S||_F.
  * zero-mean estimate was not normalised (missing / (N-1)).
  * the two datasets were drawn independently, so the true cross-covariance being estimated was 0;
    here every estimator targets the covariance of one set of samples (what the walk adapts to).
  * f = 0.8 fnorm + 0.2 funif (and Q likewise) is a weighted *sum*, not the intended 80/20 mixture;
    phi = exp(U(0, log 2pi)) is log-uniform on [1, 2pi], not uniform on [0, 2pi].
  * raw parameters differ by ~20 orders of magnitude (A ~ 1e-7, f ~ 1e3), so pooled histograms were
    dominated by f; here everything lives on the walk's integer grid (the register values).
  * waveforms (40960 samples each) were generated and never used.

Run:  python3.12 cov_methods.py   ->  three standalone covariance-method PDFs in figures/previous/
"""
import itertools
import time
from pathlib import Path

import os

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

from qcov import covariance_kick, history_pairs, regs_values, sample_circuit

# ----------------------------------------------------------------------------- settings
D = 5                      # parameters (A, f, t, tau, phi)
N = 8                      # history samples (3 index qubits)
N_BITS = 4                 # 16-point grid per parameter, signed coordinates -8..7
E_BITS = 10                # signed output register of the CLT kick (no wrap)
Q_CLT = 1 / 3              # lazy-coin probability: 3-point Gauss-Hermite rule (zero excess kurtosis)
N_ROUNDS_CLASSICAL = 256   # FINAL.ipynb
SCALE2_HAARIO = 2.38**2 / D
N_DATASETS = 300
BUDGET_B = 2**14           # total shots per method in panel (b)
BUDGETS_A = 2 ** np.arange(7, 18)
POOL_A = 2**20             # circuit shots drawn once for panel (a)
SEED = 2026
OUT = Path(__file__).parent / "figures" / "previous"
DATA = Path(__file__).parent / "data"            # run data and logs (figures/ holds PDFs only)

GRID_LO, GRID_HI = -(2 ** (N_BITS - 1)), 2 ** (N_BITS - 1) - 1


# ----------------------------------------------------------------------------- data
def random_dataset(rng):
    """N samples of a correlated d-dim Gaussian on the signed integer grid.

    Random correlation structure (Gaussian factor model), per-parameter scale 1-3 grid units
    and a random offset of the walk from the grid centre (history need not be centred).
    """
    W = rng.normal(size=(D, D))
    C0 = W @ W.T + 0.1 * np.eye(D)
    s = np.sqrt(np.diag(C0))
    R0 = C0 / np.outer(s, s)
    scales = rng.uniform(1.0, 3.0, D)
    mu = rng.uniform(-3, 3, D)
    X = rng.multivariate_normal(mu, R0 * np.outer(scales, scales), size=N)
    return np.clip(np.round(X), GRID_LO, GRID_HI).astype(int)          # (N, d)


def rel_err(C, S):
    return np.linalg.norm(C - S) / np.linalg.norm(S)


# ----------------------------------------------------------------------------- methods 1-3 as circuits
def value_oracle(qc, index_regs, target, values, lo, hi):
    """|j>|0> -> |j>(sqrt(1-p_j)|0> + sqrt(p_j)|1>),  p_j = (values[j] - lo) / (hi - lo).

    index_regs: list of registers jointly holding the (flattened) index j, little-endian.
    """
    qubits = [q for r in index_regs for q in r]
    for j, v in enumerate(values):
        p = (v - lo) / (hi - lo) if hi > lo else 0.0
        if p <= 1e-15:
            continue
        flips = [q for k, q in enumerate(qubits) if not (j >> k) & 1]
        if flips:
            qc.x(flips)
        qc.mcry(2 * np.arcsin(np.sqrt(min(p, 1.0))), qubits, target)
        if flips:
            qc.x(flips)


def variance_circuit(x):
    """Two-copy variance: P(1) = E_{i,j}[(x_i - x_j)^2] / (2C) = Var_pop / C, C = range^2 / 2."""
    n = int(np.log2(len(x)))
    I, J, a = QuantumRegister(n, "i"), QuantumRegister(n, "j"), QuantumRegister(1, "anc")
    qc = QuantumCircuit(I, J, a)
    qc.h(I); qc.h(J)                                                   # uniform over samples
    C = (x.max() - x.min()) ** 2 / 2
    vals = [(x[i] - x[j]) ** 2 / 2 for j in range(len(x)) for i in range(len(x))]   # index i + N j
    value_oracle(qc, [I, J], a[0], vals, 0.0, C)
    return qc, C


def moment_circuit(xp, xq):
    """Uncentred second moment: P(1) = (E[x_p x_q] - m) / (M - m)."""
    n = int(np.log2(len(xp)))
    I, a = QuantumRegister(n, "i"), QuantumRegister(1, "anc")
    qc = QuantumCircuit(I, a)
    qc.h(I)
    prod = xp * xq
    m, M = float(prod.min()), float(prod.max())
    value_oracle(qc, [I], a[0], prod, m, M)
    return qc, m, M


def pairwise_circuit(xp, xq):
    """Centred pairwise covariance (the variance circuit's trick, for every pair of parameters):
    P(1) = (E_ij[(x_ip - x_jp)(x_iq - x_jq)] / 2 - lo) / (hi - lo); the average over all ordered sample pairs
    (i, j) is the population covariance, so no mean has to be known or estimated."""
    n = int(np.log2(len(xp)))
    I, J, a = QuantumRegister(n, "i"), QuantumRegister(n, "j"), QuantumRegister(1, "anc")
    qc = QuantumCircuit(I, J, a)
    qc.h(I); qc.h(J)
    vals = [(xp[i] - xp[j]) * (xq[i] - xq[j]) / 2 for j in range(len(xp)) for i in range(len(xp))]   # i + N j
    lo, hi = min(vals), max(vals)
    value_oracle(qc, [I, J], a[0], vals, lo, hi)
    return qc, lo, hi


def pairwise_exact(X):
    """Analytic P(1), lo, hi of every pairwise circuit (p <= q)."""
    out = []
    for p, q in itertools.combinations_with_replacement(range(D), 2):
        x, y = X[:, p].astype(float), X[:, q].astype(float)
        v = ((x[:, None] - x[None, :]) * (y[:, None] - y[None, :]) / 2).ravel()
        lo, hi = v.min(), v.max()
        out.append((p, q, (v.mean() - lo) / (hi - lo) if hi > lo else 0.0, lo, hi))
    return out


def pairwise_estimate(X, budget, rng, exact=None):
    """Centred pairwise estimate with the budget split evenly over the D(D+1)/2 circuits; unbiased."""
    ex = exact if exact is not None else pairwise_exact(X)
    s = max(budget // len(ex), 1)
    C = np.zeros((D, D))
    for p, q, p1, lo, hi in ex:
        C[p, q] = C[q, p] = lo + (hi - lo) * rng.binomial(s, p1) / s
    return C * N / (N - 1)


def validate_pairwise(X):
    worst = 0.0
    for p, q, p1, _, _ in pairwise_exact(X):
        worst = max(worst, abs(ancilla_p1(pairwise_circuit(X[:, p].astype(float), X[:, q].astype(float))[0]) - p1))
    return worst


def swap_distance_circuit(a, b):
    """SWAP-test distance estimation (Lloyd, Mohseni & Rebentrost 2013) for real vectors a, b of length 2^n.
    P(ancilla = 0) = 1/2 + |a - b|^2 / (4 Z), Z = |a|^2 + |b|^2. The ancilla is the last qubit."""
    n = int(np.log2(len(a)))
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    Z = na**2 + nb**2
    data, flag, phi, anc = (QuantumRegister(n, "data"), QuantumRegister(1, "flag"), QuantumRegister(1, "phi"),
                            QuantumRegister(1, "anc"))
    qc = QuantumCircuit(data, flag, phi, anc)
    qc.append(StatePreparation(np.concatenate([a / na, b / nb]) / np.sqrt(2)), [*data, flag[0]])   # |psi>
    qc.append(StatePreparation(np.array([na, -nb]) / np.sqrt(Z)), [phi[0]])                          # |phi>
    qc.h(anc)
    qc.cswap(anc[0], flag[0], phi[0])
    qc.h(anc)
    return qc, Z


def swap_exact(X):
    """Analytic P(0) and Z of the SWAP-test circuit for every pair p < q of centred columns, and the norms^2."""
    A = X.astype(float) - X.mean(0)
    pairs = list(itertools.combinations(range(D), 2))
    Z = np.array([A[:, p] @ A[:, p] + A[:, q] @ A[:, q] for p, q in pairs])
    P0 = np.array([0.5 + (A[:, p] - A[:, q]) @ (A[:, p] - A[:, q]) / (4 * z) for (p, q), z in zip(pairs, Z)])
    return pairs, P0, Z, (A * A).sum(0)


def swap_estimate(X, budget, rng, exact=None):
    """SWAP-test covariance estimate with the budget split evenly over the D(D-1)/2 pair circuits (variances from
    the known norms, as the amplitude encoding requires)."""
    pairs, P0, Z, n2 = exact if exact is not None else swap_exact(X)
    s = max(budget // len(pairs), 1)
    P0_hat = rng.binomial(s, P0) / s
    dot = (Z - 4 * Z * (P0_hat - 0.5)) / 2
    C = np.diag(n2)
    for (p, q), v in zip(pairs, dot):
        C[p, q] = C[q, p] = v
    return C / (N - 1)


def validate_swap(X):
    pairs, P0, Z, _ = swap_exact(X)
    A = X.astype(float) - X.mean(0)
    return max(abs((1 - ancilla_p1(swap_distance_circuit(A[:, p], A[:, q])[0])) - p0) for (p, q), p0 in zip(pairs, P0))


def ancilla_p1(qc):
    return float(Statevector(qc).probabilities([qc.num_qubits - 1])[1])


def exact_probabilities(X):
    """Analytic |1>-probabilities of every circuit used by methods 1-3."""
    p_var, C_var = np.zeros(D), np.zeros(D)
    for p in range(D):
        x = X[:, p].astype(float)
        C_var[p] = (x.max() - x.min()) ** 2 / 2
        p_var[p] = x.var() / C_var[p] if C_var[p] > 0 else 0.0
    p_mom, lo, hi = np.zeros((D, D)), np.zeros((D, D)), np.zeros((D, D))
    for p, q in itertools.combinations_with_replacement(range(D), 2):
        prod = (X[:, p] * X[:, q]).astype(float)
        lo[p, q], hi[p, q] = prod.min(), prod.max()
        p_mom[p, q] = (prod.mean() - lo[p, q]) / (hi[p, q] - lo[p, q]) if hi[p, q] > lo[p, q] else 0.0
    return p_var, C_var, p_mom, lo, hi


def validate_circuits(X):
    p_var, C_var, p_mom, lo, hi = exact_probabilities(X)
    worst = 0.0
    for p in range(D):
        qc, _ = variance_circuit(X[:, p].astype(float))
        worst = max(worst, abs(ancilla_p1(qc) - p_var[p]))
    for p, q in itertools.combinations_with_replacement(range(D), 2):
        qc, _, _ = moment_circuit(X[:, p].astype(float), X[:, q].astype(float))
        worst = max(worst, abs(ancilla_p1(qc) - p_mom[p, q]))
    return worst


def estimates_from_shots(X, budget, rng, probs=None):
    """Max, Min, Zero-mean estimates with the budget split evenly over each method's circuits."""
    p_var, C_var, p_mom, lo, hi = probs if probs is not None else exact_probabilities(X)
    corr = N / (N - 1)
    s_var = max(budget // D, 1)                             # 5 variance circuits
    var_hat = C_var * rng.binomial(s_var, p_var) / s_var * corr
    sig = np.sqrt(var_hat)
    C_max = np.outer(sig, sig)
    C_min = -np.outer(sig, sig)
    C_min[np.diag_indices(D)] = var_hat
    s_mom = max(budget // (D * (D + 1) // 2), 1)          # 15 moment circuits
    iu = np.triu_indices(D)
    mom = lo + (hi - lo) * rng.binomial(s_mom, p_mom) / s_mom
    Z = np.zeros((D, D))
    Z[iu] = mom[iu]
    Z = Z + np.triu(Z, 1).T
    return C_max, C_min, Z * corr


def exact_estimates(X):
    """Infinite-shot limit of methods 1-3."""
    S = np.cov(X.T, ddof=1)
    sig = np.sqrt(np.diag(S))
    C_max = np.outer(sig, sig)
    C_min = -np.outer(sig, sig)
    C_min[np.diag_indices(D)] = np.diag(S)
    return C_max, C_min, X.T @ X / (N - 1)


# ----------------------------------------------------------------------------- method 4: CLT walks
def classical_clt_steps(S, n, rng, rounds=N_ROUNDS_CLASSICAL):
    """FINAL.ipynb: eigen-square-root of S, z = sum of `rounds` Rademacher signs / sqrt(rounds)."""
    w, V = np.linalg.eigh(S)
    L = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
    z = (2 * rng.binomial(rounds, 0.5, size=(n, D)) - rounds) / np.sqrt(rounds)
    return z @ L.T


def quantum_clt_circuit(X, q=Q_CLT):
    """History registers loaded with the samples, pair kick into a wide signed register."""
    params = ["A", "f", "t", "tau", "phi"]
    H = [[QuantumRegister(N_BITS, f"{p}_h{k}") for p in params] for k in range(N)]
    E = [QuantumRegister(E_BITS, f"{p}_eps") for p in params]
    n_pairs = N * (N - 1) // 2
    mv, sg = QuantumRegister(n_pairs, "move"), QuantumRegister(n_pairs, "sign")
    qc = QuantumCircuit(*E, *[r for h in H for r in h], mv, sg)
    U = X - GRID_LO                                         # unsigned register values
    for k in range(N):
        for p in range(D):
            for i in range(N_BITS):
                if (U[k, p] >> i) & 1:
                    qc.x(H[k][p][i])
    scale2 = q * N * (N - 1)
    covariance_kick(qc, [E], H, zip(mv, sg), scale2, periodic=False)
    return qc, E, scale2


def quantum_clt_steps(qc, E, shots, seed, chunk=2**15):
    out = []
    for c0 in range(0, shots, chunk):
        b = sample_circuit(qc, min(chunk, shots - c0), seed=seed + c0)
        out.append(regs_values(b, qc, E, signed=True))
    return np.concatenate(out)


def step_cov(eps, scale2):
    """Empirical step covariance (the mean is exactly 0 by construction) rescaled to estimate S."""
    return eps.T @ eps / len(eps) / scale2


def excess_kurtosis_quantum(X, q):
    """Exact excess kurtosis of each marginal of the quantum CLT step."""
    d = np.array([X[a] - X[b] for a, b in history_pairs(N)], dtype=float)    # (pairs, d)
    return (1 / q - 3) * (d**4).sum(0) / np.maximum((d**2).sum(0) ** 2, 1e-300)


def excess_kurtosis_classical(S, rounds=N_ROUNDS_CLASSICAL):
    w, V = np.linalg.eigh(S)
    L = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
    return (-2 / rounds) * (L**4).sum(1) / np.maximum((L**2).sum(1) ** 2, 1e-300)


# ----------------------------------------------------------------------------- experiments
def main():
    t00 = time.time()
    rng = np.random.default_rng(SEED)
    datasets = [random_dataset(rng) for _ in range(N_DATASETS)]
    datasets = [X for X in datasets if np.all(np.ptp(X, axis=0) > 0)]   # every parameter must vary

    # --- validation: circuits for methods 1-3 vs analytic probabilities -----------------------
    worst = max(validate_circuits(X) for X in datasets[:3])
    print(f"methods 1-3: max |P_circuit - P_analytic| over 3 datasets = {worst:.2e}")
    assert worst < 1e-10
    worst = max(validate_swap(X) for X in datasets[:3])
    print(f"SWAP test: max |P_circuit - P_analytic| over 3 datasets = {worst:.2e}")
    assert worst < 1e-10

    # --- PSD check of the min estimate ---------------------------------------------------------
    min_eigs = np.array([np.linalg.eigvalsh(exact_estimates(X)[1]).min() for X in datasets])
    print(f"min estimate not PSD in {np.mean(min_eigs < -1e-12):.0%} of datasets")

    # --- panel (b): error across datasets at a fixed budget ---------------------------------
    names = ["Max (CS)", "Min (CS)", "Zero-mean", "SWAP test", "CLT classical", "CLT quantum"]
    rng_swap = np.random.default_rng(SEED + 77)
    errs_b = {k: [] for k in names}
    bias = {k: [] for k in names[:3]}
    kurt_q_haario, kurt_q13, kurt_c = [], [], []
    t0 = time.time()
    for i, X in enumerate(datasets):
        S = np.cov(X.T, ddof=1)
        for k, C in zip(names[:3], estimates_from_shots(X, BUDGET_B, rng)):
            errs_b[k].append(rel_err(C, S))
        for k, C in zip(names[:3], exact_estimates(X)):
            bias[k].append(rel_err(C, S))
        errs_b["SWAP test"].append(rel_err(swap_estimate(X, BUDGET_B, rng_swap), S))
        eps_c = classical_clt_steps(S, BUDGET_B, rng)
        errs_b["CLT classical"].append(rel_err(step_cov(eps_c, 1.0), S))
        qc, E, scale2 = quantum_clt_circuit(X)
        eps = quantum_clt_steps(qc, E, BUDGET_B, seed=10_000 * i)
        errs_b["CLT quantum"].append(rel_err(step_cov(eps, scale2), S))
        kurt_q13.append(excess_kurtosis_quantum(X, Q_CLT))
        kurt_q_haario.append(excess_kurtosis_quantum(X, SCALE2_HAARIO / (N * (N - 1))))
        kurt_c.append(excess_kurtosis_classical(S))
        if i == 0:
            # sanity: exact covariance identity and kurtosis formula against circuit samples
            big = quantum_clt_steps(qc, E, 2**18, seed=1)
            kz = (big**4).mean(0) / (big**2).mean(0) ** 2 - 3
            print(f"dataset 0: circuit kurtosis {np.round(kz, 2)} vs formula {np.round(kurt_q13[0], 2)}; "
                  f"rel. err of 2^18-shot covariance {rel_err(step_cov(big, scale2), S):.3f}")
            print(f"quantum CLT circuit: {qc.num_qubits} qubits, {len(qc.data)} gates "
                  f"(per dataset {time.time() - t0:.1f}s)")
    print(f"panel (b) done in {time.time() - t0:.0f}s over {len(datasets)} datasets")

    # --- panel (a): error vs total shots on a representative dataset --------------------------
    med = np.median(bias["Max (CS)"])
    i_rep = int(np.argmin(np.abs(np.array(bias["Max (CS)"]) - med)))
    X = datasets[i_rep]
    S = np.cov(X.T, ddof=1)
    probs = exact_probabilities(X)
    qc, E, scale2 = quantum_clt_circuit(X)
    t0 = time.time()
    pool_q = quantum_clt_steps(qc, E, POOL_A, seed=777)
    pool_c = classical_clt_steps(S, POOL_A, rng)
    print(f"panel (a) pools drawn in {time.time() - t0:.0f}s")
    curves = {k: [] for k in names}
    for B in BUDGETS_A:
        reps = int(min(64, POOL_A // B))
        e = {k: [] for k in names}
        for r in range(64):
            for k, C in zip(names[:3], estimates_from_shots(X, int(B), rng, probs)):
                e[k].append(rel_err(C, S))
            e["SWAP test"].append(rel_err(swap_estimate(X, int(B), rng_swap), S))
        for r in range(reps):                                # disjoint blocks of the circuit pool
            blk = slice(r * B, (r + 1) * B)
            e["CLT quantum"].append(rel_err(step_cov(pool_q[blk], scale2), S))
            e["CLT classical"].append(rel_err(step_cov(pool_c[blk], 1.0), S))
        for k in names:
            curves[k].append(np.percentile(e[k], [16, 50, 84]))
    curves = {k: np.array(v) for k, v in curves.items()}

    DATA.mkdir(exist_ok=True)
    np.savez(DATA / "cov_methods_data.npz", budgets=BUDGETS_A, i_rep=i_rep,
             **{f"a_{k}": v for k, v in curves.items()}, **{f"b_{k}": np.array(v) for k, v in errs_b.items()},
             **{f"bias_{k}": np.array(v) for k, v in bias.items()},
             kurt_q13=np.array(kurt_q13), kurt_q_haario=np.array(kurt_q_haario), kurt_c=np.array(kurt_c))
    for k in names:
        print(f"  {k:14s} panel (b) median rel. error {np.median(errs_b[k]):.3f}"
              + (f"   (infinite-shot bias {np.median(bias[k]):.3f})" if k in bias else ""))

    plot(curves, errs_b, bias, datasets)
    print(f"total {time.time() - t00:.0f}s")


# ----------------------------------------------------------------------------- figure
COLORS = {                          # shared method-family palette
    "Max (CS)": "#777777", "Min (CS)": "#aaaaaa", "Zero-mean": "#d95f02", "Pairwise": "#762a83",
    "SWAP test": "#9970ab", "CLT classical": "#2166ac", "CLT quantum": "#b2182b", "CLT quantum, walk q": "#d6604d",
}
DISPLAY = {"Max (CS)": "Max (Cauchy-Schwarz bound): quantum circuit",
           "Min (CS)": "Min (Cauchy-Schwarz bound): quantum circuit",
           "Zero-mean": "Zero-mean (uncentred): quantum circuit",
           "Pairwise": "Pairwise, centred (unbiased): quantum circuit",
           "SWAP test": "SWAP-test distance estimation: quantum circuit",
           "CLT classical": "CLT walk: classical",
           "CLT quantum": "CLT walk, q = 1/3: quantum circuit",
           "CLT quantum, walk q": "CLT walk, sampler's q = 0.02: quantum circuit"}
TICK = {"Max (CS)": "Max\n(bound)", "Min (CS)": "Min\n(bound)", "Zero-mean": "Zero-\nmean", "Pairwise": "Pair-\nwise",
        "SWAP test": "SWAP\ntest", "CLT classical": "CLT\nclassical", "CLT quantum": "CLT\nq=1/3",
        "CLT quantum, walk q": "CLT\nq=0.02"}
QC_NOTE = ("quantum circuit: simulated on a classical computer (exact statevector for the bound, zero-mean, pairwise "
           "and SWAP-test ancilla probabilities, then binomial shot noise;\nan exact sampler, validated against Aer, for "
           "the CLT walk); no quantum hardware. Every circuit is compiled from the classical data, so S could also be "
           "computed classically:\nthis compares estimators under one access model, with plain repeated shots (no "
           "amplitude estimation); it is not a demonstration of quantum advantage.")
MARKERS = {"Max (CS)": "o", "Min (CS)": "v", "Zero-mean": "s", "Pairwise": "X", "SWAP test": "P", "CLT classical": "D",
           "CLT quantum": "^", "CLT quantum, walk q": "v"}
LINES = {"Max (CS)": "-", "Min (CS)": "-", "Zero-mean": "-", "Pairwise": "-", "SWAP test": "-", "CLT classical": "--",
         "CLT quantum": "-", "CLT quantum, walk q": ":"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot(curves, errs_b, bias, datasets):
    import paper_style
    paper_style.apply()
    OUT.mkdir(exist_ok=True)
    names = list(COLORS)
    # Error versus total shots.
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for k in names:
        lo, mid, hi = curves[k].T
        ax.fill_between(BUDGETS_A, lo, hi, color=COLORS[k], alpha=0.18, lw=0)
        ax.plot(BUDGETS_A, mid, LINES[k], color=COLORS[k], lw=1.5, marker=MARKERS[k], ms=3.5,
                mec="white", mew=0.6, label=TICK[k].replace("\n", " "))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel(r"total shots $B$")
    ax.set_ylabel(r"$\|\hat C - S\|_F \,/\, \|S\|_F$")
    ax.grid(color=paper_style.GRID, lw=.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.22), columnspacing=.8)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(OUT / "cov_methods_shot_error.pdf", bbox_inches="tight")
    plt.close(fig)

    # Error spread over random datasets at the fixed shot budget.
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    data = [np.array(errs_b[k]) for k in names]
    bp = ax.boxplot(data, positions=range(len(names)), widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=MUTED, lw=0.8),
                    capprops=dict(color=MUTED, lw=0.8))
    for patch, k in zip(bp["boxes"], names):
        patch.set(facecolor=COLORS[k], alpha=0.85, edgecolor="white", lw=0.8)
    for x, k in enumerate(names[:3]):
        ax.plot([x - 0.3, x + 0.3], [np.median(bias[k])] * 2, color=INK, lw=0.8, ls=(0, (2, 1.5)))
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([TICK[k].replace("\n", " ") for k in names], rotation=25, ha="right", fontsize=7)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"$\|\hat C - S\|_F \,/\, \|S\|_F$")
    ax.grid(axis="y", color=paper_style.GRID, lw=.6); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "cov_methods_dataset_error.pdf", bbox_inches="tight")
    plt.close(fig)

    # Gaussianity of the CLT step as a function of the lazy-coin probability.
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    qs = np.logspace(np.log10(0.01), 0, 200)
    X_all = datasets
    # excess kurtosis = (1/q - 3) * w,  w = sum d^4 / (sum d^2)^2 per marginal (dataset-dependent)
    w = np.concatenate([excess_kurtosis_quantum(X, 0.25) for X in X_all])   # (1/0.25 - 3) = 1 -> w
    band = np.percentile(np.outer(1 / qs - 3, w), [16, 50, 84], axis=1)
    kq = COLORS["CLT quantum"]
    ax.fill_between(qs, band[0], band[2], color=kq, alpha=0.18, lw=0)
    ax.plot(qs, band[1], color=kq, lw=1.5, label=DISPLAY["CLT quantum"])
    kc = np.concatenate([excess_kurtosis_classical(np.cov(X.T, ddof=1)) for X in X_all])
    ax.axhline(np.median(kc), color=COLORS["CLT classical"], lw=1.5, ls="--", label="Classical CLT")
    ax.axhline(0, color=MUTED, lw=0.6)
    q_h = SCALE2_HAARIO / (N * (N - 1))
    for qv, lab in [(1 / 3, r"$q=1/3$"), (q_h, r"$q$ (walk default)")]:
        y = np.median((1 / qv - 3) * w)
        ax.plot(qv, y, "o", color=kq, ms=4, mec="white", mew=0.6, zorder=3, label=lab)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 1.0)
    ax.set_ylim(-0.8, 5.2)
    ax.set_xlabel(r"kick probability $q$ (quantum)")
    ax.set_ylabel("excess kurtosis of step marginals")
    ax.set_ylabel("Excess kurtosis of step marginals")
    ax.grid(color=paper_style.GRID, lw=.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "cov_methods_clt_kurtosis.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved three standalone covariance-method plots")


def add_swap():
    """Compute only the SWAP-test results and merge them into data/cov_methods_data.npz (same datasets, same
    representative dataset and budgets as the stored results)."""
    rng = np.random.default_rng(SEED)
    datasets = [random_dataset(rng) for _ in range(N_DATASETS)]
    datasets = [X for X in datasets if np.all(np.ptp(X, axis=0) > 0)]
    worst = max(validate_swap(X) for X in datasets[:3])
    print(f"SWAP test: max |P_circuit - P_analytic| over 3 datasets = {worst:.2e}")
    assert worst < 1e-10
    d = dict(np.load(DATA / "cov_methods_data.npz"))
    rng_swap = np.random.default_rng(SEED + 77)
    d["b_SWAP test"] = np.array([rel_err(swap_estimate(X, BUDGET_B, rng_swap), np.cov(X.T, ddof=1)) for X in datasets])
    X = datasets[int(d["i_rep"])]
    S, ex = np.cov(X.T, ddof=1), swap_exact(X)
    d["a_SWAP test"] = np.array([np.percentile([rel_err(swap_estimate(X, int(B), rng_swap, ex), S) for _ in range(64)],
                                               [16, 50, 84]) for B in BUDGETS_A])
    np.savez(DATA / "cov_methods_data.npz", **d)
    print(f"SWAP test: panel (b) median rel. error {np.median(d['b_SWAP test']):.3f}")


def add_extra():
    """Add the centred pairwise estimator and the quantum CLT walk at the sampler's own coin probability
    (q = 2.38^2/d / (N (N-1)), instead of the most Gaussian q = 1/3) to data/cov_methods_data.npz."""
    rng = np.random.default_rng(SEED)
    datasets = [random_dataset(rng) for _ in range(N_DATASETS)]
    datasets = [X for X in datasets if np.all(np.ptp(X, axis=0) > 0)]
    worst = max(validate_pairwise(X) for X in datasets[:3])
    print(f"pairwise: max |P_circuit - P_analytic| over 3 datasets = {worst:.2e}")
    assert worst < 1e-10
    d = dict(np.load(DATA / "cov_methods_data.npz"))
    q_h = SCALE2_HAARIO / (N * (N - 1))
    r = np.random.default_rng(SEED + 99)
    eb_p, eb_q = [], []
    t0 = time.time()
    for i, X in enumerate(datasets):
        S = np.cov(X.T, ddof=1)
        eb_p.append(rel_err(pairwise_estimate(X, BUDGET_B, r), S))
        qc, E, scale2 = quantum_clt_circuit(X, q=q_h)
        eb_q.append(rel_err(step_cov(quantum_clt_steps(qc, E, BUDGET_B, seed=20_000 * i), scale2), S))
    print(f"panel (b) extras in {time.time() - t0:.0f}s")
    d["b_Pairwise"], d["b_CLT quantum, walk q"] = np.array(eb_p), np.array(eb_q)
    X = datasets[int(d["i_rep"])]
    S, ex = np.cov(X.T, ddof=1), pairwise_exact(X)
    qc, E, scale2 = quantum_clt_circuit(X, q=q_h)
    pool = quantum_clt_steps(qc, E, POOL_A, seed=4242)
    a_p, a_q = [], []
    for B in BUDGETS_A:
        a_p.append(np.percentile([rel_err(pairwise_estimate(X, int(B), r, ex), S) for _ in range(64)], [16, 50, 84]))
        reps = int(min(64, POOL_A // B))
        a_q.append(np.percentile([rel_err(step_cov(pool[k * B:(k + 1) * B], scale2), S) for k in range(reps)],
                                 [16, 50, 84]))
    d["a_Pairwise"], d["a_CLT quantum, walk q"] = np.array(a_p), np.array(a_q)
    d["bias_Pairwise"] = np.zeros(len(datasets))
    np.savez(DATA / "cov_methods_data.npz", **d)
    for k in ("Pairwise", "SWAP test", "CLT quantum", "CLT quantum, walk q"):
        print(f"  {k:22s} panel (b) median rel. error {np.median(d['b_' + k]):.4f}")


def qubit_counts(X):
    """(label, qubits in one circuit, circuits per covariance estimate) from the circuits themselves."""
    from qcov import covariance_kick
    import tdrjmcmc_qclt as T
    x = X[:, 0].astype(float)
    A = X.astype(float) - X.mean(0)
    rows = [("Max (CS)", variance_circuit(x)[0].num_qubits, D),
            ("Min (CS)", variance_circuit(x)[0].num_qubits, D),
            ("Zero-mean", moment_circuit(x, x)[0].num_qubits, D * (D + 1) // 2),
            ("Pairwise", pairwise_circuit(x, x)[0].num_qubits, D * (D + 1) // 2),
            ("SWAP test", swap_distance_circuit(A[:, 0], A[:, 1])[0].num_qubits, D * (D - 1) // 2),
            ("CLT quantum", quantum_clt_circuit(X)[0].num_qubits, 1),
            ("CLT classical", 0, 0)]
    for label, bits in (("QCLT-AP kick in the sampler\n(6 stored states, 16-bit registers)", T.NB),
                        ("QCLT + AMR kick in the sampler\n(6 stored states, 6-bit registers)", T.AMR_BITS)):
        H = [[QuantumRegister(bits, f"{p}_h{k}") for p in T.PARAMS] for k in range(T.K_RES)]
        from qclt_amr import out_bits
        E = [QuantumRegister(out_bits(T.K_RES, bits), f"{p}_eps") for p in T.PARAMS]
        npairs = T.K_RES * (T.K_RES - 1) // 2
        mv, sg = QuantumRegister(npairs, "move"), QuantumRegister(npairs, "sign")
        qc = QuantumCircuit(*E, *[r for h in H for r in h], mv, sg)
        covariance_kick(qc, [E], H, zip(mv, sg), 0.5 * T.K_RES * (T.K_RES - 1), periodic=False)
        rows.append((label, qc.num_qubits, 1))
    return rows


def replot():
    """Redraw the figure from data/cov_methods_data.npz without re-running the experiment."""
    rng = np.random.default_rng(SEED)
    datasets = [random_dataset(rng) for _ in range(N_DATASETS)]
    datasets = [X for X in datasets if np.all(np.ptp(X, axis=0) > 0)]
    d = np.load(DATA / "cov_methods_data.npz")
    names = list(COLORS)
    plot({k: d[f"a_{k}"] for k in names}, {k: d[f"b_{k}"] for k in names},
         {k: d[f"bias_{k}"] for k in names[:3]}, datasets)


if __name__ == "__main__":
    import sys
    OUT.mkdir(exist_ok=True)
    if "--add-swap" in sys.argv:
        add_swap()
        replot()
    elif "--add-extra" in sys.argv:
        add_extra()
        replot()
    else:
        replot() if "--plot-only" in sys.argv else main()
