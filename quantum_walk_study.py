"""The QCLT walk, from circuit construction to its Gaussian-convergence plot.

Each independent pair coin contributes r_ab (h_a-h_b). Choosing
q = SCALE2 / (K*(K-1)) gives conditional covariance SCALE2*S_history.
Repeated steps keep q fixed and use fresh coins. The convergence experiment
whitens their sum by its conditional covariance before scoring Gaussianity.
"""

import sys
import itertools
from statistics import NormalDist
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister

from qcov import (add_const, check_coherent, iso_step, lazy_coin,
                  regs_values, sample_circuit, torus_cov)


ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "quovar"
DATA = ROOT / "data" / "quantum_walk"
D = 2                         # Coordinates
K = 3                         # Preparatory history positions
SCALE2 = 2.38**2 / D           # Proposal covariance multiplier
ISO_VAR = .1                  # Independent variance in lattice units
X0 = np.array([0.69, 0.37])
# Rows are two independent physical displacement modes. The first couples x and y.
MODES = np.array([[0.0020, 0.0012], [0.0007, 0.0015]])


def history_branches(modes=MODES):
    """Enumerate the 64 sign histories and their unrounded displacements."""
    for bits in itertools.product((0, 1), repeat=K * D):
        signs = 1 - 2 * np.array(bits).reshape(K, D)
        yield signs, np.cumsum(signs @ modes, axis=0)


def grid_coordinates(x, center, axes, spacing, bits):
    """Physical x -> integer coordinates of a grid centred at 2^(bits-1)."""
    return np.rint((np.asarray(x) - center) @ axes / spacing + (1 << (bits - 1))).astype(int)


def make_walk(bits, center, axes, spacing, e_bits, track_displacement=True,
              adaptive_steps=1, modes_physical=None):
    """K coherent preparatory steps, then adaptive_steps history-covariance kicks.

    A preparatory coin string determines all K history positions exactly, so
    pair differences are computed directly from those still-coherent coins.
    Optional wide displacement registers make the adaptive kick readable.
    """
    center, axes, spacing = np.asarray(center), np.asarray(axes), np.asarray(spacing)
    modes_physical = MODES if modes_physical is None else np.asarray(modes_physical)
    n = 1 << bits
    initial = grid_coordinates(X0, center, axes, spacing, bits)
    modes = np.rint(modes_physical @ axes / spacing).astype(int)
    if np.any((initial < 0) | (initial >= n)):
        raise ValueError("initial point lies outside the grid window")
    max_step = np.abs(modes).sum(axis=0)
    if adaptive_steps < 1:
        raise ValueError("adaptive_steps must be positive")
    if np.any(max_step * K >= n // 2):
        raise ValueError("a history pair could cross half the periodic grid")
    max_kick = adaptive_steps * ((K * (K - 1) // 2) * (K - 1) * max_step + 1)
    if track_displacement and np.any(max_kick >= 1 << (e_bits - 1)):
        raise ValueError("signed diagnostic displacement register could overflow")

    P = [QuantumRegister(bits, f"position_{p}") for p in range(D)]
    disp = ([QuantumRegister(e_bits, f"displacement_{p}") for p in range(D)]
            if track_displacement else [])
    parity = AncillaRegister(1, "pair_sign_parity")
    prep = AncillaRegister(K * D, "preparation_signs")
    pairs = K * (K - 1) // 2
    move = AncillaRegister(pairs * adaptive_steps, "pair_move")
    sign = AncillaRegister(pairs * adaptive_steps, "pair_sign")
    iso_move = AncillaRegister(D * adaptive_steps, "iso_move")
    iso_sign = AncillaRegister(D * adaptive_steps, "iso_sign")
    qc = QuantumCircuit(*P, *disp, prep, move, sign, iso_move, iso_sign, parity)
    for p, val in enumerate(initial):
        for bit in range(bits):
            if (int(val) >> bit) & 1:
                qc.x(P[p][bit])

    # Prepare the history with fresh independent sign coins.
    for t in range(K):
        for mode in range(D):
            coin = prep[t * D + mode]
            qc.h(coin)
            for p, velocity in enumerate(modes[mode]):
                if velocity:
                    add_const(qc, P[p], int(velocity))
                    add_const(qc, P[p], -2 * int(velocity), [coin])
    # Accumulate the signed pair differences without storing snapshots.
    targets = [P, disp] if track_displacement else [P]
    q = SCALE2 / (K * (K - 1))
    for adaptive in range(adaptive_steps):
        for j, (a, b) in enumerate(itertools.combinations(range(K), 2)):
            pair_index = adaptive * pairs + j
            lazy_coin(qc, q, move[pair_index], sign[pair_index])
            # h_a - h_b = -sum_{t=a+1}^b sum_m (1-2*prep[t,m]) v_m.
            # XOR with the pair sign reverses this difference coherently.
            for t in range(a + 1, b + 1):
                for m in range(D):
                    qc.cx(prep[t * D + m], parity[0])
                    qc.cx(sign[pair_index], parity[0])
                    for regs in targets:
                        for p, velocity in enumerate(modes[m]):
                            if velocity:
                                add_const(qc, regs[p], -int(velocity), [move[pair_index]])
                                add_const(qc, regs[p], +2 * int(velocity),
                                          [move[pair_index], parity[0]])
                    qc.cx(sign[pair_index], parity[0])
                    qc.cx(prep[t * D + m], parity[0])
        # Add the small independent lattice regularizer.
        for p in range(D):
            lazy_coin(qc, ISO_VAR, iso_move[adaptive * D + p],
                      iso_sign[adaptive * D + p])
        iso_step(qc, targets, [1] * D,
                 iso_sign[adaptive * D:(adaptive + 1) * D],
                 iso_move[adaptive * D:(adaptive + 1) * D])
    assert check_coherent(qc) == 0
    return qc, dict(P=P, prep=prep, disp=disp, center=center, axes=axes,
                    spacing=spacing, modes=modes, initial=initial, bits=bits,
                    adaptive_steps=adaptive_steps)


def histories_from_coins(raw, qc, reg):
    """Reconstruct the basis histories for diagnostics after final measurement."""
    coins = np.stack([raw[qc.find_bit(q).index] for q in reg["prep"]], axis=1)
    coins = coins.reshape(-1, K, D).astype(int)
    increments = np.einsum("ntm,mp->ntp", 1 - 2 * coins, reg["modes"])
    return (reg["initial"] + np.cumsum(increments, axis=1)) % (1 << reg["bits"])


def sample_walk(bits, center, axes, spacing, e_bits, shots, seed,
                adaptive_steps=1, modes_physical=None):
    qc, reg = make_walk(bits, center, axes, spacing, e_bits,
                        adaptive_steps=adaptive_steps, modes_physical=modes_physical)
    raw = sample_circuit(qc, shots, seed=seed)
    hist = histories_from_coins(raw, qc, reg)
    disp = regs_values(raw, qc, reg["disp"], signed=True)
    measured_position = regs_values(raw, qc, reg["P"])
    if not np.array_equal(measured_position, (hist[:, -1] + disp) % (1 << bits)):
        raise AssertionError("measured position disagrees with history plus kick")
    basis = reg["axes"] @ np.diag(reg["spacing"])
    pred_lattice = adaptive_steps * (SCALE2 * torus_cov(hist, bits) + ISO_VAR * np.eye(D))
    pred = np.einsum("ij,njk,lk->nil", basis, pred_lattice, basis)
    physical_disp = (disp * reg["spacing"]) @ reg["axes"].T
    physical_hist = (hist - (1 << (bits - 1))) * reg["spacing"] @ reg["axes"].T + reg["center"]
    empirical = physical_disp.T @ physical_disp / shots
    predicted = pred.mean(axis=0)
    # A guard for accidental signed overflow or an incorrect coordinate map.
    mismatch = np.linalg.norm(empirical - predicted) / np.linalg.norm(predicted)
    if mismatch > 0.08:
        raise AssertionError(f"circuit covariance differs from its branch prediction: {mismatch:.3f}")
    return dict(qubits=qc.num_qubits, operational_qubits=qc.num_qubits - D * e_bits,
                circuit=qc, reg=reg, hist=hist, disp=disp,
                physical_hist=physical_hist, physical_disp=physical_disp,
                pred=pred, empirical=empirical, predicted=predicted, mismatch=mismatch,
                adaptive_steps=adaptive_steps)


def ideal_covariance(modes_physical=None):
    """Continuous-grid covariance of one adaptive kick after K random-walk snapshots."""
    modes_physical = MODES if modes_physical is None else np.asarray(modes_physical)
    return SCALE2 * ((K + 1) / 6) * (modes_physical.T @ modes_physical)


def conditional_standardized_steps(run):
    """Whiten each measured walk using its history-conditional covariance."""
    histories = run["hist"].reshape(len(run["hist"]), -1)
    _, inverse = np.unique(histories, axis=0, return_inverse=True)
    standardized = np.empty_like(run["physical_disp"], dtype=float)
    for branch in np.unique(inverse):
        indices = np.flatnonzero(inverse == branch)
        covariance = run["pred"][indices[0]]
        eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
        if np.any(eigenvalues <= 0):
            raise ValueError("A history-conditional step covariance is not positive definite")
        whitening = eigenvectors / np.sqrt(eigenvalues)
        standardized[indices] = run["physical_disp"][indices] @ whitening
    return standardized


def sliced_wasserstein_bootstrap(samples, seed, n_bootstrap=48, sample_size=8192,
                                 n_directions=48):
    """Return individual bootstrap estimates of sliced W1 distance to N(0,I).

    The Gaussian reference quantiles are evaluated deterministically, avoiding
    a second finite sample from adding noise to the distance estimate.
    """
    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, len(samples))
    angles = np.linspace(0, np.pi, n_directions, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    normal = NormalDist()
    gaussian_quantiles = np.fromiter(
        (normal.inv_cdf((i + 0.5) / sample_size) for i in range(sample_size)),
        dtype=float, count=sample_size)
    errors = np.empty(n_bootstrap)
    for replicate in range(n_bootstrap):
        observed = samples[rng.integers(len(samples), size=sample_size)]
        observed_projection = np.sort(observed @ directions.T, axis=0)
        errors[replicate] = np.abs(
            observed_projection - gaussian_quantiles[:, None]).mean(axis=0).mean()
    return errors


def plot_clt_convergence():
    """Plot Gaussian convergence of standardized coherent QCLT step sums."""
    import paper_style
    paper_style.apply()
    with np.load(DATA / "results.npz") as results:
        steps = results["multi_steps"]
        errors = results["clt_wasserstein_samples"]
    mean = errors.mean(axis=1)
    sd = errors.std(axis=1, ddof=1)
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    ax.errorbar(steps, mean, yerr=sd,
                color=paper_style.COLORS["QCLT_AMR"], marker="o", ms=5,
                capsize=3, lw=1.6,
                label=r"QCLT walk (mean $\pm$ 1 bootstrap SD)")
    guide = mean[0] * np.sqrt(steps[0] / steps)
    ax.plot(steps, guide, ls="--", color=paper_style.MUTED, lw=1.0,
            label=r"$N^{-1/2}$ reference")
    ax.set_xticks(steps)
    ax.set_xlabel("Number of QCLT steps, $N$")
    ax.set_ylabel("Sliced Wasserstein distance to Gaussian")
    paper_style.legend_below(ax)
    ax.set_axisbelow(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "qclt_clt_convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def generate(shots=100_000):
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    eye = np.eye(D)
    branch = sample_walk(6, X0, eye, np.full(D, 0.08 / 64), 8, shots, 11)
    # The pilot itself is a coherent circuit. Only its final measurements are
    # used to select the next circuit's centre, axes, and spacings.
    pooled = branch["physical_hist"].reshape(-1, D)
    amr_center = pooled.mean(axis=0)
    lam, amr_axes = np.linalg.eigh(np.cov(pooled.T))
    amr_spacing = np.maximum(np.sqrt(np.maximum(lam, 0)) / 4, 1e-6)
    amr8 = sample_walk(8, amr_center, amr_axes, amr_spacing / 4, 14, shots, 15)
    # A full coherent circuit contains all repeated kicks and fresh coins.
    # Its covariance reference is the same measured pilot history, held fixed.
    multi = [amr8] + [sample_walk(8, amr_center, amr_axes, amr_spacing / 4,
                                   14, shots, 20 + steps, adaptive_steps=steps)
                      for steps in (2, 3, 4)]
    clt_wasserstein_samples = np.stack([
        sliced_wasserstein_bootstrap(conditional_standardized_steps(r), 20261116 + i)
        for i, r in enumerate(multi)
    ])
    np.savez_compressed(
        DATA / "results.npz", multi_steps=np.arange(1, 5),
        multi_qubits=np.array([r["operational_qubits"] for r in multi]),
        multi_empirical=np.stack([r["empirical"] for r in multi]),
        multi_predicted=np.stack([r["predicted"] for r in multi]),
        clt_wasserstein_samples=clt_wasserstein_samples)
    plot_clt_convergence()


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot_clt_convergence()
    else:
        generate()
