"""Focused, measurement-free QCLT walk and between-circuit AMR experiment.

Run with a Qiskit + Matplotlib environment:
    python quantum_walk_study.py             # sample the circuits and save data + PDFs
    python quantum_walk_study.py --plot-only # redraw PDFs from saved summaries

Every walk segment is a QuantumCircuit with coherent history encoded by its
preparatory coins, fresh kick coins, and no intermediate measurement or reset.
`qcov.sample_circuit` samples its exact
computational-basis distribution because these circuits contain no interfering
branches. AMR window selection uses measurements from an earlier circuit and
is classical between circuits. There is no likelihood, acceptance rule, or
claim of a quantum speedup.
"""

import sys
from statistics import NormalDist
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister

from qcov import (add_const, check_coherent, history_pairs, iso_step,
                  kick_probability, lazy_coin, regs_values, sample_circuit, torus_cov)


ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "quantum_walk"
DATA = ROOT / "data" / "quantum_walk"
K, D, SCALE2, ISO_VAR = 3, 2, 2.38**2 / 2, 0.1
X0 = np.array([0.69, 0.37])
# Rows are two independent physical displacement modes. The first couples x and y.
MODES = np.array([[0.0020, 0.0012], [0.0007, 0.0015]])
COL = {"fixed6": "#777777", "pilot6": "#d19a24", "amr6": "#78b6a5",
       "amr8": "#3f9589", "amr12": "#176a6d", "fixed16": "#364b8f"}


def grid_coordinates(x, center, axes, spacing, bits):
    """Physical x -> integer coordinates of a grid centred at 2^(bits-1)."""
    return np.rint((np.asarray(x) - center) @ axes / spacing + (1 << (bits - 1))).astype(int)


def make_walk(bits, center, axes, spacing, e_bits, track_displacement=True,
              adaptive_steps=1, modes_physical=None, initial_physical=None):
    """K coherent preparatory steps, then adaptive_steps history-covariance kicks.

    A preparatory coin string determines all K history positions exactly, so
    pair differences are computed directly from those still-coherent coins.
    Optional wide displacement registers make the adaptive kick readable.
    """
    center, axes, spacing = np.asarray(center), np.asarray(axes), np.asarray(spacing)
    modes_physical = MODES if modes_physical is None else np.asarray(modes_physical)
    initial_physical = X0 if initial_physical is None else np.asarray(initial_physical)
    n = 1 << bits
    initial = grid_coordinates(initial_physical, center, axes, spacing, bits)
    modes = np.rint(modes_physical @ axes / spacing).astype(int)
    if np.any((initial < 0) | (initial >= n)):
        raise ValueError("initial point lies outside the grid window")
    max_step = np.abs(modes).sum(axis=0)
    if adaptive_steps < 1:
        raise ValueError("adaptive_steps must be positive")
    if np.any(max_step * K >= n // 2):
        raise ValueError("a history pair could cross half the periodic grid")
    max_kick = adaptive_steps * (len(history_pairs(K)) * (K - 1) * max_step + 1)
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

    for t in range(K):
        for mode in range(D):
            coin = prep[t * D + mode]
            qc.h(coin)
            for p, velocity in enumerate(modes[mode]):
                if velocity:
                    add_const(qc, P[p], int(velocity))
                    add_const(qc, P[p], -2 * int(velocity), [coin])
    targets = [P, disp] if track_displacement else [P]
    q = kick_probability(K, SCALE2)
    for adaptive in range(adaptive_steps):
        for j, (a, b) in enumerate(history_pairs(K)):
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
        for p in range(D):
            lazy_coin(qc, ISO_VAR, iso_move[adaptive * D + p],
                      iso_sign[adaptive * D + p])
        iso_step(qc, targets, [1] * D,
                 iso_sign[adaptive * D:(adaptive + 1) * D],
                 iso_move[adaptive * D:(adaptive + 1) * D])
    assert check_coherent(qc) == 0
    return qc, dict(P=P, prep=prep, disp=disp, center=center, axes=axes,
                    spacing=spacing, modes=modes, initial=initial, bits=bits,
                    adaptive_steps=adaptive_steps,
                    modes_physical=modes_physical, initial_physical=initial_physical)


def histories_from_coins(raw, qc, reg):
    """Reconstruct the basis histories for diagnostics after final measurement."""
    coins = np.stack([raw[qc.find_bit(q).index] for q in reg["prep"]], axis=1)
    coins = coins.reshape(-1, K, D).astype(int)
    increments = np.einsum("ntm,mp->ntp", 1 - 2 * coins, reg["modes"])
    return (reg["initial"] + np.cumsum(increments, axis=1)) % (1 << reg["bits"])


def sample_walk(bits, center, axes, spacing, e_bits, shots, seed,
                adaptive_steps=1, modes_physical=None, initial_physical=None):
    qc, reg = make_walk(bits, center, axes, spacing, e_bits,
                        adaptive_steps=adaptive_steps, modes_physical=modes_physical,
                        initial_physical=initial_physical)
    operational, op_reg = make_walk(bits, center, axes, spacing, e_bits,
                                    track_displacement=False,
                                    adaptive_steps=adaptive_steps,
                                    modes_physical=modes_physical,
                                    initial_physical=initial_physical)
    # The wide displacement tracker is only a readout aid. With identical
    # random coins, removing it must leave the physical walk unchanged.
    tracked_check = sample_circuit(qc, 128, seed=seed + 1000)
    bare_check = sample_circuit(operational, 128, seed=seed + 1000)
    if not np.array_equal(regs_values(tracked_check, qc, reg["P"]),
                          regs_values(bare_check, operational, op_reg["P"])):
        raise AssertionError("diagnostic tracker changed the quantum walk")
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
    return dict(qubits=qc.num_qubits, operational_qubits=operational.num_qubits,
                circuit=qc, reg=reg, hist=hist, disp=disp,
                physical_hist=physical_hist, physical_disp=physical_disp,
                pred=pred, empirical=empirical, predicted=predicted, mismatch=mismatch,
                adaptive_steps=adaptive_steps)


def branch_summary(run):
    """Conditional second moments for each observed quantum-history branch."""
    h = run["hist"].reshape(len(run["hist"]), -1)
    _, inv = np.unique(h, axis=0, return_inverse=True)
    count = np.bincount(inv)
    d = run["disp"]
    pred = SCALE2 * torus_cov(run["hist"], run["reg"]["bits"]) + ISO_VAR * np.eye(D)
    terms = np.stack((d[:, 0] ** 2, d[:, 1] ** 2, d[:, 0] * d[:, 1]), axis=1)
    target = np.stack((pred[:, 0, 0], pred[:, 1, 1], pred[:, 0, 1]), axis=1)
    obs = np.zeros((len(count), 3))
    expect = np.zeros_like(obs)
    np.add.at(obs, inv, terms)
    np.add.at(expect, inv, target)
    obs /= count[:, None]
    expect /= count[:, None]
    keep = count >= 100
    return obs[keep], expect[keep], count[keep]


def ideal_covariance(modes_physical=None):
    """Continuous-grid covariance of one adaptive kick after K random-walk snapshots."""
    modes_physical = MODES if modes_physical is None else np.asarray(modes_physical)
    return SCALE2 * ((K + 1) / 6) * (modes_physical.T @ modes_physical)


def exact_unconditional_covariance(reg):
    """Average physical step covariance over every preparatory coin branch."""
    basis = reg["axes"] @ np.diag(reg["spacing"])
    modes = reg["modes"]
    lattice = SCALE2 * ((K + 1) / 6) * (modes.T @ modes) + ISO_VAR * np.eye(D)
    return reg["adaptive_steps"] * basis @ lattice @ basis.T


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


def plot_clt_convergence(results=None):
    """Plot Gaussian convergence of standardized coherent QCLT step sums."""
    if results is None:
        results = np.load(DATA / "results.npz")
    if "clt_wasserstein_samples" not in results:
        raise RuntimeError("Individual convergence errors are missing; regenerate results with quantum_walk_study.py")
    import paper_style
    paper_style.apply()
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


def window_corners(center, axes, spacing, bits):
    half = (1 << bits) / 2
    local = np.array([[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]])
    return center + (local * spacing) @ axes.T


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
    fixed6 = sample_walk(6, np.full(D, 0.5), eye, np.full(D, 1 / 64), 8, shots, 12)
    amr6 = sample_walk(6, amr_center, amr_axes, amr_spacing, 8, shots, 13)
    amr8 = sample_walk(8, amr_center, amr_axes, amr_spacing / 4, 14, shots, 15)
    amr12 = sample_walk(12, amr_center, amr_axes, amr_spacing / 64, 18, shots, 16)
    fixed16 = sample_walk(16, np.full(D, 0.5), eye, np.full(D, 1 / 65536), 20, shots, 14)
    runs = {"fixed6": fixed6, "pilot6": branch, "amr6": amr6,
            "amr8": amr8, "amr12": amr12, "fixed16": fixed16}
    # A full coherent circuit contains all repeated kicks and fresh coins.
    # Its covariance reference is the same measured pilot history, held fixed.
    multi = [amr8] + [sample_walk(8, amr_center, amr_axes, amr_spacing / 4,
                                   14, shots, 20 + steps, adaptive_steps=steps)
                      for steps in (2, 3, 4)]
    clt_wasserstein_samples = np.stack([
        sliced_wasserstein_bootstrap(conditional_standardized_steps(r), 20261116 + i)
        for i, r in enumerate(multi)
    ])
    target = ideal_covariance()
    for name, r in runs.items():
        r["analytic"] = exact_unconditional_covariance(r["reg"])
        r["error"] = np.linalg.norm(r["analytic"] - target) / np.linalg.norm(target)
        print(f"{name:7s}: {r['operational_qubits']:3d} operational qubits "
              f"({r['qubits']} with diagnostic tracker), lattice modes {r['reg']['modes'].tolist()}, "
              f"ideal-cov error {r['error']:.3f}, circuit-vs-prediction {r['mismatch']:.3f}", flush=True)

    observed, expected, counts = branch_summary(branch)
    budgets = np.unique(np.geomspace(200, shots, 16).astype(int))
    covariance_errors = []
    lattice_pred = SCALE2 * torus_cov(branch["hist"], 6) + ISO_VAR * np.eye(D)
    d = branch["disp"]
    residual = np.stack((d[:, 0] ** 2 - lattice_pred[:, 0, 0],
                         d[:, 1] ** 2 - lattice_pred[:, 1, 1],
                         d[:, 0] * d[:, 1] - lattice_pred[:, 0, 1]), axis=1)
    normalizer = np.linalg.norm(lattice_pred.mean(axis=0))
    bootstrap = np.random.default_rng(688)
    for n in budgets:
        errors = []
        for _ in range(32):
            mean = residual[bootstrap.integers(shots, size=n)].mean(axis=0)
            errors.append(np.sqrt(mean[0] ** 2 + mean[1] ** 2 + 2 * mean[2] ** 2)
                          / normalizer)
        covariance_errors.append(errors)
    covariance_errors = np.array(covariance_errors)

    np.savez_compressed(DATA / "results.npz", observed=observed, expected=expected,
        counts=counts, budgets=budgets, covariance_errors=covariance_errors,
        err_curve=covariance_errors.mean(axis=1),
        err_std=covariance_errors.std(axis=1, ddof=1),
        pilot_hist=branch["physical_hist"][:500], target=target,
        amr_center=amr_center, amr_axes=amr_axes, amr_spacing=amr_spacing,
        multi_steps=np.arange(1, 5),
        multi_qubits=np.array([r["operational_qubits"] for r in multi]),
        multi_empirical=np.stack([r["empirical"] for r in multi]),
        multi_predicted=np.stack([r["predicted"] for r in multi]),
        clt_wasserstein_samples=clt_wasserstein_samples,
        **{f"{name}_{key}": r[key] for name, r in runs.items()
           for key in ("qubits", "operational_qubits", "error", "analytic",
                       "empirical", "predicted", "mismatch")})
    plot()


def plot():
    """Render the convergence figure used in the paper."""
    plot_clt_convergence()


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot()
    else:
        generate()
