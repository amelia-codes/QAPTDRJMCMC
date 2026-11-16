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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister

from qcov import (add_const, check_coherent, history_pairs, iso_step,
                  kick_probability, lazy_coin, regs_values, sample_circuit, torus_cov)


ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "quantum_walk_archive"
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
    target = ideal_covariance()
    for name, r in runs.items():
        r["analytic"] = exact_unconditional_covariance(r["reg"])
        r["error"] = np.linalg.norm(r["analytic"] - target) / np.linalg.norm(target)
        print(f"{name:7s}: {r['operational_qubits']:3d} operational qubits "
              f"({r['qubits']} with diagnostic tracker), lattice modes {r['reg']['modes'].tolist()}, "
              f"ideal-cov error {r['error']:.3f}, circuit-vs-prediction {r['mismatch']:.3f}", flush=True)

    observed, expected, counts = branch_summary(branch)
    budgets = np.unique(np.geomspace(200, shots, 16).astype(int))
    err_curve, err_low, err_high = [], [], []
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
        lo, mid, hi = np.quantile(errors, [0.1, 0.5, 0.9])
        err_low.append(lo)
        err_curve.append(mid)
        err_high.append(hi)

    np.savez_compressed(DATA / "results.npz", observed=observed, expected=expected,
        counts=counts, budgets=budgets, err_curve=np.array(err_curve),
        err_low=np.array(err_low), err_high=np.array(err_high),
        pilot_hist=branch["physical_hist"][:500], target=target,
        amr_center=amr_center, amr_axes=amr_axes, amr_spacing=amr_spacing,
        multi_steps=np.arange(1, 5),
        multi_qubits=np.array([r["operational_qubits"] for r in multi]),
        multi_empirical=np.stack([r["empirical"] for r in multi]),
        multi_predicted=np.stack([r["predicted"] for r in multi]),
        **{f"{name}_{key}": r[key] for name, r in runs.items()
           for key in ("qubits", "operational_qubits", "error", "analytic",
                       "empirical", "predicted", "mismatch")})
    plot()


def plot():
    z = np.load(DATA / "results.npz")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 11,
                         "axes.labelsize": 10})

    def branch_panel(ax, title):
        for j, (label, color) in enumerate(zip(
                ("x variance", "y variance", "x–y covariance"),
                ("#364b8f", "#c45a42", "#176a6d"))):
            ax.scatter(z["expected"][:, j], z["observed"][:, j], s=17,
                       alpha=0.68, color=color, label=label)
        values = np.r_[z["expected"].ravel(), z["observed"].ravel()]
        lo, hi = values.min(), values.max()
        pad = max((hi - lo) * 0.07, 0.8)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color="#444444", ls="--", lw=1, label="exact agreement")
        ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad),
               xlabel="Predicted next-step covariance (grid units²)",
               ylabel="Covariance of sampled steps (grid units²)", title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.16)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    def sampling_panel(ax, title):
        ax.fill_between(z["budgets"], 100 * z["err_low"], 100 * z["err_high"],
                        color="#364b8f", alpha=0.16, label="10th–90th percentile")
        ax.loglog(z["budgets"], 100 * z["err_curve"], "o-",
                  color="#364b8f", ms=3.5, label="median")
        guide = z["err_curve"][0] * np.sqrt(z["budgets"][0] / z["budgets"])
        ax.loglog(z["budgets"], 100 * guide, ":", color="#777777",
                  label="1/√samples guide")
        ax.set(xlabel="Samples from the same circuit", ylabel="Validation error (%)",
               title=title)
        ax.grid(alpha=0.16, which="major")
        ax.legend(frameon=False, fontsize=8)

    def window_panel(ax, title):
        h = z["pilot_hist"].reshape(-1, D)
        ax.scatter(h[:, 0], h[:, 1], s=8, alpha=0.17, color="#364b8f",
                   label="pilot histories")
        pilot = window_corners(X0, np.eye(D), np.full(D, 0.08 / 64), 6)
        amr = window_corners(z["amr_center"], z["amr_axes"], z["amr_spacing"], 6)
        ax.plot(pilot[:, 0], pilot[:, 1], color=COL["pilot6"], lw=2,
                label="initial window")
        ax.plot(amr[:, 0], amr[:, 1], color=COL["amr12"], lw=2,
                label="window chosen from pilot")
        ax.scatter(*X0, color="#222222", marker="x", s=42, lw=1.5,
                   label="start")
        ax.set(xlabel="physical x", ylabel="physical y", title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.16)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    def accuracy_panel(ax, title):
        order = ("fixed6", "pilot6", "amr6", "amr8", "amr12", "fixed16")
        labels = ("Full 6b", "Pilot 6b", "AMR 6b", "AMR 8b", "AMR 12b", "Full 16b")
        for i, name in enumerate(order):
            percent = 100 * float(z[f"{name}_error"])
            ax.bar(i, percent, width=0.68, color=COL[name], zorder=2)
            ax.text(i, percent * 1.2, f"{percent:.3g}%", ha="center",
                    va="bottom", fontsize=8)
        qlabels = [f"{label}\n{int(z[f'{name}_operational_qubits'])}q"
                   for label, name in zip(labels, order)]
        ax.set_xticks(range(len(order)), qlabels, fontsize=8)
        ax.set(yscale="log", ylim=(0.09, 350), ylabel="Next-step covariance error (%)",
               title=title)
        ax.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        ax.axhline(1, color="#777777", ls=":", lw=1, zorder=1)
        ax.grid(axis="y", alpha=0.16, which="major")

    def multistep_panel(ax, extract, ylabel, title, legend=True):
        n = z["multi_steps"]
        measured = np.array([extract(c) for c in z["multi_empirical"]]) * 1e6
        predicted = np.array([extract(c) for c in z["multi_predicted"]]) * 1e6
        ideal = n * extract(z["target"]) * 1e6
        ax.plot(n, ideal, "--", color="#555555", lw=1.2,
                label="continuous-grid target")
        ax.plot(n, predicted, color=COL["amr12"], lw=1.7,
                label="QCLT prediction")
        ax.scatter(n, measured, color="#364b8f", s=42, zorder=3,
                   label="circuit samples")
        ax.set_xticks(n, [f"{step}\n{q}q" for step, q in zip(n, z["multi_qubits"])])
        ax.set(xlabel="QCLT kicks in one coherent circuit", ylabel=ylabel,
               title=title)
        ax.grid(alpha=0.16)
        if legend:
            ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    branch_panel(axes[0], "Each history gives the intended next-step covariance")
    sampling_panel(axes[1], "Sampling validates the circuit output")
    fig.suptitle("QCLT: coherent, history-shaped next steps · 29 operational qubits",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.2)
    fig.savefig(OUT / "qclt_coherent_covariance.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    window_panel(axes[0], "Pilot measurements set the AMR window")
    accuracy_panel(axes[1], "AMR preserves accuracy with fewer qubits")
    fig.suptitle("QCLT+AMR: smaller grid spacing where the walk moves",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.0)
    fig.savefig(OUT / "qclt_amr_grid.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    multistep_panel(axes[0], np.trace, "Total step variance (×10⁻⁶)",
                    "Motion grows as predicted")
    multistep_panel(axes[1], lambda a: a[0, 1],
                    "x–y step covariance (×10⁻⁶)", "Correlation persists", False)
    fig.suptitle("Repeated coherent QCLT kicks · measured 8-bit AMR window",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.3)
    fig.savefig(OUT / "qclt_multistep_walk.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.3))
    branch_panel(axes[0, 0], "1. History shapes the next step")
    window_panel(axes[0, 1], "2. Pilot data select the window")
    accuracy_panel(axes[1, 0], "3. AMR saves qubits at the same accuracy")
    multistep_panel(axes[1, 1], np.trace, "Total step variance (×10⁻⁶)",
                    "4. Repeated coherent steps diffuse")
    fig.suptitle("QCLT + AMR: coherent next steps, compact grids", fontsize=14, y=0.99)
    fig.text(0.5, 0.012,
             "Circuits output steps; covariance is computed afterward only for validation.\n"
             "Ideal circuits sampled classically · AMR updated between circuits · q = logical qubits",
             ha="center", fontsize=8.2, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), h_pad=2.1, w_pad=2.4)
    fig.savefig(OUT / "qclt_amr_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved four archived quantum-walk diagnostic PDFs in", OUT, flush=True)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot()
    else:
        generate()
