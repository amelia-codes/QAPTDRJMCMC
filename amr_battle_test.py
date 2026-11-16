"""Stress-test the next-step QCLT+AMR circuit across walk shapes and pilot sizes.

The circuit itself outputs a random step. Covariances below are computed only
after the circuit, to assess whether the step law matches the intended target.
"""

from pathlib import Path
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Ellipse

ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "quantum_walk"
DATA = ROOT / "data" / "quantum_walk"
BITS = 8
PILOT_BITS = 6
PILOT_POOL = 20_000
BUDGETS = np.array([8, 16, 32, 64, 128, 512, 1000])
REPS = 80
D, K, SCALE2, ISO_VAR, X0, MODES = 2, 3, 2.38**2 / 2, 0.1, np.array([0.69, 0.37]), np.array([[0.0020, 0.0012], [0.0007, 0.0015]])

# Different widths and signs test anisotropy, alignment, and scale.
BASE_CASES = [
    ("tilted", MODES, 0.08),
    ("axis-aligned", np.diag([0.0020, 0.0015]), 0.08),
    ("anti-tilted", MODES * np.array([1, -1]), 0.08),
    ("isotropic", np.diag([0.0015, 0.0015]), 0.08),
    ("broader", 4 * MODES, 0.32),
]
CASES = BASE_CASES + [
    # Here the fixed local grid already represents the modes nearly exactly.
    # PCA rotation/re-spacing can make AMR less accurate at the same width.
    ("grid-aligned", np.diag([0.0025, 0.00125]), 0.08),
    ("isotropic-exact", np.diag([0.0025, 0.0025]), 0.08),
]


def exact_step_covariance(modes_physical, center, axes, spacing):
    """Analytic output covariance of this QCLT circuit after grid rounding."""
    lattice_modes = np.rint(modes_physical @ axes / spacing).astype(int)
    basis = axes @ np.diag(spacing)
    lattice = SCALE2 * ((K + 1) / 6) * (lattice_modes.T @ lattice_modes)
    lattice += ISO_VAR * np.eye(D)
    return basis @ lattice @ basis.T


def relative_error(modes_physical, center, axes, spacing):
    predicted = exact_step_covariance(modes_physical, center, axes, spacing)
    target = ideal_covariance(modes_physical)
    return np.linalg.norm(predicted - target) / np.linalg.norm(target)


def all_true_histories(modes):
    coins = np.array(list(itertools.product((0, 1), repeat=K * D))).reshape(-1, K, D)
    increments = np.einsum("ntm,mp->ntp", 1 - 2 * coins, modes)
    return X0 + np.cumsum(increments, axis=1)


def window_safe(modes, center, axes, spacing, histories):
    """Require all possible preparatory histories inside the physical window."""
    n = 1 << BITS
    initial = (X0 - center) @ axes / spacing + n / 2
    lattice_modes = np.rint(modes @ axes / spacing).astype(int)
    pair_safe = np.all(K * np.abs(lattice_modes).sum(axis=0) < n / 2)
    coords = (histories - center) @ axes / spacing + n / 2
    covered = np.all((coords >= 0) & (coords < n), axis=(1, 2))
    return bool(np.all((initial >= 0) & (initial < n)) and pair_safe
                and covered.all())


def window_from_pilot(histories):
    pooled = histories.reshape(-1, D)
    center = pooled.mean(axis=0)
    lam, axes = np.linalg.eigh(np.cov(pooled.T))
    spacing_6 = np.maximum(np.sqrt(np.maximum(lam, 0)) / 4, 1e-6)
    return center, axes, spacing_6 / (1 << (BITS - PILOT_BITS))


def run():
    global D, K, SCALE2, ISO_VAR, X0, MODES
    from quantum_walk_study import D, K, MODES, SCALE2, ISO_VAR, X0, ideal_covariance, sample_walk
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261114)
    names = []
    baseline = []
    curves = []
    failure = []
    win_rate = []
    heldout_mismatch = []
    for case_index, (name, modes, pilot_width) in enumerate(CASES):
        names.append(name)
        pilot_spacing = np.full(D, pilot_width / (1 << PILOT_BITS))
        pilot = sample_walk(PILOT_BITS, X0, np.eye(D), pilot_spacing, 8,
                            PILOT_POOL, 300 + case_index, modes_physical=modes)
        pool = pilot["physical_hist"]
        histories = all_true_histories(modes)
        local_spacing = np.full(D, pilot_width / (1 << BITS))
        global_spacing = np.full(D, 1 / (1 << BITS))
        # Population AMR uses the exact pooled-history covariance to show
        # where the empirical geometry rule converges. Grid rounding means
        # it is not necessarily an accuracy lower bound.
        pooled_cov = ((K + 1) / 2) * (modes.T @ modes)
        lam, oracle_axes = np.linalg.eigh(pooled_cov)
        oracle_spacing = np.maximum(np.sqrt(np.maximum(lam, 0)) / 4, 1e-6)
        oracle_spacing /= 1 << (BITS - PILOT_BITS)
        global_error = relative_error(modes, np.full(D, 0.5), np.eye(D), global_spacing)
        local_error = relative_error(modes, X0, np.eye(D), local_spacing)
        oracle_error = relative_error(modes, X0, oracle_axes, oracle_spacing)

        # Independent pilot subsets: the published workflow chooses the AMR
        # window from final measurements, then builds a separate coherent walk.
        case_curves = []
        case_failures = []
        case_win_rates = []
        for budget in BUDGETS:
            errs = []
            bad = 0
            wins = 0
            for _ in range(REPS):
                subset = pool[rng.choice(PILOT_POOL, size=int(budget), replace=False)]
                center, axes, spacing = window_from_pilot(subset)
                if not window_safe(modes, center, axes, spacing, histories):
                    bad += 1
                else:
                    error = relative_error(modes, center, axes, spacing)
                    errs.append(error)
                    wins += error < local_error
            case_failures.append(bad / REPS)
            case_win_rates.append(wins / REPS)
            case_curves.append(np.quantile(errs, [0.1, 0.5, 0.9])
                               if errs else [np.nan] * 3)
        case_curves = np.array(case_curves)
        curves.append(case_curves)
        failure.append(case_failures)
        win_rate.append(case_win_rates)
        baseline.append([global_error, local_error, oracle_error])

        # One held-out circuit per case checks that the analytically scored
        # proposal still has the expected measured output law.
        center, axes, spacing = window_from_pilot(pool[:1000])
        if not window_safe(modes, center, axes, spacing, histories):
            raise AssertionError(f"{name}: held-out AMR window is unsafe")
        heldout = sample_walk(BITS, center, axes, spacing, 14, 30_000,
                              400 + case_index, modes_physical=modes)
        heldout_mismatch.append(heldout["mismatch"])
        print(f"{name:12s} global {global_error:.3%}, local {local_error:.3%}, "
              f"AMR median {case_curves[-1, 1]:.3%}, "
              f"population-geometry {oracle_error:.3%}, "
              f"pilot failure at 8 shots {case_failures[0]:.1%}, "
              f"held-out law error {heldout['mismatch']:.2%}", flush=True)

    z = dict(names=np.array(names), budgets=BUDGETS, baseline=np.array(baseline),
             curves=np.array(curves), failure=np.array(failure),
             win_rate=np.array(win_rate),
             heldout_mismatch=np.array(heldout_mismatch),
             reps=REPS, pilot_pool=PILOT_POOL)
    np.savez_compressed(DATA / "amr_battle_test.npz", **z)
    plot(z)


def plot(z=None):
    if z is None:
        z = np.load(DATA / "amr_battle_test.npz")
    names = z["names"].tolist()
    baseline = z["baseline"]
    curves = z["curves"]
    win_rate = z["win_rate"]
    budgets = z["budgets"]
    import paper_style
    paper_style.apply()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(names))
    width = 0.24
    methods = [
        ("Full-range grid", baseline[:, 0], paper_style.COLORS["FIXED"]),
        ("Local grid", baseline[:, 1], paper_style.COLORS["ZOOM_IN"]),
        ("AMR grid", curves[:, -1, 1], paper_style.COLORS["QCLT_AMR"]),
    ]
    for j, (label, err, color) in enumerate(methods):
        ax.bar(x + (j - 1) * width, 100 * err, width,
               label=label, color=color)
    lower = 100 * (curves[:, -1, 1] - curves[:, -1, 0])
    upper = 100 * (curves[:, -1, 2] - curves[:, -1, 1])
    ax.errorbar(x + width, 100 * curves[:, -1, 1],
                yerr=np.stack((lower, upper)), fmt="none", ecolor=paper_style.INK,
                capsize=2, lw=1)
    ax.set(yscale="log", ylabel=r"Next-step covariance error (\%)")
    ax.set_xticks(x, [n.replace("-", "\n") for n in names])
    ax.grid(axis="y", color=paper_style.GRID, lw=.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=paper_style.INK, lw=1,
                          label="10th--90th percentile (up to 80 windows)"))
    labels.append("10th--90th percentile (up to 80 windows)")
    ax.legend(handles, labels, frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "qclt_amr_covariance_error.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    image = ax.imshow(100 * win_rate, aspect="auto", origin="lower",
                      cmap="Greens", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(budgets)), [str(n) for n in budgets],
                  rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_xlabel("Initial circuit samples")
    fig.colorbar(image, ax=ax, shrink=0.8, label=r"Windows improving on local grid (\%)")
    for row, values in enumerate(win_rate):
        best = int(np.flatnonzero(values == np.max(values))[0])
        ax.add_patch(Rectangle((best - .5, row - .5), 1, 1, fill=False,
                               edgecolor=paper_style.INK, linewidth=1.0))
    fig.tight_layout()
    fig.savefig(OUT / "qclt_amr_pilot_success.pdf", bbox_inches="tight")
    plt.close(fig)

    # A single geometry plot makes "tilted" concrete: rotation changes the
    # covariance ellipse while preserving its eigenvalues.
    cov_tilted = MODES.T @ MODES
    cov_tilted = cov_tilted / np.max(np.linalg.eigvalsh(cov_tilted))
    eigvals = np.linalg.eigvalsh(cov_tilted)
    cov_aligned = np.diag(eigvals)
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    for cov, color, label, ls in (
            (cov_tilted, paper_style.COLORS["QCLT_AMR"], "Tilted modes", "-"),
            (cov_aligned, paper_style.COLORS["CLASSICAL"], "Axis-aligned modes", "--")):
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        for scale, alpha in ((1.0, .95), (1.7, .72)):
            ax.add_patch(Ellipse((0, 0), 2 * scale * np.sqrt(vals[1]),
                                 2 * scale * np.sqrt(vals[0]), angle=angle,
                                 fill=False, edgecolor=color, linewidth=1.25,
                                 linestyle=ls, alpha=alpha,
                                 label=label if scale == 1.0 else None))
    ax.axhline(0, color=paper_style.GRID, lw=.6, zorder=0)
    ax.axvline(0, color=paper_style.GRID, lw=.6, zorder=0)
    ax.set(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0),
           xlabel="Parameter 1", ylabel="Parameter 2", aspect="equal")
    ax.legend(frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "qclt_amr_tilted_geometry.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    plot() if "--plot-only" in sys.argv else run()
