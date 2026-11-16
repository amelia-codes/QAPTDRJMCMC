"""Small-grid output-law replica of the Kitaev-Webb correlated-Gaussian route.

This implements the published independent-Gaussians-plus-reversible-shear
construction. The 1D normal states use Qiskit's generic StatePreparation and
the shear uses a lookup circuit, not the paper's polynomial arithmetic.
Covariance parameters are supplied classically. Therefore its gate count is
for this explicit replica, not a resource claim for a fully coherent
history-conditioned Gaussian proposal.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

from qcov import add_const
from quantum_walk_study import D, K, SCALE2, ISO_VAR
from amr_battle_test import BASE_CASES


ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "quantum_walk_archive"
DATA = ROOT / "data" / "quantum_walk"
BITS = 6
N = 1 << BITS
SIGNED = np.where(np.arange(N) >= N // 2, np.arange(N) - N, np.arange(N))


def normalized_gaussian_1d(var):
    p = np.exp(-0.5 * SIGNED.astype(float) ** 2 / var)
    return p / p.sum()


def target_lattice_covariance(modes_physical, width):
    spacing = width / N
    modes = np.rint(modes_physical / spacing).astype(int)
    return SCALE2 * ((K + 1) / 6) * (modes.T @ modes) + ISO_VAR * np.eye(D)


def direct_gaussian_prob(cov):
    y, x = np.meshgrid(SIGNED, SIGNED, indexing="ij")
    v = np.stack((x, y), axis=-1).astype(float)
    e = np.einsum("...i,ij,...j->...", v, np.linalg.inv(cov), v)
    p = np.exp(-0.5 * e)
    return p / p.sum()


def prepare_sheared_gaussian(cov):
    """Factor covariance as M diag(v0,v1) M^T and shear x <- x+round(alpha*y)."""
    v1 = cov[1, 1]
    alpha = cov[0, 1] / v1
    v0 = cov[0, 0] - cov[0, 1] ** 2 / v1
    if min(v0, v1) <= 0:
        raise ValueError("covariance must be positive definite")
    amp0 = np.sqrt(normalized_gaussian_1d(v0))
    amp1 = np.sqrt(normalized_gaussian_1d(v1))
    qc = QuantumCircuit(2 * BITS)
    qc.append(StatePreparation(amp0), qc.qubits[:BITS])
    qc.append(StatePreparation(amp1), qc.qubits[BITS:])
    controls = qc.qubits[BITS:]
    for encoded_y in range(N):
        shift = int(np.rint(alpha * SIGNED[encoded_y]))
        if shift % N == 0:
            continue
        flips = [q for bit, q in enumerate(controls) if not (encoded_y >> bit) & 1]
        for q in flips:
            qc.x(q)
        add_const(qc, qc.qubits[:BITS], shift, controls)
        for q in flips:
            qc.x(q)
    return qc


def covariance_from_prob(p):
    y, x = np.meshgrid(SIGNED, SIGNED, indexing="ij")
    v = np.stack((x, y), axis=-1).astype(float)
    mu = np.einsum("yx,yxi->i", p, v)
    c = v - mu
    return np.einsum("yx,yxi,yxj->ij", p, c, c)


def analytic_shear_result(modes, width, bits):
    """Output law of independent Gaussians followed by rounded modular shear."""
    n = 1 << bits
    signed = np.where(np.arange(n) >= n // 2, np.arange(n) - n, np.arange(n))
    spacing = width / n
    lattice_modes = np.rint(modes / spacing).astype(int)
    cov = SCALE2 * ((K + 1) / 6) * (lattice_modes.T @ lattice_modes)
    cov += ISO_VAR * np.eye(D)
    v1 = cov[1, 1]
    alpha = cov[0, 1] / v1
    v0 = cov[0, 0] - cov[0, 1] ** 2 / v1
    p0 = np.exp(-0.5 * signed.astype(float) ** 2 / v0)
    p1 = np.exp(-0.5 * signed.astype(float) ** 2 / v1)
    p0 /= p0.sum()
    p1 /= p1.sum()
    shear = np.zeros((n, n))
    for y, value in enumerate(signed):
        shear[y] = p1[y] * np.roll(p0, int(np.rint(alpha * value)))
    yy, xx = np.meshgrid(signed, signed, indexing="ij")
    vec = np.stack((xx, yy), axis=-1).astype(float)
    e = np.einsum("...i,ij,...j->...", vec, np.linalg.inv(cov), vec)
    direct = np.exp(-0.5 * e)
    direct /= direct.sum()
    mu = np.einsum("yx,yxi->i", shear, vec)
    centered = vec - mu
    shear_cov = np.einsum("yx,yxi,yxj->ij", shear, centered, centered)
    fidelity = np.sum(np.sqrt(shear * direct)) ** 2
    cov_error = np.linalg.norm(shear_cov - cov) / np.linalg.norm(cov)
    return fidelity, cov_error, shear


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    names, fidelity, ideal_error, shear_error, gates, depth = [], [], [], [], [], []
    for name, modes, width in BASE_CASES:
        cov = target_lattice_covariance(modes, width)
        ideal_p = direct_gaussian_prob(cov)
        qc = prepare_sheared_gaussian(cov)
        p = np.abs(Statevector.from_instruction(qc).data.reshape(N, N)) ** 2
        _, _, analytic_p = analytic_shear_result(modes, width, BITS)
        if not np.allclose(p, analytic_p, atol=1e-10):
            raise AssertionError("shear circuit differs from its analytic output law")
        if not np.isclose(p.sum(), 1, atol=1e-10):
            raise AssertionError("Gaussian circuit output is not normalized")
        fid = np.sum(np.sqrt(p * ideal_p)) ** 2
        ideal_cov = covariance_from_prob(ideal_p)
        shear_cov = covariance_from_prob(p)
        target = cov
        names.append(name)
        fidelity.append(fid)
        ideal_error.append(np.linalg.norm(ideal_cov - target) / np.linalg.norm(target))
        shear_error.append(np.linalg.norm(shear_cov - target) / np.linalg.norm(target))
        gates.append(qc.size())
        depth.append(qc.depth())
        print(f"{name:12s} Gaussian fidelity {fid:.5f}, "
              f"ideal lattice covariance error {ideal_error[-1]:.2%}, "
              f"shear output covariance error {shear_error[-1]:.2%}, "
              f"replica {qc.num_qubits}q/{qc.size()} high-level gates", flush=True)
    sweep_bits = np.arange(6, 11)
    sweep = np.array([analytic_shear_result(BASE_CASES[0][1], BASE_CASES[0][2], int(b))[:2]
                      for b in sweep_bits])
    z = dict(names=np.array(names), fidelity=np.array(fidelity),
             ideal_error=np.array(ideal_error), shear_error=np.array(shear_error),
             gates=np.array(gates), depth=np.array(depth), bits=BITS,
             sweep_bits=sweep_bits, sweep_fidelity=sweep[:, 0],
             sweep_cov_error=sweep[:, 1])
    np.savez_compressed(DATA / "gaussian_shear_replica.npz", **z)
    plot(z)


def plot(z=None):
    if z is None:
        z = np.load(DATA / "gaussian_shear_replica.npz")
    names = z["names"].tolist()
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8))
    ax = axes[0]
    ax.bar(x, 100 * z["shear_error"], .6, color="#176a6d",
           label="prepared shear circuit")
    ax.set(ylabel="Covariance error after grid encoding (%)",
           title="Same 6-bit local grids\nIdeal grid Gaussian: < $10^{-8}$% error")
    ax.set_xticks(x, [n.replace("-", "\n") for n in names], fontsize=8)
    ax.grid(axis="y", alpha=.16)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.bar(x, 100 * (1 - z["fidelity"]), color="#176a6d")
    ax.set(ylabel="Infidelity to ideal grid Gaussian (%)",
           title="Independent Gaussians + reversible shear")
    ax.set_xticks(x, [n.replace("-", "\n") for n in names], fontsize=8)
    ax.grid(axis="y", alpha=.16)
    ax = axes[2]
    ax.plot(z["sweep_bits"], 100 * z["sweep_cov_error"], "o-",
            color="#176a6d", label="covariance error")
    ax.plot(z["sweep_bits"], 100 * (1 - z["sweep_fidelity"]), "s--",
            color="#364b8f", label="infidelity")
    ax.set(yscale="log", xlabel="Bits per coordinate",
           ylabel="Error on tilted case (%)", title="Grid resolution controls shear error")
    ax.set_xticks(z["sweep_bits"])
    ax.grid(alpha=.16)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Kitaev–Webb-style Gaussian proposal: output-law replica", fontsize=12, y=.99)
    fig.text(.5, .01,
             "Covariance supplied classically; 1D states use generic loading and shear uses lookup gates. "
             "This is not the paper's optimized arithmetic circuit.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .05, 1, .94), w_pad=2)
    fig.savefig(OUT / "qclt_vs_gaussian_shear.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
