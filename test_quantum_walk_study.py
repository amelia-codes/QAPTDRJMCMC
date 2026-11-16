"""Independent history-register reference for the compressed coherent walk."""

import numpy as np
import itertools
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister

from qcov import (add_const, check_coherent, covariance_kick, iso_step,
                  lazy_coin, regs_values, sample_circuit, snapshot, torus_cov)
from quantum_walk_study import (D, K, MODES, SCALE2, ISO_VAR, X0,
                                exact_unconditional_covariance, grid_coordinates,
                                histories_from_coins, make_walk)


def explicit_history_reference(bits, center, axes, spacing):
    """Store every snapshot and use the original register-difference kick."""
    initial = grid_coordinates(X0, center, axes, spacing, bits)
    modes = np.rint(MODES @ axes / spacing).astype(int)
    position = [QuantumRegister(bits, f"position_{p}") for p in range(D)]
    history = [[QuantumRegister(bits, f"history_{t}_{p}") for p in range(D)]
               for t in range(K)]
    prep = AncillaRegister(K * D, "preparation_signs")
    pairs = K * (K - 1) // 2
    move = AncillaRegister(pairs, "pair_move")
    sign = AncillaRegister(pairs, "pair_sign")
    iso_move = AncillaRegister(D, "iso_move")
    iso_sign = AncillaRegister(D, "iso_sign")
    qc = QuantumCircuit(*position, *[r for row in history for r in row],
                        prep, move, sign, iso_move, iso_sign)
    for p, value in enumerate(initial):
        for bit in range(bits):
            if (int(value) >> bit) & 1:
                qc.x(position[p][bit])
    for t in range(K):
        for m in range(D):
            coin = prep[t * D + m]
            qc.h(coin)
            for p, velocity in enumerate(modes[m]):
                if velocity:
                    add_const(qc, position[p], int(velocity))
                    add_const(qc, position[p], -2 * int(velocity), [coin])
        snapshot(qc, position, history[t])
    covariance_kick(qc, [position], history, zip(move, sign), SCALE2,
                    periodic=True)
    for p in range(D):
        lazy_coin(qc, ISO_VAR, iso_move[p], iso_sign[p])
    iso_step(qc, [position], [1] * D, iso_sign, iso_move)
    return qc, position, history


def test_compressed_walk_matches_explicit_quantum_history():
    bits = 6
    axes = np.eye(D)
    spacing = np.full(D, 0.08 / 64)
    compressed, reg = make_walk(bits, X0, axes, spacing, 8,
                                track_displacement=False)
    explicit, position, history = explicit_history_reference(bits, X0, axes, spacing)
    assert compressed.num_qubits == 29
    assert explicit.num_qubits == 64
    assert check_coherent(compressed) == check_coherent(explicit) == 0
    a = sample_circuit(compressed, 512, seed=4242)
    b = sample_circuit(explicit, 512, seed=4242)
    assert np.array_equal(regs_values(a, compressed, reg["P"]),
                          regs_values(b, explicit, position))
    reconstructed = histories_from_coins(a, compressed, reg)
    stored = np.stack([regs_values(b, explicit, row) for row in history], axis=1)
    assert np.array_equal(reconstructed, stored)


def test_fresh_coins_across_repeated_kicks():
    axes = np.eye(D)
    for bits, expected in [(6, (29, 39, 49, 59)),
                           (8, (33, 43, 53, 63))]:
        spacing = np.full(D, 0.08 / (1 << bits))
        for steps, expected_qubits in enumerate(expected, start=1):
            qc, _ = make_walk(bits, X0, axes, spacing, 14,
                              track_displacement=False, adaptive_steps=steps)
            assert qc.num_qubits == expected_qubits
            assert check_coherent(qc) == 0


def test_analytic_covariance_enumerates_all_history_branches():
    bits = 6
    axes = np.eye(D)
    spacing = np.full(D, 0.08 / 64)
    _, reg = make_walk(bits, X0, axes, spacing, 8,
                       track_displacement=False)
    coins = np.array(list(itertools.product((0, 1), repeat=K * D))).reshape(-1, K, D)
    increments = np.einsum("ntm,mp->ntp", 1 - 2 * coins, reg["modes"])
    histories = (reg["initial"] + np.cumsum(increments, axis=1)) % (1 << bits)
    lattice = SCALE2 * torus_cov(histories, bits).mean(axis=0) + ISO_VAR * np.eye(D)
    basis = axes @ np.diag(spacing)
    assert np.allclose(exact_unconditional_covariance(reg), basis @ lattice @ basis.T)


if __name__ == "__main__":
    test_compressed_walk_matches_explicit_quantum_history()
    test_fresh_coins_across_repeated_kicks()
    test_analytic_covariance_enumerates_all_history_branches()
    print("coherent-walk equivalence and resource checks passed")
