"""Independent history-register reference for the compressed coherent walk."""

import numpy as np
import itertools
import unittest
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister

from amr_battle_test import exact_step_covariance
from qcov import (add_const, check_coherent, iso_step, lazy_coin,
                  regs_values, sample_circuit, torus_cov)
from quantum_walk_study import (D, K, MODES, SCALE2, ISO_VAR, X0,
                                grid_coordinates,
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
        for register, saved in zip(position, history[t]):
            for qubit, copy in zip(register, saved):
                qc.cx(qubit, copy)
    # Independent reference: subtract the actual stored registers, rather
    # than reconstructing their differences from the preparatory signs.
    for j, (a, b) in enumerate(itertools.combinations(range(K), 2)):
        lazy_coin(qc, SCALE2 / (K * (K - 1)), move[j], sign[j])
        for p in range(D):
            for bit in range(bits):
                ha, hb = history[a][p][bit], history[b][p][bit]
                add_const(qc, position[p], 1 << bit, [move[j], ha])
                add_const(qc, position[p], -(1 << bit), [move[j], hb])
                add_const(qc, position[p], -2 * (1 << bit), [move[j], sign[j], ha])
                add_const(qc, position[p], 2 * (1 << bit), [move[j], sign[j], hb])
    for p in range(D):
        lazy_coin(qc, ISO_VAR, iso_move[p], iso_sign[p])
    iso_step(qc, [position], [1] * D, iso_sign, iso_move)
    return qc, position, history


class QuantumWalkTests(unittest.TestCase):
    def test_compressed_walk_matches_explicit_quantum_history(self):
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


    def test_fresh_coins_across_repeated_kicks(self):
        axes = np.eye(D)
        for bits, expected in [(6, (29, 39, 49, 59)),
                               (8, (33, 43, 53, 63))]:
            spacing = np.full(D, 0.08 / (1 << bits))
            for steps, expected_qubits in enumerate(expected, start=1):
                qc, _ = make_walk(bits, X0, axes, spacing, 14,
                                  track_displacement=False, adaptive_steps=steps)
                assert qc.num_qubits == expected_qubits
                assert check_coherent(qc) == 0


    def test_analytic_covariance_enumerates_all_history_branches(self):
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
        assert np.allclose(exact_step_covariance(MODES, X0, axes, spacing), basis @ lattice @ basis.T)


    def test_diagnostic_register_does_not_change_position(self):
        spacing = np.full(D, .08 / 64)
        tracked, tracked_reg = make_walk(6, X0, np.eye(D), spacing, 8)
        bare, bare_reg = make_walk(6, X0, np.eye(D), spacing, 8, track_displacement=False)
        a, b = sample_circuit(tracked, 512, seed=4242), sample_circuit(bare, 512, seed=4242)
        np.testing.assert_array_equal(regs_values(a, tracked, tracked_reg["P"]),
                                      regs_values(b, bare, bare_reg["P"]))

    def test_sampler_matches_a_small_statevector(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        circuit = QuantumCircuit(4)
        lazy_coin(circuit, .3, circuit.qubits[0], circuit.qubits[1])
        add_const(circuit, circuit.qubits[2:], 1, [circuit.qubits[0]])
        add_const(circuit, circuit.qubits[2:], -2, circuit.qubits[:2])
        actual = sample_circuit(circuit, 50_000, seed=240)
        labels = (actual * (1 << np.arange(4))[:, None]).sum(axis=0)
        observed = np.bincount(labels, minlength=16) / len(labels)
        expected = Statevector.from_instruction(circuit).probabilities()
        self.assertLess(np.abs(observed - expected).sum(), .015)

    def test_sampler_rejects_interference(self):
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.h(0)
        with self.assertRaisesRegex(ValueError, "fresh"):
            sample_circuit(circuit, 10)


if __name__ == "__main__":
    unittest.main()
