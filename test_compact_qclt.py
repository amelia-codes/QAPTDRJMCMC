"""Distribution and resource checks for the classical-history QCLT circuit."""

import itertools
import unittest

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from compact_qclt import build_compiled_kick, output_widths
from qcov import (check_coherent, covariance_kick, exact_distribution,
                  iso_step, lazy_coin, regs_values)


def reference_distribution(dm, q, p_iso):
    """Enumerate the sampler's independent three-point choices directly."""
    law = {-1: q / 2, 0: 1 - q, 1: q / 2}
    iso_law = {-1: p_iso / 2, 0: 1 - p_iso, 1: p_iso / 2}
    result = {}
    for pair_choice in itertools.product((-1, 0, 1), repeat=len(dm)):
        pair_weight = np.prod([law[x] for x in pair_choice])
        if not pair_weight:
            continue
        pair_step = np.asarray(pair_choice) @ dm
        for iso_choice in itertools.product((-1, 0, 1), repeat=dm.shape[1]):
            weight = pair_weight * np.prod([iso_law[x] for x in iso_choice])
            if weight:
                step = tuple(pair_step + iso_choice)
                result[step] = result.get(step, 0.0) + weight
    return result


class CompactQCLTTests(unittest.TestCase):
    def check_case(self, dm, q, p_iso):
        qc, output = build_compiled_kick(dm, q, p_iso)
        self.assertEqual(check_coherent(qc), 0)
        bits, weights = exact_distribution(qc)
        observed = {}
        for step, weight in zip(regs_values(bits, qc, output, signed=True), weights):
            key = tuple(step)
            observed[key] = observed.get(key, 0.0) + weight
        expected = reference_distribution(dm, q, p_iso)
        self.assertEqual(set(observed), set(expected))
        self.assertLess(max(abs(observed[k] - expected[k]) for k in expected), 1e-12)

    def test_mixed_coins_and_zero_pair(self):
        dm = np.array([[2, -1], [0, 0], [-3, 1]])
        self.check_case(dm, 0.25, 0.2)

    def test_certain_moves_and_no_moves(self):
        dm = np.array([[1, 2], [-2, 1]])
        self.check_case(dm, 1.0, 1.0)
        self.check_case(dm, 0.0, 0.0)

    def test_small_quantum_statevector(self):
        qc, output = build_compiled_kick(np.array([[2]]), 0.25, 0.2)
        self.assertEqual(qc.num_qubits, 7)
        probs = Statevector(qc).probabilities(
            qargs=[qc.find_bit(q).index for q in output[0]])
        values = {(i if i < 4 else i - 8,): float(p)
                  for i, p in enumerate(probs) if p > 1e-14}
        reference = reference_distribution(np.array([[2]]), 0.25, 0.2)
        self.assertEqual(set(values), set(reference))
        self.assertLess(max(abs(values[k] - reference[k]) for k in values), 1e-12)

    def test_wavelet_amr_worst_case_fits_under_100(self):
        history = np.repeat(np.array([0, 1, 2, 61, 62, 63])[:, None], 5, axis=1)
        dm = np.array([history[a] - history[b]
                       for a, b in itertools.combinations(range(6), 2)])
        self.assertEqual(output_widths(dm, 0.5, 0.2), [11] * 5)
        qc, _ = build_compiled_kick(dm, 0.2, 0.5)
        self.assertEqual(qc.num_qubits, 95)

    def test_same_distribution_as_history_register_circuit(self):
        history = np.array([[0, 3], [4, 1], [2, 6]])
        dm = np.array([history[a] - history[b]
                       for a, b in itertools.combinations(range(3), 2)])
        q, p_iso = 0.25, 0.2
        compact, compact_out = build_compiled_kick(dm, q, p_iso)
        k, d, bits = 3, 2, 3
        old_out = [QuantumRegister(w, f"old_step_{p}")
                   for p, w in enumerate(output_widths(dm, p_iso, q))]
        hregs = [[QuantumRegister(bits, f"h_{j}_{p}") for p in range(d)]
                 for j in range(k)]
        move, sign = QuantumRegister(3, "move"), QuantumRegister(3, "sign")
        iso_move, iso_sign = QuantumRegister(d, "iso_move"), QuantumRegister(d, "iso_sign")
        old = QuantumCircuit(*old_out, *[r for row in hregs for r in row],
                             move, sign, iso_move, iso_sign)
        for j in range(k):
            for p in range(d):
                for i in range(bits):
                    if (history[j, p] >> i) & 1:
                        old.x(hregs[j][p][i])
        covariance_kick(old, [old_out], hregs, zip(move, sign), q * k * (k - 1))
        for p in range(d):
            lazy_coin(old, p_iso, iso_move[p], iso_sign[p])
        iso_step(old, [old_out], [1] * d, iso_sign, iso_move)

        def distribution(qc, outputs):
            bits_out, weights = exact_distribution(qc)
            law = {}
            for step, weight in zip(regs_values(bits_out, qc, outputs, signed=True), weights):
                key = tuple(step)
                law[key] = law.get(key, 0.0) + weight
            return law

        a, b = distribution(compact, compact_out), distribution(old, old_out)
        self.assertEqual(set(a), set(b))
        self.assertLess(max(abs(a[x] - b[x]) for x in a), 1e-12)


if __name__ == "__main__":
    unittest.main()
