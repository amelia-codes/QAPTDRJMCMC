"""Circuit, normalization, and method-selection checks for the paper comparison."""

import itertools
import unittest

import numpy as np
from qiskit.quantum_info import Statevector, partial_trace

from covariance_benchmarks import (
    READOUT_KINDS, amplitude_readout_circuit,
    covariance_preparation_circuit, hadamard_readout_circuit, pairwise_exact,
    pairwise_readout_circuit, qae_phase_states, qae_sample, readout_estimate,
    single_pair_step_circuit, swap_readout_circuit,
)


class CovarianceBenchmarkTests(unittest.TestCase):
    def test_interference_circuits_preserve_covariance_sign(self):
        for sign in (-1, 1):
            history = np.array([[2., -1.], [4., 2.], [7., 4.]]) * [1, sign]
            for builder in (hadamard_readout_circuit, swap_readout_circuit):
                circuit, expected = builder(history)
                actual = Statevector.from_instruction(circuit).probabilities([circuit.num_qubits - 1])[0]
                self.assertAlmostEqual(actual, expected, places=11)
            for p, q, probability, lo, hi in pairwise_exact(history):
                circuit, _ = pairwise_readout_circuit(history, p, q)
                actual = Statevector.from_instruction(circuit).probabilities([circuit.num_qubits - 1])[1]
                self.assertAlmostEqual(actual, probability, places=11)
                estimate = len(history) / (len(history) - 1) * (lo + (hi - lo) * actual)
                self.assertAlmostEqual(estimate, np.cov(history.T)[p, q], places=10)

    def test_covariance_state_uses_norm_weights_and_retains_scale(self):
        rng = np.random.default_rng(4200)
        for shape in ((3, 2), (5, 3)):
            history = rng.normal(size=shape) * np.arange(1, shape[0] + 1)[:, None] + 7
            circuit, trace = covariance_preparation_circuit(history)
            data_bits = max(1, (shape[1] - 1).bit_length())
            rho = partial_trace(Statevector.from_instruction(circuit),
                                list(range(data_bits, circuit.num_qubits))).data
            expected = np.zeros_like(rho)
            expected[:shape[1], :shape[1]] = np.cov(history.T)
            np.testing.assert_allclose(trace * rho, expected, atol=1e-11)

    def test_readout_laws_are_unbiased_and_translation_invariant(self):
        history = np.array([[0., 4.], [2., 1.], [5., -2.]])
        target = np.cov(history.T)
        for kind in ("swap", "hadamard", "qpca", "pairwise"):
            estimates = readout_estimate(history, kind, 384, np.random.default_rng(83), 30_000)
            self.assertLess(np.linalg.norm(estimates.mean(axis=0) - target) / np.linalg.norm(target), .004)
            shifted = readout_estimate(history + [100., -77.], kind, 384,
                                       np.random.default_rng(83), 30_000)
            np.testing.assert_allclose(estimates, shifted, atol=1e-10)
        for kind in READOUT_KINDS:
            np.testing.assert_allclose(
                readout_estimate(np.ones((3, 2)), kind, 384, np.random.default_rng(2), 10), 0)

    def test_amplitude_budget_counts_preparation_and_inverse(self):
        for budget in (1, 2, 3, 4, 12, 64, 128, 384):
            states = qae_phase_states(budget)
            self.assertEqual(states & (states - 1), 0)
            self.assertLessEqual(2 * (states - 1) + 1, budget)
            self.assertGreater(2 * (2 * states - 1) + 1, budget)
        self.assertEqual(qae_phase_states(128), 64)
        values = qae_sample(.5, 4, np.random.default_rng(5), 100)
        np.testing.assert_allclose(values, .5)

    def test_amplitude_law_matches_coherent_phase_estimation(self):
        circuit, probability = amplitude_readout_circuit(np.array([[0., 1.], [1., 0.]]), 0, 0, 2)
        self.assertAlmostEqual(probability, .5)
        actual = Statevector.from_instruction(circuit).probabilities(
            [circuit.num_qubits - 2, circuit.num_qubits - 1])
        np.testing.assert_allclose(actual, [0, .5, 0, .5], atol=1e-10)

    def test_single_pair_step_matches_rounded_covariance(self):
        history = np.array([[0, 0], [1, 0], [0, 1]])
        scale2 = 2.38 ** 2 / 2
        circuit, output = single_pair_step_circuit(history, bits=3, scale2=scale2)
        indices = [circuit.find_bit(qubit).index for register in output for qubit in register]
        probabilities = Statevector.from_instruction(circuit).probabilities(indices)
        unsigned = np.arange(64)
        coordinates = np.column_stack((unsigned % 8, unsigned // 8))
        coordinates = np.where(coordinates < 4, coordinates, coordinates - 8)
        mean = probabilities @ coordinates
        actual = (coordinates.T * probabilities) @ coordinates - np.outer(mean, mean)
        differences = np.rint(np.sqrt(scale2 / 2) * np.array(
            [history[a] - history[b] for a, b in itertools.combinations(range(3), 2)]))
        expected = differences.T @ differences / 3 + .1 * np.eye(2)
        np.testing.assert_allclose(actual, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
