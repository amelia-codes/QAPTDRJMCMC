"""Checks that each readout law matches its circuit and estimates the covariance.

    python3.12 -m unittest
"""

import itertools
import unittest

import numpy as np
from qiskit.quantum_info import Statevector, partial_trace

from circuit_costs import (amplitude_estimation, covariance_preparation, hadamard_test, pairwise_moment,
                           single_pair_step, swap_test)
from covariance_methods import amplitude_estimates, phase_states, readout
from walk import SCALE2


def ancilla_probability(qc, outcome):
    return Statevector.from_instruction(qc).probabilities([qc.num_qubits - 1])[outcome]


class CovarianceMethodTests(unittest.TestCase):
    def test_circuits_give_the_probabilities_the_readout_laws_use(self):
        for sign in (-1, 1):
            history = np.array([[2., -1.], [4., 2.], [7., 4.]]) * [1, sign]
            for build in (hadamard_test, swap_test):
                qc, probability = build(history)
                self.assertAlmostEqual(ancilla_probability(qc, 0), probability, places=11)
            for p, q in [(0, 0), (0, 1), (1, 1)]:
                qc, probability = pairwise_moment(history, p, q)
                self.assertAlmostEqual(ancilla_probability(qc, 1), probability, places=11)
                values = (history[:, None, p] - history[None, :, p]) * (history[:, None, q] - history[None, :, q]) / 2
                estimate = 3 / 2 * (values.min() + (values.max() - values.min()) * probability)
                self.assertAlmostEqual(estimate, np.cov(history.T)[p, q], places=10)

    def test_covariance_preparation_holds_the_scaled_covariance(self):
        rng = np.random.default_rng(4200)
        for shape in ((3, 2), (5, 3)):
            history = rng.normal(size=shape) * np.arange(1, shape[0] + 1)[:, None] + 7
            qc, trace = covariance_preparation(history)
            data_bits = max(1, (shape[1] - 1).bit_length())
            rho = partial_trace(Statevector.from_instruction(qc), list(range(data_bits, qc.num_qubits))).data
            expected = np.zeros_like(rho)
            expected[:shape[1], :shape[1]] = np.cov(history.T)
            np.testing.assert_allclose(trace * rho, expected, atol=1e-11)

    def test_readouts_are_unbiased_and_translation_invariant(self):
        history = np.array([[0., 4.], [2., 1.], [5., -2.]])
        for kind in ("swap", "hadamard", "qpca", "pairwise"):
            estimates = readout(history, kind, 384, np.random.default_rng(83), 30_000)
            target = np.cov(history.T)
            self.assertLess(np.linalg.norm(estimates.mean(axis=0) - target) / np.linalg.norm(target), .004)
            shifted = readout(history + [100., -77.], kind, 384, np.random.default_rng(83), 30_000)
            np.testing.assert_allclose(estimates, shifted, atol=1e-10)

    def test_amplitude_estimation_budget(self):
        for budget in (1, 2, 3, 4, 12, 64, 128, 384):
            states = phase_states(budget)
            self.assertEqual(states & (states - 1), 0)             # a power of two
            self.assertLessEqual(2 * (states - 1) + 1, budget)     # fits the budget
            self.assertGreater(2 * (2 * states - 1) + 1, budget)   # the next one would not
        np.testing.assert_allclose(amplitude_estimates(.5, 4, np.random.default_rng(5), 100), .5)

    def test_amplitude_estimation_law_matches_its_circuit(self):
        qc, probability = amplitude_estimation(np.array([[0., 1.], [1., 0.]]), 0, 0, 2)
        self.assertAlmostEqual(probability, .5)
        phase = Statevector.from_instruction(qc).probabilities([qc.num_qubits - 2, qc.num_qubits - 1])
        np.testing.assert_allclose(phase, [0, .5, 0, .5], atol=1e-10)

    def test_single_pair_step_has_the_rounded_covariance(self):
        history = np.array([[0, 0], [1, 0], [0, 1]])
        qc, step = single_pair_step(history, bits=3)
        probabilities = Statevector.from_instruction(qc).probabilities(
            [qc.find_bit(qubit).index for register in step for qubit in register])
        coordinates = np.column_stack((np.arange(64) % 8, np.arange(64) // 8))
        coordinates = np.where(coordinates < 4, coordinates, coordinates - 8)
        mean = probabilities @ coordinates
        covariance = (coordinates.T * probabilities) @ coordinates - np.outer(mean, mean)
        differences = np.rint(np.sqrt(SCALE2 / 2) * np.array(
            [history[a] - history[b] for a, b in itertools.combinations(range(3), 2)]))
        np.testing.assert_allclose(covariance, differences.T @ differences / 3 + .1 * np.eye(2), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
