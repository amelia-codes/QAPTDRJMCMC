# Checks that each readout law matches its circuit and estimates the covariance
#
#   python3.12 -m unittest

import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt
from qiskit.quantum_info import Statevector, partial_trace

from circuit_costs import (amplitude_estimation, covariance_preparation, hadamard_test, pair_product_circuit,
                           swap_test)
from covariance_methods import READOUTS, amplitude_estimates, pair_products, phase_states, readout
import covariance_methods as methods


# Exact probability that a circuit's last (ancilla) qubit reads `outcome`
def ancilla_probability(qc, outcome):
    return Statevector.from_instruction(qc).probabilities([qc.num_qubits - 1])[outcome]


class CovarianceMethodTests(unittest.TestCase):
    def test_skewed_errors_still_use_mean_and_standard_deviation(self):
        # Percent errors [1, 1, 1, 9] have mean 3 and sample SD 4, but median 1.
        sample = np.array([.01, .01, .01, .09])
        errors = {key + suffix: sample for key in methods.METHODS for suffix in methods.GRIDS}
        budgets = np.broadcast_to(sample, (len(methods.READOUTS), len(methods.BUDGETS), 4))
        for draw, arguments in ((methods.plot_all_grids, (errors,)),
                                (methods.plot_budgets, (errors, budgets))):
            with patch.object(methods.style, "save") as save:
                draw(*arguments)
                figure = save.call_args.args[0]
                axes = figure.axes[0]
                segments = axes.collections[0].get_segments()[0]
                if draw is methods.plot_all_grids:
                    self.assertAlmostEqual(axes.patches[0].get_width(), 3)
                    np.testing.assert_allclose(segments[:, 0], [-1, 7])
                else:
                    np.testing.assert_allclose(axes.lines[0].get_ydata(), 3)
                    np.testing.assert_allclose(segments[:, 1], [-1, 7])
                plt.close(figure)

    # Each circuit's exact outcome probability is the one its readout law uses
    def test_circuits_give_the_probabilities_the_readout_laws_use(self):
        for sign in (-1, 1):
            history = np.array([[2., -1.], [4., 2.], [7., 4.]]) * [1, sign]
            S = np.cov(history.T)

            # Hadamard and SWAP tests, whose probabilities give the correlation and Sigma_01 / tr Sigma_N
            for build in (hadamard_test, swap_test):
                qc, probability = build(history)
                self.assertAlmostEqual(ancilla_probability(qc, 0), probability, places=11)
            self.assertAlmostEqual(hadamard_test(history)[1], (1 + S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])) / 2)
            self.assertAlmostEqual(swap_test(history)[1], 3 / 4 - S[0, 1] / (2 * np.trace(S)))

            # Pair-product circuits for each entry and for the trace, whose rescaled probability gives
            # back that entry or the trace
            tables = [(pair_products(history, p, q), S[p, q]) for p, q in [(0, 0), (0, 1), (1, 1)]]
            tables.append((pair_products(history, 0, 0) + pair_products(history, 1, 1), np.trace(S)))
            for values, expected in tables:
                qc, probability = pair_product_circuit(values)
                self.assertAlmostEqual(ancilla_probability(qc, 1), probability, places=11)

                estimate = 3 / 2 * (values.min() + (values.max() - values.min()) * probability)
                self.assertAlmostEqual(estimate, expected, places=10)

    # Classical and hybrid AMR appear in the all-grids chart only with --classical-amr
    def test_classical_amr_switch(self):
        sample = np.array([.01, .02])
        errors = {key + suffix: sample for key in methods.METHODS for suffix in methods.GRIDS}
        for switch, grids in ((False, 2), (True, 4)):
            with patch.object(methods.style, "CLASSICAL_AMR", switch), patch.object(methods.style, "save") as save:
                methods.plot_all_grids(errors)
                self.assertEqual(len(save.call_args.args[0].axes[0].patches), grids * len(methods.SHOWN))
            plt.close("all")

    # The prepared coordinate state is Sigma_N / tr(Sigma_N)
    def test_covariance_preparation_holds_the_scaled_covariance(self):
        rng = np.random.default_rng(4200)
        for shape in ((3, 2), (5, 3)):
            history = rng.normal(size=shape) * np.arange(1, shape[0] + 1)[:, None] + 7
            qc, rho = covariance_preparation(history)
            np.testing.assert_allclose(rho, np.cov(history.T) / np.trace(np.cov(history.T)), atol=1e-12)

            # Trace out the history-index qubits
            data_bits = max(1, (shape[1] - 1).bit_length())
            reduced = partial_trace(Statevector.from_instruction(qc), list(range(data_bits, qc.num_qubits))).data

            expected = np.zeros_like(reduced)
            expected[:shape[1], :shape[1]] = rho
            np.testing.assert_allclose(reduced, expected, atol=1e-11)

    # Averaged over many repetitions the readouts recover the covariance, and shifting the
    # history does not change them
    def test_readouts_are_unbiased_and_translation_invariant(self):
        history = np.array([[0., 4.], [2., 1.], [5., -2.]])
        target = np.cov(history.T)

        for kind in ("swap", "hadamard", "qpca"):
            estimates = readout(history, kind, 384, np.random.default_rng(83), 30_000)
            self.assertLess(np.linalg.norm(estimates.mean(axis=0) - target) / np.linalg.norm(target), .004, kind)

        for kind in READOUTS:
            estimates = readout(history, kind, 384, np.random.default_rng(83), 1000)
            shifted = readout(history + [100., -77.], kind, 384, np.random.default_rng(83), 1000)
            np.testing.assert_allclose(estimates, shifted, atol=1e-10)

    # Cauchy-Schwarz measures the variances, then sets every correlation to +1 or to 0
    def test_cauchy_schwarz_readouts(self):
        history = np.array([[0., 4.], [2., 1.], [5., -2.]])
        for kind in ("cs_upper", "cs_average"):
            estimates = readout(history, kind, 384, np.random.default_rng(83), 30_000)
            variances = estimates[:, [0, 1], [0, 1]]
            np.testing.assert_allclose(variances.mean(axis=0), np.diag(np.cov(history.T)), rtol=.004)

            expected = np.sqrt(variances[:, 0] * variances[:, 1]) if kind == "cs_upper" else 0
            np.testing.assert_array_equal(estimates[:, 0, 1], expected)

    def test_constant_histories_have_zero_covariance(self):
        history = np.ones((3, 2))
        for kind in READOUTS:
            np.testing.assert_array_equal(readout(history, kind, 384, np.random.default_rng(3), 10), 0)
        for build in (swap_test, hadamard_test, covariance_preparation):
            qc, _ = build(history)
            self.assertTrue(np.isfinite(Statevector.from_instruction(qc).data).all())
        qc, probability = pair_product_circuit(pair_products(history, 0, 1))
        self.assertEqual(probability, 0)
        self.assertAlmostEqual(ancilla_probability(qc, 1), 0)

    def test_constant_coordinate_keeps_readout_finite(self):
        history = np.array([[1., 0.], [1., 1.], [1., 2.]])
        for kind in READOUTS:
            estimates = readout(history, kind, 384, np.random.default_rng(3), 10)
            self.assertTrue(np.isfinite(estimates).all(), kind)
        for build in (swap_test, hadamard_test):
            qc, probability = build(history)
            self.assertAlmostEqual(ancilla_probability(qc, 0), probability, places=11)

    # The number of phase states is the largest power of two that fits the oracle-call budget
    def test_amplitude_estimation_budget(self):
        for budget in (1, 2, 3, 4, 12, 64, 128, 384):
            states = phase_states(budget)
            self.assertEqual(states & (states - 1), 0)             # a power of two
            self.assertLessEqual(2 * (states - 1) + 1, budget)     # fits the budget
            self.assertGreater(2 * (2 * states - 1) + 1, budget)   # the next one would not

        # Probability 1/2 is estimated exactly with 4 phase states
        np.testing.assert_allclose(amplitude_estimates(.5, 4, np.random.default_rng(5), 100), .5)

    # The phase register of the amplitude-estimation circuit has the law's distribution
    def test_amplitude_estimation_law_matches_its_circuit(self):
        qc, probability = amplitude_estimation(pair_products(np.array([[0., 1.], [1., 0.]]), 0, 0), 2)
        self.assertAlmostEqual(probability, .5)

        phase = Statevector.from_instruction(qc).probabilities([qc.num_qubits - 2, qc.num_qubits - 1])
        np.testing.assert_allclose(phase, [0, .5, 0, .5], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
