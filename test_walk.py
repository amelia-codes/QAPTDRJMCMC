# Checks that the walk circuit does what walk.py and amr.py say it does
#
#   python3.12 -m unittest

import unittest
import runpy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from amr import CASES, fits, step_covariance
from walk import (BITS, D, ISO_VAR, K, MODES, PAIRS, SCALE2, X0, add_const, add_signed, all_histories,
                  amr_window, lazy_coin, make_walk, pilot_histories, quantum_amr_modes,
                  quantum_amr_step_covariance, read, run_walk, sample_circuit, to_grid)


# The 6-bit pilot grid, with the start and the modes rounded onto it
GRID_6 = (X0, np.eye(D), np.full(D, .08 / 64))
START_6 = to_grid(X0, GRID_6, 6)
MODES_6 = np.rint(MODES / GRID_6[2]).astype(int)


# Reference circuit: copy each history position into its own registers, then add every
# pair difference from those copies
def stored_history_walk(bits, start, grid_modes):
    # Registers, including K copies of the position
    position = [QuantumRegister(bits, f"position_{p}") for p in range(D)]
    history = [[QuantumRegister(bits, f"history_{t}_{p}") for p in range(D)] for t in range(K)]
    prep = QuantumRegister(K * D, "preparation_signs")
    move, sign = QuantumRegister(len(PAIRS), "pair_move"), QuantumRegister(len(PAIRS), "pair_sign")
    iso_move, iso_sign = QuantumRegister(D, "iso_move"), QuantumRegister(D, "iso_sign")
    qc = QuantumCircuit(*position, *sum(history, []), prep, move, sign, iso_move, iso_sign)

    # Start, then take each preparatory step and copy the new position
    for p in range(D):
        add_const(qc, position[p], int(start[p]))
    for t in range(K):
        for m in range(D):
            qc.h(prep[t * D + m])
            for p in range(D):
                add_signed(qc, position[p], int(grid_modes[m, p]), prep[t * D + m])

        for p in range(D):
            for qubit, copy in zip(position[p], history[t][p]):
                qc.cx(qubit, copy)

    # Add r_ab (h_a - h_b) bit by bit from the stored copies
    for j, (a, b) in enumerate(PAIRS):
        lazy_coin(qc, SCALE2 / (K * (K - 1)), move[j], sign[j])
        for p in range(D):
            for bit in range(bits):
                add_signed(qc, position[p], 1 << bit, sign[j], [move[j], history[a][p][bit]])
                add_signed(qc, position[p], -(1 << bit), sign[j], [move[j], history[b][p][bit]])

    # The extra +-1 step
    for p in range(D):
        lazy_coin(qc, ISO_VAR, iso_move[p], iso_sign[p])
    for p in range(D):
        add_signed(qc, position[p], 1, iso_sign[p], [iso_move[p]])

    return qc, position, history


class WalkTests(unittest.TestCase):
    def test_importing_entry_point_does_not_run_experiments(self):
        with patch("covariance_methods.run", side_effect=AssertionError("experiment ran on import")):
            runpy.run_path(str(Path(__file__).parent / "generate_paper_figures.py"))

    # Same coins, same positions and histories as the reference that stores every history position
    def test_compressed_walk_matches_stored_history_walk(self):
        compressed, record, _, _ = make_walk(6, START_6, MODES_6, record_bits=8)
        reference, position, stored = stored_history_walk(6, START_6, MODES_6)
        self.assertEqual(reference.num_qubits, 64)

        # Final positions agree
        a, b = sample_circuit(compressed, 512, 4242), sample_circuit(reference, 512, 4242)
        final = read(a, compressed, compressed.qregs[:D])
        np.testing.assert_array_equal(final, read(b, reference, position))

        # Decoded histories agree with the stored copies
        history, _ = run_walk(6, GRID_6, 512, 4242)
        np.testing.assert_array_equal(history, np.stack([read(b, reference, row) for row in stored], axis=1))

        # The record holds exactly the step that the position register took
        np.testing.assert_array_equal(final, (history[:, -1] + read(a, compressed, record, signed=True)) % 64)

    # 6-bit and 8-bit walks with 1 to 4 coin-sum steps, without record registers
    def test_qubit_counts(self):
        for bits, expected in [(6, [29, 39, 49, 59]), (8, [33, 43, 53, 63])]:
            for steps, qubits in enumerate(expected, start=1):
                self.assertEqual(make_walk(bits, START_6, MODES_6, steps)[0].num_qubits, qubits)
        self.assertEqual(make_walk(8, START_6, MODES_6, steps=16, record_bits=14)[0].num_qubits, 211)

        # Quantum AMR adds only its 2-qubit counter
        quantum_modes = quantum_amr_modes(MODES, np.full(D, .08 / 256))[1]
        self.assertEqual(make_walk(8, START_6, quantum_modes)[0].num_qubits, 35)

    # amr.step_covariance equals the average over the 64 possible histories
    def test_exact_covariance_is_the_average_over_all_histories(self):
        signs, _ = all_histories()
        histories = np.cumsum(signs @ MODES_6, axis=1)
        lattice = SCALE2 * np.mean([np.cov(h.T) for h in histories], axis=0) + ISO_VAR * np.eye(D)
        basis = np.diag(GRID_6[2])
        np.testing.assert_allclose(step_covariance(MODES, GRID_6), basis @ lattice @ basis.T)

    # On each geometry's AMR grid, 30,000 sampled steps match the exact covariance
    def test_sampled_steps_match_exact_covariance_on_amr_grids(self):
        for c, (name, modes, width) in enumerate(CASES):
            grid = amr_window(pilot_histories(modes, width, 1000, seed=300 + c))
            self.assertTrue(fits(modes, grid, all_histories(modes)[1]), name)

            _, step = run_walk(BITS, grid, 30_000, 400 + c, modes, record_bits=14)
            center, axes, spacing = grid
            physical = (step * spacing) @ axes.T

            exact = step_covariance(modes, grid)
            sampled = physical.T @ physical / len(physical)
            self.assertLess(np.linalg.norm(sampled - exact) / np.linalg.norm(exact), .05, name)

    # Quantum AMR on each geometry: every branch fits its own grid, the circuit's counter and
    # positions are what the branch's grid says, and 30,000 sampled steps match the exact covariance
    def test_quantum_amr_walk(self):
        for c, (name, modes, width) in enumerate(CASES):
            spacing = np.full(D, width / (1 << BITS))
            levels, grid_modes = quantum_amr_modes(modes, spacing)
            start = np.full(D, 1 << (BITS - 1))
            qc, record, prep, counter = make_walk(BITS, start, grid_modes, record_bits=14)

            # Every branch's history, and every pair difference, stays within half the register
            signs, _ = all_histories(modes)
            n = np.sum(signs[..., 0] != signs[..., 1], axis=1)
            histories = np.cumsum(np.einsum("btm,bmp->btp", signs, grid_modes[n]), axis=1)
            differences = histories[:, :, None] - histories[:, None, :]
            self.assertLess(max(np.abs(histories).max(), np.abs(differences).max()), 1 << (BITS - 1), name)

            # The counter holds each shot's number of disagreeing coin pairs
            bits = sample_circuit(qc, 30_000, 500 + c)
            coins = bits[[qc.find_bit(q).index for q in prep]].T.reshape(-1, K, D)
            shot_n = np.sum(coins[..., 0] != coins[..., 1], axis=1)
            np.testing.assert_array_equal(read(bits, qc, [counter])[:, 0], shot_n)

            # The final position is the start, plus the history on the shot's grid, plus the recorded step
            step = read(bits, qc, record, signed=True)
            history = np.einsum("stm,smp->sp", 1 - 2 * coins.astype(int), grid_modes[shot_n])
            np.testing.assert_array_equal(read(bits, qc, qc.qregs[:D]), (start + history + step) % (1 << BITS))

            # Steps in physical units, each on its shot's grid, match the exact covariance
            physical = step * spacing / 2.0 ** levels[shot_n]
            exact = np.mean([quantum_amr_step_covariance(s, modes, spacing) for s in signs], axis=0)
            sampled = physical.T @ physical / len(physical)
            self.assertLess(np.linalg.norm(sampled - exact) / np.linalg.norm(exact), .05, name)

    # The coin sampler reproduces a small circuit's exact statevector probabilities
    def test_sampler_matches_statevector(self):
        qc = QuantumCircuit(4)
        lazy_coin(qc, .3, qc.qubits[0], qc.qubits[1])
        add_const(qc, qc.qubits[2:], 1, [qc.qubits[0]])
        add_const(qc, qc.qubits[2:], -2, qc.qubits[:2])

        outcomes = (sample_circuit(qc, 50_000, 240) * (1 << np.arange(4))[:, None]).sum(axis=0)
        observed = np.bincount(outcomes, minlength=16) / len(outcomes)
        self.assertLess(np.abs(observed - Statevector.from_instruction(qc).probabilities()).sum(), .015)

    # H after H would interfere, which the sampler cannot represent, so it must refuse
    def test_sampler_rejects_interference(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        with self.assertRaisesRegex(ValueError, "fresh"):
            sample_circuit(qc, 10, 0)


if __name__ == "__main__":
    unittest.main()
