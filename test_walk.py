# Checks that the walk circuit does what walk.py and amr.py say it does
#
#   python3.12 -m unittest

import unittest
import runpy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

from amr import CASES, fits, step_covariance
from walk import (BITS, D, GAMMA2, ISO_VAR, MODES, N, PAIRS, UNIFORM_PAIRS, XI0, add_const, add_signed,
                  all_histories, amr_window, disagreements, lazy_coin, make_walk, pilot_histories,
                  quantum_amr_modes, quantum_amr_step_covariances, random_pair_covariance, random_pair_draws,
                  read, run_walk, sample_circuit, scaled_modes, to_grid)


# The 6-bit pilot grid, with the start and the modes rounded onto it
GRID_6 = (XI0, np.eye(D), np.full(D, .08 / 64))
START_6 = to_grid(XI0, GRID_6, 6)
MODES_6 = np.rint(MODES / GRID_6[2]).astype(int)


# Reference circuit: copy each history position into its own registers, then add every
# pair difference from those copies
def stored_history_walk(bits, start, grid_modes):
    # Registers, including N copies of the position
    position = [QuantumRegister(bits, f"position_{p}") for p in range(D)]
    history = [[QuantumRegister(bits, f"history_{t}_{p}") for p in range(D)] for t in range(N)]
    prep = QuantumRegister(N * D, "preparation_signs")
    move, sign = QuantumRegister(len(PAIRS), "pair_move"), QuantumRegister(len(PAIRS), "pair_sign")
    iso_move, iso_sign = QuantumRegister(D, "iso_move"), QuantumRegister(D, "iso_sign")
    qc = QuantumCircuit(*position, *sum(history, []), prep, move, sign, iso_move, iso_sign)

    # Start, then take each preparatory step and copy the new position
    for p in range(D):
        add_const(qc, position[p], int(start[p]))
    for t in range(N):
        for m in range(D):
            qc.h(prep[t * D + m])
            for p in range(D):
                add_signed(qc, position[p], int(grid_modes[m, p]), prep[t * D + m])

        for p in range(D):
            for qubit, copy in zip(position[p], history[t][p]):
                qc.cx(qubit, copy)

    # Add R_ab (xi^(a) - xi^(b)) bit by bit from the stored copies
    for j, (a, b) in enumerate(PAIRS):
        lazy_coin(qc, GAMMA2 / (N * (N - 1)), move[j], sign[j])
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
        history, _, _ = run_walk(6, GRID_6, 512, 4242)
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
        lattice = GAMMA2 * np.mean([np.cov(h.T) for h in histories], axis=0) + ISO_VAR * np.eye(D)
        basis = np.diag(GRID_6[2])
        np.testing.assert_allclose(step_covariance(MODES, GRID_6), basis @ lattice @ basis.T)

    # On each geometry's classical AMR window, 30,000 sampled steps match the exact covariance
    def test_sampled_steps_match_exact_covariance_on_amr_grids(self):
        for c, (name, modes, width) in enumerate(CASES):
            grid = amr_window(pilot_histories(modes, width, 1000, seed=300 + c))
            self.assertTrue(fits(modes, grid, all_histories(modes)[1]), name)

            _, step, _ = run_walk(BITS, grid, 30_000, 400 + c, modes, record_bits=14)
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
            coins = bits[[qc.find_bit(q).index for q in prep]].T.reshape(-1, N, D)
            shot_n = np.sum(coins[..., 0] != coins[..., 1], axis=1)
            np.testing.assert_array_equal(read(bits, qc, [counter])[:, 0], shot_n)

            # The final position is the start, plus the history on the shot's grid, plus the recorded step
            step = read(bits, qc, record, signed=True)
            history = np.einsum("stm,smp->sp", 1 - 2 * coins.astype(int), grid_modes[shot_n])
            np.testing.assert_array_equal(read(bits, qc, qc.qregs[:D]), (start + history + step) % (1 << BITS))

            # Steps in physical units, each on its shot's grid, match the exact covariance
            physical = step * spacing / 2.0 ** levels[shot_n]
            exact = quantum_amr_step_covariances((XI0, np.eye(D), spacing), modes).mean(axis=0)
            sampled = physical.T @ physical / len(physical)
            self.assertLess(np.linalg.norm(sampled - exact) / np.linalg.norm(exact), .05, name)

    # run_walk with quantum AMR, from the local grid (quantum AMR) and from a rotated classical AMR
    # window (hybrid AMR): every shot's grid is the base grid refined by a power of two, and
    # 30,000 sampled steps match the exact covariance
    def test_run_walk_with_quantum_and_hybrid_amr(self):
        window = amr_window(pilot_histories(MODES, .08, 1000, seed=300))
        for seed, grid in enumerate([(XI0, np.eye(D), np.full(D, .08 / 256)), window]):
            center, axes, base = grid
            _, step, spacing = run_walk(BITS, grid, 30_000, 600 + seed, record_bits=14, quantum_amr=True)
            self.assertTrue(np.all(np.isin(np.log2(base / spacing), np.arange(13))))

            physical = (step * spacing) @ axes.T
            exact = quantum_amr_step_covariances(grid).mean(axis=0)
            sampled = physical.T @ physical / len(physical)
            self.assertLess(np.linalg.norm(sampled - exact) / np.linalg.norm(exact), .05, seed)

    # The coin sampler reproduces a small circuit's exact statevector probabilities
    def test_sampler_matches_statevector(self):
        qc = QuantumCircuit(4)
        lazy_coin(qc, .3, qc.qubits[0], qc.qubits[1])
        add_const(qc, qc.qubits[2:], 1, [qc.qubits[0]])
        add_const(qc, qc.qubits[2:], -2, qc.qubits[:2])

        outcomes = (sample_circuit(qc, 50_000, 240) * (1 << np.arange(4))[:, None]).sum(axis=0)
        observed = np.bincount(outcomes, minlength=16) / len(outcomes)
        self.assertLess(np.abs(observed - Statevector.from_instruction(qc).probabilities()).sum(), .015)

    # A prepared index register and open-controlled X gates, against the statevector
    def test_sampler_handles_joint_coins_and_open_controls(self):
        qc = QuantumCircuit(5)
        qc.append(StatePreparation(UNIFORM_PAIRS), [0, 1])
        qc.h(2)
        for state in range(4):
            qc.mcx([0, 1], 3, ctrl_state=state)
        qc.mcx([0, 1, 2], 4, ctrl_state=0b101)
        qc.cx(2, 3, ctrl_state=0)

        outcomes = (sample_circuit(qc, 50_000, 241) * (1 << np.arange(5))[:, None]).sum(axis=0)
        observed = np.bincount(outcomes, minlength=32) / len(outcomes)
        self.assertLess(np.abs(observed - Statevector.from_instruction(qc).probabilities()).sum(), .015)

    # Random pairs with 1, 2 and 3 draws, with quantum AMR: every shot records exactly the step its
    # coins give, each pair index is uniform over the pairs, and the sampled covariance matches the
    # exact one
    def test_random_pair_walks(self):
        spacing = np.full(D, .08 / 256)
        levels, grid_modes = quantum_amr_modes(MODES, spacing)
        start = np.full(D, 1 << (BITS - 1))
        for seed, spec in enumerate(random_pair_draws(draws) for draws in (1, 2, 3)):
            step_modes = scaled_modes(MODES, spacing, levels, spec[2])
            qc, record, prep, _ = make_walk(BITS, start, grid_modes, record_bits=14, random_pairs=spec[:2],
                                            step_modes=step_modes)
            bits = sample_circuit(qc, 20_000, 700 + seed)
            registers = {register.name: register for register in qc.qregs}
            value = lambda name: read(bits, qc, [registers[name]])[:, 0]

            # The history each shot's step modes give, from its preparatory coins
            signs = 1 - 2 * bits[[qc.find_bit(q).index for q in prep]].T.reshape(-1, N, D).astype(int)
            g = disagreements(signs)
            history = np.cumsum(np.einsum("stm,smp->stp", signs, step_modes[g]), axis=1)

            # Each draw adds u R (history[A] - history[B]), then the isotropic step
            expected = np.zeros((len(g), D), dtype=int)
            for k in range(spec[0]):
                index = value(f"pair_index_{k}")
                self.assertLess(np.abs(np.bincount(index, minlength=4) / len(index) - UNIFORM_PAIRS**2).max(), .01)
                a, b = np.array(PAIRS)[index].T
                r = 1 - 2 * ((value("pair_sign") >> k) & 1)
                u = (value("pair_move") >> k) & 1 if spec[1] < 1 else 1
                expected += (u * r)[:, None] * (history[np.arange(len(g)), a] - history[np.arange(len(g)), b])
            for p in range(D):
                expected[:, p] += ((value("iso_move") >> p) & 1) * (1 - 2 * ((value("iso_sign") >> p) & 1))
            np.testing.assert_array_equal(read(bits, qc, record, signed=True), expected)

            # Sampled against exact covariance, in physical units on each shot's grid
            shot_spacing = spacing / 2.0 ** levels[g]
            physical = read(bits, qc, record, signed=True) * shot_spacing
            all_signs = all_histories()[0]
            exact = np.mean([np.diag(spacing / 2.0 ** levels[n]) @ random_pair_covariance(s, step_modes[n], spec)
                             @ np.diag(spacing / 2.0 ** levels[n])
                             for s, n in zip(all_signs, disagreements(all_signs))], axis=0)
            sampled = physical.T @ physical / len(physical)
            self.assertLess(np.linalg.norm(sampled - exact) / np.linalg.norm(exact), .05, spec)

    # H after H would interfere, which the sampler cannot represent, so it must refuse
    def test_sampler_rejects_interference(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        with self.assertRaisesRegex(ValueError, "fresh"):
            sample_circuit(qc, 10, 0)


if __name__ == "__main__":
    unittest.main()
