"""Compile a QCLT proposal from classical history into a smaller coherent circuit.

The MCMC samplers in this repository already store their history as classical
integers.  For one proposal, their pair differences are therefore known before
the circuit is built.  This circuit uses those differences as gate constants
instead of loading them into quantum history registers.  It produces exactly
the same signed step distribution, including the isotropic lattice term.

The output is still a quantum register until final measurement.  Mapping an
AMR step back to the 16-bit parameter grid and the Metropolis decision remain
classical, as they do in the existing samplers.  This is not a complete
coherent MCMC implementation or a quantum speedup.
"""

import numpy as np
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister

from qcov import add_const, lazy_coin


def output_widths(differences, isotropic_probability=0.0, pair_probability=1.0):
    """Exact signed widths for the worst step possible with these differences.

    The bound is a sum of absolute pair differences plus at most one isotropic
    lattice step.  Widths depend on the *actual* history, so they may be much
    smaller than a bound based only on input register size.
    """
    dm = np.asarray(differences, dtype=np.int64)
    if dm.ndim != 2 or dm.shape[1] == 0:
        raise ValueError("differences must have shape (pairs, parameters)")
    if not 0 <= pair_probability <= 1 or not 0 <= isotropic_probability <= 1:
        raise ValueError("coin probabilities must lie in [0, 1]")
    return [1 + (sum(abs(int(v)) for v in dm[:, p]) * (pair_probability > 0)
                 + (isotropic_probability > 0)).bit_length()
            for p in range(dm.shape[1])]


def build_compiled_kick(differences, pair_probability, isotropic_probability=0.0):
    """Return (circuit, signed output registers) for the sampler's QCLT step.

    ``differences`` is the sampler's array of pairwise history differences,
    after any periodic minimal-image conversion and AMR window mapping.  A
    nonzero pair uses a fresh move/sign coin, except at probability 1 where a
    sign coin alone suffices.  Zero pairs need no coins.  No measurement or
    reset occurs in the circuit.
    """
    dm = np.asarray(differences, dtype=np.int64)
    widths = output_widths(dm, isotropic_probability, pair_probability)
    active = [j for j, row in enumerate(dm) if pair_probability > 0 and np.any(row)]
    d = dm.shape[1]
    outputs = [QuantumRegister(width, f"step_{p}") for p, width in enumerate(widths)]
    pair_move = (AncillaRegister(len(active), "pair_move")
                 if active and pair_probability < 1 else None)
    pair_sign = AncillaRegister(len(active), "pair_sign") if active else None
    iso_move = (AncillaRegister(d, "iso_move")
                if 0 < isotropic_probability < 1 else None)
    iso_sign = AncillaRegister(d, "iso_sign") if isotropic_probability > 0 else None
    coins = [r for r in (pair_move, pair_sign, iso_move, iso_sign) if r is not None]
    qc = QuantumCircuit(*outputs, *coins)

    for i, j in enumerate(active):
        sign = pair_sign[i]
        if pair_move is None:
            qc.h(sign)
            controls = []
        else:
            lazy_coin(qc, pair_probability, pair_move[i], sign)
            controls = [pair_move[i]]
        for p, value in enumerate(dm[j]):
            value = int(value)
            if value:
                add_const(qc, outputs[p], value, controls)
                add_const(qc, outputs[p], -2 * value, [*controls, sign])

    if iso_sign is not None:
        for p in range(d):
            sign = iso_sign[p]
            if iso_move is None:
                qc.h(sign)
                controls = []
            else:
                lazy_coin(qc, isotropic_probability, iso_move[p], sign)
                controls = [iso_move[p]]
            add_const(qc, outputs[p], 1, controls)
            add_const(qc, outputs[p], -2, [*controls, sign])
    return qc, outputs
