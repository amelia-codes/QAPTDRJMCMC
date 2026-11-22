"""Integer arithmetic, independent quantum coins, and their exact output sampler.

The walk uses fresh H/RY coins followed by basis permutations. Sampling those
coins and applying the same gates therefore gives its exact measurement law.
The sampler rejects coin reuse and gates that could introduce interference.
"""

import itertools

import numpy as np
from qiskit.circuit import ControlledGate


def add_const(circuit, register, value, controls=()):
    """Add an integer modulo 2**len(register), using little-endian qubits."""
    value %= 1 << len(register)
    for bit in range(len(register)):
        if (value >> bit) & 1:
            for target in reversed(range(bit, len(register))):
                carry = list(controls) + list(register[bit:target])
                if carry:
                    circuit.mcx(carry, register[target])
                else:
                    circuit.x(register[target])


def lazy_coin(circuit, probability, move, sign):
    """Prepare r in {-1, 0, +1}: probabilities q/2, 1-q, q/2."""
    if not 0 <= probability <= 1:
        raise ValueError("Move probability must lie in [0, 1]")
    if np.isclose(probability, 1):
        circuit.x(move)
    else:
        circuit.ry(2 * np.arcsin(np.sqrt(probability)), move)
    circuit.h(sign)


def iso_step(circuit, targets, velocities, signs, moves=None):
    """Add an independent signed coordinate step to each target register set."""
    for registers in targets:
        for p, (register, velocity) in enumerate(zip(registers, velocities)):
            controls = [] if moves is None else [moves[p]]
            add_const(circuit, register, velocity, controls)
            add_const(circuit, register, -2 * velocity, controls + [signs[p]])


def check_coherent(circuit):
    """Allow only the walk's primitive gates, with measurements at the end."""
    measurements = 0
    allowed = {"h", "ry", "x", "cx", "ccx", "mcx", "swap", "cswap"}
    for instruction in circuit.data:
        gate = instruction.operation
        if gate.name == "measure":
            measurements += 1
        elif gate.name != "barrier":
            controlled_x = isinstance(gate, ControlledGate) and gate.base_gate.name == "x"
            if measurements or (gate.name not in allowed and not controlled_x):
                raise ValueError(f"Unsupported coherent-walk gate: {gate.name}")
            if getattr(gate, "condition", None) is not None:
                raise ValueError("Classically conditioned gates are not supported")
    return measurements


def sample_circuit(circuit, shots, seed=None):
    """Evaluate the actual walk circuit; return bits with shape (qubits, shots)."""
    check_coherent(circuit)
    rng = np.random.default_rng(seed)
    bits = np.zeros((circuit.num_qubits, shots), dtype=bool)
    touched = np.zeros(circuit.num_qubits, dtype=bool)
    coin = np.zeros(circuit.num_qubits, dtype=bool)
    for instruction in circuit.data:
        gate = instruction.operation
        name = gate.name
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if name in ("barrier", "measure"):
            continue
        if name in ("h", "ry"):
            target = qubits[0]
            if touched[target]:
                raise ValueError("A random coin must be a fresh zero qubit")
            probability = .5 if name == "h" else np.sin(float(gate.params[0]) / 2) ** 2
            bits[target] = rng.random(shots) < probability
            coin[target] = touched[target] = True
        elif name in ("swap", "cswap"):
            a, b = qubits[-2:]
            if coin[a] or coin[b]:
                raise ValueError("A prepared coin may only be used as a control")
            difference = bits[a] ^ bits[b]
            if name == "cswap":
                difference &= bits[qubits[0]]
            bits[a] ^= difference
            bits[b] ^= difference
            touched[a] = touched[b] = True
        else:
            if isinstance(gate, ControlledGate):
                if gate.ctrl_state != (1 << gate.num_ctrl_qubits) - 1:
                    raise ValueError("Open controls are not supported by the walk sampler")
                if len(qubits) != gate.num_ctrl_qubits + 1:
                    raise ValueError("Controlled X must not use extra workspace")
            *controls, target = qubits
            if coin[target]:
                raise ValueError("A prepared coin may only be used as a control")
            bits[target] ^= np.all(bits[controls], axis=0) if controls else True
            touched[target] = True
    return bits


def regs_values(bits, circuit, registers, signed=False):
    """Decode little-endian registers into a (shots, coordinates) array."""
    values = []
    for register in registers:
        indices = [circuit.find_bit(q).index for q in register]
        value = (bits[indices] * (1 << np.arange(len(register)))[:, None]).sum(axis=0)
        if signed:
            value = np.where(value >= 1 << (len(register) - 1), value - (1 << len(register)), value)
        values.append(value)
    return np.stack(values, axis=1)


def torus_cov(history, bits):
    """Sample covariance from shortest signed history differences on the grid."""
    k = history.shape[-2]
    covariance = 0
    for a, b in itertools.combinations(range(k), 2):
        difference = (history[..., a, :] - history[..., b, :]) % (1 << bits)
        difference = np.where(difference >= 1 << (bits - 1), difference - (1 << bits), difference)
        covariance += difference[..., :, None] * difference[..., None, :]
    return covariance / (k * (k - 1))
