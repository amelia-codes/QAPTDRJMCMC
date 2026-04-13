from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
import numpy as np

# Create a circuit with a register of three qubits
# change this to actual A later

# beginning dimension
m_0 = 8 # of variables
v_0 = np.random.rand(m_0)
v_0_norm = np.linalg.norm(v_0)
n_0 = int(np.ceil(np.log2(m_0)))
print(n_0)
A = np.eye(2**(n_0*3))

qreg_prev = QuantumRegister(n_0)
qreg_next = QuantumRegister(n_0)
creg = ClassicalRegister(n_0)
qreg_a = QuantumRegister(n_0)

circ = QuantumCircuit(creg,qreg_prev,qreg_next,qreg_a)
circ.prepare_state(v_0/v_0_norm,qubits=qreg_next)
A=UnitaryGate(A, label='A')
circ.append(A, range(n_0*3))
circ.x(qreg_a)
circ.measure(qreg_a,creg)
with circ.if_test((creg,0)):
    circ.swap(qreg_prev,qreg_next)
# Draw the circuit
circ.decompose(reps=5).draw('mpl')

# understand TD vs. RJ
# draw the decomposed circuit to assess the complexity of state_preparation for large inputs (done)
# figure out the information that we are going to simulate (parameters)
