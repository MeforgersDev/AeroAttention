from .qubit import Qubit
from .quantum_gate import hadamard, pauli_x

class QuantumCircuit:
    def __init__(self, num_qubits):
        """
        Initializes a quantum circuit with the specified number of qubits.

        Parameters:
        - num_qubits (int): Number of qubits in the circuit.
        """
        self.qubits = [Qubit() for _ in range(num_qubits)]

    def apply_gate(self, gate, qubit_index):
        """
        Applies a gate to a specific qubit in the circuit.

        Parameters:
        - gate (function): The quantum gate function.
        - qubit_index (int): Index of the qubit to apply the gate to.
        """
        self.qubits[qubit_index].apply_gate(gate())

    def entangle(self, control_index, target_index):
        """
        Creates an entanglement between two qubits using a CNOT-like operation.

        Parameters:
        - control_index (int): Index of the control qubit.
        - target_index (int): Index of the target qubit.
        """
        control_qubit = self.qubits[control_index]
        target_qubit = self.qubits[target_index]
        
        if control_qubit.measure() == 1:
            target_qubit.apply_gate(pauli_x())

    def measure_all(self):
        """
        Measures all qubits in the circuit and returns their values.

        Returns:
        - results (list of int): Measurement results of the qubits.
        """
        return [qubit.measure() for qubit in self.qubits]
