# quantum_circuit.py

import numpy as np
from .qubit import Qubit
from .quantum_gate import hadamard, pauli_x

class QuantumCircuit:
    def __init__(self, num_qubits):
        """
        Initializes a quantum circuit with the specified number of qubits.
        """
        self.num_qubits = num_qubits
        self.qubits = [Qubit() for _ in range(num_qubits)]

    def apply_gate(self, gate, qubit_index):
        """
        Applies a quantum gate to a specific qubit in the circuit.

        Parameters:
        - gate (function): Quantum gate function to apply.
        - qubit_index (int): Index of the qubit.
        """
        self.qubits[qubit_index].apply_gate(gate())

    def entangle(self, control_index, target_index):
        """
        Creates entanglement between two qubits using a CNOT-like operation.

        Parameters:
        - control_index (int): Index of the control qubit.
        - target_index (int): Index of the target qubit.
        """
        control_qubit = self.qubits[control_index]
        target_qubit = self.qubits[target_index]

        # Apply Pauli-X gate to the target qubit if the control qubit is in state |1>
        if control_qubit.measure() == 1:
            target_qubit.apply_gate(pauli_x())

    def optimized_entangle(self, control_indices, target_indices):
        """
        Creates entanglement between multiple pairs of qubits using optimized memory access.

        Parameters:
        - control_indices (list of int): Indices of control qubits.
        - target_indices (list of int): Indices of target qubits.
        """
        for c_idx, t_idx in zip(control_indices, target_indices):
            control_qubit = self.qubits[c_idx]
            target_qubit = self.qubits[t_idx]

            # Directly apply operation without unnecessary memory access
            if control_qubit.state[1] != 0:
                target_qubit.apply_gate(pauli_x())

    def apply_tiled_operations(self, operations, tile_size):
        """
        Applies operations on the circuit using tiling to optimize memory usage.

        Parameters:
        - operations (list): List of quantum gate functions to apply.
        - tile_size (int): Size of the tile/block.
        """
        num_qubits = self.num_qubits
        for i in range(0, num_qubits, tile_size):
            # Get the qubits in the current tile
            tile_qubits = self.qubits[i:i+tile_size]
            for op in operations:
                for qubit in tile_qubits:
                    # Apply the gate operation to the qubit
                    qubit.apply_gate(op())

    def measure_all(self):
        """
        Measures all qubits in the circuit and returns their values.

        Returns:
        - results (list of int): Measurement results of the qubits.
        """
        return [qubit.measure() for qubit in self.qubits]
