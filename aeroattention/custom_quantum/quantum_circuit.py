from __future__ import annotations

# quantum_circuit.py

from .qubit import Qubit
from .quantum_gate import pauli_x


class QuantumCircuit:
    def __init__(self, num_qubits):
        """Initialise a quantum circuit with ``num_qubits`` qubits."""

        self.num_qubits = num_qubits
        self.qubits = [Qubit() for _ in range(num_qubits)]

    def apply_gate(self, gate, qubit_index):
        """Apply a quantum ``gate`` to ``qubit_index``."""

        self.qubits[qubit_index].apply_gate(gate())

    def entangle(self, control_index, target_index):
        """Create entanglement between two qubits using a CNOT-like operation."""

        control_qubit = self.qubits[control_index]
        target_qubit = self.qubits[target_index]

        if control_qubit.measure() == 1:
            target_qubit.apply_gate(pauli_x())

    def optimized_entangle(self, control_indices, target_indices):
        """Create entanglement between multiple qubit pairs."""

        for c_idx, t_idx in zip(control_indices, target_indices):
            control_qubit = self.qubits[c_idx]
            target_qubit = self.qubits[t_idx]

            if control_qubit.state[1] != 0:
                target_qubit.apply_gate(pauli_x())

    def apply_tiled_operations(self, operations, tile_size):
        """Apply gate ``operations`` across the circuit using tiling."""

        for i in range(0, self.num_qubits, tile_size):
            tile_qubits = self.qubits[i:i + tile_size]
            for op in operations:
                for qubit in tile_qubits:
                    qubit.apply_gate(op())

    def measure_all(self):
        """Measure all qubits in the circuit."""

        return [qubit.measure() for qubit in self.qubits]
