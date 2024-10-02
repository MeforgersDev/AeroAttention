import numpy as np

class Qubit:
    def __init__(self, state=None):
        """
        Initializes a qubit in a given state.

        Parameters:
        - state (np.ndarray): A 2-element complex numpy array representing the qubit state.
        """
        if state is not None:
            self.state = state
        else:
            # Default state |0>
            self.state = np.array([1, 0], dtype=complex)
    
    def apply_gate(self, gate_matrix):
        """
        Applies a quantum gate to the qubit.

        Parameters:
        - gate_matrix (np.ndarray): A 2x2 numpy array representing the quantum gate.
        """
        self.state = np.dot(gate_matrix, self.state)
    
    def measure(self):
        """
        Simulates the measurement of the qubit.

        Returns:
        - result (int): 0 or 1 based on the probability amplitudes.
        """
        probabilities = np.abs(self.state) ** 2
        result = np.random.choice([0, 1], p=probabilities)
        return result
