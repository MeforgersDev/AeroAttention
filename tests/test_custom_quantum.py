import unittest
try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

from aeroattention.custom_quantum.qubit import Qubit
from aeroattention.custom_quantum.quantum_gate import hadamard
from aeroattention.custom_quantum.quantum_fourier import quantum_fourier_transform

@unittest.skipUnless(np is not None, "NumPy is required for quantum tests")
class TestCustomQuantum(unittest.TestCase):
    
    def test_qubit_initialization(self):
        q = Qubit()
        self.assertTrue(np.array_equal(q.state, np.array([1, 0], dtype=complex)))
    
    def test_hadamard_gate(self):
        q = Qubit()
        q.apply_gate(hadamard())
        expected_state = np.array([1, 1], dtype=complex) / np.sqrt(2)
        self.assertTrue(np.allclose(q.state, expected_state))
    
    def test_quantum_fourier_transform(self):
        token_matrix = np.random.rand(8, 8)
        qft_result = quantum_fourier_transform(token_matrix)
        self.assertEqual(qft_result.shape, token_matrix.shape)

if __name__ == '__main__':
    unittest.main()