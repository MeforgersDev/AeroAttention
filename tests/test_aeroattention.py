import sys
import os
import unittest
import numpy as np
from aeroattention import AeroAttention

class TestAeroAttention(unittest.TestCase):
    
    def test_compute_attention(self):
        attention = AeroAttention()
        token_matrix = np.random.rand(128, 128)
        result = attention.compute_attention(token_matrix)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, token_matrix.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
    
    def test_compression_level(self):
        attention = AeroAttention(compression_level=0.5)
        real_matrix = np.random.rand(8, 8).astype(np.float32)
        compressed_real = attention.dynamic_compression(real_matrix)
        self.assertEqual(compressed_real.shape, real_matrix.shape)
        self.assertFalse(np.iscomplexobj(compressed_real))
        self.assertEqual(compressed_real.dtype, real_matrix.dtype)

        complex_matrix = (np.random.rand(6, 6) + 1j * np.random.rand(6, 6)).astype(np.complex128)
        compressed_complex = attention.dynamic_compression(complex_matrix)
        self.assertEqual(compressed_complex.shape, complex_matrix.shape)
        self.assertTrue(np.iscomplexobj(compressed_complex))
        self.assertEqual(compressed_complex.dtype, complex_matrix.dtype)

        zero_compression = AeroAttention(compression_level=0.0)
        zero_result = zero_compression.dynamic_compression(real_matrix)
        self.assertEqual(zero_result.shape, real_matrix.shape)
        self.assertTrue(np.allclose(zero_result, 0, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
