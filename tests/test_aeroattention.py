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
        token_matrix = np.random.rand(8, 8)
        compressed = attention.dynamic_compression(token_matrix)
        self.assertEqual(compressed.shape, token_matrix.shape)

if __name__ == '__main__':
    unittest.main()
