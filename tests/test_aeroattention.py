import sys
import os
import unittest
import numpy as np
from aeroattention import AeroAttention
from aeroattention.utilities import block_diagonalize

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


class TestBlockDiagonalize(unittest.TestCase):

    def test_block_diagonalize_square_matrix(self):
        matrix = np.arange(16).reshape(4, 4)
        blocks = block_diagonalize(matrix, 2)

        self.assertEqual(len(blocks), 2)
        self.assertTrue(np.array_equal(blocks[0], matrix[:2, :2]))
        self.assertTrue(np.array_equal(blocks[1], matrix[2:, 2:]))

    def test_block_diagonalize_rectangular_matrix(self):
        matrix = np.arange(15).reshape(3, 5)
        blocks = block_diagonalize(matrix, 2)

        self.assertEqual(len(blocks), 2)
        self.assertTrue(np.array_equal(blocks[0], matrix[:2, :2]))
        self.assertTrue(np.array_equal(blocks[1], matrix[2:, 2:3]))

    def test_block_diagonalize_invalid_block_size(self):
        matrix = np.arange(9).reshape(3, 3)

        with self.assertRaises(ValueError):
            block_diagonalize(matrix, 0)

        with self.assertRaises(ValueError):
            block_diagonalize(matrix, 4)

    def test_block_diagonalize_complex_values(self):
        matrix = np.array(
            [[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j, 9 + 9j]]
        )
        blocks = block_diagonalize(matrix, 2)

        self.assertEqual(len(blocks), 2)
        self.assertTrue(np.allclose(blocks[0], matrix[:2, :2]))
        self.assertTrue(np.allclose(blocks[1], matrix[2:, 2:]))

if __name__ == '__main__':
    unittest.main()
