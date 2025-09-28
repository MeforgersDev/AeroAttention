import unittest
from typing import Any, List

import pytest

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised when NumPy is absent
    np = None  # type: ignore[assignment]
    AeroAttention = None  # type: ignore[assignment]
else:  # pragma: no cover - exercised when NumPy is present
    from aeroattention import AeroAttention

from aeroattention.utilities import block_diagonalize


def _as_list(block: Any) -> List[List[Any]]:
    if np is not None and isinstance(block, np.ndarray):
        return block.tolist()
    return [list(row) for row in block]


@pytest.mark.skipif(np is None, reason="NumPy is required for AeroAttention tests")
class TestAeroAttention(unittest.TestCase):
    def test_compute_attention(self):
        attention = AeroAttention()
        token_matrix = np.random.rand(128, 128)
        result = attention.compute_attention(token_matrix)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, token_matrix.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))


class TestBlockDiagonalize(unittest.TestCase):
    def test_block_diagonalize_square_matrix(self):
        matrix = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]
        blocks = block_diagonalize(matrix, 2)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(_as_list(blocks[0]), [[0, 1], [4, 5]])
        self.assertEqual(_as_list(blocks[1]), [[10, 11], [14, 15]])

    def test_block_diagonalize_rectangular_matrix(self):
        matrix = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
        ]
        blocks = block_diagonalize(matrix, 2)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(_as_list(blocks[0]), [[0, 1], [5, 6]])
        self.assertEqual(_as_list(blocks[1]), [[12]])

    def test_block_diagonalize_invalid_block_size(self):
        matrix = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        with self.assertRaises(ValueError):
            block_diagonalize(matrix, 0)

        with self.assertRaises(ValueError):
            block_diagonalize(matrix, 4)

    def test_block_diagonalize_complex_values(self):
        matrix = [
            [1 + 1j, 2 + 2j, 3 + 3j],
            [4 + 4j, 5 + 5j, 6 + 6j],
            [7 + 7j, 8 + 8j, 9 + 9j],
        ]
        blocks = block_diagonalize(matrix, 2)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(_as_list(blocks[0]), [[1 + 1j, 2 + 2j], [4 + 4j, 5 + 5j]])
        self.assertEqual(_as_list(blocks[1]), [[9 + 9j]])

    @pytest.mark.skipif(np is None, reason="NumPy-specific block diagonalization check")
    def test_block_diagonalize_numpy_round_trip(self):
        matrix = np.arange(16).reshape(4, 4)
        blocks = block_diagonalize(matrix, 2)

        self.assertTrue(np.array_equal(blocks[0], matrix[:2, :2]))
        self.assertTrue(np.array_equal(blocks[1], matrix[2:, 2:]))


if __name__ == "__main__":
    unittest.main()
