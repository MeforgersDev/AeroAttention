import unittest

import pytest

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed when NumPy missing
    np = None  # type: ignore[assignment]
    fft = None  # type: ignore[assignment]
    ifft = None  # type: ignore[assignment]
    svd = None  # type: ignore[assignment]
else:  # pragma: no cover - executed when NumPy available
    from aeroattention.math_functions.fft import fft, ifft
    from aeroattention.math_functions.svd import svd


@pytest.mark.skipif(np is None, reason="NumPy is required for math function tests")
class TestMathFunctions(unittest.TestCase):
    def test_fft(self):
        x = np.random.rand(16)
        X = fft(x)
        X_numpy = np.fft.fft(x)
        self.assertTrue(np.allclose(X, X_numpy))

    def test_ifft(self):
        X = np.random.rand(16) + 1j * np.random.rand(16)
        x = ifft(X)
        x_numpy = np.fft.ifft(X)
        self.assertTrue(np.allclose(x, x_numpy))

    def test_svd(self):
        matrix = np.random.rand(8, 5)
        U, S, Vh = svd(matrix)
        reconstructed = np.dot(U, np.dot(np.diag(S), Vh))
        self.assertTrue(np.allclose(matrix, reconstructed))


if __name__ == '__main__':
    unittest.main()
