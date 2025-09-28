from __future__ import annotations

# quantum_fourier.py

from multiprocessing import Pool

from .._compat import require_numpy


def quantum_fourier_transform(token_matrix, num_threads=4):
    """Apply a basic Quantum Fourier Transform to ``token_matrix``.

    The implementation continues to rely on NumPy when available. If the
    dependency is missing, a helpful error message is raised to aid debugging.
    """

    numpy = require_numpy("quantum_fourier_transform")

    def qft_row(row):
        n = len(row)
        qft_values = numpy.zeros(n, dtype=complex)
        for k in range(n):
            phases = [row[j] * numpy.exp(2j * numpy.pi * j * k / n) for j in range(n)]
            qft_values[k] = numpy.sum(phases) / numpy.sqrt(n)
        return qft_values

    with Pool(num_threads) as pool:
        qft_matrix = pool.map(qft_row, token_matrix)
    return numpy.array(qft_matrix)
