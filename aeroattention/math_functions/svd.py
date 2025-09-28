from __future__ import annotations

from .._compat import require_numpy


def svd(matrix):
    """Compute the Singular Value Decomposition (SVD) of ``matrix``.

    The implementation relies on :mod:`numpy`. A clear error message is emitted
    when the dependency is not available so that callers can handle the
    situation gracefully.
    """

    numpy = require_numpy("svd")

    # Compute eigenvalues and eigenvectors of A^T A
    ata = numpy.dot(matrix.T, matrix)
    eigenvalues, v = numpy.linalg.eigh(ata)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    idx = numpy.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    v = v[:, idx]

    # Compute singular values
    singular_values = numpy.sqrt(eigenvalues)

    # Compute U = A V S^{-1}
    s_inv = numpy.diag(1 / (singular_values + 1e-10))  # Avoid divide-by-zero
    u = numpy.dot(matrix, numpy.dot(v, s_inv))

    # Transpose V for compatibility
    vh = v.T

    return u, singular_values, vh