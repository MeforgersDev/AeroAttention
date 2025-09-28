from __future__ import annotations

# entanglement.py

from .._compat import require_numpy


def create_entanglement(matrix):
    """Simulate entanglement by transposing ``matrix``."""

    numpy = require_numpy("create_entanglement")
    return numpy.transpose(matrix)
