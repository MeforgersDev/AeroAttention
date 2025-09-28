from __future__ import annotations

# custom_quantum/qaoa.py

from .._compat import require_numpy


def qaoa_layer(gamma, beta, hamiltonian, state_vector):
    """Apply one layer of the Quantum Approximate Optimization Algorithm."""

    numpy = require_numpy("qaoa_layer")

    phase_operator = numpy.diag(numpy.exp(-1j * gamma * numpy.diag(hamiltonian)))
    state_vector = numpy.dot(phase_operator, state_vector)

    mixing_operator = numpy.full_like(state_vector, numpy.cos(beta)) + 1j * numpy.sin(beta)
    state_vector = state_vector * mixing_operator

    return state_vector
