from __future__ import annotations

from .._compat import require_numpy


def hadamard():
    numpy = require_numpy("hadamard gate")
    return numpy.array([[1, 1], [1, -1]]) / numpy.sqrt(2)


def pauli_x():
    numpy = require_numpy("pauli_x gate")
    return numpy.array([[0, 1], [1, 0]])


def phase(theta):
    numpy = require_numpy("phase gate")
    return numpy.array([[1, 0], [0, numpy.exp(1j * theta)]])