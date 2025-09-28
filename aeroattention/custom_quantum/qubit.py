from __future__ import annotations

from .._compat import require_numpy


class Qubit:
    def __init__(self, state=None):
        """Initialise a qubit with ``state`` or the default ``|0>`` state."""

        numpy = require_numpy("Qubit")
        if state is not None:
            self.state = state
        else:
            self.state = numpy.array([1, 0], dtype=complex)

    def apply_gate(self, gate_matrix):
        """Apply ``gate_matrix`` to the qubit state."""

        numpy = require_numpy("Qubit.apply_gate")
        self.state = numpy.dot(gate_matrix, self.state)

    def measure(self):
        """Simulate measurement of the qubit."""

        numpy = require_numpy("Qubit.measure")
        probabilities = numpy.abs(self.state) ** 2
        result = numpy.random.choice([0, 1], p=probabilities)
        return result