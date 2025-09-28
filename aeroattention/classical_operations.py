from __future__ import annotations

from ._compat import require_numpy


def classical_attention(matrix):
    """Apply a softmax operation to ``matrix`` along the last axis."""

    numpy = require_numpy("classical_attention")
    exp_matrix = numpy.exp(matrix - numpy.max(matrix, axis=-1, keepdims=True))
    attention_weights = exp_matrix / numpy.sum(exp_matrix, axis=-1, keepdims=True)
    return attention_weights
