from __future__ import annotations

from ._compat import require_numpy


def entropy_sparse_filter(attention_matrix, threshold=0.1):
    """Filter out components with low entropy."""

    numpy = require_numpy("entropy_sparse_filter")
    epsilon = 1e-9
    probabilities = numpy.abs(attention_matrix) + epsilon
    probabilities /= numpy.sum(probabilities, axis=-1, keepdims=True)

    entropy = -numpy.sum(probabilities * numpy.log(probabilities), axis=-1)
    entropy = entropy / numpy.log(probabilities.shape[-1])

    mask = (entropy > threshold).astype(float)
    sparse_matrix = attention_matrix * mask[:, numpy.newaxis]
    return sparse_matrix
