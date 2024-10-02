import numpy as np

def classical_attention(matrix):
    """
    Applies classical attention computation using softmax.

    Parameters:
    - matrix (np.ndarray): Input matrix.

    Returns:
    - attention_weights (np.ndarray): Attention weights matrix.
    """
    # Compute softmax along the last axis
    exp_matrix = np.exp(matrix - np.max(matrix, axis=-1, keepdims=True))
    attention_weights = exp_matrix / np.sum(exp_matrix, axis=-1, keepdims=True)
    return attention_weights
