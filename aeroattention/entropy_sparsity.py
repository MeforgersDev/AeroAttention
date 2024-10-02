import numpy as np

def entropy_sparse_filter(attention_matrix, threshold=0.1):
    """
    Filters out components with low entropy.

    Parameters:
    - attention_matrix (np.ndarray): Input attention matrix.
    - threshold (float): Entropy threshold.

    Returns:
    - sparse_matrix (np.ndarray): Sparsified attention matrix.
    """
    # Calculate probabilities
    epsilon = 1e-9
    probabilities = np.abs(attention_matrix) + epsilon
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)

    # Compute entropy
    entropy = -np.sum(probabilities * np.log(probabilities), axis=-1)
    entropy = entropy / np.log(probabilities.shape[-1])  # Normalize entropy

    # Sparsify the matrix
    mask = (entropy > threshold).astype(float)
    sparse_matrix = attention_matrix * mask[:, np.newaxis]
    return sparse_matrix
