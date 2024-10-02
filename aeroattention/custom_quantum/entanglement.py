import numpy as np

def create_entanglement(matrix):
    """
    Simulates entanglement by transposing the matrix.

    Parameters:
    - matrix (np.ndarray): Input matrix.

    Returns:
    - entangled_matrix (np.ndarray): Entangled matrix.
    """
    # Simple simulation by transposing the matrix
    entangled_matrix = np.transpose(matrix)
    return entangled_matrix
