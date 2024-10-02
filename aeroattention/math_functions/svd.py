import numpy as np

def svd(matrix):
    """
    Computes the Singular Value Decomposition (SVD) of a matrix.

    Parameters:
    - matrix (np.ndarray): The input matrix to decompose.

    Returns:
    - U (np.ndarray): Left singular vectors.
    - S (np.ndarray): Singular values (as a 1D array).
    - Vh (np.ndarray): Right singular vectors (transposed).
    """
    # Compute eigenvalues and eigenvectors of A^T A
    ATA = np.dot(matrix.T, matrix)
    eigenvalues, V = np.linalg.eigh(ATA)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    # Compute singular values
    singular_values = np.sqrt(eigenvalues)

    # Compute U = A V S^{-1}
    S_inv = np.diag(1 / (singular_values + 1e-10))  # Add epsilon to avoid division by zero
    U = np.dot(matrix, np.dot(V, S_inv))

    # Transpose V for compatibility
    Vh = V.T

    return U, singular_values, Vh
