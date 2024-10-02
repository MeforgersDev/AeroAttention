import numpy as np
from math_functions.svd import svd

def compress_matrix(matrix, compression_level):
    """
    Compresses the input matrix using low-rank approximation with custom SVD.

    Parameters:
    - matrix (np.ndarray): The input matrix.
    - compression_level (float): Compression ratio (0 to 1).

    Returns:
    - compressed_matrix (np.ndarray): Compressed matrix.
    """
    U, S, Vh = svd(matrix)
    total_energy = np.sum(S ** 2)
    energy = 0
    k = 0
    while energy / total_energy < compression_level and k < len(S):
        energy += S[k] ** 2
        k += 1
    # Use the top k singular values and vectors
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :]
    compressed_matrix = np.dot(U_k, np.dot(np.diag(S_k), Vh_k))
    return compressed_matrix

def block_diagonalize(matrix, block_size):
    """
    Divides the matrix into block diagonal components.

    Parameters:
    - matrix (np.ndarray): The input matrix.
    - block_size (int): Size of each block.

    Returns:
    - blocks (list of np.ndarray): List of block matrices.
    """
    n = matrix.shape[0]
    blocks = []
    for i in range(0, n, block_size):
        block = matrix[i:i+block_size, i:i+block_size]
        blocks.append(block)
    return blocks
