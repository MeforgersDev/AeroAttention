# utilities.py
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
    # Assuming existing implementation of custom SVD
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

def adaptive_factorization(matrix, compression_level):
    """
    Applies adaptive factorization to the matrix.

    Parameters:
    - matrix (np.ndarray): The input matrix.
    - compression_level (float): Compression ratio.

    Returns:
    - factored_matrix (np.ndarray): Factorized matrix.
    """
    # Assuming existing implementation
    pass

def strassen_matrix_multiply(A, B):
    """
    Multiplies two matrices using the Strassen algorithm.

    Parameters:
    - A (np.ndarray): Matrix A.
    - B (np.ndarray): Matrix B.

    Returns:
    - C (np.ndarray): Resulting matrix C = A * B.
    """
    # Ensure matrices are square and have dimensions that are powers of two
    assert A.shape == B.shape
    n = A.shape[0]
    if n == 1:
        # Base case: single element multiplication
        return A * B
    else:
        # Divide the matrices into quadrants
        mid = n // 2

        A11 = A[:mid, :mid]
        A12 = A[:mid, mid:]
        A21 = A[mid:, :mid]
        A22 = A[mid:, mid:]

        B11 = B[:mid, :mid]
        B12 = B[:mid, mid:]
        B21 = B[mid:, :mid]
        B22 = B[mid:, mid:]

        # Compute the 7 products using Strassen's formulas
        M1 = strassen_matrix_multiply(A11 + A22, B11 + B22)
        M2 = strassen_matrix_multiply(A21 + A22, B11)
        M3 = strassen_matrix_multiply(A11, B12 - B22)
        M4 = strassen_matrix_multiply(A22, B21 - B11)
        M5 = strassen_matrix_multiply(A11 + A12, B22)
        M6 = strassen_matrix_multiply(A21 - A11, B11 + B12)
        M7 = strassen_matrix_multiply(A12 - A22, B21 + B22)

        # Combine the products to get the final result
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6

        # Combine quadrants into a full matrix
        top = np.hstack((C11, C12))
        bottom = np.hstack((C21, C22))
        C = np.vstack((top, bottom))
        return C
