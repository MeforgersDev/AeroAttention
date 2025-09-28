from __future__ import annotations

# utilities.py
from typing import List, Sequence, Tuple, TypeVar

from ._compat import has_numpy, is_numpy_array, np, require_numpy
from .math_functions.svd import svd

def compress_matrix(matrix, compression_level):
    """
    Compresses the input matrix using low-rank approximation with custom SVD.

    Parameters:
    - matrix (np.ndarray): The input matrix.
    - compression_level (float): Compression ratio (0 to 1).

    Returns:
    - compressed_matrix (np.ndarray): Compressed matrix.
    """
    numpy = require_numpy("compress_matrix")

    # Assuming existing implementation of custom SVD
    U, S, Vh = svd(matrix)
    total_energy = numpy.sum(S ** 2)
    energy = 0
    k = 0
    while energy / total_energy < compression_level and k < len(S):
        energy += S[k] ** 2
        k += 1
    # Use the top k singular values and vectors
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :]
    compressed_matrix = numpy.dot(U_k, numpy.dot(numpy.diag(S_k), Vh_k))
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
    numpy = require_numpy("strassen_matrix_multiply")

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
        top = numpy.hstack((C11, C12))
        bottom = numpy.hstack((C21, C22))
        C = numpy.vstack((top, bottom))
        return C


T = TypeVar("T")


def _coerce_sequence(matrix: Sequence[Sequence[T]]) -> Tuple[List[List[T]], int, int]:
    """Validate and normalise a sequence-based matrix representation."""

    if not isinstance(matrix, Sequence):
        raise TypeError("matrix must be a 2D sequence when NumPy is unavailable")

    rows: List[List[T]] = []
    expected_cols: int | None = None
    for row in matrix:
        if not isinstance(row, Sequence):
            raise ValueError("matrix must be a sequence of sequences")
        current_row = list(row)
        if expected_cols is None:
            expected_cols = len(current_row)
            if expected_cols == 0:
                raise ValueError("matrix must have non-zero dimensions")
        elif len(current_row) != expected_cols:
            raise ValueError("matrix rows must all be the same length")
        rows.append(current_row)

    if expected_cols is None:
        raise ValueError("matrix must have non-zero dimensions")

    return rows, len(rows), expected_cols


def block_diagonalize(matrix, block_size: int) -> List[object]:
    """Split a matrix into consecutive square blocks along its diagonal.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix to be partitioned.
    block_size : int
        Desired size for each square block.

    Returns
    -------
    List[np.ndarray]
        A list of square ``numpy`` arrays representing the diagonal blocks.

    Raises
    ------
    ValueError
        If ``matrix`` is not two-dimensional, ``block_size`` is not positive,
        ``block_size`` exceeds the smallest matrix dimension, or the matrix is
        empty.
    """

    is_numpy_input = is_numpy_array(matrix)
    if is_numpy_input:
        assert np is not None  # for mypy
        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        rows, cols = matrix.shape
        if rows == 0 or cols == 0:
            raise ValueError("matrix must have non-zero dimensions")
        source = matrix
    else:
        if has_numpy() and isinstance(matrix, np.ndarray):  # type: ignore[attr-defined]
            # NumPy is available but the array type was not recognised (unlikely).
            rows, cols = matrix.shape
            source = matrix
        else:
            source, rows, cols = _coerce_sequence(matrix)

    if block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    min_dim = min(rows, cols)
    if block_size > min_dim:
        raise ValueError(
            "block_size cannot be larger than the smallest matrix dimension"
        )

    blocks: List[object] = []
    row_start = 0
    col_start = 0

    while row_start < rows and col_start < cols:
        remaining_rows = rows - row_start
        remaining_cols = cols - col_start

        current_block_size = block_size
        if remaining_rows < block_size or remaining_cols < block_size:
            current_block_size = min(remaining_rows, remaining_cols)

        if current_block_size == 0:
            break

        if is_numpy_input:
            block = matrix[
                row_start : row_start + current_block_size,
                col_start : col_start + current_block_size,
            ]
            blocks.append(block)
        else:
            # ``source`` is a sequence of rows at this point.
            block_rows: List[List[object]] = []
            for idx in range(row_start, row_start + current_block_size):
                row_slice = source[idx][col_start : col_start + current_block_size]
                block_rows.append(list(row_slice))
            if has_numpy():
                blocks.append(np.array(block_rows))  # type: ignore[arg-type]
            else:
                blocks.append(block_rows)

        row_start += current_block_size
        col_start += current_block_size

    return blocks
