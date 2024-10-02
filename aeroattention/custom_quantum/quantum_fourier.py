import numpy as np
from math_functions.fft import fft

def quantum_fourier_transform(token_matrix):
    """
    Applies Quantum Fourier Transform to the token matrix using custom FFT.

    Parameters:
    - token_matrix (np.ndarray): Input token matrix.

    Returns:
    - qft_matrix (np.ndarray): Transformed matrix.
    """
    # Apply FFT along the first axis
    qft_matrix = np.apply_along_axis(fft, axis=0, arr=token_matrix)
    qft_matrix = qft_matrix / np.sqrt(token_matrix.shape[0])
    return qft_matrix
