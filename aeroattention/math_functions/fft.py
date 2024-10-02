import numpy as np

def fft(x):
    """
    Computes the discrete Fourier Transform of the 1D array x using the Cooley-Tukey algorithm.

    Parameters:
    - x (np.ndarray): Input array.

    Returns:
    - X (np.ndarray): The Fourier Transform of the input array.
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    
    if N <= 1:
        return x
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])

def ifft(X):
    """
    Computes the inverse discrete Fourier Transform using the Cooley-Tukey algorithm.

    Parameters:
    - X (np.ndarray): Input array in the frequency domain.

    Returns:
    - x (np.ndarray): The inverse Fourier Transform of the input array.
    """
    X_conj = np.conjugate(X)
    x = fft(X_conj)
    x = np.conjugate(x)
    x = x / X.shape[0]
    return x
