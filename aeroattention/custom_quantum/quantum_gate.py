import numpy as np

def hadamard():
    """
    Returns the Hadamard gate matrix.
    """
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def pauli_x():
    """
    Returns the Pauli-X (NOT) gate matrix.
    """
    return np.array([[0, 1], [1, 0]])

def phase(theta):
    """
    Returns the phase gate matrix.
    
    Parameters:
    - theta (float): The phase angle in radians.
    """
    return np.array([[1, 0], [0, np.exp(1j * theta)]])
