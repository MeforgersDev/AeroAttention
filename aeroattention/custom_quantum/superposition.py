from .quantum_gate import hadamard

def apply_superposition(qubit):
    """
    Using Hadamard door for superposition.
    """
    qubit.apply_gate(hadamard())
