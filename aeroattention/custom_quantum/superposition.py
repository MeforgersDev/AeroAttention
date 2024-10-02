from .quantum_gate import hadamard

def apply_superposition(qubit):
    """
    Hadamard kapısı ile qubit'e süperpozisyon uygular.
    """
    qubit.apply_gate(hadamard())
