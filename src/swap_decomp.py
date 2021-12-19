import numpy as np
from params import CNOT, CNOT_P, SWAP 


def is_swap(U):
    return np.allclose(SWAP, U)


def swap_decomp(U):
    return [CNOT, CNOT_P, CNOT]