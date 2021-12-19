from cnot_decomp import cnot_decomp
from kron_decomp import kron_decomp
from params import CZ
from swap_decomp import swap_decomp, is_swap
from ctrl_u_decomp import ctrl_u_decomp, is_ctrl_u
import numpy as np
from utils import is_unitary


def check_input(U):
    """
    U should be a 4x4 unitary matrix
    """
    if not isinstance(U, np.ndarray):
        raise ValueError("U should be a numpy ndarray")
    if not U.shape == (4, 4):
        raise ValueError(f"The shape of input matrix must be (4, 4). Got {U.shape}.")
    if not is_unitary(U, 4):
        raise ValueError("The input matrix must be a unitary matrix.")


def gate_decomp(U):
    """
    :param U: ndarray of two qubit gates
    :return: a list of single qubit gates and CNOT gates
    """
    check_input(U)
    is_tensor_prod, A, B = kron_decomp(U)
    if is_tensor_prod:
        return A, B
    if is_swap:
        return swap_decomp(U)
    if is_ctrl_u is not None:
        return ctrl_u_decomp(U)
    else:
        return cnot_decomp(U)


if __name__ == "__main__":
    U = CZ
    result = gate_decomp(U)
