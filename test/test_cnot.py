from cnot_decomp import cnot_decomp, cnot_reconstruct, remove_global_phase
from test_utils import gen_4x4_unitary
import numpy as np
import pytest


def test_decomp_random():
    U = gen_4x4_unitary()
    output = cnot_decomp(U)
    recons = cnot_reconstruct(*output)
    assert is_equal_wo_phase(U, recons)


def is_equal_wo_phase(U, V):
    U1 = remove_global_phase(U)
    V1 = remove_global_phase(V)
    return np.allclose(U1, V1)