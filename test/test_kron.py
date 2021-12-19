import numpy as np
from numpy.lib.shape_base import kron
from kron_decomp import kron_decomp
from test_utils import gen_2x2_unitary
from params import CNOT, X, Z
import pytest

def test_random():
    U1 = gen_2x2_unitary()
    U2 = gen_2x2_unitary()
    check_decomp(U1, U2)


def test_X_Z():
    check_decomp(X, Z)


def test_false():
    U = CNOT
    is_tensor_prod, _, _ = kron_decomp(U)
    assert not is_tensor_prod


def check_decomp(U1, U2):
    U = np.kron(U1, U2)
    T, A, B = kron_decomp(U)
    assert T
    assert np.allclose(np.kron(A, B), U)