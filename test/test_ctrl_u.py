import numpy as np
from ctrl_u_decomp import ctrl_u_decomp, is_ctrl_u
from params import CNOT, CZ, I
from test_utils import gen_2x2_unitary
from utils import is_unitary, reconstruct
from functools import reduce


def test_is_ctrl_0():
    U = gen_2x2_unitary()
    ctrl_0_gate = np.identity(4, dtype=np.complex)
    ctrl_0_gate[2:, 2:] = U
    assert is_ctrl_u(ctrl_0_gate) == 0


def test_is_ctrl_1():
    U = gen_2x2_unitary()
    ctrl_1_gate = np.identity(4, dtype=np.complex)
    ctrl_1_gate[:2, :2] = U
    assert is_ctrl_u(ctrl_1_gate) == 1   


def test_not_ctrl_unitary():
    U = gen_2x2_unitary()
    U[0][0] = -U[0][0]
    G = np.identity(4, dtype=np.complex)
    G[:2, :2] = U
    assert not is_ctrl_u(G)


def test_not_ctrl_identity():
    U = gen_2x2_unitary()
    G = np.identity(4, dtype=np.complex)
    G[0][1] = 1
    G[2:, 2:] = U
    assert not is_ctrl_u(G)


def test_decomp_random():
    U = gen_2x2_unitary()
    ctrl_0_gate = np.identity(4, dtype=np.complex)
    ctrl_0_gate[2:, 2:] = U
    output = ctrl_u_decomp(ctrl_0_gate)
    recons = reconstruct(output)
    assert np.allclose(recons, ctrl_0_gate)


def test_decomp_cz():
    output = ctrl_u_decomp(CZ)
    recons = reconstruct(output)
    assert np.allclose(recons, CZ)
