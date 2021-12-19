import numpy as np
from params import CNOT, CNOT_P, H
from utils import is_unitary
from scipy import linalg
from math import sqrt, pi, atan2


def is_ctrl_u(U):
    """
    Whether the input gate is ctrl gate
    :return: if ctrl_0, return 0; if ctrl_1 return 1; else return None
    """
    # ctrl 0
    U0 = np.identity(4, dtype=np.complex)
    U0[2:, 2:] = U[2:, 2:]
    is_ctrl_0 = is_unitary(U[2:, 2:], 2) and np.allclose(U, U0)
    if is_ctrl_0:
        return 0

    # ctrl 1
    U1 = np.identity(4, dtype=np.complex)
    U1[:2, :2] = U[:2, :2]
    is_ctrl_1 = is_unitary(U[:2, :2], 2) and np.allclose(U, U1)
    if is_ctrl_1:
        return 1

    return None


def get_global_phase(U):
    """
    Compute the global phase a in U = e^(ja)*V
    :return: a, float in (-pi, pi]
    """
    exp_a = linalg.det(U) ** (1 / 2)
    a = np.angle(exp_a)
    return a


def get_abcd(U):
    """
    Get the a, b, c, d in notions below.
    U = e^(ja)*V, V = S(b)@R(c)@S(d)
    :param U: 2x2 unitary matrix
    :return: a, b, c, d
    """
    # get a, where a is the global phase
    exp_a = linalg.det(U) ** (-1 / 2)
    a = - np.angle(exp_a)

    V = exp_a * U

    c = atan2(abs(V[1, 0]), abs(V[0, 0]))
    h = - np.angle(V[1, 1])
    k = - np.angle(V[1, 0])
    b = - (h + k) / 2
    d = (k - h) / 2
    return a, b, c, d


def ctrl_u_decomp(U):
    # decompose the Control-U gate with two CNOT gates and some single-qubit gates
    is_ctrl_0 = (is_ctrl_u(U) == 0)
    cnot = CNOT if is_ctrl_0 else CNOT_P
    U22 = U[2:, 2:] if is_ctrl_0 else U[:2, :2]
    
    def S(b):
        return np.array([[np.exp(-1j*b), 0],
                         [0, np.exp(1j * b)]], dtype=np.complex)
    
    def R(c):
        return np.array([[np.cos(c), -np.sin(c)],
                         [np.sin(c), np.cos(c)]], dtype=np.complex)
    
    a, b, c, d = get_abcd(U22)
    if c == 0:
        # U is CZ
        return [H, cnot, H]

    C = S(b) @ R(c/2)
    A = S((d - b) / 2)
    B = R(-c/2) @ S(-(d + b) / 2)
    E = np.diag([1, np.exp(a * 1j)])
    return [E, A, cnot, B, cnot, C]
