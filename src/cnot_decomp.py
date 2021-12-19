# Download and install pygsvd: https://github.com/bnaecker/pygsvd
# Reference: https://arxiv.org/ftp/quant-ph/papers/0602/0602174.pdf

import numpy as np
from scipy import linalg
from kron_decomp import kron_decomp
import pygsvd
from params import X, Y, Z, SWAP, CNOT, I
from functools import reduce


over_sqrt_2 = 1 / np.sqrt(2)
# matrics M
M = over_sqrt_2 * np.array([
    [1, 0, 0, 1j],
    [0, 1j, 1, 0],
    [0, 1j, -1, 0],
    [1, 0, 0, -1j]
    ])
M_conjugate_transpose = np.matrix(M).getH()

Lamaba = np.array([
    [1, 1, -1, 1],
    [1, 1, 1, -1],
    [1, -1, -1, -1],
    [1, -1, 1, 1]
])

Lamaba_inverse = np.linalg.inv(Lamaba)


def remove_global_phase(U):
    alpha = np.angle(linalg.det(U) ** (1 / 4))
    return U * np.exp(- 1j * alpha)


def decompose_ua_ub_va_vb(U):
    V = remove_global_phase(U)
    U_ = M_conjugate_transpose @ V @ M

    UR = np.real(U_)
    UI = np.imag(U_)

    C, S, X, V1, V2 = pygsvd.gsvd(UR, UI, extras='uv')

    assert(np.allclose(UR, V1 @ (np.diag(C) @ np.matrix(X).getH())))
    assert(np.allclose(UI, V2 @ (np.diag(S) @ np.matrix(X).getH())))

    UA_tensordot_UB = M @ (V1 @ M_conjugate_transpose)
    VA_tensordot_VB = M @ (np.matrix(X).getH() @ M_conjugate_transpose)

    F = np.linalg.inv(V1) @ V2
    exp_i_phy = np.diag(np.diag(C) + 1j * (F @ np.diag(S)))
    phy = np.angle(exp_i_phy)
    theta = - Lamaba_inverse @ phy

    return UA_tensordot_UB, VA_tensordot_VB, theta

def cnot_decomp(U):
    UAB, VAB, theta = decompose_ua_ub_va_vb(U)
    s1, UA, UB = kron_decomp(UAB)
    s2, VA, VB = kron_decomp(VAB)
    assert s1
    assert s2
    
    Ra1, Rb1 = VA, VB

    Ra2 = 1j * over_sqrt_2 * (X + Z) @ linalg.expm(-1j * (theta[1] - np.pi / 4) * X)
    Rb2 = linalg.expm(-1j * theta[3] * Z)

    Ra3 = -1j * over_sqrt_2 * (X + Z)
    Rb3 = linalg.expm(1j * theta[2] * Z)

    Ra4 = UA @ (over_sqrt_2 * (np.diag([1, 1]) - 1j * X))
    Rb4 = UB @ (over_sqrt_2 * (np.diag([1, 1]) - 1j * X)).conj().T

    return [Ra1, Rb1, Ra2, Rb2, Ra3, Rb3, Ra4, Rb4]


def cnot_reconstruct(*gates):
    Ra1, Rb1, Ra2, Rb2, Ra3, Rb3, Ra4, Rb4 = gates
    return reduce(np.dot, [np.kron(Ra4, Rb4), CNOT, np.kron(Ra3, Rb3), CNOT, np.kron(Ra2, Rb2), CNOT, np.kron(Ra1, Rb1)])