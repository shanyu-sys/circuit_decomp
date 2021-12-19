import numpy as np
from params import I
from functools import reduce


def is_unitary(U, n):
    return np.allclose(U @ np.matrix(U).getH(), np.identity(n))


def tensor_prod_i(G):
    return np.kron(I, G)


def reconstruct(gates):
    gates44 = []
    for g in gates:
        if g.shape[0] == 2:
            g44 = tensor_prod_i(g)
            gates44.append(g44)
        else:
            gates44.append(g) 
    return reduce(np.dot, reversed(gates44))
