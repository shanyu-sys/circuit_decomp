from scipy.stats import unitary_group
import numpy as np


def gen_2x2_unitary():
    A = unitary_group.rvs(2)
    return A * np.linalg.det(A) ** (-1 / 2)

def gen_4x4_unitary():
    return unitary_group.rvs(4, random_state=123)