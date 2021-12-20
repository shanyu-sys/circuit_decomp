import numpy as np


def kron_decomp(U):
    """
    Decompose the input two-qubit gates into two single-qubit gates, whose tensor product is the input.
    :param U: a 4x4 unitary matrix
    :return: A, B. 
        If U cannot be decomposed, both A and B are None
        else, both A and B are 2x2 unitary matrix. 
    """
    grids = [(0, 0), (0, 1), (1, 0), (1, 1)]
    A = np.zeros(shape=(2, 2), dtype=np.complex)
    B = None
    for i, j in grids: 
        U_block = U[2*i:2*i + 2, 2*j:2*j + 2]
        det_U_block = np.linalg.det(U_block)
        # U_block = Aij*B
        # det(U_block) = det(Aij*B) = (Aij)^2*det(B) = (Aij)^2
        if np.abs(det_U_block) > 0:
            a = np.sqrt(det_U_block)
            B_new = U_block / a
            if B is None:
                B = B_new
            if np.allclose(B_new, B):
                A[i][j] = a
            elif np.allclose(B_new, -B):
                A[i][j] = -a
            else:
                return False, None, None
    if B is None:
        return False, None, None
    else:
        assert np.allclose(np.kron(A, B), U)
        return True, A, B