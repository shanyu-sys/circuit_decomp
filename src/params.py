import numpy as np

# single-qubit gates
I = np.array([[1, 0], 
              [0, 1]],
             dtype=np.complex)
X = np.array([[0, 1], 
              [1, 0]],
             dtype=np.complex)
Y = np.array([[0, -1j], 
              [1j, 0]],
             dtype=np.complex)
Z = np.array([[1, 0],
              [0, -1]],
             dtype=np.complex)
H = np.array([[1, 1],
              [1, -1]], 
             dtype=complex) / np.sqrt(2)

# multi-qubit gates
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]],
                dtype=np.complex)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]],
                 dtype=np.complex)
# CNOT_P is CNOT with q2 as control bit and q1 as target bit
CNOT_P = np.array([[1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0]],
                  dtype=np.complex)
CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]],
              dtype=np.complex)
