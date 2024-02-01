import random

import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random
import numpy as np
import matplotlib.pyplot as plt

def m(a ,b, c, d):
    return np.kron(np.kron(np.kron(a, b), c), d)


z = np.array([[1,0,0]]).T
e = np.array([[0,1,0]]).T
f = np.array([[0,0,1]]).T
A = [z, e, f]

B = []
for i1 in range(3):
    for i2 in range(3):
        for i3 in range(3):
            for i4 in range(3):
                B.append(m(A[i1], A[i2], A[i3], A[i4]))


ro = (np.kron(1/(2**0.5) * (e + z), m(f, z, e, f))) @ (np.kron(1/(2**0.5) * (e + z), m(f, z, e, f)).T)
tr = np.eye(3) - np.eye(3)
for i in range(3):
    for j in range(3):
        #tr1 = 0
        for k in range(81):
            #tr1 = 0
            tr = tr + np.kron(A[i].T, B[k].T) @ ro @ np.kron(A[j], B[k])  * (A[i] @ A[j].T)
        #tr += tr1 * (A[i] @ A[j].T)
print(tr)
