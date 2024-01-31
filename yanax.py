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


z = np.array([[1,0]]).T
e = np.array([[0,1]]).T
b1 = np.kron(z,z)
b2 = np.kron(z,e)
b3 = np.kron(e,z)
b4 = np.kron(e,e)
b = [b1,b2,b3,b4]

ro = np.eye(16)
tr = np.eye(4) - np.eye(4)
for i in range(4):
    for j in range(4):
        for k in range(4):
            tr = tr + np.kron(b[i].T , b[k].T) @ ro @ np.kron(b[j] , b[k]) * b[i] @ b[j].T
print(tr)

