import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random
N = 100

PMS1 = 0.99

def adde(circuit, gate, qud, ind):
    if ind == 1:
        primen = [gate[i](qud[i]) for i in range(len(qud))]
    else:
        primen = [gate[0](qud[0], qud[1])]
    circuit.append(primen, strategy=InsertStrategy.INLINE)

    error(circuit, qud, ind)


def R(fi, hi, i=0, j=1):
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    x01_for_ms = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]])
    y01_for_ms = np.array([[0, complex(0, -1), 0],
                           [complex(0, 1), 0, 0],
                           [0, 0, 0]])
    x12_for_ms = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0]])
    y12_for_ms = np.array([[0, 0, 0],
                           [0, 0, complex(0, -1)],
                           [0, complex(0, 1), 0]])
    x02_for_ms = np.array([[0, 0, 1],
                           [0, 0, 0],
                           [1, 0, 0]])
    y02_for_ms = np.array([[0, 0, complex(0, -1)],
                           [0, 0, 0],
                           [complex(0, 1), 0, 0]])
    if (i, j) == (0, 1):
        x_for_ms = x01_for_ms
        y_for_ms = y01_for_ms
    elif (i, j) == (1, 2):
        x_for_ms = x12_for_ms
        y_for_ms = y12_for_ms
    else:
        x_for_ms = x02_for_ms
        y_for_ms = y02_for_ms
    m = np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms

    return linalg.expm(complex(0, -1) * m * hi / 2)


class U(cirq.Gate):
    def __init__(self, mat, diag_i='R'):
        self.mat = mat
        self.diag_info = diag_i


    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return self.mat

    def _circuit_diagram_info_(self, args):
        return self.diag_info


class H(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi / 2, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'H'


class X1_conj(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, complex(0, -1), 0], [complex(0, -1), 0, 0], [0, 0, 1]])

    def _circuit_diagram_info_(self, args):
        return 'X1_c'


class X2_conj(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.conj(np.array([[0, 0, complex(0, -1)],
                                 [0, 1, 0],
                                 [complex(0, -1), 0, 0]]))

    def _circuit_diagram_info_(self, args):
        return 'X2_c'


class Z1(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'Z1'

class Y1(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(np.pi /2, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'Y1'


class X2(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 2)

    def _circuit_diagram_info_(self, args):
        return 'X2'


class X1(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):

        return 'X1'


def time_error(circuit, qutrits, t):

    x = X1()
    y = Y1()
    z = Z1()

    p = t/(t+5)

    n = len(qutrits)
    for i in range(n):
        rvv = random.randint(0, 10000)
        if rvv > p * 10000:
            return
        else:
            rv = random.randint(0, 2)
            #rv = 2
            if rv == 0:
                circuit.append([x(qutrits[i])], strategy=InsertStrategy.INLINE)
            if rv == 2:
                circuit.append([y(qutrits[i])], strategy=InsertStrategy.INLINE)
            if rv == 1:
                circuit.append([z(qutrits[i])], strategy=InsertStrategy.INLINE)


def error(circuit, qutrits):
    p = PMS1

    x = X1()
    y = Y1()
    z = Z1()
    rv = random.randint(0, 10000)
    n = len(qutrits)
    if rv < p * 10000:
        return
    else:
        for i in range(n):
            rv = random.randint(0 ,3)

            if rv == 1:
                circuit.append(x(qutrits[i]), strategy=InsertStrategy.INLINE)
            if rv == 2:
                circuit.append(y(qutrits[i]), strategy=InsertStrategy.INLINE)
            if rv == 0:
                circuit.append(z(qutrits[i]), strategy=InsertStrategy.INLINE)

# Основная операция
x = X1()
x2 = X2()
z = Z1()
y = Y1()
x_conj = X1_conj()
x2_conj = X2_conj()
h = H()

def m(a ,b, c, d, e):
    return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)

sps1 = []
sps2 = []
z = np.array([1,0,0])
'''
for t in range(0, 200, 10):
    sps2.append(t)
    sch = 0
    for i in range(N):
        x = X1()
        y = Y1()
        sim = cirq.Simulator()
        circuit1 = cirq.Circuit()
        qutrits1 = []
        for j in range(1):
            qutrits1.append(cirq.LineQid(j, dimension=3))
        circuit1.append([x(qutrits1[0])])
        circuit1.append([h(qutrits1[0])])
        time_error(circuit1, qutrits1, t)


        res1 = sim.simulate(circuit1)
        vec = circuit1.final_state_vector(initial_state=0, qubit_order=qutrits1)
        ssum = 0
        razn = vec - z
        if abs(np.dot(razn, razn)) < 0.001:
            sch += 1

    sps1.append(1 - sch / N)
'''
t = np.arange(0,100,8)
T0 = 25
f = np.exp(-t/T0)
print(list(f))

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot()
ax.scatter(t, f, color='r', s = 5)
print(sps1)
print(sps2)
plt.show()

'''[0.64, 0.538, 0.446, 0.428, 0.458, 0.378, 0.314, 0.376, 0.332, 0.336, 0.34, 0.36, 0.29]
[0.642, 0.542, 0.45, 0.442, 0.396, 0.454, 0.384, 0.41, 0.37, 0.428, 0.386, 0.372, 0.394]
[0.76, 0.66, 0.592, 0.518, 0.484, 0.458, 0.484, 0.412, 0.49, 0.424, 0.416, 0.384, 0.42]
[0.86, 0.716, 0.682, 0.574, 0.578, 0.5, 0.514, 0.486, 0.476, 0.518, 0.448, 0.456, 0.422]
[0.936, 0.856, 0.694, 0.636, 0.65, 0.6, 0.558, 0.57, 0.556, 0.59, 0.544, 0.504, 0.498]'''