import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
import scipy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random

# Базовые обозначения
T0 = 25

z = np.array([[1,0,0]]).T
e = np.array([[0,1,0]]).T
f = np.array([[0,0,1]]).T

basis = [z,e,f]

B = []
for i1 in range(3):
    for i2 in range(3):
        for i3 in range(3):
            for i4 in range(3):
                B.append(np.kron(np.kron(np.kron(basis[i1], basis[i2]), basis[i3]), basis[i4]))

X = np.array([[0,1,0], [1,0,0], [0,0,1]])
Y = np.array([[0,complex(0,-1), 0], [complex(0,1), 0, 0], [0,0,1]])
Z = np.array([[1,0,0],[0,-1,0], [0,0,1]])
id = np.eye(3)

paulies1 = [id, X, Y, Z]

def dag(matrix):
    return np.conj(matrix.T)

# Код для деполяризующего канала
def E(bas, i, j, p0, paulies):
    v1 = bas[i]
    v2 = bas[j]
    id = paulies[0]
    x = paulies[1]
    y = paulies[2]
    z = paulies[3]
    K0 = id / 2
    K1 = x / 2
    K2 = y / 2
    K3 = z / 2

    _rho = v1 @ (v2.T)

    if i == 0 and j == 0:
        cgjjjfjk = 0
    return K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3)

def nice_repr(parameter):
    """Nice parameter representation
        SymPy symbol - as is
        float number - 3 digits after comma
    """
    if isinstance(parameter, float):
        return f'{parameter:.3f}'
    else:
        return f'{parameter}'

def levels_connectivity_check(l1, l2):
    """Check ion layers connectivity for gates"""
    connected_layers_list = [{0, i} for i in range(max(l1, l2) + 1)]
    assert {l1, l2} in connected_layers_list, "Layers are not connected"

def generalized_sigma(index, i, j, dimension=4):
    """Generalized sigma matrix for qudit gates implementation"""

    sigma = np.zeros((dimension, dimension), dtype='complex')

    if index == 0:
        # identity matrix elements
        sigma[i][i] = 1
        sigma[j][j] = 1
    elif index == 1:
        # sigma_x matrix elements
        sigma[i][j] = 1
        sigma[j][i] = 1
    elif index == 2:
        # sigma_y matrix elements
        sigma[i][j] = -1j
        sigma[j][i] = 1j
    elif index == 3:
        # sigma_z matrix elements
        sigma[i][i] = 1
        sigma[j][j] = -1

    return sigma

class QuditGate(cirq.Gate):
    """Base class for qudits gates"""

    def __init__(self, dimension=4, num_qubits=1):
        self.d = dimension
        self.n = num_qubits
        self.symbol = None

    def _num_qubits_(self):
        return self.n

    def _qid_shape_(self):
        return (self.d,) * self.n

    def _circuit_diagram_info_(self, args):
        return (self.symbol,) * self.n

#деполяризующий
class QutritDepolarizingChannel(QuditGate):

    def __init__(self,PP, p_matrix=None):
        super().__init__(dimension=3, num_qubits=1)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        f1 = 0.9
        self.p1 = (1 - f1) / (1 - 1 / self.d ** 2)
        self.p1 = PP

        # Choi matrix initialization

        if p_matrix is None:
            self.p_matrix = (1 - self.p1) / (self.d ** 2) * np.ones((self.d, self.d))
            self.p_matrix = np.zeros_like(self.p_matrix)
        else:
            self.p_matrix = p_matrix
        for o in range(3):
            for oo in range(3):
                self.p_matrix[o, oo] = 1 / 9

        if p_matrix is None:
            self.p_matrix = self.p1 / (self.d ** 2) * np.ones((self.d, self.d))
        else:
            self.p_matrix = p_matrix
        self.p_matrix[0, 0] += (1 - self.p1)  # identity probability
        self.p_matrix = np.array([[(1 - self.p1), self.p1 / 3], [self.p1 / 3, self.p1 / 3]])

    def _mixture_(self):
        ps = []
        for i in range(self.d):
            for j in range(self.d):
                pinv = np.linalg.inv(self.p_matrix)
                op = E(basis, i, j, self.p1, paulies1)
                ps.append(op)

        X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        Y = np.array([[0, complex(0, -1), 0], [complex(0, 1), 0, 0], [0, 0, 1]])
        Z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        id = np.eye(3)
        shiz_massiv = [id, X, Y, Z]
        return tuple(zip(self.p_matrix.flatten(), shiz_massiv))

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"

'''
#АМПЛИТУДНЫЙ
class QutritDepolarizingChannel(QuditGate):

    def __init__(self,PP, p_matrix=None):
        super().__init__(dimension=3, num_qubits=1)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        f1 = 0.9
        self.p1 = (1 - f1) / (1 - 1 / self.d ** 2)
        self.p1 = PP
        #print(self.d)
        #print((1 / self.d ** 2))

        # Choi matrix initialization


        if p_matrix is None:
            self.p_matrix = self.p1 / (self.d ** 2) * np.ones((self.d, self.d))
        else:
            self.p_matrix = p_matrix
        self.p_matrix[0, 0] += (1 - self.p1)  # identity probability
        self.p_matrix = np.array([[1/3, 1 / 3], [1 / 3, 0]])
        #print('prob[0,0]', self.p_matrix[0, 0])
        #print('prob_sum', self.p_matrix.sum())

        #print('prob_sum', self.p_matrix.sum())

    def _mixture_(self):
        ps = []
        for i in range(self.d):
            for j in range(self.d):
                pinv = np.linalg.inv(self.p_matrix)
                op = E(basis, i, j, self.p1, paulies1)
                #print(np.trace(op))
                ps.append(op)
        #print('total_sum', (np.trace(np.array(ps)) * self.p_matrix).sum())
        #chm = np.kron(np.ones(3), ps)
        X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        Y = np.array([[0, complex(0, -1), 0], [complex(0, 1), 0, 0], [0, 0, 1]])
        Z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        m1 = 3**0.5*np.array([[1,0,0],[0,(1 - self.p1)**0.5, 0], [0,0,(1 - self.p1)**0.5]])
        m2 = 3**0.5*np.array([[0,(self.p1)**0.5,0],[0,0, 0], [0,0,0]])
        m3 = 3**0.5*np.array([[0,0,0],[0,0,(self.p1)**0.5], [0,0,0]])
        id = np.eye(3) - np.eye(3)
        shiz_massiv = [m1, m2, m3, id]
        return tuple(zip(self.p_matrix.flatten(), shiz_massiv))

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"
        
        
# фазовый
class QutritDepolarizingChannel(QuditGate):

    def __init__(self, PP, p_matrix=None):
        super().__init__(dimension=3, num_qubits=1)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        f1 = 0.9
        self.p1 = (1 - f1) / (1 - 1 / self.d ** 2)
        self.p1 = PP
     
        # Choi matrix initialization

        if p_matrix is None:
            self.p_matrix = self.p1 / (self.d ** 2) * np.ones((self.d, self.d))
        else:
            self.p_matrix = p_matrix
        self.p_matrix[0, 0] += (1 - self.p1)  # identity probability
        self.p_matrix = np.array([[1 / 2, 0], [0, 1 / 2]])

    def _mixture_(self):
        ps = []
        for i in range(self.d):
            for j in range(self.d):
                pinv = np.linalg.inv(self.p_matrix)
                op = E(basis, i, j, self.p1, paulies1)
                ps.append(op)

        X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        Y = np.array([[0, complex(0, -1), 0], [complex(0, 1), 0, 0], [0, 0, 1]])
        Z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        m1 = 3 ** 0.5 * np.array([[1, 0, 0], [0, (1 - self.p1) ** 0.5, 0], [0, 0, (1 - self.p1) ** 0.5]])
        m2 = 3 ** 0.5 * np.array([[0, (self.p1) ** 0.5, 0], [0, 0, 0], [0, 0, 0]])
        m3 = 3 ** 0.5 * np.array([[0, 0, 0], [0, 0, (self.p1) ** 0.5], [0, 0, 0]])
        m4 = 2 ** 0.5 * (1 - self.p1) ** 0.5 * np.eye(3)
        omega = np.exp(complex(0, 2 * np.pi / 3))
        m5 = 2 ** 0.5 * (self.p1) ** 0.5 * np.array([[1, 0, 0], [0, omega, 0], [0, 0, omega ** 2]])
        id = np.eye(3) - np.eye(3)
        shiz_massiv = [m4, id, id, m5]
        return tuple(zip(self.p_matrix.flatten(), shiz_massiv))

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"


'''

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

def make_ms_matrix(fi, hi, i=0, j=1, k=0, l=1):
    x_for_ms = np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 0]])
    y_for_ms = np.array([[0, complex(0, -1), 0],
                         [complex(0, 1), 0, 0],
                         [0, 0, 0]])
    m = np.kron((np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms), (np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms))
    m = complex(0, -1) * m * hi
    return linalg.expm(m)

class TwoQuditMSGate3_c(gate_features.TwoQubitGate
                        ):

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__
        }

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        return cls()

    def _qid_shape_(self):
        return (3, 3,)

    def _unitary_(self):
        matrix = make_ms_matrix(0, -np.pi / 2)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101_c',
                          'XX0101_c'))
class TwoQuditMSGate3(gate_features.TwoQubitGate
                      ):

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__
        }

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        return cls()

    def _qid_shape_(self):
        return (3, 3,)

    def _unitary_(self):
        matrix = make_ms_matrix(0, np.pi / 2)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))
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

def U1(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u6 = U(R(np.pi / 2, -np.pi, 0, 2), 'Ry(-π)02')
    adde(cirquit, [u1, u6], [q1, q2], 1)
    adde(cirquit, [u2], [q1], 1)
    xx = TwoQuditMSGate3()
    adde(cirquit, [xx], [q1, q2], 2)

def U1_clear(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u6 = U(R(np.pi / 2, -np.pi, 0, 2), 'Ry(-π)02')
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    xx = TwoQuditMSGate3()
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)

def U1_c(cirquit, q1, q2):
    u1 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    u2 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u6 = U(R(np.pi / 2, np.pi, 0, 2), 'Ry(π)02')
    xx_c = TwoQuditMSGate3_c()
    adde(cirquit, [xx_c], [q1, q2], 2)
    adde(cirquit, [u2], [q1], 1)
    adde(cirquit, [u1, u6], [q1, q2], 1)

def U1_c_clear(cirquit, q1, q2):
    u1 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    u2 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u6 = U(R(np.pi / 2, np.pi, 0, 2), 'Ry(π)02')
    xx_c = TwoQuditMSGate3_c()
    cirquit.append([xx_c(q1, q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)

def CX_clear(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 1), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u5 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    xx = TwoQuditMSGate3()
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u3(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u5(q1)], strategy=InsertStrategy.INLINE)

def CX(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 1), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u5 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    adde(cirquit, [u1], [q1], 1)
    adde(cirquit, [u2], [q1], 1)
    xx = TwoQuditMSGate3()
    adde(cirquit, [xx], [q1, q2], 2)
    adde(cirquit, [u3, u3], [q1, q2], 1)
    adde(cirquit, [u4], [q1], 1)
    adde(cirquit, [u5], [q1], 1)

def CCX(cirquit, q1, q2, q3):
    U1(cirquit, q1, q2)
    CX(cirquit, q2, q3)
    U1_c(cirquit, q1, q2)

def CZ(cirquit, q1, q2):
    h = H()
    adde(cirquit, [h], [q2], 1)
    CX(cirquit, q1, q2)
    adde(cirquit, [h], [q2], 1)

def CCZ(cirquit, q1, q2, q3):
    h = H()
    adde(cirquit, [h], [q3], 1)
    CCX(cirquit, q1, q2, q3)
    adde(cirquit, [h], [q3], 1)

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

def encoding_qubit(circuit, log_qubit):
    x = X1()
    h = H()
    q1, q2, q3, q4, q5 = log_qubit[0], log_qubit[1], log_qubit[2], log_qubit[3], log_qubit[4]
    adde(circuit, [h, h, h], [q2, q3, q4], 1)
    CCZ(circuit, q1, q3, q4)
    adde(circuit, [x, x], [q3, q4], 1)
    CCZ(circuit, q4, q3, q1)
    adde(circuit, [x, x], [q3, q4], 1)
    CX(circuit, q1, q5)
    CX(circuit, q2, q5)
    CX(circuit, q2, q1)
    CX(circuit, q4, q1)
    CX(circuit, q3, q5)
    CZ(circuit, q4, q5)

def decoding_qubit(circuit, log_qubit):
    x = X1()
    h = H()
    q1, q2, q3, q4, q5 = log_qubit[0], log_qubit[1], log_qubit[2], log_qubit[3], log_qubit[4]
    CZ(circuit, q4, q5)
    CX(circuit, q3, q5)
    CX(circuit, q4, q1)
    CX(circuit, q2, q1)
    CX(circuit, q2, q5)
    CX(circuit, q1, q5)
    adde(circuit, [x, x], [q3, q4], 1)
    CCZ(circuit, q1, q3, q4)
    adde(circuit, [x, x], [q3, q4], 1)
    CCZ(circuit, q1, q3, q4)
    adde(circuit, [h,h,h], [q2, q3, q4], 1)

def CCCCX(cirquit, q1, q2, q3, q4, q5):
    U1_clear(cirquit, q1, q2)
    U1_clear(cirquit, q2, q3)
    U1_clear(cirquit, q3, q4)
    CX_clear(cirquit, q4, q5)
    U1_c_clear(cirquit, q3, q4)
    U1_c_clear(cirquit, q2, q3)
    U1_c_clear(cirquit, q1, q2)

def CCCCZ(cirquit, q1, q2, q3, q4, q5):
    h = H()
    cirquit.append(h(q5), strategy=InsertStrategy.INLINE)
    CCCCX(cirquit, q1, q2, q3, q4, q5)
    cirquit.append(h(q5), strategy=InsertStrategy.INLINE)

def CCCCY(cirquit, q1, q2, q3, q4, q5):
    h = H()
    CCCCZ(cirquit, q1, q2, q3, q4, q5)
    CCCCX(cirquit, q1, q2, q3, q4, q5)

def error_correction(circuit, qutrits1):

    q0 = qutrits1[0]
    q1 = qutrits1[1]
    q2 = qutrits1[2]
    q3 = qutrits1[3]
    q4 = qutrits1[4]

    # Операции для исправления ошибок X
    circuit.append([x(q1)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q1)], strategy=InsertStrategy.INLINE)

    circuit.append([x(q1), x(q4)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q1), x(q4)], strategy=InsertStrategy.INLINE)

    circuit.append([x(q1), x(q2), x(q3)], strategy=InsertStrategy.INLINE)
    CCCCZ(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q1), x(q2), x(q3)], strategy=InsertStrategy.INLINE)

    circuit.append([x(q2)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q2)], strategy=InsertStrategy.INLINE)

    # Операции для исправления ошибок Y
    circuit.append([x(q3)], strategy=InsertStrategy.INLINE)
    CCCCY(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q3)], strategy=InsertStrategy.INLINE)

    circuit.append([x(q4)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q4)], strategy=InsertStrategy.INLINE)

    circuit.append([x(q1), x(q3)], strategy=InsertStrategy.INLINE)
    CCCCZ(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q1), x(q3)], strategy=InsertStrategy.INLINE)

    circuit.append([x(q2), x(q3)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q2), x(q3)], strategy=InsertStrategy.INLINE)

    CCCCZ(circuit, q1, q2, q3, q4, q0)

    # Операции для исправления ошибок Z
    circuit.append([x(q2), x(q4)], strategy=InsertStrategy.INLINE)
    CCCCZ(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q2), x(q4)], strategy=InsertStrategy.INLINE)


    circuit.append([x(q3), x(q4)], strategy=InsertStrategy.INLINE)
    CCCCZ(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q3), x(q4)], strategy=InsertStrategy.INLINE)

def time_error(circuit, qutrits, t):

    p = np.exp(-t / T0)
    dpg_t = QutritDepolarizingChannel(1.000001 - p)
    for q in qutrits:
        circuit.append([dpg_t.on(q)], strategy=InsertStrategy.INLINE)

def error(circuit, qutrits, ind):
    if ind == 1:
        p = PMS1
    else:
        p = PMS2
    dpg = QutritDepolarizingChannel(1.0001 - p)
    for q in qutrits:
        circuit.append([dpg.on(q)], strategy=InsertStrategy.INLINE)

def X1_l(circuit, lqubits):
    x = X1()
    z = Z1()
    q1, q2, q3, q4, q5 = lqubits[0], lqubits[1], lqubits[2], lqubits[3], lqubits[4]
    adde(circuit, [z,z], [q1, q4], 1)
    adde(circuit, [x, x, x, x, x], [q1, q2, q3, q4, q5], 1)

def Z1_l(circuit, lqubits):
    x = X1()
    z = Z1()
    q1, q2, q3, q4, q5 = lqubits[0], lqubits[1], lqubits[2], lqubits[3], lqubits[4]
    adde(circuit, [x, x], [q1, q4], 1)
    adde(circuit, [z], [q5], 1)

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

def partial_trace(rho_ab):
    tr = np.eye(3) - np.eye(3)
    for i in range(3):
        for j in range(3):
            for k in range(81):
                tr = tr + np.kron(A[i].T, B[k].T) @ rho_ab @ np.kron(A[j], B[k]) * A[i] @ A[j].T
    return tr

sps1 = []
sps2 = []

def run_circit(t, N):

    fidelity = 0
    sch = 0
    for alf1 in np.linspace(0, 2 * np.pi, N):
        for alf2 in np.linspace(0, np.pi, N//2):
            alf2 = alf2 + np.pi / N
            sch += 1
            x = X1()
            y = Y1()
            sim = cirq.Simulator()
            circuit1 = cirq.Circuit()
            qutrits1 = []
            for j in range(5):
                qutrits1.append(cirq.LineQid(j, dimension=3))

            povorot = R(alf1, alf2, 0, 1)

            pg = U(povorot)
            circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)

            encoding_qubit(circuit1, qutrits1)
            er_q1 = int(random.randint(0, 4))
            er_q2 = int(random.randint(0, 4))
            if er_q2 == er_q1:
                er_q2 += 1
                er_q2 = er_q2 % 5
            er_q3 = int(random.randint(0, 4))
            if er_q3 == er_q2 or er_q3 == er_q1:
                er_q3 = (max(er_q1, er_q2) + 1) % 5
            time_error(circuit1, qutrits1, t)
            decoding_qubit(circuit1, qutrits1)
            error_correction(circuit1, qutrits1)

            povorot_r = R(alf1, -alf2, 0, 1)
            pg_r = U(povorot_r)
            circuit1.append([pg_r(qutrits1[0])], strategy=InsertStrategy.INLINE)

            ro_ab = cirq.final_density_matrix(circuit1, qubit_order = qutrits1)

            mat_0 = partial_trace(np.array(ro_ab))
            #print(mat_0)
            fidelity += abs(mat_0[0][0])
    return fidelity / sch

def run_single_qudit(t, N):
    fidelity = 0
    sch = 0
    for alf1 in np.linspace(0, np.pi, N // 2):
        for alf2 in np.linspace(0, 2 * np.pi, N):
            alf2 += 2 * np.pi / N / 2
            sch += 1
            circuit1 = cirq.Circuit()
            qutrits1 = []
            qutrits1.append(cirq.LineQid(0, dimension=3))

            povorot = R(alf1, alf2, 0, 1)

            pg = U(povorot)
            circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)

            time_error(circuit1, qutrits1, t)

            povorot_r = R(alf1, -alf2, 0, 1)
            pg_r = U(povorot_r)
            circuit1.append([pg_r(qutrits1[0])], strategy=InsertStrategy.INLINE)

            ro_ab = cirq.final_density_matrix(circuit1)

            fidelity += abs(ro_ab[0][0])
    return fidelity / sch

def main(T, k, N):
    code_line = []
    single_qudit_line = []
    for t_ in np.linspace(0, T, k):
        N = 2
        code_line.append(run_circit(t_, N))

        N = 20
        single_qudit_line.append(run_single_qudit(t_, N))
    '''
    print(code_line)
    print(single_qudit_line)
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    ax.scatter(np.linspace(0, T, k), single_qudit_line, color='b', s=5, label='без коррекции')
    ax.scatter(np.linspace(0, T, k), code_line, color='r', s=5, label='c коррекции')

    ax.set_xlabel('t, mcs (T0 = 25 mcs)')
    ax.set_ylabel('fid')
    plt.title('P1 = 0.999, P2 = 0.99')
    plt.legend()
    plt.grid()

    plt.show()
    '''
    return code_line, single_qudit_line, np.linspace(0, T, k)

def graph(c,s,t):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    ax.scatter(t, s, color='b', s=5, label='без коррекции')
    ax.scatter(t, c, color='r', s=5, label='c коррекции')

    ax.set_xlabel('t, mcs (T0 = 25 mcs)')
    ax.set_ylabel('fidelity')
    plt.title('P1 = 1, P2 = 1, amplitude damping (two-qutrit error)')
    plt.legend()
    plt.grid()

    plt.show()

PMS1 = 1
PMS2 = 1

nn = 2
T = 100
t = np.linspace(0, 200, 10)

def graph_3d(ms1, ms2, kms, T, nn, k):
    code_surf = []
    single_qudit_surf = []
    for ms in np.linspace(ms1, ms2, kms):
        PMS2 = ms
        cl, sl, tl = main(T,k,N)
        code_surf.append(cl)
        single_qudit_surf.append(sl)
    print(code_surf)
    print()
    print(single_qudit_surf)
    code_surf = np.array(code_surf)
    single_qudit_surf = np.array(single_qudit_surf)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    T, MS = np.meshgrid(np.linspace(0,T,k), np.linspace(ms1, ms2, kms))
    ax.plot_surface(T, MS, code_surf, color = 'r')
    ax.plot_surface(T, MS, single_qudit_surf, color='b')
    ax.set_xlabel('t, mcs (T0 = 25 mcs)')
    ax.set_ylabel('ms2')
    ax.set_zlabel('fid')
    plt.title('P1 = 0.99')
    plt.legend()
    plt.grid()
    plt.show()


PMS1 = 1
ms1, ms2, kms, T, N, k = 0.9, 1, 10, 100, 2, 10
code_surf = []
single_qudit_surf = []

code_surf = np.array([[0.4159785087977071, 0.4114580290624876, 0.40920070371066686, 0.4080947985785316, 0.40754517295863485, 0.4072693338384852, 0.40712799024186097, 0.40705383024760544, 0.407013245043345, 0.4069904488133035], [0.4268464300985215, 0.4192282166113728, 0.4154616741143402, 0.41369657751783984, 0.4128741306485608, 0.4124959340697388, 0.41232386639603646, 0.41224550904007634, 0.41220934939337894, 0.4121923191851239], [0.4416369726022822, 0.4289647735349718, 0.42272389842401026, 0.4199342461724883, 0.41873114718328014, 0.4182442731253106, 0.418067355218227, 0.41801528498035634, 0.41800966628215985, 0.4180172305277665], [0.4622791024339677, 0.4414806194545236, 0.43120942668247153, 0.4268442084794517, 0.42513339391734917, 0.4245694015698973, 0.4244615694769894, 0.42450867304432904, 0.4245904819836142, 0.42466488266654867], [0.4916605999769673, 0.45800805887483875, 0.44121232014731504, 0.4344518574307586, 0.4321076922951761, 0.4315849602608069, 0.43171865426302264, 0.4320200643978751, 0.4323086915755994, 0.43253509777969174], [0.5340299184608739, 0.48041540942631433, 0.45312108306188753, 0.44276716143758676, 0.4397175337726367, 0.43953739163589495, 0.44027436024134664, 0.44114019414791994, 0.4418702692364608, 0.44241367764880124], [0.59549998802504, 0.5115393312835295, 0.46745632366742074, 0.4517888132972985, 0.44812937453252755, 0.44895244954705055, 0.4510082363271977, 0.4530323156495801, 0.45464501231867877, 0.45581508099303414], [0.6846066153461265, 0.5556866675007086, 0.4849225064144776, 0.4615272851478949, 0.4577541057176407, 0.4609157361555278, 0.46564117920297593, 0.4699214068160141, 0.4732241397882722, 0.4755838943475515], [0.8127489576717154, 0.6193481665276722, 0.506451492701416, 0.47204371348651364, 0.46948757733175484, 0.4775447972045162, 0.48740894853023087, 0.4959160739249669, 0.5023470513158587, 0.5068941769905564], [0.9941992398824199, 0.7122600082578208, 0.5332742174885884, 0.483558676213165, 0.48518589299043746, 0.5028475150098706, 0.5222349650665131, 0.5384193887581991, 0.5504785304685376, 0.5589424014076114]])
single_qudit_surf = np.array([[0.9999996486306191, 0.8718408632278443, 0.7896679708361626, 0.73698033452034, 0.703198035210371, 0.6815375140309334, 0.6676491458714008, 0.6587442290782929, 0.6530346171557904, 0.6493736876547337], [0.9999996486306191, 0.8718408632278443, 0.7896679708361626, 0.73698033452034, 0.703198035210371, 0.6815375140309334, 0.6676491458714008, 0.6587442290782929, 0.6530346171557904, 0.6493736876547337], [0.9999996486306191, 0.8718408632278443, 0.7896679708361626, 0.73698033452034, 0.703198035210371, 0.6815375140309334, 0.6676491458714008, 0.6587442290782929, 0.6530346171557904, 0.6493736876547337], [0.9999996486306191, 0.8718408632278443, 0.7896679708361626, 0.73698033452034, 0.703198035210371, 0.6815375140309334, 0.6676491458714008, 0.6587442290782929, 0.6530346171557904, 0.6493736876547337], [0.9999996486306191, 0.8718408632278443, 0.7896679708361626, 0.73698033452034, 0.703198035210371, 0.6815375140309334, 0.6676491458714008, 0.6587442290782929, 0.6530346171557904, 0.6493736876547337], [0.9999996486306191, 0.8718408632278443, 0.7896679708361626, 0.73698033452034, 0.703198035210371, 0.6815375140309334, 0.6676491458714008, 0.6587442290782929, 0.6530346171557904, 0.6493736876547337], [0.9999996486306191, 0.8718408632278443, 0.7896679708361626, 0.73698033452034, 0.703198035210371, 0.6815375140309334, 0.6676491458714008, 0.6587442290782929, 0.6530346171557904, 0.6493736876547337], [0.9999996486306191, 0.8718408632278443, 0.7896679708361626, 0.73698033452034, 0.703198035210371, 0.6815375140309334, 0.6676491458714008, 0.6587442290782929, 0.6530346171557904, 0.6493736876547337], [0.9999996486306191, 0.8718408632278443, 0.7896679708361626, 0.73698033452034, 0.703198035210371, 0.6815375140309334, 0.6676491458714008, 0.6587442290782929, 0.6530346171557904, 0.6493736876547337], [0.9999996486306191, 0.8718408632278443, 0.7896679708361626, 0.73698033452034, 0.703198035210371, 0.6815375140309334, 0.6676491458714008, 0.6587442290782929, 0.6530346171557904, 0.6493736876547337]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
T, MS = np.meshgrid(np.linspace(0,T,k), np.linspace(ms1, ms2, kms))
ax.plot_surface(T, MS, code_surf, cmap='viridis')
ax.plot_wireframe(T, MS, single_qudit_surf, color='r')
ax.set_xlabel('t, mcs (T0 = 25 mcs)')
ax.set_ylabel('ms2')
ax.set_zlabel('fid')
plt.title('P1 = 1')
plt.legend()
plt.grid()
plt.show()


cs = np.array([[(0.40646447538165376+1.949782557142484e-09j), (0.4346093237400055-2.356950194662039e-10j), (0.4039295100956224+1.0974464715419805e-10j), (0.40801449870923534-6.719451659277042e-10j), (0.39478514075744897-5.355242920419393e-10j), (0.4231309476890601-6.257925290414942e-10j), (0.3936230191611685+2.864057320769087e-10j), (0.3689566606772132-5.286170869158083e-10j), (0.37357087596319616+1.1826238341486775e-09j), (0.3775536030298099+3.897093205167515e-10j), (0.39938682434149086-2.169119994759434e-09j), (0.43758267303928733+1.3848786843943951e-09j), (0.4048577412031591+2.5533473112598e-09j), (0.4065297738998197+1.027563256628263e-09j), (0.4164427744690329-6.039975325355159e-10j), (0.4168452605372295+1.2290511107409702e-09j), (0.38759166188538074-2.9371913792449e-09j), (0.3793499371386133+1.0591361101815569e-10j), (0.4166476971004158-2.712720714695266e-09j), (0.4168987662997097-5.968232573172766e-10j)], [(0.4603690844960511-6.412650757757475e-09j), (0.4587688036262989-1.3568625463382135e-09j), (0.37829225044697523+1.9996317936340257e-09j), (0.39715484163025394-4.40278125064391e-09j), (0.404340774693992-1.3970578027547867e-09j), (0.39170807891059667-1.6160659480779612e-09j), (0.41311119589954615-7.963184487953203e-10j), (0.44026599737117067+3.7989763568633307e-10j), (0.4116287280921824+1.1705914299300533e-09j), (0.3770142531138845-4.768936150037083e-10j), (0.40021976217394695+1.9862550909466776e-09j), (0.4311069755931385+1.2326997756583473e-09j), (0.41183589375577867-5.849843838083334e-10j), (0.38695709977764636+5.310101301851118e-11j), (0.39370169589528814-7.560796434997716e-10j), (0.3862380536738783+1.3728836114334484e-10j), (0.4098314273869619+8.088194837285262e-10j), (0.40383011143421754+2.1683841389105826e-09j), (0.4043603159952909+1.0453573562062877e-09j), (0.41678759292699397+9.526872627231873e-10j)], [(0.47680025291629136+9.066815078673731e-10j), (0.3923743258055765-1.7291426067645524e-09j), (0.41147872112924233+3.0930278337401683e-09j), (0.42878526845015585-7.228213027781303e-10j), (0.4454627587692812-8.173150295949446e-10j), (0.40995706274406984+9.66945135096557e-10j), (0.4332875575637445+1.1095019834339407e-09j), (0.4109244800056331-9.570478933333226e-11j), (0.39321497242781334+1.9041031538344e-09j), (0.41553122585173696+1.7016808935831527e-09j), (0.44320495112333447+2.1465505205658536e-09j), (0.37258100119652227+6.0047421180833e-10j), (0.4328438179800287+1.2138719687043994e-09j), (0.39883945920155384+1.196172327922735e-09j), (0.39961632364429533-2.498773743495397e-09j), (0.3797127684520092+7.019183568573809e-10j), (0.3810075960645918+6.827333837443878e-10j), (0.410268884152174+1.0125586481520674e-09j), (0.40582912566605955+1.9531272157399205e-09j), (0.429477215919178+7.704416150046787e-10j)], [(0.4724881036381703+4.316792256803938e-09j), (0.397931304294616-8.84570694705208e-10j), (0.42976331437239423-1.0986553336739412e-09j), (0.39355577845708467+3.9382497261948864e-09j), (0.4067768156528473-1.3689888112084407e-09j), (0.3777256909525022-6.30633403812012e-10j), (0.39040415547788143-3.01137397303162e-10j), (0.41139193225535564-1.6683680748522303e-09j), (0.40742820099694654-1.9879141666399078e-09j), (0.4299364443286322+4.56970897235272e-09j), (0.42584637395339087+8.260589432933654e-10j), (0.3921753661124967-1.9701729690300602e-09j), (0.384740260313265-3.7129214594227044e-09j), (0.40177052485523745+1.538863258866819e-09j), (0.40577361575560644-2.3713058182486e-09j), (0.39977470348821953-3.100803527219212e-09j), (0.4110770290135406+1.5846560175253842e-09j), (0.39640633371891454-1.2308280534619027e-12j), (0.4251743286731653-8.722444979356566e-11j), (0.4129206249199342-2.0324616940599565e-10j)], [(0.4287443759967573+1.4929279978039148e-11j), (0.45017824036767706-1.9629247396756535e-09j), (0.4530842445092276+3.296194872958438e-09j), (0.39740637625800446-8.797251036308403e-10j), (0.41065506386803463+1.5788660379152589e-09j), (0.4248182426090352+1.915569147932043e-09j), (0.4149425973300822-1.3852235121409478e-09j), (0.4033580193936359+2.7616316318918387e-09j), (0.42225327383494005+1.7431668198084454e-09j), (0.4449216205975972-1.703181422469911e-09j), (0.3938670679635834-3.013993497317423e-10j), (0.388310567883309-1.2010796801622355e-09j), (0.43228369212010875-8.987516557766676e-10j), (0.3807969229237642+1.9637902561878895e-10j), (0.40207501366967335+3.37073785536783e-09j), (0.4167054308927618+3.814004667821634e-09j), (0.4002585801354144+2.446038090896387e-09j), (0.4303837029146962-1.2765808408746308e-09j), (0.3967903334123548+2.3017394657481344e-09j), (0.41980897646863014+2.9067303984626775e-11j)], [(0.4407654279493727+6.6460223991543204e-09j), (0.45407585936482064-4.899014530580857e-10j), (0.41914294127491303-6.785594346171997e-11j), (0.4321016071771737-1.1060595778735134e-09j), (0.3981755619170144-2.668082387192883e-09j), (0.40738566397340037+2.8781240462592357e-10j), (0.4132399578811601-1.784526900240213e-10j), (0.43478326668264344-1.2524294021541797e-09j), (0.4169050650962163-4.1028555413989395e-09j), (0.4045347274513915+2.0101175171791423e-09j), (0.44847143872175366-1.7901511372402874e-10j), (0.4386463266564533+1.1574950442052088e-09j), (0.39538262999849394-8.424263230965193e-10j), (0.4099400107515976+4.848295098451972e-10j), (0.3942832263710443+6.302047774228745e-10j), (0.41829804205917753-2.8539057783000143e-09j), (0.38723617140203714-3.8065994787149336e-10j), (0.38329861240345053-6.445012299865917e-10j), (0.42091708281077445-2.6131878887307857e-09j), (0.398779904760886+1.5474971305758746e-09j)], [(0.5165518499561585-1.5353115173266744e-09j), (0.4971804771339521+8.248752848441826e-09j), (0.4524991920334287-1.306510674330387e-12j), (0.4174719273869414-7.633342845704737e-10j), (0.43732022793847136-2.2320744145665814e-09j), (0.42722231478546746-2.101480230274414e-09j), (0.4309578419197351-5.212171821762284e-10j), (0.39360892586410046-1.4905313320245381e-09j), (0.43470675725257024+1.778650525487857e-09j), (0.4315440631180536-5.552697729619131e-10j), (0.41615458225714974+1.396601334685241e-09j), (0.4015933560440317-7.408733429003183e-10j), (0.3972705166670494+5.962095265007878e-10j), (0.416523119318299-2.1186510325014394e-09j), (0.4222395774559118+1.6552146322779184e-09j), (0.39810725999996066+4.046476488359537e-10j), (0.4060313611989841+2.411614352671772e-09j), (0.3960011369199492+7.773693357356177e-10j), (0.3949565274233464-1.8493931370948847e-10j), (0.419269266189076-1.0255546369856777e-10j)], [(0.51301144019817-1.9710455493879173e-09j), (0.48921879625413567+3.699578865299722e-09j), (0.40707585198106244-5.631609071394385e-10j), (0.44720059545943514-1.9642079078116496e-09j), (0.4559990401030518+7.142289155714976e-10j), (0.41255159501452+2.7944005623164366e-10j), (0.4254875839978922-1.1979110642887909e-09j), (0.3955773736233823-4.0503052439587573e-10j), (0.4427835848473478+5.357811282442581e-10j), (0.4014656686631497+1.7793655438608862e-10j), (0.4213382338930387-3.788767077754192e-10j), (0.4245370542339515+8.799263534986273e-10j), (0.42496221329201944-5.677403133350488e-10j), (0.41720766000798903+3.738680393153697e-09j), (0.4180496388289612+4.3518886272701884e-09j), (0.40705121573409997-9.180423646247582e-10j), (0.44023069157265127+3.800785041061616e-09j), (0.4153005451662466-2.4462580846207367e-09j), (0.4241874498256948+3.140272485960398e-10j), (0.396776496025268-6.549916188790048e-11j)], [(0.5276658600923838-5.41751121745356e-09j), (0.4761913654219825-1.9626213988359154e-09j), (0.45770257245749235+3.0452272926429822e-09j), (0.43841497210087255-3.487022448638751e-09j), (0.42769637765013613+2.4504354099751615e-09j), (0.41577361058443785-2.284992639629091e-09j), (0.43107759454869665+3.293842560022606e-10j), (0.40444627893157303+1.7601795107630244e-10j), (0.44883916145772673-4.443545661505152e-09j), (0.3965674020291772+3.485470661779314e-10j), (0.4292611246346496-2.122552717243106e-09j), (0.4086554395908024-1.1689365448982052e-09j), (0.4035837879637256-1.472420657804305e-10j), (0.41998858549050055+3.5707137240221836e-10j), (0.4202306145161856-2.5197754992971657e-10j), (0.4418152311409358+1.742612276673799e-09j), (0.43304459570208564+2.557776213055016e-11j), (0.4395025482517667+1.2767525920080876e-09j), (0.4299365149636287+4.757026328640943e-09j), (0.4052120980049949-4.237567630448794e-09j)], [(0.474119284626795+3.633091605081768e-10j), (0.46259508025832474-2.5664229629388832e-09j), (0.4278991803585086+3.3954830059030517e-09j), (0.4363883239857387-1.0462268271508814e-09j), (0.41078835912048817+1.2848093142620802e-09j), (0.4273523987212684-1.6797400133336218e-10j), (0.41242206892638933-5.4223152981521e-09j), (0.4540787332516629+9.052096452535592e-10j), (0.45739691928611137+7.997542678099086e-10j), (0.42988821129256394-1.9444231224924983e-09j), (0.43732162375818007+5.418564485622695e-10j), (0.42335999039642047+9.097230049452024e-10j), (0.4141825025435537-7.65621028474451e-10j), (0.43640413126558997-6.416584485819649e-10j), (0.4265215945924865+1.2188661624235565e-09j), (0.3974140864011133-1.0335010645085552e-10j), (0.41268362346454524+2.7285160792560597e-09j), (0.4348682672716677-4.731219860413701e-10j), (0.4092097994289361-2.755136323875686e-09j), (0.45293908211169764-7.875699122455759e-10j)], [(0.5615002640988678+4.308690252756623e-09j), (0.5160105275572278-8.242984078906372e-10j), (0.45516398041218054-8.282512452889641e-10j), (0.4422264608583646+4.841800737764308e-09j), (0.41319150051276665-2.808569377088981e-09j), (0.4344443841109751-1.346468491508161e-10j), (0.4399049067287706-3.18129216553347e-10j), (0.4517763591138646+1.1503807023584614e-09j), (0.41961569151317235-2.051764592336189e-09j), (0.41317802856792696-2.080397469081616e-09j), (0.43181514988827985+2.5375264388729295e-10j), (0.459166563290637-3.5684870665834357e-10j), (0.42164402223716024+3.316718844235696e-09j), (0.4184394290932687+1.040363802104688e-09j), (0.43460054966271855+3.627040889853106e-09j), (0.42771571705816314+1.0418710341444622e-09j), (0.4333148727455409-2.7571719888261967e-09j), (0.4558992679230869+1.6952406140224217e-10j), (0.43802041086019017-1.7350911542209236e-09j), (0.41947462399548385+2.262562248761669e-09j)], [(0.5855257219227497+5.886214847604555e-10j), (0.4976321025606012-5.173421690820646e-09j), (0.4446670222823741-4.8347743588244116e-09j), (0.44745266290556174+8.546040600051621e-11j), (0.4453364537184825-6.943464096178355e-09j), (0.4531908554636175+1.0672395016005442e-09j), (0.4388508855190594+3.60928938115041e-09j), (0.42998747459205333-2.773271513963329e-10j), (0.42703427046944853+4.693592186150324e-10j), (0.45233770814957097-1.46411471869477e-09j), (0.42256952344905585+2.7175294931081806e-09j), (0.4297422494564671+1.6297709848468978e-09j), (0.4074492208601441+1.8124848825024744e-10j), (0.40837082905636635-6.910666761789282e-10j), (0.4390740820381325+1.2151871141443338e-10j), (0.4376170872274088-5.357347454240897e-09j), (0.40621410838502925+4.624730738172937e-10j), (0.43187504399975296-1.0722018831532799e-09j), (0.4382828577363398+2.7041754947316963e-10j), (0.41253971672267653-3.502631074085882e-09j)], [(0.5043936653746641-3.5359271067013054e-09j), (0.5165361391409533+3.215575403035337e-09j), (0.45766754337819293+2.1809061223439442e-09j), (0.43391079967841506+1.3478516079250406e-09j), (0.44849461258854717+2.03738048622388e-09j), (0.41564118016685825-2.6041672133227703e-10j), (0.43076561190537177+1.3720878668204155e-09j), (0.4188713108887896-1.8242409789244738e-09j), (0.41603891203703824+5.193762279951404e-10j), (0.43874468650028575+2.0435293458873933e-09j), (0.46420702055911534+2.571744859833212e-09j), (0.4483548019488808+9.195663811940705e-10j), (0.45856239715067204+1.1524292601495641e-10j), (0.4374612769315718-1.752460371895521e-09j), (0.4135210563108558+1.5126975224633125e-09j), (0.4415514548163628+2.784389496459676e-09j), (0.43243708403315395-4.822866553072278e-10j), (0.431685901407036+1.2394689719148394e-09j), (0.4382028801192064-8.999190329334823e-11j), (0.43391159326711204-1.2016687642144065e-10j)], [(0.5622097491868772+3.762892442093425e-09j), (0.5126303429278778-2.0075585639314524e-09j), (0.48377681577403564-1.6352794675873243e-09j), (0.45654370296688285-9.189132789666836e-10j), (0.46641983807785437+7.806911776501838e-10j), (0.4544486146332929-3.786803475081718e-09j), (0.46045962344214786-4.4806220215859655e-10j), (0.46461127830843907+7.229374400495338e-10j), (0.4583839876577258+4.092370895406329e-11j), (0.43454412945720833+1.4417499494833572e-10j), (0.4607519990677247+7.854441541761616e-10j), (0.44639888349047396+2.0811210534509424e-09j), (0.4411199916576152-1.8574889680654803e-09j), (0.4239607848066953-5.022028342759242e-10j), (0.4299549088755157-3.530293835372769e-10j), (0.4166995335035608-1.2649498881369808e-09j), (0.4347499737632461-2.130334464502082e-09j), (0.43189808596798684+1.5825452615079737e-09j), (0.4496995371300727-1.0897833748663022e-09j), (0.4224613529149792+2.1140422764649374e-10j)], [(0.6172275147546316-4.737611525353389e-09j), (0.5089501140028005-3.8616915777958455e-09j), (0.495679871433822-2.9941800986362895e-09j), (0.4595940739818616-5.470547360998938e-10j), (0.4570629001100315-2.7617579337345677e-09j), (0.4549773530452512-2.8391099474585774e-09j), (0.4680725306097884+2.894615557504379e-09j), (0.4349413125382853+1.5864167174129842e-09j), (0.4297534096331219+3.4773606146539147e-09j), (0.4492203154586605+6.575696095657531e-10j), (0.4657975167647237+2.880607969043446e-10j), (0.4478397926868638-1.043006270531159e-09j), (0.4459203172737034+1.908889860356895e-09j), (0.4475440356109175+1.38495703778679e-09j), (0.428070556728926-2.6479756165116537e-09j), (0.4553958466567565-3.917062675165743e-09j), (0.464034599965089+8.408352902826894e-10j), (0.44950257646269165-5.058743138857125e-10j), (0.44176781980058877-2.0249182212771068e-10j), (0.42863175513775786-1.2739791441834518e-09j)], [(0.6358998304640409+3.7492560167606825e-09j), (0.5571662225120235-4.6128212332768145e-10j), (0.48464073019567877+2.4608945990636903e-09j), (0.480114676123776-1.1058127750882666e-09j), (0.4693898505429388+9.989683865644283e-10j), (0.45422696956666186+3.1104389377405363e-10j), (0.4668198607978411-1.534578019413728e-09j), (0.4444493242481258+1.2091720240776581e-09j), (0.463434462697478-1.816984099553632e-09j), (0.469961455106386+2.4447395149556958e-09j), (0.45244107833423186+1.054244691967065e-09j), (0.4706113449574332+1.5398862252689265e-09j), (0.44441009996808134+2.928963738608043e-09j), (0.46265016122197267-4.4624601239212795e-10j), (0.4479126433725469+1.677722405288345e-09j), (0.4536541429333738-1.5312053844317527e-09j), (0.4600255975383334-4.341189546593812e-09j), (0.44475000933744013-3.5164348102550518e-09j), (0.429288733190333+1.0266112403604458e-09j), (0.4270880066687823+1.735488417769773e-10j)], [(0.6057604900070146-1.6844448059046784e-09j), (0.5643450293973729-5.247728030141524e-09j), (0.49102803949426743-5.911344610386934e-10j), (0.4668007837390178-5.365332179094154e-09j), (0.4822346462897258-1.1190370713278886e-09j), (0.4569797685107915-6.353119549381378e-10j), (0.4571033983156667-2.419656280913788e-09j), (0.453668985159311-2.5462543461228557e-09j), (0.460671896697022-7.151019778112616e-10j), (0.4623882184459944-4.827673262096319e-10j), (0.4515916265445412-2.3733153780455e-09j), (0.4648642545362236-5.249218838617866e-10j), (0.46055440110649215+8.199399603637108e-09j), (0.4632505460467655-2.1127840431286122e-10j), (0.46156317827262683-2.049699965800428e-09j), (0.45133681206789333+2.3721220534199763e-09j), (0.4483453737520904-4.659427850128511e-10j), (0.4549571247844142-4.963172622916873e-09j), (0.4489842483490065+1.7168551023141613e-09j), (0.46145135643018875-4.762410244235282e-09j)], [(0.6635651819924533+7.407374269513205e-09j), (0.5639755891170353+3.839041973250991e-09j), (0.5012753016853821-1.9351691362081883e-09j), (0.46925682733854046+1.4764906241405719e-09j), (0.4622722407548281-1.8340145801109964e-09j), (0.45434208874212345+2.7189812067292558e-09j), (0.45156644671806134-5.494504713258319e-09j), (0.4665069675029372+7.677617666293007e-09j), (0.47437134257052094+5.985543781874133e-09j), (0.4589937343989732+2.1008679448214446e-09j), (0.47250469808932394-3.340892007904944e-10j), (0.44405354031550814+1.48817319083741e-10j), (0.46874595408371533+2.8273530037920477e-09j), (0.44699116945048445-8.774546855844726e-11j), (0.4565151143760886+7.232792138630187e-10j), (0.45753186385627487+1.0935332637063383e-09j), (0.45275432085691136-5.826805703293793e-10j), (0.4545976501394762+2.113499631871684e-11j), (0.44456840639395523+6.440603609239501e-10j), (0.46192072312987875-2.6868932622592114e-10j)], [(0.6929881106134417-1.3607731830234685e-08j), (0.5769850083197525+3.348764820737609e-09j), (0.5154564952845249-2.278003702526568e-09j), (0.4920591226145916+3.3888447521704656e-10j), (0.47222934207457-8.674843510419196e-09j), (0.4713129683477746-1.0528611176002426e-09j), (0.48245091269564+1.5365259974572475e-09j), (0.4564032856033009-7.022160285011503e-10j), (0.4652447252992715-1.9726845348249015e-09j), (0.47736135041486705+4.13477809495766e-09j), (0.4663282448218524+5.087419933697764e-10j), (0.4647809025536844-5.25294467575959e-10j), (0.46239300105298753+1.1499436400445275e-09j), (0.4609509461588459-2.0654207236114896e-09j), (0.45061349908246484+5.292942001163555e-09j), (0.4699732081826369+1.136949422658548e-09j), (0.4608713529960369+1.5699608278204225e-09j), (0.45874239041950204+9.312888236980064e-10j), (0.46954494258534396+5.794669186484698e-10j), (0.4651178763851931-1.6921264399242303e-09j)], [(0.7645633181282392-1.502429996382607e-08j), (0.6385556895584159-3.624533562434096e-09j), (0.5443711416064616+9.12388386845886e-10j), (0.49996834325793316-2.5340751723900037e-09j), (0.4811883581405709-2.404842290049612e-10j), (0.47296372301752854-9.602724483357806e-10j), (0.48574788447876927+7.781082736088488e-10j), (0.4623655543764471-7.189933648651633e-10j), (0.46988132445949304-5.56794346734256e-10j), (0.4745217056952242-2.4770642199000484e-09j), (0.4727052213202114+1.2447802782001753e-09j), (0.4695304553642927+1.0291819029699326e-09j), (0.4658152639094624-1.2083887226986816e-09j), (0.45590815649120486+3.515595589939108e-10j), (0.46632662956108106+3.9368019830138403e-10j), (0.4536143725272268+1.5683681296682482e-09j), (0.4689862510531384-2.4393047585440378e-09j), (0.4600703586147574-1.4013049059001439e-09j), (0.4600628273747134-1.5580869952296494e-10j), (0.4539118320117268+9.135570909432571e-10j)]])
ss = np.array([[(0.9999993741512299+1.3155790345997787e-09j), (0.8734378814697266+2.9909186232183744e-09j), (0.7709031105041504-4.0277188251280904e-09j), (0.6878336668014526-1.3701515720240026e-10j), (0.6205344200134277-1.8732366387219646e-09j), (0.5660113096237183+1.455697881680429e-09j), (0.5218391120433807+4.626597549517442e-09j), (0.48605261743068695-6.96159063728885e-09j), (0.45705994963645935-8.25699501827426e-10j), (0.43357129395008087-1.8014517308428957e-10j), (0.4145417660474777-3.5541909831904306e-09j), (0.3991248160600662-7.27236808421747e-12j), (0.3866347074508667-3.770203694380969e-09j), (0.3765156716108322+1.4708219531200939e-08j), (0.3683176785707474+1.3960695756983638e-09j), (0.3616761267185211+2.249545882904158e-09j), (0.35629527270793915-5.125043729776923e-10j), (0.35193607211112976+9.07041308728651e-09j), (0.34840431809425354+5.97783333944335e-09j), (0.3455430418252945-5.043973873991581e-09j)], [(0.9999994337558746+1.4831007531508902e-08j), (0.8734377920627594-1.9205799173249716e-09j), (0.7709029912948608+4.802293562811144e-10j), (0.687833696603775+1.9320611499562546e-09j), (0.6205343902111053-8.867654610611453e-09j), (0.566011369228363-1.0410847739450446e-09j), (0.5218391716480255-1.6504112659854187e-09j), (0.48605261743068695-7.456404244043924e-10j), (0.4570598751306534+2.7153070902841137e-09j), (0.4335712641477585-1.4562882913460307e-09j), (0.41454170644283295+1.2650222384370802e-10j), (0.3991248607635498+4.43823273076592e-09j), (0.3866347074508667-2.9103830456733704e-10j), (0.376515731215477-6.249240057723537e-11j), (0.36831775307655334+2.5856758401054947e-09j), (0.36167609691619873-1.2475753208285312e-09j), (0.35629524290561676+4.268437450716256e-09j), (0.3519360423088074-7.978058769175789e-10j), (0.34840428829193115-6.657504769690244e-09j), (0.34554311633110046+7.45058059692341e-09j)], [(0.9999993145465851-1.564318267976983e-08j), (0.8734378218650818-4.688865412667699e-10j), (0.7709030210971832-5.587935447692871e-09j), (0.6878336668014526-1.4767040923402419e-08j), (0.6205343306064606-5.2731586076826265e-09j), (0.5660114288330078-6.976647592971119e-10j), (0.5218391418457031+2.995180561904398e-10j), (0.48605260252952576+3.200631620847716e-09j), (0.45705990493297577-1.4347034671397184e-09j), (0.4335712492465973+9.005854603727492e-10j), (0.41454170644283295+4.325906813318348e-09j), (0.39912480115890503+2.2802773003149923e-09j), (0.3866346925497055+3.9065977119889794e-09j), (0.3765157461166382+1.1827800975163875e-09j), (0.36831770837306976+2.6069204572885718e-09j), (0.3616761118173599+1.3333464887743673e-09j), (0.35629531741142273+1.2701474403492563e-09j), (0.351936012506485-7.3278538790343125e-09j), (0.34840433299541473-1.8605167684260238e-09j), (0.3455430418252945-2.502749683285263e-10j)], [(0.9999993741512299+1.4663883440846348e-09j), (0.8734379410743713+1.9249881688665482e-09j), (0.7709029912948608-2.163486723105734e-09j), (0.6878337264060974+1.0476657857747962e-18j), (0.6205344498157501+7.653596756362901e-09j), (0.5660113394260406-5.983583947766213e-09j), (0.5218391120433807+3.3478310124124278e-09j), (0.48605260252952576-2.1443948006183433e-10j), (0.45705990493297577+8.835325110423398e-10j), (0.4335712790489197+1.3505279117254076e-09j), (0.41454172134399414+4.945364118214002e-10j), (0.39912478625774384+2.1432704500942257e-09j), (0.3866346925497055-2.738463899730397e-10j), (0.37651580572128296-5.772547321768631e-11j), (0.36831772327423096-5.957124488142895e-09j), (0.3616761416196823+4.072259966736436e-09j), (0.35629531741142273-2.276965034705633e-10j), (0.35193605720996857+1.4901161193828825e-08j), (0.34840428829193115-4.759115768958926e-09j), (0.3455430567264557-1.3875757997875595e-11j)], [(0.9999994039535522-1.511676506193993e-08j), (0.8734378218650818+2.6627643919675334e-18j), (0.7709030508995056-1.5267414000203066e-08j), (0.6878336668014526+2.8729330026067146e-08j), (0.6205343902111053+7.463009765729112e-09j), (0.5660114586353302+7.085911579718385e-10j), (0.5218391418457031+2.5422192129198606e-10j), (0.48605261743068695+3.0062452527346295e-11j), (0.45705994963645935-8.268642742725874e-09j), (0.4335712492465973-5.902533739554627e-10j), (0.41454170644283295+1.1414794387487603e-09j), (0.3991248309612274-6.706023969164584e-11j), (0.3866346925497055+2.2210334960082179e-10j), (0.3765157014131546-2.614377382151163e-09j), (0.36831772327423096+4.967423361534884e-09j), (0.36167609691619873+5.497003967806802e-10j), (0.35629530251026154-7.334165275096893e-09j), (0.3519360274076462+2.266310972220964e-09j), (0.34840430319309235+3.9613573532548685e-09j), (0.3455430418252945+1.1646715362967353e-10j)], [(0.9999994337558746+6.026263932401334e-10j), (0.8734378218650818-1.5204096509569e-08j), (0.7709031105041504+2.820118027990759e-09j), (0.687833696603775+5.497819974032013e-10j), (0.6205344200134277-1.4338542408953714e-08j), (0.566011369228363-2.748742622404876e-09j), (0.5218390822410583+2.5316105548706114e-11j), (0.48605264723300934-6.95374671039195e-10j), (0.45705991983413696-7.45131172649044e-09j), (0.4335712343454361+8.756060546760702e-11j), (0.41454169154167175-1.9399803985464814e-10j), (0.39912480115890503-5.999456713822206e-19j), (0.3866346925497055-2.1548475226396135e-09j), (0.3765157461166382+3.606305615244665e-09j), (0.36831773817539215-7.568804474410627e-10j), (0.3616761267185211-8.242100918431916e-09j), (0.35629527270793915+1.1088988616236861e-09j), (0.3519359827041626+2.5501445399811473e-10j), (0.34840431809425354+1.5721860140524376e-18j), (0.3455430716276169+7.450580596968696e-09j)], [(0.9999993443489075+1.4650343382882625e-08j), (0.8734378814697266+2.0059820471381176e-09j), (0.7709030508995056+7.831967990812316e-09j), (0.6878337264060974+7.1961391290287224e-09j), (0.6205344200134277+2.5691582195008777e-09j), (0.566011369228363-2.7048430162324166e-09j), (0.5218391418457031-6.401890351170891e-09j), (0.48605264723300934-5.066386520383159e-09j), (0.45705990493297577+4.5894635314347454e-11j), (0.43357130885124207-2.961931366840531e-09j), (0.4145417660474777+5.210944870048806e-10j), (0.3991248607635498+4.666151662106877e-10j), (0.3866346776485443+1.3040345692161281e-08j), (0.3765157610177994+4.656612868277607e-10j), (0.36831772327423096-9.313225746603287e-10j), (0.36167606711387634+2.506428398163507e-09j), (0.35629527270793915+5.36360145186876e-09j), (0.35193607211112976-2.4087008776429997e-09j), (0.34840431809425354-2.328306470829626e-10j), (0.3455430418252945-7.450580597194715e-09j)], [(0.9999993145465851+1.034864323105289e-17j), (0.8734378516674042+9.339704565292095e-10j), (0.7709029912948608+8.181780457492602e-11j), (0.6878337860107422+6.7915928436690365e-09j), (0.6205343902111053-9.201079342879837e-10j), (0.5660114884376526-6.101786263756903e-11j), (0.5218391418457031+2.2154034162724656e-09j), (0.48605260252952576+3.7306026323680186e-11j), (0.45705993473529816-1.2834240337156189e-09j), (0.43357130885124207-7.450580596811898e-09j), (0.41454169154167175-2.0321227686692644e-10j), (0.3991248309612274+1.534949253912225e-11j), (0.3866347074508667+1.3354774673277059e-09j), (0.37651577591896057-4.6695827204956686e-09j), (0.36831769347190857+8.110811577921595e-09j), (0.3616761118173599+8.187833522306917e-09j), (0.35629525780677795+6.371594085674559e-09j), (0.35193605720996857-4.038133091799345e-09j), (0.34840427339076996+1.5364222782920933e-08j), (0.3455430269241333-2.559287329246781e-12j)], [(0.9999992549419403-1.1175870895385742e-08j), (0.8734378516674042-3.0400983952461047e-09j), (0.7709030508995056+4.625462236257262e-09j), (0.6878337562084198-1.902058391583528e-18j), (0.6205343902111053+6.608696272517034e-11j), (0.5660114586353302-6.067696345368745e-11j), (0.5218392312526703+3.6374899620161827e-10j), (0.48605263233184814+1.6168107495190265e-09j), (0.45705990493297577+9.694627045586657e-11j), (0.4335712194442749-7.904367826938596e-09j), (0.41454175114631653-3.5399794207080504e-09j), (0.3991248160600662-6.450796452561747e-09j), (0.3866346925497055+2.824607214790831e-10j), (0.3765156865119934-1.4594169622794695e-09j), (0.36831772327423096-4.381762674920964e-09j), (0.3616761416196823+8.596541260824211e-10j), (0.35629524290561676-1.8482800429042712e-09j), (0.3519359976053238+5.6802131886968255e-09j), (0.34840428829193115-1.862645149230957e-09j), (0.3455430418252945+5.742471177522688e-14j)], [(0.9999992847442627+1.629726076313176e-08j), (0.873437762260437-1.0791160631740127e-08j), (0.7709029912948608-1.5456529389723528e-09j), (0.6878336668014526+4.734224421341915e-09j), (0.6205344498157501-7.42471772952058e-09j), (0.5660113990306854+5.435907626805125e-10j), (0.5218392014503479+3.2306950448202088e-09j), (0.48605261743068695+4.796956831754073e-09j), (0.45705991983413696+7.657193767940385e-10j), (0.4335712641477585-2.0621082619243225e-10j), (0.4145417660474777+1.7893196633271565e-10j), (0.3991248309612274+1.2145258132534305e-09j), (0.3866347223520279+8.74085250230572e-11j), (0.376515731215477-3.512256602103072e-10j), (0.36831776797771454-1.4564606187761342e-09j), (0.3616761565208435-1.33212958546014e-09j), (0.35629530251026154+1.1762882334024597e-10j), (0.3519359976053238-4.704988482151506e-10j), (0.34840431809425354-4.8403474561808935e-09j), (0.3455430865287781+4.382259842496644e-11j)], [(0.9999993145465851-3.887384747436329e-09j), (0.8734377324581146-6.134334928908913e-09j), (0.7709029614925385-4.0435801373917e-09j), (0.6878337562084198-1.715213542350343e-19j), (0.6205344796180725-5.890539667152694e-09j), (0.5660113990306854+3.1680509882114904e-09j), (0.5218391120433807-2.763413109452273e-11j), (0.48605257272720337-3.455503660987347e-10j), (0.45705996453762054+4.2908676611830277e-10j), (0.43357129395008087+9.402658918394174e-10j), (0.41454172134399414+3.779190807517985e-09j), (0.3991248905658722+1.296339311937511e-09j), (0.3866346478462219-4.36479869608819e-10j), (0.3765157163143158+7.511845027952102e-09j), (0.3683176785707474+8.912793170168243e-09j), (0.3616761714220047-1.3197449799662309e-09j), (0.35629530251026154-1.0915558040780482e-10j), (0.35193607211112976-9.312179378301311e-10j), (0.34840430319309235+2.6345169101826826e-09j), (0.3455430567264557+6.636598381959402e-09j)], [(0.9999994337558746-6.769908189596663e-09j), (0.873437762260437+3.577313362201906e-09j), (0.7709031105041504-7.095423719979123e-10j), (0.6878337562084198-5.49173773123357e-09j), (0.620534360408783+4.798612725925855e-09j), (0.5660113990306854-1.6640833155889823e-10j), (0.5218391716480255+8.298138988449555e-09j), (0.48605263233184814+3.0845539455981452e-09j), (0.45705993473529816+1.548417505325972e-10j), (0.4335712790489197-5.465160712869732e-09j), (0.41454175114631653-8.103168136486261e-11j), (0.3991248607635498+1.7071485752806481e-09j), (0.3866347372531891+6.541406860627319e-10j), (0.37651579082012177+1.2413868552599658e-10j), (0.36831773817539215+1.16415321816365e-10j), (0.36167605221271515+2.0559778324269473e-09j), (0.35629528760910034+9.313225733106358e-10j), (0.351936012506485+3.0551190466354683e-09j), (0.34840433299541473-3.045090598077782e-09j), (0.3455430269241333-5.587935447692871e-09j)], [(0.9999991953372955+9.337975670486998e-10j), (0.8734378218650818-1.1744290303747427e-18j), (0.7709030210971832+7.45058059690243e-09j), (0.6878336369991302+1.9372161097486185e-09j), (0.6205344796180725+2.1848577391736512e-09j), (0.566011369228363+1.4548862129911914e-10j), (0.5218391716480255-4.536212960815078e-09j), (0.48605260252952576+4.150887480958154e-10j), (0.45705991983413696-2.328306436994022e-10j), (0.4335712641477585+8.614879087875948e-10j), (0.41454173624515533-4.949081977567715e-10j), (0.39912478625774384-3.8218828102287716e-10j), (0.3866347074508667-1.5052001589577202e-09j), (0.3765157163143158-4.30530661121864e-09j), (0.36831770837306976+3.3081417466496177e-10j), (0.3616761416196823+1.4068523235266639e-08j), (0.35629528760910034+4.65661291277901e-10j), (0.35193605720996857+1.0814962259075855e-09j), (0.34840431809425354-2.3283064221977792e-10j), (0.3455430418252945-5.128735214299395e-10j)], [(0.9999993741512299-6.009572839360544e-10j), (0.8734378218650818-1.0283313760828234e-08j), (0.7709030508995056+1.980301256310213e-09j), (0.6878337562084198-4.461971888840932e-10j), (0.6205344200134277-1.4264561754018246e-11j), (0.566011369228363-3.7262959384776195e-09j), (0.5218391716480255+6.570260092930713e-09j), (0.48605263233184814-7.1382477151438195e-09j), (0.45705994963645935-3.04797270755941e-10j), (0.4335712492465973+2.2143510679569984e-10j), (0.41454169154167175+1.735316335071957e-09j), (0.3991248309612274+2.3934328963193252e-09j), (0.3866347074508667-1.4935947478811329e-09j), (0.3765157014131546+7.868914964070939e-10j), (0.36831775307655334-4.4682941791052144e-09j), (0.3616761416196823+9.313225785004563e-10j), (0.35629524290561676+3.858667607659072e-09j), (0.3519360423088074+3.150453786737728e-09j), (0.34840428829193115-3.2952043871325998e-09j), (0.3455430567264557+8.047951198003262e-10j)], [(0.9999993145465851-2.6111628415037558e-08j), (0.8734377324581146+1.050785678398874e-09j), (0.7709029912948608-8.125779216161533e-10j), (0.6878337860107422-7.450580600376654e-09j), (0.6205344200134277-2.758603234731538e-09j), (0.5660113990306854+3.436359252706467e-10j), (0.5218391418457031+8.504460691227678e-09j), (0.48605260252952576+1.406377542201298e-09j), (0.45705996453762054-1.4881545093436976e-08j), (0.4335712343454361+1.7865309231979154e-10j), (0.41454175114631653+5.386349699598725e-09j), (0.39912478625774384-9.524584121400892e-09j), (0.3866347223520279-7.5373451334515e-09j), (0.376515731215477-3.691506267333722e-10j), (0.36831770837306976+1.446403591983848e-11j), (0.36167609691619873-2.98152702526977e-09j), (0.35629528760910034-7.176562372146478e-09j), (0.3519360423088074+1.436318163364203e-09j), (0.34840431809425354+2.688848033116642e-10j), (0.3455430865287781-9.436397080397896e-10j)], [(0.9999994039535522+2.910382792038205e-11j), (0.8734378218650818+4.9046583600054205e-09j), (0.7709029912948608+1.2159330209371433e-09j), (0.6878336668014526+1.5062434355339605e-09j), (0.6205343902111053-5.665811707864479e-10j), (0.5660114288330078+7.886374220308972e-09j), (0.5218391418457031+3.3335894669694888e-09j), (0.48605261743068695-1.023080073783711e-09j), (0.45705990493297577+1.3935069763300081e-10j), (0.4335712492465973-6.966462962217485e-09j), (0.41454175114631653+1.0683350037954398e-11j), (0.3991248607635498+3.0133207040705656e-09j), (0.3866346627473831-2.6020152699146593e-09j), (0.3765157461166382-5.602292518780416e-10j), (0.36831775307655334+5.820766092441566e-11j), (0.3616761118173599-5.296772048746234e-10j), (0.35629527270793915+3.0500074688077916e-10j), (0.351936012506485+3.970415113403643e-10j), (0.34840428829193115+4.6566128730773926e-09j), (0.3455430567264557+1.821736247964445e-09j)], [(0.9999992549419403+2.6522981588783523e-09j), (0.8734377026557922-6.003376907690362e-09j), (0.7709030508995056+2.1759558155309833e-08j), (0.687833696603775+2.0682940915506265e-09j), (0.6205343902111053-6.059561741267316e-09j), (0.5660114586353302+4.322342816998059e-09j), (0.5218391716480255-1.559311083032533e-10j), (0.48605263233184814+1.7329149226696927e-09j), (0.45705996453762054+1.5030638844670818e-08j), (0.4335712194442749+3.5272368341745705e-10j), (0.4145417809486389+5.041681724882174e-10j), (0.3991248160600662-2.234852193388169e-09j), (0.3866346925497055+3.59728912429864e-10j), (0.3765157163143158+3.7252902984605153e-09j), (0.36831770837306976-3.6088749766349792e-09j), (0.36167608201503754+9.313225746154785e-09j), (0.3562953472137451+5.70352681750208e-10j), (0.35193605720996857-3.942526305422689e-09j), (0.34840433299541473+5.364960697979404e-10j), (0.3455430269241333+6.885861603134202e-09j)], [(0.9999992847442627-4.733461733685829e-09j), (0.8734377920627594+1.1816356282334795e-08j), (0.7709030210971832-5.2852944243397815e-11j), (0.687833696603775+9.312657314846089e-10j), (0.6205344200134277-4.2762393001605226e-09j), (0.5660114884376526-1.566462474924396e-09j), (0.5218391120433807+2.0510623754965707e-10j), (0.48605263233184814-3.5087807104261515e-09j), (0.4570598900318146+8.783114346933019e-09j), (0.4335712790489197-4.704360734297808e-09j), (0.41454173624515533+6.631884993425763e-11j), (0.39912480115890503+1.7959028708602887e-10j), (0.3866346627473831-7.40034211688112e-09j), (0.376515731215477+1.2835744690469893e-09j), (0.36831772327423096-3.372655266176139e-09j), (0.36167609691619873+4.2052196169706235e-09j), (0.35629528760910034+7.870601503867647e-09j), (0.3519360274076462-9.725871885635229e-09j), (0.34840427339076996+7.335009120923441e-09j), (0.34554310142993927+1.8626451492309568e-09j)], [(0.9999994933605194-1.3665912068638875e-08j), (0.8734377920627594-1.639505862831481e-09j), (0.7709029316902161+1.2137957750013584e-08j), (0.6878336668014526-7.899349396822686e-09j), (0.6205344200134277+1.5834410727055825e-09j), (0.5660113394260406+2.2569662805337274e-10j), (0.5218391716480255-1.997690031898505e-11j), (0.48605258762836456+7.103224675120146e-10j), (0.45705994963645935+1.3447087584064765e-10j), (0.43357130885124207+4.157540312021979e-09j), (0.41454173624515533+8.247014739593989e-10j), (0.3991248309612274-5.873584951743283e-10j), (0.3866347074508667+4.733818614877094e-10j), (0.3765157014131546+3.988128708367242e-12j), (0.36831772327423096-1.0380289405809151e-09j), (0.36167609691619873-5.917569989566681e-09j), (0.35629531741142273+3.732558179203593e-09j), (0.35193605720996857+1.963977314112242e-09j), (0.34840427339076996+5.329985341973043e-10j), (0.3455430269241333+1.0872331802219692e-09j)], [(0.9999993145465851-5.7230373218253305e-11j), (0.873437911272049+1.9364698932919366e-09j), (0.7709030508995056-3.4428265793806645e-09j), (0.6878337860107422-1.4911398330821233e-08j), (0.6205344200134277-2.0125201505219564e-09j), (0.5660113990306854+7.313565309719449e-09j), (0.5218391120433807-7.287159695112955e-10j), (0.48605260252952576-3.538667969760212e-11j), (0.4570598900318146-7.412362388188907e-10j), (0.4335712641477585-3.643383303386649e-09j), (0.41454169154167175-2.097358869912469e-09j), (0.3991248160600662-3.725290298618076e-09j), (0.3866346776485443+2.5983068475464724e-10j), (0.37651579082012177-6.812761466079564e-11j), (0.36831770837306976+6.4973848522746595e-09j), (0.3616761118173599-5.470073560776001e-10j), (0.35629528760910034+6.419474955490888e-10j), (0.3519360423088074+7.083559780407533e-11j), (0.3484043478965759-4.357935958289091e-10j), (0.3455430865287781-7.645167165985412e-09j)]])
cs = np.array([[(0.40913504536729306+2.935110155037424e-09j), (0.403768727905117-7.387491507554644e-09j), (0.4009171579964459-2.5144686361827677e-10j), (0.39965810102876276-5.476388233892598e-10j), (0.39918756391853094-3.0894813375027013e-09j), (0.39904815377667546+3.9456006099307295e-09j), (0.39902223041281104-2.058316100894686e-09j), (0.39901925274170935+2.4553877955183043e-09j), (0.39901010401081294+3.362376849289523e-09j), (0.39898863004054874-3.1008040027604865e-09j), (0.39895765425171703-6.736434632145645e-09j), (0.398921694024466+4.965715616200496e-09j), (0.3988849165616557+6.505928569340011e-10j), (0.3988496921956539-9.326813765619012e-09j), (0.398817305220291-1.1327112584102463e-09j), (0.39878873783163726-4.785957297243015e-09j), (0.39876400399953127-1.362008728291383e-09j), (0.39874300453811884-1.3049841211383694e-09j), (0.398725452250801+3.5195348854858556e-09j)], [(0.4178527002222836-4.5319481501457845e-09j), (0.4109545375686139+9.205308626546507e-09j), (0.40716320206411183-1.0036549369470201e-09j), (0.4054772867821157+2.001883103955709e-09j), (0.40485246235039085-7.894249697614725e-10j), (0.40467256878037006+3.1714917916681517e-09j), (0.40464182931464165+6.107212028329221e-11j), (0.4046387415146455+3.4674195187665545e-09j), (0.40462392137851566-2.3511969034452086e-09j), (0.4045908027328551+4.463800218844458e-09j), (0.40454417769797146-8.340382827094813e-10j), (0.40449057274963707+3.0674950361220286e-09j), (0.40443600330036134+6.823817074481505e-10j), (0.40438420232385397+3.445669972124852e-09j), (0.40433681185822934+2.4508156622034993e-09j), (0.4042950478615239-1.1764300638048386e-10j), (0.4042588312877342+1.354974799449897e-09j), (0.4042283802991733-2.304664015232614e-09j), (0.4042027310933918+1.7216883472721878e-09j)], [(0.42859146813862026-4.669377773401988e-09j), (0.4194664820097387+3.173866058470606e-09j), (0.41427199641475454+3.6806113700873477e-10j), (0.41194193076808006+4.138905634934596e-09j), (0.411082279169932-7.29438723349849e-11j), (0.41084001038689166-7.246348187830662e-09j), (0.4108016254613176-2.720385431375485e-09j), (0.41079770668875426-6.664220370083119e-09j), (0.41077456367202103+1.0761299357230943e-09j), (0.41072392696514726+2.6785166862610807e-09j), (0.41065376659389585+6.893209225678167e-09j), (0.4105739425867796-1.518796088870242e-08j), (0.41049298003781587+7.140159576182773e-10j), (0.4104160802671686+1.154074580371823e-08j), (0.4103462138446048-2.0300503495393557e-09j), (0.41028452501632273+3.4823548492792932e-09j), (0.41023137839511037-3.3488360967835634e-09j), (0.4101864444091916-2.1055290839190923e-09j), (0.4101487231673673+9.730931620264205e-10j)], [(0.44207754562376067+2.9347217988509143e-09j), (0.42973758559674025+3.336196208214805e-09j), (0.42245490063214675+1.0971756830647765e-09j), (0.41915582679212093-2.530151532367238e-09j), (0.4179404456517659+8.529870143164345e-10j), (0.41760257055284455-9.838575509367084e-10j), (0.41755189350806177-7.884201722060683e-09j), (0.4175465978332795+2.9896346230667527e-09j), (0.41751080652466044+4.677758180881068e-09j), (0.41743405867600814-1.3583391631792522e-09j), (0.41732863237848505+5.422368776400603e-09j), (0.4172092346707359-1.9969228491381736e-09j), (0.4170885195489973+7.83899212243393e-09j), (0.41697416902752593-1.0415736049595101e-09j), (0.4168702973984182-8.603382673016676e-09j), (0.4167788731283508-5.261876268065508e-09j), (0.41670029622036964+4.020545230133078e-09j), (0.41663375339703634+7.473823781047989e-10j), (0.4165781650226563+7.4176491618740566e-09j)], [(0.45932331518270075-3.4045575247887427e-09j), (0.44238286023028195-4.759920443927771e-09j), (0.43200549366883934+1.3351941775389311e-09j), (0.4272530963062309-1.1716600190059268e-09j), (0.4255002942518331+2.4781139917860372e-09j), (0.42501736688427627-1.103825046643455e-08j), (0.4249475251417607+2.0876836715781436e-10j), (0.4249400718254037+2.525983588915583e-09j), (0.4248850886942819-4.983741253023361e-09j), (0.42476926383096725+7.585312711249188e-09j), (0.4246107583749108-2.6438506383599346e-09j), (0.4244321142323315+4.928556229761265e-09j), (0.42425166442990303-1.3467201576133904e-08j), (0.4240808536997065-8.436813470500879e-10j), (0.4239260546746664+7.175313675309304e-10j), (0.42378994409227744-5.965664406934205e-09j), (0.4236728720134124-2.6062549895963055e-09j), (0.4235740912845358+7.325964612706333e-09j), (0.4234914444386959-6.786516570971876e-09j)], [(0.481729956867639-1.3198257153841025e-08j), (0.4582775341696106-9.461328609384698e-09j), (0.4433339539973531-6.123373808009092e-09j), (0.43640973858418874-1.1849579872918488e-08j), (0.43384853383759037-4.395188421508183e-09j), (0.4331459257809911-5.794170102027804e-09j), (0.4330466295941733-4.675090849147103e-09j), (0.4330357907747384-1.5249677215253683e-09j), (0.43295212724478915-6.465913728048722e-09j), (0.43277707090601325-7.216679587271083e-09j), (0.43253889435436577-4.9522129452558705e-09j), (0.43227122715325095-8.115681460196811e-10j), (0.43200089025776833-5.875415709127566e-10j), (0.4317454931733664+2.841458846011572e-09j), (0.4315143798594363-3.2657939130764646e-09j), (0.43131110235117376+2.700702716998667e-09j), (0.4311364663008135-8.949644580999185e-09j), (0.4309890216391068-2.171015423030405e-09j), (0.43086571869207546+2.660116682496649e-09j)], [(0.5112283816124545+2.1759070989443034e-08j), (0.4786706033628434+1.1306462432950345e-08j), (0.45701954918331467-8.722377043011272e-09j), (0.4468599612591788-7.280304359580267e-09j), (0.4430874931567814+3.379541090930732e-09j), (0.44205378962215036+1.5731505076214127e-10j), (0.44191011984366924+1.3317987189716188e-09j), (0.44189360618474893-9.19830426783351e-10j), (0.441766563890269-7.344935992249672e-09j), (0.4415030067320913+4.258848552416437e-09j), (0.4411453068314586-1.9750984183028733e-10j), (0.4407437119516544+1.886785339127758e-09j), (0.4403387512138579+3.382698454705554e-09j), (0.4399560443707742+1.1119491744423974e-08j), (0.43960979851544835-3.767024247674524e-09j), (0.43930593045661226-4.199318674190055e-10j), (0.4390449099591933+6.385235116889077e-10j), (0.4388244140718598+1.2235038539783286e-09j), (0.4386401880183257-2.811342658249012e-09j)], [(0.5504632222291548+6.781625927492558e-09j), (0.5053545639675576+7.218432074329829e-09j), (0.4738952655898174-4.742803482975273e-09j), (0.45893124279973563+3.790000312983469e-09j), (0.453347492904868+3.1330198435238e-09j), (0.45181743716239+1.7253688344764407e-09j), (0.45160598440270405+6.425327045265632e-09j), (0.4515808985743206+5.130560185316825e-09j), (0.4513892012037104+6.47514700217487e-09j), (0.4509928451734595-9.116031596169402e-09j), (0.4504553994338494+1.3430841153500805e-09j), (0.4498531445860863+3.2361180313898544e-10j), (0.4492457970191026+4.2231036440704666e-09j), (0.4486725363822188+1.8588983968141604e-09j), (0.44815405036206357-5.208174558432385e-09j), (0.4476991136325523-2.807629865179675e-09j), (0.4473083776392741+2.981785235030357e-09j), (0.4469783697568346+9.80690240159174e-09j), (0.44670294840761926-5.092748000759214e-09j)], [(0.6030157102140947-1.5080500459431154e-08j), (0.5408941139030503+1.1171827907219267e-09j), (0.49515634970157407-4.664966857248487e-09j), (0.4730773799237795-5.330651920174104e-09j), (0.4647929490020033+2.243902646718905e-09j), (0.4625196719280211+4.770680239740699e-09j), (0.46220683594583534+4.913808526712088e-09j), (0.46216926656779833-9.57280435476271e-10j), (0.46188000931579154-6.073566316544543e-09j), (0.46128421337198233-1.1139128577959752e-09j), (0.46047751772857737+7.60308273257494e-09j), (0.45957424298103433-4.911367202127497e-09j), (0.4586641195201082-5.644088085787757e-09j), (0.45780493938946165+9.274908514513434e-09j), (0.4570283557259245-3.0164729464218046e-09j), (0.4563472311710939-4.527786035000849e-09j), (0.4557625142188044-7.115200653953367e-10j), (0.45526867194712395-6.807406333384915e-09j), (0.45485643293795874-8.077570973510249e-09j)], [(0.6736849296794389+1.3033572443532648e-08j), (0.5889657839907159-1.705674268492885e-09j), (0.5225327253574505+5.663579827341891e-09j), (0.48994465967916767+4.879963099685831e-09j), (0.4776408957222884+1.0065993594823772e-08j), (0.47425825202299166+5.261269974878855e-09j), (0.47379381603605-3.202231778370374e-09j), (0.4737369341601152-4.932267004763045e-09j), (0.47330205244361423-2.1182095688758433e-08j), (0.4724074294936145-1.5028169042575755e-08j), (0.4711977625520376-1.0906633377717913e-09j), (0.4698439589155896-7.078691855149558e-09j), (0.4684806043151184-9.69237136132431e-10j), (0.46719398081768304+7.229306930407108e-09j), (0.46603155643242644+5.959705173348072e-09j), (0.46501212368821143+1.6424799289316185e-09j), (0.4641372899277485+5.173607763819789e-09j), (0.4633986776243546-1.1880476159013881e-08j), (0.46278226425056346-2.5033499720650888e-09j)]])

ss = np.array([[(0.9999993145465851-5.048908846561062e-09j), (0.8671576380729675+9.027865107030534e-10j), (0.7607862949371338-4.942420036061455e-10j), (0.67561075091362+1.9896768677085674e-09j), (0.6074075400829315+7.317297004427736e-09j), (0.5527946949005127-1.9545243190848688e-11j), (0.5090640485286713+1.1633419921830424e-09j), (0.47404739260673523+3.6411890169674166e-10j), (0.44600820541381836-4.680831278136566e-10j), (0.4235561788082123+3.2344666181516324e-09j), (0.40557803213596344-2.606920456955044e-09j), (0.39118219912052155+2.6362368954480075e-09j), (0.37965500354766846+1.7752500580050804e-09j), (0.37042464315891266-3.725290298356115e-09j), (0.36303360760211945-5.218325328081803e-09j), (0.35711534321308136+1.8253593081415518e-09j), (0.3523763120174408+6.7956127947077505e-09j), (0.34858161211013794-2.3283064365368227e-10j), (0.3455430120229721-1.016193906799856e-09j)], [(0.9999993145465851-6.6037060264534375e-09j), (0.8671576380729675-2.488320670231368e-10j), (0.7607862651348114+1.4609779303009773e-08j), (0.6756108105182648-2.8640632088183793e-09j), (0.6074075400829315+1.0177387821495068e-09j), (0.5527946352958679-9.591800964603436e-11j), (0.5090640783309937+2.535202645037593e-09j), (0.47404737770557404-2.1677523109886465e-09j), (0.44600823521614075-1.994431370055949e-09j), (0.42355620861053467-6.931949736443954e-09j), (0.40557795763015747+2.5456481367314154e-09j), (0.39118218421936035-4.4770225304802125e-09j), (0.3796549588441849+7.316581411911938e-09j), (0.37042468786239624+3.7249245713735432e-09j), (0.36303362250328064+7.451816941284051e-13j), (0.3571152985095978-9.49394122345959e-12j), (0.3523763120174408-1.1641532182693481e-10j), (0.34858161211013794-3.6953952653803833e-19j), (0.3455430418252945-7.457162887741151e-09j)], [(0.9999994337558746+6.311278885938744e-18j), (0.8671574890613556-1.5192934809532543e-09j), (0.7607862651348114-1.4478228171910246e-08j), (0.6756107807159424-1.2584608910515271e-08j), (0.6074075698852539+7.274221669462122e-09j), (0.5527946352958679+7.452408943642644e-09j), (0.5090641677379608+6.556074017005642e-10j), (0.4740474075078964-9.514220938866558e-10j), (0.44600823521614075-1.3159777711990728e-09j), (0.4235561788082123+7.218278752496587e-10j), (0.40557798743247986+1.1704708313686751e-09j), (0.39118219912052155+1.2929022293593008e-09j), (0.37965497374534607+6.784617338238142e-10j), (0.37042468786239624-9.83447323062913e-10j), (0.36303360760211945-5.587935447692871e-09j), (0.35711534321308136-2.40153802399945e-09j), (0.3523763120174408+4.824022015181839e-10j), (0.34858161211013794-2.539857768546483e-10j), (0.34554310142993927+1.2526979441139323e-11j)], [(0.9999993443489075+2.9937242129740582e-09j), (0.8671575486660004+8.478101110043212e-09j), (0.760786235332489-2.2361642537660487e-09j), (0.6756107807159424+2.191739914558788e-10j), (0.6074075698852539-1.2000619109553412e-09j), (0.5527946054935455+2.1138640837747857e-09j), (0.5090641379356384+7.5058015913676e-09j), (0.47404736280441284+1.418686529364166e-09j), (0.44600819051265717-3.8668142255879334e-10j), (0.4235561490058899-2.0739238104638957e-09j), (0.40557801723480225+4.053286610401326e-10j), (0.39118218421936035-9.0978179183027e-13j), (0.3796549439430237+9.528768885047612e-10j), (0.37042465806007385-4.147765242379364e-09j), (0.36303359270095825-6.254397010643442e-10j), (0.35711535811424255-7.881401545284383e-09j), (0.3523762822151184-1.172802502899875e-09j), (0.3485816419124603-7.521947509303573e-09j), (0.3455430567264557-4.477213766396204e-09j)], [(0.9999994039535522-1.4675075044046082e-10j), (0.8671576082706451-2.2602598903631588e-08j), (0.7607862651348114-1.7411725394822497e-09j), (0.6756108105182648-4.775231322435086e-10j), (0.6074075400829315-4.0440099925449086e-09j), (0.5527946650981903-1.3054354298935866e-10j), (0.5090641677379608-4.66925320630196e-09j), (0.47404733300209045-3.1329506003374696e-10j), (0.44600822031497955-2.0066640831623017e-09j), (0.42355623841285706+2.2958286383101267e-09j), (0.4055780619382858+1.8074788332711478e-09j), (0.39118216931819916-2.7922788526218745e-09j), (0.3796549439430237-2.2247332275016074e-10j), (0.37042465806007385+4.615915440636183e-10j), (0.36303363740444183+5.255170213921967e-11j), (0.35711534321308136-2.175689948242397e-20j), (0.3523762971162796+2.6843428591050156e-09j), (0.3485816419124603-2.9103834308711984e-11j), (0.3455429971218109+1.174381314417161e-09j)], [(0.9999993741512299+4.142014731374898e-09j), (0.8671575784683228+1.7622212570887473e-09j), (0.7607863247394562+1.8047806304988256e-11j), (0.6756107211112976-1.9488070046591588e-09j), (0.6074075400829315-4.070957727298064e-10j), (0.5527946650981903+2.0486916611000723e-09j), (0.509064108133316+5.877892964023523e-11j), (0.47404736280441284-1.699414831099787e-09j), (0.44600822031497955-8.584929656052509e-09j), (0.4235561788082123-1.1067075034803846e-09j), (0.40557803213596344+5.042379402988969e-09j), (0.39118216931819916-3.638994747895419e-11j), (0.3796549439430237+1.7372979270458e-10j), (0.37042465806007385-3.220999411635006e-09j), (0.36303360760211945+8.29700164128866e-09j), (0.35711531341075897-1.584761988304706e-09j), (0.3523763120174408+2.4180157922397455e-10j), (0.34858162701129913+3.094341727294392e-09j), (0.3455430567264557+2.5586555102572168e-09j)], [(0.9999994039535522-6.1804632522921565e-09j), (0.8671576082706451+1.3643282874981066e-09j), (0.7607862651348114+5.9913901341360585e-12j), (0.67561075091362-3.423629046928056e-09j), (0.6074075698852539+6.710584310454237e-10j), (0.5527946352958679-4.3725272014238925e-09j), (0.509064108133316-7.3091502295064e-10j), (0.47404734790325165+8.005904911234651e-09j), (0.44600819051265717-1.016436605022486e-09j), (0.4235561937093735+1.0162595209725933e-09j), (0.40557801723480225+1.6079289174767348e-11j), (0.39118216931819916+4.308092771410088e-11j), (0.3796549439430237+7.45058059692367e-09j), (0.37042461335659027+4.73336037032368e-10j), (0.36303360760211945-1.164153220946354e-10j), (0.35711534321308136+1.054020870583372e-09j), (0.35237637162208557-8.876117618683566e-10j), (0.34858159720897675+7.978890992355048e-09j), (0.3455430567264557-3.859410790951756e-09j)], [(0.9999995231628418+3.725290298461914e-09j), (0.8671576380729675-1.462458210088613e-10j), (0.7607863247394562-7.84547538046354e-09j), (0.6756106913089752+8.328354117459469e-10j), (0.6074075698852539+1.3597727499536916e-08j), (0.5527946949005127+3.2875921429642574e-09j), (0.5090640485286713+3.0872535083936725e-10j), (0.47404737770557404-2.722500558347329e-10j), (0.44600822031497955+7.44930064681959e-09j), (0.42355620861053467+5.389956814205732e-10j), (0.40557801723480225-7.440801030877964e-10j), (0.39118218421936035-2.3283064365527827e-10j), (0.37965497374534607-1.980857408656611e-09j), (0.37042468786239624-7.824706438990948e-10j), (0.36303362250328064+2.0198005767912974e-10j), (0.35711532831192017-1.2735182908230058e-09j), (0.352376326918602-2.353753303374617e-10j), (0.3485816866159439-9.299719833047441e-11j), (0.3455430865287781+6.7560714800407595e-09j)], [(0.9999993443489075-9.320611712992921e-10j), (0.8671576082706451-2.165512824614524e-09j), (0.760786235332489+5.471046171656724e-09j), (0.67561075091362-1.898345259557678e-09j), (0.6074075400829315-1.5360049288037914e-08j), (0.5527946650981903+4.783890079863504e-10j), (0.5090640187263489+3.7075167658162655e-09j), (0.47404739260673523-1.73356652991008e-09j), (0.44600822031497955-3.4246981639451946e-10j), (0.4235561788082123+7.515141519504054e-09j), (0.40557801723480225-1.2845338126510342e-09j), (0.39118222892284393-2.1406736938622828e-09j), (0.3796549141407013-1.2284835371190184e-09j), (0.37042467296123505-2.7676009262123102e-09j), (0.36303363740444183-5.248650403189004e-10j), (0.35711532831192017-1.943752485356054e-10j), (0.352376326918602-2.3374013835564256e-10j), (0.3485816568136215+5.0849350771708934e-11j), (0.3455430567264557-4.744917792010028e-09j)], [(0.9999994039535522+1.7140653272596703e-09j), (0.8671575486660004+9.259136577030244e-09j), (0.7607862949371338-1.4088302802139907e-08j), (0.6756107211112976+6.44581110709197e-09j), (0.6074075996875763+2.223695449998786e-09j), (0.5527945458889008-9.102510412084541e-11j), (0.509064108133316-1.5727116050440193e-10j), (0.47404734790325165-4.6452689210409875e-10j), (0.44600823521614075+3.72529029841935e-09j), (0.4235561937093735+2.1001039813406004e-10j), (0.40557804703712463-7.50699655216458e-10j), (0.39118222892284393+2.015833722168735e-09j), (0.3796549588441849+9.381503629501964e-10j), (0.37042470276355743-7.769500431908938e-09j), (0.36303360760211945-4.75978145786371e-09j), (0.35711534321308136+1.3758009176356722e-09j), (0.3523763418197632-1.2007681513482544e-09j), (0.34858158230781555-5.40455791231409e-10j), (0.3455430865287781+1.3574863455545483e-11j)]])

'''
T, MS = np.linspace(0,100,20), np.linspace(0.9, 1, 20)
T, MS = np.meshgrid(np.linspace(0,100,19), np.linspace(0.9, 1, 10))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = cs - np.min(cs)
colors /= np.max(colors)
ax.plot_surface(T, MS, cs, cmap='viridis')
ax.plot_wireframe(T, MS, ss, color='r')
ax.set_xlabel('t, mcs (T0 = 25 mcs)')
ax.set_ylabel('ms2')
ax.set_zlabel('fid')
plt.title('P1 = 0.99')
plt.legend()
plt.grid()
plt.show()
'''
#graph_3d(0.95,1,4, 100, 2, 5)

'''
cs = [[(0.40913504536729306+2.935110155037424e-09j), (0.403768727905117-7.387491507554644e-09j), (0.4009171579964459-2.5144686361827677e-10j), (0.39965810102876276-5.476388233892598e-10j), (0.39918756391853094-3.0894813375027013e-09j), (0.39904815377667546+3.9456006099307295e-09j), (0.39902223041281104-2.058316100894686e-09j), (0.39901925274170935+2.4553877955183043e-09j), (0.39901010401081294+3.362376849289523e-09j), (0.39898863004054874-3.1008040027604865e-09j), (0.39895765425171703-6.736434632145645e-09j), (0.398921694024466+4.965715616200496e-09j), (0.3988849165616557+6.505928569340011e-10j), (0.3988496921956539-9.326813765619012e-09j), (0.398817305220291-1.1327112584102463e-09j), (0.39878873783163726-4.785957297243015e-09j), (0.39876400399953127-1.362008728291383e-09j), (0.39874300453811884-1.3049841211383694e-09j), (0.398725452250801+3.5195348854858556e-09j)], [(0.4178527002222836-4.5319481501457845e-09j), (0.4109545375686139+9.205308626546507e-09j), (0.40716320206411183-1.0036549369470201e-09j), (0.4054772867821157+2.001883103955709e-09j), (0.40485246235039085-7.894249697614725e-10j), (0.40467256878037006+3.1714917916681517e-09j), (0.40464182931464165+6.107212028329221e-11j), (0.4046387415146455+3.4674195187665545e-09j), (0.40462392137851566-2.3511969034452086e-09j), (0.4045908027328551+4.463800218844458e-09j), (0.40454417769797146-8.340382827094813e-10j), (0.40449057274963707+3.0674950361220286e-09j), (0.40443600330036134+6.823817074481505e-10j), (0.40438420232385397+3.445669972124852e-09j), (0.40433681185822934+2.4508156622034993e-09j), (0.4042950478615239-1.1764300638048386e-10j), (0.4042588312877342+1.354974799449897e-09j), (0.4042283802991733-2.304664015232614e-09j), (0.4042027310933918+1.7216883472721878e-09j)], [(0.42859146813862026-4.669377773401988e-09j), (0.4194664820097387+3.173866058470606e-09j), (0.41427199641475454+3.6806113700873477e-10j), (0.41194193076808006+4.138905634934596e-09j), (0.411082279169932-7.29438723349849e-11j), (0.41084001038689166-7.246348187830662e-09j), (0.4108016254613176-2.720385431375485e-09j), (0.41079770668875426-6.664220370083119e-09j), (0.41077456367202103+1.0761299357230943e-09j), (0.41072392696514726+2.6785166862610807e-09j), (0.41065376659389585+6.893209225678167e-09j), (0.4105739425867796-1.518796088870242e-08j), (0.41049298003781587+7.140159576182773e-10j), (0.4104160802671686+1.154074580371823e-08j), (0.4103462138446048-2.0300503495393557e-09j), (0.41028452501632273+3.4823548492792932e-09j), (0.41023137839511037-3.3488360967835634e-09j), (0.4101864444091916-2.1055290839190923e-09j), (0.4101487231673673+9.730931620264205e-10j)], [(0.44207754562376067+2.9347217988509143e-09j), (0.42973758559674025+3.336196208214805e-09j), (0.42245490063214675+1.0971756830647765e-09j), (0.41915582679212093-2.530151532367238e-09j), (0.4179404456517659+8.529870143164345e-10j), (0.41760257055284455-9.838575509367084e-10j), (0.41755189350806177-7.884201722060683e-09j), (0.4175465978332795+2.9896346230667527e-09j), (0.41751080652466044+4.677758180881068e-09j), (0.41743405867600814-1.3583391631792522e-09j), (0.41732863237848505+5.422368776400603e-09j), (0.4172092346707359-1.9969228491381736e-09j), (0.4170885195489973+7.83899212243393e-09j), (0.41697416902752593-1.0415736049595101e-09j), (0.4168702973984182-8.603382673016676e-09j), (0.4167788731283508-5.261876268065508e-09j), (0.41670029622036964+4.020545230133078e-09j), (0.41663375339703634+7.473823781047989e-10j), (0.4165781650226563+7.4176491618740566e-09j)], [(0.45932331518270075-3.4045575247887427e-09j), (0.44238286023028195-4.759920443927771e-09j), (0.43200549366883934+1.3351941775389311e-09j), (0.4272530963062309-1.1716600190059268e-09j), (0.4255002942518331+2.4781139917860372e-09j), (0.42501736688427627-1.103825046643455e-08j), (0.4249475251417607+2.0876836715781436e-10j), (0.4249400718254037+2.525983588915583e-09j), (0.4248850886942819-4.983741253023361e-09j), (0.42476926383096725+7.585312711249188e-09j), (0.4246107583749108-2.6438506383599346e-09j), (0.4244321142323315+4.928556229761265e-09j), (0.42425166442990303-1.3467201576133904e-08j), (0.4240808536997065-8.436813470500879e-10j), (0.4239260546746664+7.175313675309304e-10j), (0.42378994409227744-5.965664406934205e-09j), (0.4236728720134124-2.6062549895963055e-09j), (0.4235740912845358+7.325964612706333e-09j), (0.4234914444386959-6.786516570971876e-09j)], [(0.481729956867639-1.3198257153841025e-08j), (0.4582775341696106-9.461328609384698e-09j), (0.4433339539973531-6.123373808009092e-09j), (0.43640973858418874-1.1849579872918488e-08j), (0.43384853383759037-4.395188421508183e-09j), (0.4331459257809911-5.794170102027804e-09j), (0.4330466295941733-4.675090849147103e-09j), (0.4330357907747384-1.5249677215253683e-09j), (0.43295212724478915-6.465913728048722e-09j), (0.43277707090601325-7.216679587271083e-09j), (0.43253889435436577-4.9522129452558705e-09j), (0.43227122715325095-8.115681460196811e-10j), (0.43200089025776833-5.875415709127566e-10j), (0.4317454931733664+2.841458846011572e-09j), (0.4315143798594363-3.2657939130764646e-09j), (0.43131110235117376+2.700702716998667e-09j), (0.4311364663008135-8.949644580999185e-09j), (0.4309890216391068-2.171015423030405e-09j), (0.43086571869207546+2.660116682496649e-09j)], [(0.5112283816124545+2.1759070989443034e-08j), (0.4786706033628434+1.1306462432950345e-08j), (0.45701954918331467-8.722377043011272e-09j), (0.4468599612591788-7.280304359580267e-09j), (0.4430874931567814+3.379541090930732e-09j), (0.44205378962215036+1.5731505076214127e-10j), (0.44191011984366924+1.3317987189716188e-09j), (0.44189360618474893-9.19830426783351e-10j), (0.441766563890269-7.344935992249672e-09j), (0.4415030067320913+4.258848552416437e-09j), (0.4411453068314586-1.9750984183028733e-10j), (0.4407437119516544+1.886785339127758e-09j), (0.4403387512138579+3.382698454705554e-09j), (0.4399560443707742+1.1119491744423974e-08j), (0.43960979851544835-3.767024247674524e-09j), (0.43930593045661226-4.199318674190055e-10j), (0.4390449099591933+6.385235116889077e-10j), (0.4388244140718598+1.2235038539783286e-09j), (0.4386401880183257-2.811342658249012e-09j)], [(0.5504632222291548+6.781625927492558e-09j), (0.5053545639675576+7.218432074329829e-09j), (0.4738952655898174-4.742803482975273e-09j), (0.45893124279973563+3.790000312983469e-09j), (0.453347492904868+3.1330198435238e-09j), (0.45181743716239+1.7253688344764407e-09j), (0.45160598440270405+6.425327045265632e-09j), (0.4515808985743206+5.130560185316825e-09j), (0.4513892012037104+6.47514700217487e-09j), (0.4509928451734595-9.116031596169402e-09j), (0.4504553994338494+1.3430841153500805e-09j), (0.4498531445860863+3.2361180313898544e-10j), (0.4492457970191026+4.2231036440704666e-09j), (0.4486725363822188+1.8588983968141604e-09j), (0.44815405036206357-5.208174558432385e-09j), (0.4476991136325523-2.807629865179675e-09j), (0.4473083776392741+2.981785235030357e-09j), (0.4469783697568346+9.80690240159174e-09j), (0.44670294840761926-5.092748000759214e-09j)], [(0.6030157102140947-1.5080500459431154e-08j), (0.5408941139030503+1.1171827907219267e-09j), (0.49515634970157407-4.664966857248487e-09j), (0.4730773799237795-5.330651920174104e-09j), (0.4647929490020033+2.243902646718905e-09j), (0.4625196719280211+4.770680239740699e-09j), (0.46220683594583534+4.913808526712088e-09j), (0.46216926656779833-9.57280435476271e-10j), (0.46188000931579154-6.073566316544543e-09j), (0.46128421337198233-1.1139128577959752e-09j), (0.46047751772857737+7.60308273257494e-09j), (0.45957424298103433-4.911367202127497e-09j), (0.4586641195201082-5.644088085787757e-09j), (0.45780493938946165+9.274908514513434e-09j), (0.4570283557259245-3.0164729464218046e-09j), (0.4563472311710939-4.527786035000849e-09j), (0.4557625142188044-7.115200653953367e-10j), (0.45526867194712395-6.807406333384915e-09j), (0.45485643293795874-8.077570973510249e-09j)], [(0.6736849296794389+1.3033572443532648e-08j), (0.5889657839907159-1.705674268492885e-09j), (0.5225327253574505+5.663579827341891e-09j), (0.48994465967916767+4.879963099685831e-09j), (0.4776408957222884+1.0065993594823772e-08j), (0.47425825202299166+5.261269974878855e-09j), (0.47379381603605-3.202231778370374e-09j), (0.4737369341601152-4.932267004763045e-09j), (0.47330205244361423-2.1182095688758433e-08j), (0.4724074294936145-1.5028169042575755e-08j), (0.4711977625520376-1.0906633377717913e-09j), (0.4698439589155896-7.078691855149558e-09j), (0.4684806043151184-9.69237136132431e-10j), (0.46719398081768304+7.229306930407108e-09j), (0.46603155643242644+5.959705173348072e-09j), (0.46501212368821143+1.6424799289316185e-09j), (0.4641372899277485+5.173607763819789e-09j), (0.4633986776243546-1.1880476159013881e-08j), (0.46278226425056346-2.5033499720650888e-09j)]]

ss = [[(0.9999993145465851-5.048908846561062e-09j), (0.8671576380729675+9.027865107030534e-10j), (0.7607862949371338-4.942420036061455e-10j), (0.67561075091362+1.9896768677085674e-09j), (0.6074075400829315+7.317297004427736e-09j), (0.5527946949005127-1.9545243190848688e-11j), (0.5090640485286713+1.1633419921830424e-09j), (0.47404739260673523+3.6411890169674166e-10j), (0.44600820541381836-4.680831278136566e-10j), (0.4235561788082123+3.2344666181516324e-09j), (0.40557803213596344-2.606920456955044e-09j), (0.39118219912052155+2.6362368954480075e-09j), (0.37965500354766846+1.7752500580050804e-09j), (0.37042464315891266-3.725290298356115e-09j), (0.36303360760211945-5.218325328081803e-09j), (0.35711534321308136+1.8253593081415518e-09j), (0.3523763120174408+6.7956127947077505e-09j), (0.34858161211013794-2.3283064365368227e-10j), (0.3455430120229721-1.016193906799856e-09j)], [(0.9999993145465851-6.6037060264534375e-09j), (0.8671576380729675-2.488320670231368e-10j), (0.7607862651348114+1.4609779303009773e-08j), (0.6756108105182648-2.8640632088183793e-09j), (0.6074075400829315+1.0177387821495068e-09j), (0.5527946352958679-9.591800964603436e-11j), (0.5090640783309937+2.535202645037593e-09j), (0.47404737770557404-2.1677523109886465e-09j), (0.44600823521614075-1.994431370055949e-09j), (0.42355620861053467-6.931949736443954e-09j), (0.40557795763015747+2.5456481367314154e-09j), (0.39118218421936035-4.4770225304802125e-09j), (0.3796549588441849+7.316581411911938e-09j), (0.37042468786239624+3.7249245713735432e-09j), (0.36303362250328064+7.451816941284051e-13j), (0.3571152985095978-9.49394122345959e-12j), (0.3523763120174408-1.1641532182693481e-10j), (0.34858161211013794-3.6953952653803833e-19j), (0.3455430418252945-7.457162887741151e-09j)], [(0.9999994337558746+6.311278885938744e-18j), (0.8671574890613556-1.5192934809532543e-09j), (0.7607862651348114-1.4478228171910246e-08j), (0.6756107807159424-1.2584608910515271e-08j), (0.6074075698852539+7.274221669462122e-09j), (0.5527946352958679+7.452408943642644e-09j), (0.5090641677379608+6.556074017005642e-10j), (0.4740474075078964-9.514220938866558e-10j), (0.44600823521614075-1.3159777711990728e-09j), (0.4235561788082123+7.218278752496587e-10j), (0.40557798743247986+1.1704708313686751e-09j), (0.39118219912052155+1.2929022293593008e-09j), (0.37965497374534607+6.784617338238142e-10j), (0.37042468786239624-9.83447323062913e-10j), (0.36303360760211945-5.587935447692871e-09j), (0.35711534321308136-2.40153802399945e-09j), (0.3523763120174408+4.824022015181839e-10j), (0.34858161211013794-2.539857768546483e-10j), (0.34554310142993927+1.2526979441139323e-11j)], [(0.9999993443489075+2.9937242129740582e-09j), (0.8671575486660004+8.478101110043212e-09j), (0.760786235332489-2.2361642537660487e-09j), (0.6756107807159424+2.191739914558788e-10j), (0.6074075698852539-1.2000619109553412e-09j), (0.5527946054935455+2.1138640837747857e-09j), (0.5090641379356384+7.5058015913676e-09j), (0.47404736280441284+1.418686529364166e-09j), (0.44600819051265717-3.8668142255879334e-10j), (0.4235561490058899-2.0739238104638957e-09j), (0.40557801723480225+4.053286610401326e-10j), (0.39118218421936035-9.0978179183027e-13j), (0.3796549439430237+9.528768885047612e-10j), (0.37042465806007385-4.147765242379364e-09j), (0.36303359270095825-6.254397010643442e-10j), (0.35711535811424255-7.881401545284383e-09j), (0.3523762822151184-1.172802502899875e-09j), (0.3485816419124603-7.521947509303573e-09j), (0.3455430567264557-4.477213766396204e-09j)], [(0.9999994039535522-1.4675075044046082e-10j), (0.8671576082706451-2.2602598903631588e-08j), (0.7607862651348114-1.7411725394822497e-09j), (0.6756108105182648-4.775231322435086e-10j), (0.6074075400829315-4.0440099925449086e-09j), (0.5527946650981903-1.3054354298935866e-10j), (0.5090641677379608-4.66925320630196e-09j), (0.47404733300209045-3.1329506003374696e-10j), (0.44600822031497955-2.0066640831623017e-09j), (0.42355623841285706+2.2958286383101267e-09j), (0.4055780619382858+1.8074788332711478e-09j), (0.39118216931819916-2.7922788526218745e-09j), (0.3796549439430237-2.2247332275016074e-10j), (0.37042465806007385+4.615915440636183e-10j), (0.36303363740444183+5.255170213921967e-11j), (0.35711534321308136-2.175689948242397e-20j), (0.3523762971162796+2.6843428591050156e-09j), (0.3485816419124603-2.9103834308711984e-11j), (0.3455429971218109+1.174381314417161e-09j)], [(0.9999993741512299+4.142014731374898e-09j), (0.8671575784683228+1.7622212570887473e-09j), (0.7607863247394562+1.8047806304988256e-11j), (0.6756107211112976-1.9488070046591588e-09j), (0.6074075400829315-4.070957727298064e-10j), (0.5527946650981903+2.0486916611000723e-09j), (0.509064108133316+5.877892964023523e-11j), (0.47404736280441284-1.699414831099787e-09j), (0.44600822031497955-8.584929656052509e-09j), (0.4235561788082123-1.1067075034803846e-09j), (0.40557803213596344+5.042379402988969e-09j), (0.39118216931819916-3.638994747895419e-11j), (0.3796549439430237+1.7372979270458e-10j), (0.37042465806007385-3.220999411635006e-09j), (0.36303360760211945+8.29700164128866e-09j), (0.35711531341075897-1.584761988304706e-09j), (0.3523763120174408+2.4180157922397455e-10j), (0.34858162701129913+3.094341727294392e-09j), (0.3455430567264557+2.5586555102572168e-09j)], [(0.9999994039535522-6.1804632522921565e-09j), (0.8671576082706451+1.3643282874981066e-09j), (0.7607862651348114+5.9913901341360585e-12j), (0.67561075091362-3.423629046928056e-09j), (0.6074075698852539+6.710584310454237e-10j), (0.5527946352958679-4.3725272014238925e-09j), (0.509064108133316-7.3091502295064e-10j), (0.47404734790325165+8.005904911234651e-09j), (0.44600819051265717-1.016436605022486e-09j), (0.4235561937093735+1.0162595209725933e-09j), (0.40557801723480225+1.6079289174767348e-11j), (0.39118216931819916+4.308092771410088e-11j), (0.3796549439430237+7.45058059692367e-09j), (0.37042461335659027+4.73336037032368e-10j), (0.36303360760211945-1.164153220946354e-10j), (0.35711534321308136+1.054020870583372e-09j), (0.35237637162208557-8.876117618683566e-10j), (0.34858159720897675+7.978890992355048e-09j), (0.3455430567264557-3.859410790951756e-09j)], [(0.9999995231628418+3.725290298461914e-09j), (0.8671576380729675-1.462458210088613e-10j), (0.7607863247394562-7.84547538046354e-09j), (0.6756106913089752+8.328354117459469e-10j), (0.6074075698852539+1.3597727499536916e-08j), (0.5527946949005127+3.2875921429642574e-09j), (0.5090640485286713+3.0872535083936725e-10j), (0.47404737770557404-2.722500558347329e-10j), (0.44600822031497955+7.44930064681959e-09j), (0.42355620861053467+5.389956814205732e-10j), (0.40557801723480225-7.440801030877964e-10j), (0.39118218421936035-2.3283064365527827e-10j), (0.37965497374534607-1.980857408656611e-09j), (0.37042468786239624-7.824706438990948e-10j), (0.36303362250328064+2.0198005767912974e-10j), (0.35711532831192017-1.2735182908230058e-09j), (0.352376326918602-2.353753303374617e-10j), (0.3485816866159439-9.299719833047441e-11j), (0.3455430865287781+6.7560714800407595e-09j)], [(0.9999993443489075-9.320611712992921e-10j), (0.8671576082706451-2.165512824614524e-09j), (0.760786235332489+5.471046171656724e-09j), (0.67561075091362-1.898345259557678e-09j), (0.6074075400829315-1.5360049288037914e-08j), (0.5527946650981903+4.783890079863504e-10j), (0.5090640187263489+3.7075167658162655e-09j), (0.47404739260673523-1.73356652991008e-09j), (0.44600822031497955-3.4246981639451946e-10j), (0.4235561788082123+7.515141519504054e-09j), (0.40557801723480225-1.2845338126510342e-09j), (0.39118222892284393-2.1406736938622828e-09j), (0.3796549141407013-1.2284835371190184e-09j), (0.37042467296123505-2.7676009262123102e-09j), (0.36303363740444183-5.248650403189004e-10j), (0.35711532831192017-1.943752485356054e-10j), (0.352376326918602-2.3374013835564256e-10j), (0.3485816568136215+5.0849350771708934e-11j), (0.3455430567264557-4.744917792010028e-09j)], [(0.9999994039535522+1.7140653272596703e-09j), (0.8671575486660004+9.259136577030244e-09j), (0.7607862949371338-1.4088302802139907e-08j), (0.6756107211112976+6.44581110709197e-09j), (0.6074075996875763+2.223695449998786e-09j), (0.5527945458889008-9.102510412084541e-11j), (0.509064108133316-1.5727116050440193e-10j), (0.47404734790325165-4.6452689210409875e-10j), (0.44600823521614075+3.72529029841935e-09j), (0.4235561937093735+2.1001039813406004e-10j), (0.40557804703712463-7.50699655216458e-10j), (0.39118222892284393+2.015833722168735e-09j), (0.3796549588441849+9.381503629501964e-10j), (0.37042470276355743-7.769500431908938e-09j), (0.36303360760211945-4.75978145786371e-09j), (0.35711534321308136+1.3758009176356722e-09j), (0.3523763418197632-1.2007681513482544e-09j), (0.34858158230781555-5.40455791231409e-10j), (0.3455430865287781+1.3574863455545483e-11j)]]

'''

def get_critic_line(c,s, MSS, TT):
    ar = c - s
    #fst = 0
    crt = []
    crms = []
    for ms in range(len(MSS)):
        fst = 0
        for t in range(len(TT)):
            if ar[ms][t] > 0 and fst == 0:
                fst = 1
                crt.append(TT[t])
                crms.append(MSS[ms])
    return crt, crms
'''
T, MS = np.linspace(0,100,19), np.linspace(0.9, 1, 10)
print((c-s)[0])
crt, crms = get_critic_line(c,s, MS, T)
print(crms)
fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot()
ax.scatter(crms, crt, color='b', s=5)
#ax.scatter(t, c, color='r', s=5, label='c коррекции')
k, b = np.polyfit(crms, crt, 1)[0], np.polyfit(crms, crt, 1)[1]
ax.plot([0.9, 1], [0.9 * k + b, k + b], color='r', label='линия эффективности')
ax.set_ylabel('t critical, mcs (T0 = 25 mcs)')
ax.set_xlabel('MS')
plt.title('P1 = 0.99, depolarize')
plt.legend()
plt.grid()

plt.show()
'''
