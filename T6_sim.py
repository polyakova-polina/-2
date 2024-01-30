
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
N = 5000
PMS1 = 1


T0 = 25







X = np.array([[0,1,0], [1,0,0], [0,0,1]])
Y = np.array([[0,complex(0,-1), 0], [complex(0,1), 0, 0], [0,0,1]])
Z = np.array([[1,0,0],[0,-1,0], [0,0,1]])
id = np.eye(3)

z = np.array([[1,0,0]]).T
e = np.array([[0,1,0]]).T
f = np.array([[0,0,1]]).T
basis = [z,e,f]
paulies1 = [id, X, Y, Z]

def dag(matrix):
    return np.conj(matrix.T)

def EE(bas, i, j, p0, paulies, pinv):
    v1 = bas[i]
    v2 = bas[j]
    id = paulies[0]
    x = paulies[1]
    y = paulies[2]
    z = paulies[3]
    K0 = (1-p0)**0.5  * id
    #K0 = id
    K1 = p0**0.5 / 3**0.5 * x
    #K1 = x
    K2 = p0**0.5 / 3**0.5 * y
    #K2 = y
    K3 = p0 ** 0.5 / 3**0.5 * z
    #K3 = z
    #mat_sum = K0 @ dag(K0) + K1 @ dag(K1) + K2 @ dag(K2) + K3 @ dag(K3)
    #print(mat_sum)
    #print(np.trace(mat_sum))
    #print()
    #print(dag(K3))
    _rho = v1 @ (v2.T)
    #print(_rho)
    if i == 0 and j == 0:
        ksgfsg = 0
        #print('eij', K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3))
    #print('ee', np.trace(K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3)))
    #print()
    return (K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3))


def E(bas, i, j, p0, paulies):
    v1 = bas[i]
    v2 = bas[j]
    id = paulies[0]
    x = paulies[1]
    y = paulies[2]
    z = paulies[3]
    #K0 = (1-p0)**0.5 * id
    K0 = id / 2
    #K1 = p0**0.5 / 3**0.5 * x
    K1 = x / 2
    #K2 = p0**0.5 / 3**0.5 * y
    K2 = y / 2
    #K3 = p0 ** 0.5 / 3**0.5 * z
    K3 = z / 2
    #mat_sum = K0 @ dag(K0) + K1 @ dag(K1) + K2 @ dag(K2) + K3 @ dag(K3)
    #print(mat_sum)
    #print()
    #print(dag(K3))
    _rho = v1 @ (v2.T)

    #print(_rho)
    if i == 0 and j == 0:
        cgjjjfjk = 0
    #print('e', np.trace(K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3)))
    #print()
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


class QuditRGate(QuditGate):
    """Rotation between two specified qudit levels: l1 and l2"""

    def __init__(self, l1, l2, theta, phi, dimension=4):
        super().__init__(dimension=dimension)
        levels_connectivity_check(l1, l2)
        self.l1 = l1
        self.l2 = l2
        self.theta = theta
        self.phi = phi

    def _unitary_(self):
        sigma_x = generalized_sigma(1, self.l1, self.l2, dimension=self.d)
        sigma_y = generalized_sigma(2, self.l1, self.l2, dimension=self.d)

        s = np.sin(self.phi)
        c = np.cos(self.phi)

        u = scipy.linalg.expm(-1j * self.theta / 2 * (c * sigma_x + s * sigma_y))

        return u

    def _is_parameterized_(self) -> bool:
        return cirq.protocols.is_parameterized(any((self.theta, self.phi)))

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool):
        return self.__class__(self.l1, self.l2, resolver.value_of(self.theta, recursive), resolver.value_of(self.phi, recursive), dimension=self.d)

    def _circuit_diagram_info_(self, args):
        self.symbol = 'R'
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return f'{self.symbol}{str(self.l1).translate(SUB)}{str(self.l2).translate(SUP)}' + f'({nice_repr(self.theta)}, {nice_repr(self.phi)})'


class QuditXXGate(QuditGate):
    """Two qudit rotation for two specified qudit levels: l1 and l2"""

    def __init__(self, l1, l2, theta, dimension=4):
        levels_connectivity_check(l1, l2)
        super().__init__(dimension=dimension, num_qubits=2)
        self.l1 = l1
        self.l2 = l2
        self.theta = theta

    def _unitary_(self):
        sigma_x = generalized_sigma(1, self.l1, self.l2, dimension=self.d)
        u = scipy.linalg.expm(-1j * self.theta / 2 * np.kron(sigma_x, sigma_x))

        return u

    def _is_parameterized_(self) -> bool:
        return cirq.protocols.is_parameterized(self.theta)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool):
        return self.__class__(self.l1, self.l2, resolver.value_of(self.theta, recursive), dimension=self.d)

    def _circuit_diagram_info_(self, args):
        self.symbol = 'XX'
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        info = f'{self.symbol}{str(self.l1).translate(SUB)}{str(self.l2).translate(SUP)}'.translate(
            SUB) + f'({nice_repr(self.theta)})'
        return info, info


class QuditZZGate(QuditGate):
    """Two qudit rotation for two specified qudit levels: l1 and l2"""

    def __init__(self, l1, l2, theta, dimension=4):
        levels_connectivity_check(l1, l2)
        super().__init__(dimension=dimension, num_qubits=2)
        self.l1 = l1
        self.l2 = l2
        self.theta = theta

    def _unitary_(self):
        sigma_z = generalized_sigma(3, self.l1, self.l2, dimension=self.d)
        u = scipy.linalg.expm(-1j * self.theta / 2 * np.kron(sigma_z, sigma_z))

        return u

    def _is_parameterized_(self) -> bool:
        return cirq.protocols.is_parameterized(self.theta)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool):
        return self.__class__(self.l1, self.l2, resolver.value_of(self.theta, recursive), dimension=self.d)

    def _circuit_diagram_info_(self, args):
        self.symbol = 'ZZ'
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        info = f'{self.symbol}{str(self.l1).translate(SUB)}{str(self.l2).translate(SUP)}'.translate(
            SUB) + f'({nice_repr(self.theta)})'
        return info, info


class QuditBarrier(QuditGate):
    """Just barrier for visual separation in circuit diagrams. Does nothing"""

    def __init__(self, dimension=4, num_qudits=2):
        super().__init__(dimension=dimension, num_qubits=num_qudits)
        self.symbol = '|'

    def _unitary_(self):
        return np.eye(self.d * self.d)


class QuditArbitraryUnitary(QuditGate):
    """Random unitary acts on qubits"""

    def __init__(self, dimension=4, num_qudits=2):
        super().__init__(dimension=dimension, num_qubits=num_qudits)
        self.unitary = np.array(scipy.stats.unitary_group.rvs(self.d ** self.n))
        self.symbol = 'U'

    def _unitary_(self):
        return self.unitary

'''
if __name__ == '__main__':
    n = 3  # number of qudits
    d = 4  # dimension of qudits

    qudits = cirq.LineQid.range(n, dimension=d)

    alpha = sympy.Symbol('alpha')
    beta = sympy.Symbol('beta')

    print('Qudit R Gate')
    circuit = cirq.Circuit(QuditRGate(0, 1, alpha, beta, dimension=d).on(qudits[0]))
    param_resolver = cirq.ParamResolver({'alpha': 0.2, 'beta': 0.3})
    resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)
    print(resolved_circuit)
    print()

    print('Qudit XX Gate')
    circuit = cirq.Circuit(QuditXXGate(0, 2, beta, dimension=d).on(*qudits[:2]))
    param_resolver = cirq.ParamResolver({'alpha': 0.2, 'beta': 0.3})
    resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)
    print(resolved_circuit)
    print()

    print('Qudit Barrier')
    circuit = cirq.Circuit(QuditBarrier(num_qudits=n, dimension=d).on(*qudits))
    print(circuit)
    print()

    print('Qudit Arbitrary Unitary Gate')
    circuit = cirq.Circuit(QuditArbitraryUnitary(num_qudits=n, dimension=d).on(*qudits))
    print(circuit)
'''




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
        '''
        if p_matrix is None:
            self.p_matrix = (1 - self.p1) / (self.d ** 2) * np.ones((self.d, self.d))
            self.p_matrix = np.zeros_like(self.p_matrix)
            #self.p_matrix = np.ones((self.d, self.d))
        else:
            self.p_matrix = p_matrix
        #self.p_matrix[0, 0] += (1 - self.p1)  # identity probability
        for o in range(3):
            for oo in range(3):
                #self.p_matrix[o, oo] = 1 / np.trace(E(basis, o, oo, self.p1, paulies1))
                self.p_matrix[o, oo] = 1 / 9
        #self.p_matrix[0, 0] += 1
        '''

        if p_matrix is None:
            self.p_matrix = self.p1 / (self.d ** 2) * np.ones((self.d, self.d))
        else:
            self.p_matrix = p_matrix
        self.p_matrix[0, 0] += (1 - self.p1)  # identity probability
        self.p_matrix = np.array([[(1 - self.p1), self.p1 / 3], [self.p1 / 3, self.p1 / 3]])
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
        id = np.eye(3)
        shiz_massiv = [id, X, Y, Z]
        return tuple(zip(self.p_matrix.flatten(), shiz_massiv))

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"









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
    #cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)
    adde(cirquit, [u1, u6], [q1, q2], 1)
    #cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    adde(cirquit, [u2], [q1], 1)
    xx = TwoQuditMSGate3()
    #cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    #error(cirquit, [q1, q2], PMS)
    adde(cirquit, [xx], [q1, q2], 2)

def U1_clear(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u6 = U(R(np.pi / 2, -np.pi, 0, 2), 'Ry(-π)02')
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)
    #adde(cirquit, [u1, u6], [q1, q2], 1)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    #adde(cirquit, [u2], [q1], 1)
    xx = TwoQuditMSGate3()
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    #error(cirquit, [q1, q2], PMS)
    #adde(cirquit, [xx], [q1, q2], 2)



def U1_c(cirquit, q1, q2):
    u1 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    u2 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u6 = U(R(np.pi / 2, np.pi, 0, 2), 'Ry(π)02')
    xx_c = TwoQuditMSGate3_c()
    #cirquit.append([xx_c(q1, q2)], strategy=InsertStrategy.INLINE)
    adde(cirquit, [xx_c], [q1, q2], 2)
    #error(cirquit, [q1, q2], PMS)
    adde(cirquit, [u2], [q1], 1)
    #cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    adde(cirquit, [u1, u6], [q1, q2], 1)
    #cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)

def U1_c_clear(cirquit, q1, q2):
    u1 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    u2 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u6 = U(R(np.pi / 2, np.pi, 0, 2), 'Ry(π)02')
    xx_c = TwoQuditMSGate3_c()
    cirquit.append([xx_c(q1, q2)], strategy=InsertStrategy.INLINE)
    #adde(cirquit, [xx_c], [q1, q2], 2)
    #error(cirquit, [q1, q2], PMS)
    #adde(cirquit, [u2], [q1], 1)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    #adde(cirquit, [u1, u6], [q1, q2], 1)
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)

def CX_clear(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 1), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u5 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    #adde(cirquit, [u1], [q1], 1)
    #adde(cirquit, [u2], [q1], 1)
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    xx = TwoQuditMSGate3()
    #adde(cirquit, [xx], [q1, q2], 2)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    #error(cirquit, [q1, q2], 2)
    #adde(cirquit, [u3, u3], [q1, q2], 1)
    #adde(cirquit, [u4], [q1], 1)
    #adde(cirquit, [u5], [q1], 1)
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
    #cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    #cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    xx = TwoQuditMSGate3()
    adde(cirquit, [xx], [q1, q2], 2)
    #cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    #error(cirquit, [q1, q2], 2)
    adde(cirquit, [u3, u3], [q1, q2], 1)
    adde(cirquit, [u4], [q1], 1)
    adde(cirquit, [u5], [q1], 1)
    #cirquit.append([u3(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    #cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    #cirquit.append([u5(q1)], strategy=InsertStrategy.INLINE)



def CCX(cirquit, q1, q2, q3):
    U1(cirquit, q1, q2)
    CX(cirquit, q2, q3)
    U1_c(cirquit, q1, q2)


def CZ(cirquit, q1, q2):
    h = H()
    #cirquit.append(h(q2), strategy=InsertStrategy.INLINE)
    adde(cirquit, [h], [q2], 1)
    CX(cirquit, q1, q2)
    #cirquit.append(h(q2), strategy=InsertStrategy.INLINE)
    adde(cirquit, [h], [q2], 1)

def CCZ(cirquit, q1, q2, q3):
    h = H()
    #cirquit.append(h(q3), strategy=InsertStrategy.INLINE)
    adde(cirquit, [h], [q3], 1)
    CCX(cirquit, q1, q2, q3)
    #cirquit.append(h(q3), strategy=InsertStrategy.INLINE)
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
    #gates = [h(q2), h(q3), h(q4)]
    adde(circuit, [h, h, h], [q2, q3, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CCZ(circuit, q1, q3, q4)
    #gates = [x(q3), x(q4)]
    adde(circuit, [x, x], [q3, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CCZ(circuit, q4, q3, q1)
    #gates = [x(q3), x(q4)]
    adde(circuit, [x, x], [q3, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
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
    #gates = [x(q3), x(q4)]
    adde(circuit, [x, x], [q3, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CCZ(circuit, q1, q3, q4)
    #gates = [x(q3), x(q4)]
    adde(circuit, [x, x], [q3, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CCZ(circuit, q1, q3, q4)
    #gates = [h(q2), h(q3), h(q4)]
    adde(circuit, [h,h,h], [q2, q3, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)

def XZZXI(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CX(cirquit, a1, qudits[0])
    CZ(cirquit,  a1, qudits[1])
    CZ(cirquit, a1, qudits[2])
    CX(cirquit, a1, qudits[3])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def ZZXIX(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CZ(cirquit,  a1, qudits[0])
    CZ(cirquit, a1, qudits[1])
    CX(cirquit, a1, qudits[2])
    CX(cirquit, a1, qudits[4])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def XXIZX(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CX(cirquit, a1, qudits[0])
    CX(cirquit, a1, qudits[1])
    CZ(cirquit, a1, qudits[3])
    CX(cirquit, a1, qudits[4])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def IXXXZ(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CX(cirquit, a1, qudits[1])
    CX(cirquit, a1, qudits[2])
    CX(cirquit, a1, qudits[3])
    CZ(cirquit, a1, qudits[4])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def XZZXI_r(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CX(cirquit, a1, qudits[3])
    CZ(cirquit, a1, qudits[2])
    CZ(cirquit,  a1, qudits[1])
    CX(cirquit, a1, qudits[0])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def ZZXIX_r(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CX(cirquit, a1, qudits[4])
    CX(cirquit, a1, qudits[2])
    CZ(cirquit, a1, qudits[1])
    CZ(cirquit, a1, qudits[0])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def XXIZX_r(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CX(cirquit, a1, qudits[4])
    CZ(cirquit, a1, qudits[3])
    CX(cirquit, a1, qudits[1])
    CX(cirquit, a1, qudits[0])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    # cirquit.append([cirq.measure(a1)])

def IXXXZ_r(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CZ(cirquit, a1, qudits[4])
    CX(cirquit, a1, qudits[3])
    CX(cirquit, a1, qudits[2])
    CX(cirquit, a1, qudits[1])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def get_syndrome(circuit, qutrits):
    q0 = qutrits1[0]
    q1 = qutrits1[1]
    q2 = qutrits1[2]
    q3 = qutrits1[3]
    q4 = qutrits1[4]
    a1 = qutrits1[5]
    a2 = qutrits1[6]
    a3 = qutrits1[7]
    a4 = qutrits1[8]

    XZZXI(circuit1, [q0, q1, q2, q3, q4], qutrits1[5])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[5])][0]
    print(f'Measured bit: {measured_bit}')

    ZZXIX(circuit1, [q0, q1, q2, q3, q4], qutrits1[6])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[6])][0]
    print(f'Measured bit: {measured_bit}')

    XXIZX(circuit1, [q0, q1, q2, q3, q4], qutrits1[7])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[7])][0]
    print(f'Measured bit: {measured_bit}')

    IXXXZ(circuit1, [q0, q1, q2, q3, q4], qutrits1[8])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[8])][0]
    print(f'Measured bit: {measured_bit}')

def get_syndrome_r(circuit, qutrits):
    q0 = qutrits1[0]
    q1 = qutrits1[1]
    q2 = qutrits1[2]
    q3 = qutrits1[3]
    q4 = qutrits1[4]
    a1 = qutrits1[5]
    a2 = qutrits1[6]
    a3 = qutrits1[7]
    a4 = qutrits1[8]

    IXXXZ(circuit1, [q0, q1, q2, q3, q4], qutrits1[8])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[8])][0]
    print(f'Measured bit: {measured_bit}')

    XXIZX(circuit1, [q0, q1, q2, q3, q4], qutrits1[7])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[7])][0]
    print(f'Measured bit: {measured_bit}')
    ZZXIX(circuit1, [q0, q1, q2, q3, q4], qutrits1[6])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[6])][0]
    print(f'Measured bit: {measured_bit}')
    XZZXI(circuit1, [q0, q1, q2, q3, q4], qutrits1[5])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[5])][0]
    print(f'Measured bit: {measured_bit}')

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

def ec(circuit, qutrits):
    circuit.append([cirq.measure(qutrits[1])])
    circuit.append([cirq.measure(qutrits[2])])
    circuit.append([cirq.measure(qutrits[3])])
    circuit.append([cirq.measure(qutrits[4])])
    res1 = sim.simulate(circuit1)
    r1,r2,r3,r4 = res1.measurements[str(qutrits1[1])][0], res1.measurements[str(qutrits1[2])][0], res1.measurements[str(qutrits1[3])][0], res1.measurements[str(qutrits1[4])][0]
    if r1 == 0 and r2 == 1 and r3 == 1 and r4 == 1:
        circuit.append([x(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 0 and r2 == 1 and r3 == 1 and r4 == 0:
        circuit.append([x(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 1 and r2 == 0 and r3 == 1 and r4 == 1:
        circuit.append([x(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 1 and r2 == 1 and r3 == 1 and r4 == 0:
        circuit.append([x(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 1 and r2 == 0 and r3 == 0 and r4 == 1:
        circuit.append([x(qutrits[0])], strategy=InsertStrategy.INLINE)

    elif r1 == 0 and r2 == 0 and r3 == 0 and r4 == 1:
        circuit.append([z(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 0 and r2 == 1 and r3 == 0 and r4 == 1:
        circuit.append([z(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 1 and r2 == 1 and r3 == 1 and r4 == 1:
        circuit.append([z(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 1 and r2 == 0 and r3 == 1 and r4 == 0:
        circuit.append([z(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 1 and r2 == 1 and r3 == 0 and r4 == 0:
        circuit.append([z(qutrits[0])], strategy=InsertStrategy.INLINE)


    elif r1 == 1 and r2 == 1 and r3 == 0 and r4 == 1:
        circuit.append([y(qutrits[0])], strategy=InsertStrategy.INLINE)




def error_correction(circuit, qutrits):
    #get_syndrome(circuit, qutrits)
    # get_syndrome_r(circuit1, qutrits1)
    circuit.append([cirq.measure(qutrits[1])])
    circuit.append([cirq.measure(qutrits[2])])
    circuit.append([cirq.measure(qutrits[3])])
    circuit.append([cirq.measure(qutrits[4])])

    q0 = qutrits1[0]
    q1 = qutrits1[1]
    q2 = qutrits1[2]
    q3 = qutrits1[3]
    q4 = qutrits1[4]


    # Операции для исправления ошибок X
    circuit1.append([x(q1)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit1.append([x(q1)], strategy=InsertStrategy.INLINE)

    circuit1.append([x(q1), x(q4)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit1.append([x(q1), x(q4)], strategy=InsertStrategy.INLINE)

    circuit1.append([x(q1), x(q2), x(q3)], strategy=InsertStrategy.INLINE)
    CCCCZ(circuit, q1, q2, q3, q4, q0)
    circuit1.append([x(q1), x(q2), x(q3)], strategy=InsertStrategy.INLINE)

    circuit1.append([x(q2)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit1.append([x(q2)], strategy=InsertStrategy.INLINE)


    # Операции для исправления ошибок Y
    circuit1.append([x(q3)], strategy=InsertStrategy.INLINE)
    CCCCY(circuit, q1, q2, q3, q4, q0)
    circuit1.append([x(q3)], strategy=InsertStrategy.INLINE)

    circuit1.append([x(q4)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit1.append([x(q4)], strategy=InsertStrategy.INLINE)

    circuit1.append([x(q1), x(q3)], strategy=InsertStrategy.INLINE)
    CCCCZ(circuit, q1, q2, q3, q4, q0)
    circuit1.append([x(q1), x(q3)], strategy=InsertStrategy.INLINE)

    circuit1.append([x(q2), x(q3)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit1.append([x(q2), x(q3)], strategy=InsertStrategy.INLINE)

    CCCCZ(circuit, q1, q2, q3, q4, q0)

    # Операции для исправления ошибок Z
    circuit1.append([x(q2), x(q4)], strategy=InsertStrategy.INLINE)
    CCCCZ(circuit, q1, q2, q3, q4, q0)
    circuit1.append([x(q2), x(q4)], strategy=InsertStrategy.INLINE)


    circuit1.append([x(q3), x(q4)], strategy=InsertStrategy.INLINE)
    CCCCZ(circuit, q1, q2, q3, q4, q0)
    circuit1.append([x(q3), x(q4)], strategy=InsertStrategy.INLINE)


def time_error(circuit, qutrits, t):

    p = np.exp(-t / T0)
    dpg_t = QutritDepolarizingChannel(1.000001 - p)
    for q in qutrits:
        circuit.append([dpg_t.on(q)], strategy=InsertStrategy.INLINE)
        '''
        if random.randint(0,1000) > p * 1000:
            a1 = random.randint(0, 1000)
            a2 = random.randint(0, 1000)
            a3 = random.randint(0, 1000)
            a4 = random.randint(0, 1000)
            if 1 - p == 0:
                p = 0.99999999
            sss = (a1 + a2 + a3 + a4) ** 0.5

            mx = R(0, np.pi, 0, 1)
            my = R(np.pi / 2, np.pi, 0, 1)
            mz = R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi, 0, 1)
            mi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            mat = (a1 ** 0.5 * mi + a2 ** 0.5 * mx + a3 ** 0.5 * my + a4 ** 0.5 * mz) / sss
            er_gate = U(mat)
            circuit.append([er_gate(q)], strategy=InsertStrategy.INLINE)
        '''

def error(circuit, qutrits, ind):
    if ind == 1:
        p = PMS1
    else:
        p = PMS2
    dpg = QutritDepolarizingChannel(1.0001 - p)
    for q in qutrits:
        circuit.append([dpg.on(q)], strategy=InsertStrategy.INLINE)
    '''
    if ind == 1:
        p = PMS1
    else:
        p = PMS2

    for q in qutrits:
        rv_e = random.randint(0,1000)
        if rv_e > 1000 * p:
            a1 = random.randint(0, 1000)
            a2 = random.randint(0, 1000)
            a3 = random.randint(0, 1000)
            a4 = random.randint(0, 1000)
            sss = (a1 + a2 + a3 + a4) ** 0.5
            mx = R(0, np.pi, 0, 1)
            my = R(np.pi / 2, np.pi, 0, 1)
            mz = R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi, 0, 1)
            mi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            mat = (a1 ** 0.5 * mi + a2 ** 0.5 * mx + a3 ** 0.5 * my + a4 ** 0.5 * mz) / sss
            er_gate = U(mat)
            circuit.append([er_gate(q)], strategy=InsertStrategy.INLINE)
    '''

def X1_l(circuit, lqubits):
    x = X1()
    z = Z1()
    q1, q2, q3, q4, q5 = lqubits[0], lqubits[1], lqubits[2], lqubits[3], lqubits[4]
    #gates = [z(q1), z(q4)]
    adde(circuit, [z,z], [q1, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    #gates = [x(q1), x(q2) ,x(q3) ,x(q4) ,x(q5)]
    adde(circuit, [x, x, x, x, x], [q1, q2, q3, q4, q5], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)

def Z1_l(circuit, lqubits):
    x = X1()
    z = Z1()
    q1, q2, q3, q4, q5 = lqubits[0], lqubits[1], lqubits[2], lqubits[3], lqubits[4]
    #gates = [x(q1), x(q4)]
    adde(circuit, [x, x], [q1, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    adde(circuit, [z], [q5], 1)
    #gates = [z(q5)]
    #circuit.append(gates, strategy=InsertStrategy.INLINE)


# def make_error(p):


# Основная операция
x = X1()
x2 = X2()
z = Z1()
y = Y1()
x_conj = X1_conj()
x2_conj = X2_conj()
h = H()
'''
sim = cirq.Simulator()
circuit1 = cirq.Circuit()
qutrits1 = []



for i in range(10):
    qutrits1.append(cirq.LineQid(i, dimension=3))
'''
# кодируемое состояние
#gates1 = [h(qutrits1[0])]
#circuit1.append(gates1)
#encoding_qubit(circuit1, qutrits1)
# ошибка
#gates1 = [z(qutrits1[4])]
# circuit1.append(gates1)
#xxx = TwoQuditMSGate3()
#adde(circuit1, [xxx], [qutrits1[1], qutrits1[2]], 2)
#adde(circuit1, [h, h, h], [qutrits1[3], qutrits1[2], qutrits1[4]], 1)
# error_correction(circuit1, qutrits1)

# error(circuit1, qutrits1, 0.5)
#decoding_qubit(circuit1, qutrits1)
# circuit1.append([cirq.measure(qutrits1[1])])
# circuit1.append([cirq.measure(qutrits1[2])])
# circuit1.append([cirq.measure(qutrits1[3])])
# circuit1.append([cirq.measure(qutrits1[4])])
'''
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[1])][0]
print(f'Measured bit: {measured_bit}')
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[2])][0]
print(f'Measured bit: {measured_bit}')
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[3])][0]
print(f'Measured bit: {measured_bit}')
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutirts1[4])][0]
print(f'Measured bit: {measured_bit}')
'''
def m(a ,b, c, d, e):
    return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)

sps1 = []
sps2 = []
for PMS2 in np.arange(0.1, 1.005, 0.005):
    PMS2 = 1
    print(sps1)
    sps1 = []

    for t in np.arange(0,5,0.2):
        print(t)
        #sps1.append(PMS2)
        sch = 0
        for i in range(N):
            x = X1()
            y = Y1()
            sim = cirq.Simulator()
            circuit1 = cirq.Circuit()
            qutrits1 = []
            for j in range(5):
                qutrits1.append(cirq.LineQid(j, dimension=3))
            alf1 = random.randint(0,1000) / 1000 * 2 * np.pi
            alf2 = random.randint(0,1000) / 1000 * 2 * np.pi
            povorot = R(0, alf1, 0, 1) @ R(np.pi / 2, alf2, 0, 1)
            pg = U(povorot)
            circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)
            encoding_qubit(circuit1, qutrits1)
            time_error(circuit1, qutrits1, t)
            #circuit1.append([y(qutrits1[2])])
            decoding_qubit(circuit1, qutrits1)
            '''
            res1 = sim.simulate(circuit1)
            print('88888888')
            print('res1', res1)
            print('88888888')
            '''
            error_correction(circuit1, qutrits1)
            povorot_r = R(np.pi / 2, -alf2, 0, 1) @ R(0, -alf1, 0, 1)
            pg_r = U(povorot_r)
            circuit1.append([pg_r(qutrits1[0])], strategy=InsertStrategy.INLINE)



            circuit1.append([cirq.measure(qutrits1[0])])

            res1 = sim.simulate(circuit1)
            measured_bit = res1.measurements[str(qutrits1[0])][0]
            if measured_bit == 0:
                sch = sch + 1


        sps1.append(sch / N)
        print('zn', sch / N)


print(sps1)
'''
fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot()
ax.scatter(sps1, sps2, color='b', s = 5)
print(sps1)
print(sps2)
plt.show()
'''
# print(res1.final_state_vector)
#res1 = sim.simulate(circuit1)
#print(circuit1)
#print(res1)
'''
[0.219, 0.203, 0.167, 0.163, 0.143, 0.152, 0.145, 0.14, 0.134, 0.109, 0.125, 0.127, 0.095]
[0.278, 0.274, 0.208, 0.211, 0.193, 0.178, 0.161, 0.129, 0.147, 0.145, 0.151, 0.139, 0.136]
[0.379, 0.296, 0.287, 0.252, 0.203, 0.226, 0.189, 0.2, 0.179, 0.213, 0.191, 0.203, 0.188]
[0.453, 0.392, 0.362, 0.3, 0.311, 0.295, 0.24, 0.244, 0.245, 0.252, 0.214, 0.216, 0.231]
[0.608, 0.54, 0.483, 0.436, 0.395, 0.34, 0.31, 0.318, 0.308, 0.287, 0.286, 0.277, 0.272]


[1.0, 1.0, 1.0, 0.75, 0.8, 0.55, 0.55, 0.55, 0.6, 0.6, 0.45, 0.65, 0.5, 0.6, 0.3, 0.4, 0.4, 0.55, 0.65, 0.65, 0.45, 0.8, 0.5, 0.5, 0.4]
[1.0, 1.0, 0.9, 0.8, 0.8, 0.6, 0.8, 0.85, 0.65, 0.45, 0.7, 0.45, 0.55, 0.55, 0.6, 0.5, 0.6, 0.4, 0.4, 0.5, 0.6, 0.6, 0.55, 0.5, 0.55]

'''
print('31.01.2024')