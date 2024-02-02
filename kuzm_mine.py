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
PMS1 = 0.999
PMS2 = 0.99


T0 = 25

zZ = np.array([[1,0,0]]).T
eE = np.array([[0,1,0]]).T
fF = np.array([[0,0,1]]).T
A = [zZ, eE, fF]

B = []
for i1 in range(3):
    for i2 in range(3):
        for i3 in range(3):
            for i4 in range(3):
                B.append(np.kron(np.kron(np.kron(A[i1], A[i2]), A[i3]), A[i4]))






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


if __name__ == '__main__':
    n = 1  # number of qudits
    d = 3  # dimension of qudits


    q0 = cirq.LineQid(0, dimension=d)
    #print(np.kron(generalized_sigma(3, 1, 1, dimension=2), generalized_sigma(1, 0, 0, dimension=2)))
    print('Qutrit single depolarization channel. f1 = 0.99')
    circuit = cirq.Circuit()

    #circuit.append([h(q1)])
    circuit.append(QutritDepolarizingChannel(0.99).on(q0))
    #print(circuit)
    #print()

    sim = cirq.Simulator()
    #res1 = sim.simulate(circuit)

    #circuit.append([cirq.measure(q0)])

    res1 = sim.simulate(circuit)
    #measured_bit = res1.measurements[str(q0)][0]
    #print(circuit)
    #print('measured_bit', measured_bit)

    print(cirq.final_density_matrix(circuit))

    ro = (np.array([[1,0,0]]).T) @ np.array([[1,0,0]])

    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    Y = np.array([[0, complex(0, -1), 0], [complex(0, 1), 0, 0], [0, 0, 1]])
    Z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])

    print(1/3 * (X@ro@dag(X) + Y@ro@dag(Y) + Z@ro@dag(Z)))
    #print(E(basis, 2, 2,0.59, paulies1))
    #print(np.kron(generalized_sigma(1, 0, 1, dimension=2), generalized_sigma(1, 0, 1, dimension=2)))