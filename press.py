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

zZ = np.array([[1, 0, 0]]).T
eE = np.array([[0, 1, 0]]).T
fF = np.array([[0, 0, 1]]).T
A = [zZ, eE, fF]

B = []
for i1 in range(3):
    for i2 in range(3):
        for i3 in range(3):
            for i4 in range(3):
                B.append(np.kron(np.kron(np.kron(A[i1], A[i2]), A[i3]), A[i4]))

X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
Y = np.array([[0, complex(0, -1), 0], [complex(0, 1), 0, 0], [0, 0, 1]])
Z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
id = np.eye(3)

z = np.array([[1, 0, 0]]).T
e = np.array([[0, 1, 0]]).T
f = np.array([[0, 0, 1]]).T
basis = [z, e, f]
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
    K0 = (1 - p0) ** 0.5 * id
    # K0 = id
    K1 = p0 ** 0.5 / 3 ** 0.5 * x
    # K1 = x
    K2 = p0 ** 0.5 / 3 ** 0.5 * y
    # K2 = y
    K3 = p0 ** 0.5 / 3 ** 0.5 * z
    # K3 = z
    # mat_sum = K0 @ dag(K0) + K1 @ dag(K1) + K2 @ dag(K2) + K3 @ dag(K3)
    # print(mat_sum)
    # print(np.trace(mat_sum))
    # print()
    # print(dag(K3))
    _rho = v1 @ (v2.T)
    # print(_rho)
    if i == 0 and j == 0:
        ksgfsg = 0
        # print('eij', K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3))
    # print('ee', np.trace(K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3)))
    # print()
    return (K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3))


def E(bas, i, j, p0, paulies):
    v1 = bas[i]
    v2 = bas[j]
    id = paulies[0]
    x = paulies[1]
    y = paulies[2]
    z = paulies[3]
    # K0 = (1-p0)**0.5 * id
    K0 = id / 2
    # K1 = p0**0.5 / 3**0.5 * x
    K1 = x / 2
    # K2 = p0**0.5 / 3**0.5 * y
    K2 = y / 2
    # K3 = p0 ** 0.5 / 3**0.5 * z
    K3 = z / 2
    # mat_sum = K0 @ dag(K0) + K1 @ dag(K1) + K2 @ dag(K2) + K3 @ dag(K3)
    # print(mat_sum)
    # print()
    # print(dag(K3))
    _rho = v1 @ (v2.T)

    # print(_rho)
    if i == 0 and j == 0:
        cgjjjfjk = 0
    # print('e', np.trace(K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3)))
    # print()
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
        return self.__class__(self.l1, self.l2, resolver.value_of(self.theta, recursive),
                              resolver.value_of(self.phi, recursive), dimension=self.d)

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
'''
#фазовый
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
        self.p_matrix = np.array([[1/2, 0], [0, 1/2]])
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
        m4 = 2**0.5 * (1-self.p1)**0.5 * np.eye(3)
        omega = np.exp(complex(0, 2 * np.pi / 3))
        m5 =2**0.5 *  (self.p1)**0.5*np.array([[1,0,0],[0,omega,0], [0,0,omega**2]])
        id = np.eye(3) - np.eye(3)
        shiz_massiv = [m4, id, id, m5]
        return tuple(zip(self.p_matrix.flatten(), shiz_massiv))

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"

'''


# деполяризующий
class QutritDepolarizingChannel(QuditGate):

    def __init__(self, PP, p_matrix=None):
        super().__init__(dimension=3, num_qubits=1)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        f1 = 0.9
        self.p1 = (1 - f1) / (1 - 1 / self.d ** 2)
        self.p1 = PP
        # print(self.d)
        # print((1 / self.d ** 2))

        # Choi matrix initialization

        if p_matrix is None:
            self.p_matrix = (1 - self.p1) / (self.d ** 2) * np.ones((self.d, self.d))
            self.p_matrix = np.zeros_like(self.p_matrix)
            # self.p_matrix = np.ones((self.d, self.d))
        else:
            self.p_matrix = p_matrix
        # self.p_matrix[0, 0] += (1 - self.p1)  # identity probability
        for o in range(3):
            for oo in range(3):
                # self.p_matrix[o, oo] = 1 / np.trace(E(basis, o, oo, self.p1, paulies1))
                self.p_matrix[o, oo] = 1 / 9
        # self.p_matrix[0, 0] += 1

        if p_matrix is None:
            self.p_matrix = self.p1 / (self.d ** 2) * np.ones((self.d, self.d))
        else:
            self.p_matrix = p_matrix
        self.p_matrix[0, 0] += (1 - self.p1)  # identity probability
        self.p_matrix = np.array([[(1 - self.p1), self.p1 / 3], [self.p1 / 3, self.p1 / 3]])
        # print('prob[0,0]', self.p_matrix[0, 0])
        # print('prob_sum', self.p_matrix.sum())

        # print('prob_sum', self.p_matrix.sum())

    def _mixture_(self):
        ps = []
        for i in range(self.d):
            for j in range(self.d):
                pinv = np.linalg.inv(self.p_matrix)
                op = E(basis, i, j, self.p1, paulies1)
                # print(np.trace(op))
                ps.append(op)
        # print('total_sum', (np.trace(np.array(ps)) * self.p_matrix).sum())
        # chm = np.kron(np.ones(3), ps)
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
'''


class QutritAmplitudeChannel(QuditGate):

    def __init__(self, PP, p_matrix=None):
        super().__init__(dimension=3, num_qubits=1)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        f1 = 0.9
        self.p1 = (1 - f1) / (1 - 1 / self.d ** 2)
        self.p1 = PP
        # print(self.d)
        # print((1 / self.d ** 2))

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
        # print('prob[0,0]', self.p_matrix[0, 0])
        # print('prob_sum', self.p_matrix.sum())

        # print('prob_sum', self.p_matrix.sum())

    def _mixture_(self):
        ps = []
        for i in range(self.d):
            for j in range(self.d):
                pinv = np.linalg.inv(self.p_matrix)
                op = E(basis, i, j, self.p1, paulies1)
                # print(np.trace(op))
                ps.append(op)
        # print('total_sum', (np.trace(np.array(ps)) * self.p_matrix).sum())
        # chm = np.kron(np.ones(3), ps)
        X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        Y = np.array([[0, complex(0, -1), 0], [complex(0, 1), 0, 0], [0, 0, 1]])
        Z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        Ea1 = np.array([[1, 0, 0], [0, (1 - self.p1) ** 0.5, 0], [0, 0, (1 - self.p1) ** 0.5]])
        Ea2 = np.array([[0, self.p1 ** 0.5, 0], [0, 0, 0], [0, 0, 0]])
        Ea3 = np.array([[0, 0, self.p1 ** 0.5], [0, 0, 0], [0, 0, 0]])
        id = np.eye(3)
        shiz_massiv = [Ea1, Ea2, Ea3]
        return tuple(zip(self.p_matrix.flatten(), shiz_massiv))

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"





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


def make_ms_matrix(N, fi, hi, i, j, k, l):
    if i == j:
        return np.eye(N)
    if i > j:
        i, j = j, i
    x_for_ms1 = np.zeros((N, N))
    x_for_ms1[i][j] = 1
    x_for_ms1[j][i] = 1
    y_for_ms1 = np.zeros((N, N))
    y_for_ms1[i][j] = -1
    y_for_ms1[j][i] = 1
    y_for_ms1 = 1j * y_for_ms1
    if k == l:
        return np.eye(N * N)
    if k > l:
        k, l = l, k
    x_for_ms2 = np.zeros((N, N))
    x_for_ms2[k][l] = 1
    x_for_ms2[l][k] = 1
    y_for_ms2 = np.zeros((N, N))
    y_for_ms2[k][l] = -1
    y_for_ms2[l][k] = 1
    y_for_ms1 = 1j * y_for_ms1

    m = np.kron((np.cos(fi) * x_for_ms1 + np.sin(fi) * y_for_ms1), (np.cos(fi) * x_for_ms2 + np.sin(fi) * y_for_ms2))
    m = -1j * m * hi
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


class TwoQuditMSGate02(gate_features.TwoQubitGate
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
        matrix = make_ms_matrix(3, 0, np.pi / 2,0,1,0,2)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class TwoQuditMSGate01(gate_features.TwoQubitGate
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
        matrix = make_ms_matrix(3, 0, np.pi / 2,0,1,0,1)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class TwoQuditMSGate12(gate_features.TwoQubitGate
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
        matrix = make_ms_matrix(3, 0, np.pi / 2, 0,1,1,2)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class TwoQuditMSGate01_c(gate_features.TwoQubitGate
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
        matrix = make_ms_matrix(3, 0, -np.pi / 2,0,1,0,1)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class U_press(gate_features.TwoQubitGate
              ):

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__
        }

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        return cls()

    def _qid_shape_(self):
        return (3, 3, 3,)

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




def U1_clear(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u6 = U(R(np.pi / 2, -np.pi, 0, 2), 'Ry(-π)02')
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)
    # adde(cirquit, [u1, u6], [q1, q2], 1)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    # adde(cirquit, [u2], [q1], 1)
    xx = TwoQuditMSGate01()
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    # error(cirquit, [q1, q2], PMS)
    # adde(cirquit, [xx], [q1, q2], 2)




def U1_c_clear(cirquit, q1, q2):
    u1 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    u2 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u6 = U(R(np.pi / 2, np.pi, 0, 2), 'Ry(π)02')
    xx_c = TwoQuditMSGate01_c()
    cirquit.append([xx_c(q1, q2)], strategy=InsertStrategy.INLINE)
    # adde(cirquit, [xx_c], [q1, q2], 2)
    # error(cirquit, [q1, q2], PMS)
    # adde(cirquit, [u2], [q1], 1)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    # adde(cirquit, [u1, u6], [q1, q2], 1)
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)


def CX_clear01(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 1), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u5 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    # adde(cirquit, [u1], [q1], 1)
    # adde(cirquit, [u2], [q1], 1)
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    xx = TwoQuditMSGate01()
    # adde(cirquit, [xx], [q1, q2], 2)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    # error(cirquit, [q1, q2], 2)
    # adde(cirquit, [u3, u3], [q1, q2], 1)
    # adde(cirquit, [u4], [q1], 1)
    # adde(cirquit, [u5], [q1], 1)
    cirquit.append([u3(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u5(q1)], strategy=InsertStrategy.INLINE)

def CX_clear02(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 1), 'Rx(-π)01')
    u35 = U(R(0, -np.pi, 0, 2), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u5 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    # adde(cirquit, [u1], [q1], 1)
    # adde(cirquit, [u2], [q1], 1)
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    xx = TwoQuditMSGate02()
    # adde(cirquit, [xx], [q1, q2], 2)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    # error(cirquit, [q1, q2], 2)
    # adde(cirquit, [u3, u3], [q1, q2], 1)
    # adde(cirquit, [u4], [q1], 1)
    # adde(cirquit, [u5], [q1], 1)
    cirquit.append([u3(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u35(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u5(q1)], strategy=InsertStrategy.INLINE)


def CX_clear12(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 1), 'Rx(-π)01')
    u35 = U(R(0, -np.pi, 1, 2), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u5 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    # adde(cirquit, [u1], [q1], 1)
    # adde(cirquit, [u2], [q1], 1)
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    xx = TwoQuditMSGate12()
    # adde(cirquit, [xx], [q1, q2], 2)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    # error(cirquit, [q1, q2], 2)
    # adde(cirquit, [u3, u3], [q1, q2], 1)
    # adde(cirquit, [u4], [q1], 1)
    # adde(cirquit, [u5], [q1], 1)
    cirquit.append([u3(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u35(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u5(q1)], strategy=InsertStrategy.INLINE)






def CCX01(cirquit, q1, q2, q3):
    U1_clear(cirquit, q1, q2)
    CX_clear01(cirquit, q2, q3)
    U1_c_clear(cirquit, q1, q2)

def CCX02(cirquit, q1, q2, q3):
    U1_clear(cirquit, q1, q2)
    CX_clear02(cirquit, q2, q3)
    U1_c_clear(cirquit, q1, q2)

def CCX12(cirquit, q1, q2, q3):
    U1_clear(cirquit, q1, q2)
    CX_clear12(cirquit, q2, q3)
    U1_c_clear(cirquit, q1, q2)






class H(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi / 2, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'U_enc'


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
        return R(np.pi / 2, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'Y1'


class X12(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 1, 2)

    def _circuit_diagram_info_(self, args):
        return 'X2'


class X1(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'X1'



def run_single_qudit(t, N):
    fidelity = 0
    sch = 0
    for alf1 in np.linspace(0, np.pi, N // 2):
        for alf2 in np.linspace(0, 2 * np.pi, N):
            alf2 += 2 * np.pi / N / 2
            # alf1 = random.randint(0, 1000) / 1000 * 2 * np.pi
            # alf2 = random.randint(0, 1000) / 1000 * 2 * np.pi
            sch += 1
            circuit1 = cirq.Circuit()
            qutrits1 = []
            qutrits1.append(cirq.LineQid(0, dimension=3))

            povorot = R(alf1, alf2, 0, 1)
            # !

            pg = U(povorot)
            # circuit1.append([h(qutrits1[0])], strategy=InsertStrategy.INLINE)
            circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)
            # print(cirq.final_density_matrix(circuit1, qubit_order=qutrits1))
            # print()



            povorot_r = R(alf1, -alf2, 0, 1)
            pg_r = U(povorot_r)
            circuit1.append([pg_r(qutrits1[0])], strategy=InsertStrategy.INLINE)
            # circuit1.append([h(qutrits1[0])], strategy=InsertStrategy.INLINE)

            ro_ab = cirq.final_density_matrix(circuit1)

            # print(mat_0)
            fidelity += abs(ro_ab[0][0])
    return fidelity / sch

zZ = np.array([[1,0,0]]).T
eE = np.array([[0,1,0]]).T
fF = np.array([[0,0,1]]).T
A = [zZ, eE, fF]

B = []

def m(a ,b, c, d, e):
    return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)

for i1 in range(3):
    for i2 in range(3):
        B.append(np.kron(A[i1], A[i2]))

def partial_trace(rho_ab):
    tr = np.eye(3) - np.eye(3)
    for i in range(3):
        for j in range(3):
            for k in range(9):
                tr = tr + np.kron(A[i].T, B[k].T) @ rho_ab @ np.kron(A[j], B[k]) * A[i] @ A[j].T
    return tr


sim = cirq.Simulator()

circuit1 = cirq.Circuit()
qutrits1 = []
qutrits1.append(cirq.LineQid(0, dimension=3))
qutrits1.append(cirq.LineQid(1, dimension=3))
qutrits1.append(cirq.LineQid(2, dimension=3))
qutrits1.append(cirq.LineQid(3, dimension=3))
q1, q2, q3= qutrits1[0], qutrits1[1], qutrits1[2]

q4 = qutrits1[3]

alf11 = random.randint(0, 1000) / 1000 * 2 * np.pi
alf21 = random.randint(0, 1000) / 1000 * 2 * np.pi
alf11 = 1
alf21 = 1
povorot = R(alf11, alf21, 0, 1)
pg = U(povorot)
circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)
alf12 = random.randint(0, 1000) / 1000 * 2 * np.pi
alf22 = random.randint(0, 1000) / 1000 * 2 * np.pi
alf21 = 2
alf21 = 2
povorot = R(alf12, alf22, 0, 1)
pg = U(povorot)
circuit1.append([pg(qutrits1[1])], strategy=InsertStrategy.INLINE)
alf13 = random.randint(0, 1000) / 1000 * 2 * np.pi
alf23 = random.randint(0, 1000) / 1000 * 2 * np.pi
alf13 = 3
alf23 = 32q
povorot = R(alf13, alf23, 0, 1)
pg = U(povorot)
circuit1.append([pg(qutrits1[2])], strategy=InsertStrategy.INLINE)




'''
x01 = X1()
x12 = X12()
circuit1.append([x01(q1)], strategy=InsertStrategy.INLINE)
circuit1.append([x01(q2)], strategy=InsertStrategy.INLINE)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)
CCX12(circuit1,q1,q2, q3)
res1 = sim.simulate(circuit1)
print(res1)

'''
x01 = X1()
x12 = X12()
#circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)
#2-7
circuit1.append([x01(q1)], strategy=InsertStrategy.INLINE)
CCX02(circuit1,q1,q3,q2)
circuit1.append([x01(q1)], strategy=InsertStrategy.INLINE)

#5-22
CCX02(circuit1,q3,q2,q1)
circuit1.append([x12(q1)], strategy=InsertStrategy.INLINE)
CCX12(circuit1,q1,q3,q2)
circuit1.append([x12(q1)], strategy=InsertStrategy.INLINE)
CCX02(circuit1,q3,q2,q1)

#11-19
circuit1.append([x01(q2)], strategy=InsertStrategy.INLINE)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)
CCX12(circuit1,q3,q2,q1)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)
CCX01(circuit1,q1,q2,q3)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)
CCX12(circuit1,q3,q2,q1)
circuit1.append([x01(q2)], strategy=InsertStrategy.INLINE)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)

#14-16
CCX01(circuit1,q1,q2,q3)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)
CCX12(circuit1,q3,q1,q2)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)
CCX01(circuit1,q1,q2,q3)

#circuit1.append([cirq.measure(qutrits1[2])])
res1 = sim.simulate(circuit1)
print(res1)
#print(circuit1)
#print(abs(cirq.final_density_matrix(circuit1, qubit_order=qutrits1)))
#print(res1.measurements[str(qutrits1[0])][0])
#print(abs(partial_trace(cirq.final_density_matrix(circuit1, qubit_order=[q3,q2,q1]))))


'''
q3, q4 = q4, q3

CCX01(circuit1,q1,q2,q3)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)
CCX12(circuit1,q3,q1,q2)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)
CCX01(circuit1,q1,q2,q3)

circuit1.append([x01(q2)], strategy=InsertStrategy.INLINE)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)
CCX12(circuit1,q3,q2,q1)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)
CCX01(circuit1,q1,q2,q3)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)
CCX12(circuit1,q3,q2,q1)
circuit1.append([x01(q2)], strategy=InsertStrategy.INLINE)
circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)

CCX02(circuit1,q3,q2,q1)
circuit1.append([x12(q1)], strategy=InsertStrategy.INLINE)
CCX12(circuit1,q1,q3,q2)
circuit1.append([x12(q1)], strategy=InsertStrategy.INLINE)
CCX02(circuit1,q3,q2,q1)

circuit1.append([x01(q1)], strategy=InsertStrategy.INLINE)
CCX02(circuit1,q1,q3,q2)
circuit1.append([x01(q1)], strategy=InsertStrategy.INLINE)

povorot = R(alf13, -alf23, 0, 1)
pg = U(povorot)
circuit1.append([pg(qutrits1[2])], strategy=InsertStrategy.INLINE)

povorot = R(alf12, -alf22, 0, 1)
pg = U(povorot)
circuit1.append([pg(qutrits1[1])], strategy=InsertStrategy.INLINE)

povorot = R(alf11, -alf21, 0, 1)
pg = U(povorot)
circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)

q3, q4 = q4, q3
circuit1.append([cirq.measure(qutrits1[0])])
circuit1.append([cirq.measure(qutrits1[1])])
circuit1.append([cirq.measure(qutrits1[2])])
circuit1.append([cirq.measure(qutrits1[3])])
res1 = sim.simulate(circuit1)
#print(circuit1)
#print(abs(cirq.final_density_matrix(circuit1, qubit_order=qutrits1)))
print(res1.measurements[str(qutrits1[0])][0])
print(res1.measurements[str(qutrits1[1])][0])
print(res1.measurements[str(qutrits1[2])][0])
print(res1.measurements[str(qutrits1[3])][0])
#print(res1)
'''