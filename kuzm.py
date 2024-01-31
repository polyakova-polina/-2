import numpy as np
import sympy
import scipy.stats
from scipy import linalg
import cirq

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


class QuditGeneralizedXGate(QuditGate):
    """Generalized X Gate"""

    def __init__(self, dimension=4):
        super().__init__(dimension=dimension)

    def _unitary_(self):
        N = self.d
        u = np.eye(N)

        for i in range(N):
            u[:][i] = np.eye(N)[:][(i + 1) % N]

        return u

    def get_unitary(self):
        return self._unitary_()

    def _circuit_diagram_info_(self, args):
        self.symbol = 'Xgen'
        return self.symbol


class QuditGeneralizedZGate(QuditGate):
    """Generalized Z Gate"""

    def __init__(self, dimension=4):
        super().__init__(dimension=dimension)

    def _unitary_(self):
        N = self.d
        w = np.exp(2 * np.pi * 1j / N)
        u = np.diag([w ** k for k in range(N)])
        return u

    def get_unitary(self):
        return self._unitary_()

    def _circuit_diagram_info_(self, args):
        self.symbol = 'Zgen'
        return self.symbol


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

    def __init__(self, dimension=3, num_qudits=2):
        super().__init__(dimension=dimension, num_qubits=num_qudits)
        self.unitary = np.array(scipy.stats.unitary_group.rvs(self.d ** self.n))
        self.symbol = 'U'

    def _unitary_(self):
        return self.unitary




#from qudit_gates import QuditGate, QuditGeneralizedXGate, QuditGeneralizedZGate


class QuquartDepolarizingChannel(QuditGate):

    def __init__(self, p1=None):
        super().__init__(dimension=3, num_qubits=1)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        if p1 is None:
            f1 = 0.99
            self.p1 = (1 - f1)
        else:
            self.p1 = p1

        self.mixture_probabilities = np.ones(self.d ** 2) * self.p1 / (self.d ** 2 - 1)
        self.mixture_probabilities[0] = (1 - self.p1)  # identity probability

    def _mixture_(self):

        x_unitary = R(0,np.pi,0,1)
        y_unitary = R(np.pi / 2, np.pi, 0, 1)
        z_unitary = x_unitary @ y_unitary

        ps = []
        for alpha in range(self.d):
            for beta in range(self.d):
                op = np.linalg.matrix_power(x_unitary, alpha) @ np.linalg.matrix_power(z_unitary, beta)
                ps.append(op)

        return tuple(zip(self.mixture_probabilities, ps))

    def get_mixture(self):
        return self._mixture_()

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"


class DoubleQuquartDepolarizingChannel(QuditGate):
    def __init__(self, p2=None):
        super().__init__(dimension=3, num_qubits=2)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        if p2 is None:
            f2 = 0.96
            self.p2 = (1 - f2)
        else:
            self.p2 = p2

        self.mixture_probabilities = np.ones(self.d ** 4) * self.p2 / (self.d ** 4 - 1)
        self.mixture_probabilities[0] = (1 - self.p2)  # identity probability

    def _mixture_(self):
        ps = []

        x_unitary = QuditGeneralizedXGate(dimension=self.d ** 2).get_unitary()
        z_unitary = QuditGeneralizedZGate(dimension=self.d ** 2).get_unitary()

        for alpha in range(self.d ** 2):
            for beta in range(self.d ** 2):

                op = np.linalg.matrix_power(x_unitary, alpha) @ np.linalg.matrix_power(z_unitary, beta)
                ps.append(op)

        return tuple(zip(self.mixture_probabilities, ps))

    def get_mixture(self):
        return self._mixture_()

    def _circuit_diagram_info_(self, args):
        return f"ΦΦ(p2={self.p2:.3f})", f"ΦΦ(p2={self.p2:.3f})"


if __name__ == '__main__':
    n = 2  # number of qudits
    d = 3  # dimension of qudits

    q0, q1 = cirq.LineQid.range(n, dimension=d)

    print('Ququart single depolarization channel. f1 = 0.99')
    dpg = QuquartDepolarizingChannel(0.5)
    circuit = cirq.Circuit(dpg.on(q0))
    print(circuit)
    sim = cirq.Simulator()
    res1 = sim.simulate(circuit)
    for i in range(d**2):
        print(dpg.get_mixture()[i][0])
    #print(dpg.get_mixture()[8])
