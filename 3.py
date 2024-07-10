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
gvot = 50
n_t = 4
N_X = n_t * gvot
nn = 10
ans = []
T0 = 100
T = 100
N = 2
PMS1 = 0.005
PMS2 = 0.04
PMS0 = 0.04
#PMS0 = 1 / 50
rrange = range(gvot, N_X, gvot)
#n_t = 4

zZ = np.array([[1,0]]).T
eE = np.array([[0,1]]).T

A = [zZ, eE]

B = []

def make_traspose(sps):
    ar = np.eye(27)
    for par in sps:
        ar[par[0], par[0]], ar[par[1], par[0]] = 0, 1
        #ar[par[1], par[1]], ar[par[1], par[0]] = ar[par[1], par[0]], ar[par[1], par[1]]
    return ar

Pres_mat = make_traspose(((3,6), (4,2), (9,5), (10,7), (12,4), (13, 3),       (2,9), (5,10), (6,12), (7,13), (1,2), (2,1), (4,5), (5,4)))
Pres_mat = make_traspose(((3,1), (9,2), (1,3), (10,5), (12,6), (13,7), (2,9), (5,10), (6, 12), (7,13)))

ccx_mat = make_traspose(((12,13),(13,12)))

Unpres_mat = Pres_mat.T


for i1 in range(2):
    for i2 in range(2):
        B.append(np.kron(A[i1], A[i2]))

def partial_trace(rho_ab):
    tr = np.eye(2) - np.eye(2)
    for i in range(2):
        for j in range(2):
            for k in range(4):
                tr = tr + np.kron(A[i].T, B[k].T) @ rho_ab @ np.kron(A[j], B[k]) * A[i] @ A[j].T
    return tr

def R(fi, hi, i=0, j=1):
    N = 3
    if i == j:
        return np.eye(N)
    if i > j:
        i, j = j, i
    x_for_ms = np.zeros((N, N))
    x_for_ms[i][j] = 1
    x_for_ms[j][i] = 1
    y_for_ms = np.zeros((N, N))
    y_for_ms[i][j] = -1
    y_for_ms[j][i] = 1
    y_for_ms = y_for_ms * 1j

    m = np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms

    return linalg.expm(-1j * m * hi / 2)


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
        return
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

class TwoQS(gate_features.TwoQubitGate
                      ):

    def __init__(self, coaf, diag_i='XX'):
        #self.mat = mat
        self.diag_info = diag_i
        self.coaf = coaf

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
        matrix = make_ms_matrix(3, 0, -np.pi / 2,self.coaf[0],self.coaf[1],self.coaf[2],self.coaf[3])
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class cxg(gate_features.TwoQubitGate
                      ):

    def __init__(self, coaf, diag_i='XX'):
        #self.mat = mat
        self.diag_info = diag_i
        self.coaf = coaf

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
        matrix = np.eye(9)
        matrix[3][3], matrix[3][4] = matrix[3][4], matrix[3][3]
        matrix[4][4], matrix[4][3] = matrix[4][3], matrix[4][4]
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

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


def CXz(cirquit, q1, q2):
    u1 = U(R(np.pi / 2, np.pi / 2, 0, 2), 'Rx(-π)12')
    u2 = U(R(0, - np.pi, 0, 2), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 1), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 2), 'Ry(-π/2)01')
    xx = TwoQS((0,2,0,1))
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)

def CX(cirquit, q1, q2):

    xx = cxg((0,2,0,1))

    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)


class rTwoQS(gate_features.TwoQubitGate
                      ):

    def __init__(self, coaf, diag_i='XX'):
        #self.mat = mat
        self.diag_info = diag_i
        self.coaf = coaf

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
        matrix = make_ms_matrix(3, 0, np.pi / 2,self.coaf[0],self.coaf[1],self.coaf[2],self.coaf[3])
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))


def rCXz(cirquit, q1, q2):
    u1 = U(R(np.pi / 2, - np.pi / 2, 0, 2), 'Rx(-π)12')
    u2 = U(R(0, np.pi, 0, 2), 'Ry(π/2)01')
    u3 = U(R(0, np.pi, 0, 1), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, np.pi / 2, 0, 2), 'Ry(-π/2)01')
    xx = rTwoQS((0,2,0,1))
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)



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

class Press_gate(cirq.Gate):
    def __init__(self, mat,  diag_i='R'):
        self.mat = mat
        self.diag_info = diag_i


    def _qid_shape_(self):
        return (3,3,3,)

    def _unitary_(self):
        return self.mat

    def _circuit_diagram_info_(self, args):
        return self.diag_info



def U1_clear(cirquit, Q1, Q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u6 = U(R(np.pi / 2, -np.pi, 0, 2), 'Ry(-π)02')
    cirquit.append([u1(Q1), u6(Q2)], strategy=InsertStrategy.INLINE)
    #adde(cirquit, [u1, u6], [q1, q2], 1)
    cirquit.append([u2(Q1)], strategy=InsertStrategy.INLINE)
    #adde(cirquit, [u2], [q1], 1)
    xx = TwoQS((0,1,0,1))
    cirquit.append([xx(Q1, Q2)], strategy=InsertStrategy.INLINE)
    #error(cirquit, [q1, q2], PMS)
    #adde(cirquit, [xx], [q1, q2], 2)

def U1_c_clear(cirquit, q1, q2):
    u1 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    u2 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u6 = U(R(np.pi / 2, np.pi, 0, 2), 'Ry(π)02')
    xx_c = rTwoQS((0,1,0,1))
    cirquit.append([xx_c(q1, q2)], strategy=InsertStrategy.INLINE)
    #adde(cirquit, [xx_c], [q1, q2], 2)
    #error(cirquit, [q1, q2], PMS)
    #adde(cirquit, [u2], [q1], 1)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    #adde(cirquit, [u1, u6], [q1, q2], 1)
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)

def CCX(cirquit, q1, q2, q3):
    ccx_gate = Press_gate(ccx_mat)
    cirquit.append([ccx_gate(q1, q2, q3)], strategy=InsertStrategy.INLINE)

def code(circuit, Q0, Q1, Q2):
    CX(circuit, Q0, Q1)
    CX(circuit, Q0, Q2)


def decode(circuit, Q0, Q1, Q2):
    CX(circuit, Q0, Q1)
    CX(circuit, Q0, Q2)
    CCX(circuit, Q2, Q1, Q0)

class QutritAmplitudeChannel(QuditGate):

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
        self.p_matrix = np.ones((2,2))
        self.p_matrix[0][0] = self.p1
        self.p_matrix[0][1] = 1 - self.p1
        self.p_matrix[1][0] = 1 - self.p1/3
        self.p_matrix[1][1] = 1 - self.p1/3

        #print('prob[0,0]', self.p_matrix[0, 0])
        #print('prob_sum', self.p_matrix.sum())

        #print('prob_sum', self.p_matrix.sum())

    def _mixture_(self):

        Ea1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        Ea2 = np.array([[0, self.p1**0.5, 0], [0, 0, 0], [0, 0, 0]])
        Ea3 = np.array([[0, 0, self.p1**0.5], [0, 0, 0], [0, 0, 0]])
        id = np.eye(3)
        shiz_massiv = [Ea1, id]
        return tuple(zip(self.p_matrix.flatten(), shiz_massiv))

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"



def main():
    answer = []
    for P in np.linspace(0.001,0.999,100):
        circuit1 = cirq.Circuit()
        alf1 = random.randint(0, 1000) / 1000 * 2 * np.pi
        alf2 = random.randint(-1000, 1000) / 1000 * np.pi
        alf1 = 0
        alf2 = 0
        qutrits1 = []
        for j in range(3):
            qutrits1.append(cirq.LineQid(j, dimension=3))
        povorot = R(alf1, alf2, 0, 1)
        pg = U(povorot)
        circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)



        q0 = qutrits1[0]
        q1 = qutrits1[1]
        q2 = qutrits1[2]
        code(circuit1, q0, q1, q2)
        '''
        CX(circuit1, q0, q1)
        CX(circuit1, q0, q2)
        '''

        press_gate = Press_gate(Pres_mat)
        circuit1.append([press_gate(q0, q1, q2)], strategy=InsertStrategy.INLINE)

        dpg_t = QutritAmplitudeChannel((P))
        #circuit1.append([dpg_t.on(q2)], strategy=InsertStrategy.INLINE)
        ind = random.randint(0, 1000)

        if ind < P * 1000:
            povoro = R(0, np.pi, 0, 1)
            demp = U(povoro)
            circuit1.append([demp(q1)], strategy=InsertStrategy.INLINE)

        ind = random.randint(0, 1000)
        if ind < P * 1000:
            povoro = R(0, np.pi, 0, 1)
            demp = U(povoro)
            circuit1.append([demp(q2)], strategy=InsertStrategy.INLINE)
        '''
        circuit1.append([dpg_t.on(q1)], strategy=InsertStrategy.INLINE)
        circuit1.append([dpg_t.on(q2)], strategy=InsertStrategy.INLINE)
        '''
        press_gate = Press_gate(Unpres_mat)
        circuit1.append([press_gate(q0, q1, q2)], strategy=InsertStrategy.INLINE)

        decode(circuit1, q0, q1, q2)
        '''
        CX(circuit1, q0, q1)
        CX(circuit1, q0, q2)
        CCX(circuit1, q2, q1, q0)
        '''
        #povoro = R(0, 0, 0, 1)
        #demp = U(povoro)
        #circuit1.append([demp(q0)], strategy=InsertStrategy.INLINE)
        #circuit1.append([demp(q1)], strategy=InsertStrategy.INLINE)
        #circuit1.append([demp(q2)], strategy=InsertStrategy.INLINE)

        povorot = R(alf1, -alf2, 0, 1)
        pg = U(povorot)
        circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)

        #circuit1.append([demp(q0)], strategy=InsertStrategy.INLINE)
        answer.append(np.trace(abs(cirq.final_density_matrix(circuit1, qubit_order=qutrits1))[0:9][0:9]))

    print(cirq.final_density_matrix(circuit1, qubit_order=qutrits1))
    return answer

plt.scatter(np.linspace(0,1,100), main())
plt.show()

'''
circuit1 = cirq.Circuit()

qutrits1 = []
for j in range(3):
    qutrits1.append(cirq.LineQid(j, dimension=3))
q0 = qutrits1[0]
q1 = qutrits1[1]
q2 = qutrits1[2]
povoro = R(0, np.pi, 0, 1)
demp = U(povoro)
#circuit1.append([demp(q0)], strategy=InsertStrategy.INLINE)

#circuit1.append([demp(q1)], strategy=InsertStrategy.INLINE)

#circuit1.append([demp(q2)], strategy=InsertStrategy.INLINE)
#CCX(circuit1, q0,q1,q2)
pg = Press_gate(Pres_mat)
#print(abs(cirq.final_density_matrix(circuit1, qubit_order=qutrits1)[0][9]))
print(Pres_mat)
'''
