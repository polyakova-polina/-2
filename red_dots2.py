
import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random
N = 1
PMS1 = 1


T0 = 25

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

def error_correction(circuit, qutrits):
    #get_syndrome(circuit, qutrits)
    # get_syndrome_r(circuit1, qutrits1)


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

    for q in qutrits:
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
            matchek = ((a2 ** 0.5 / sss) * mx) @ (np.conj((a2 ** 0.5 / sss) * mx).T) - mi + ((a3 ** 0.5 / sss) * my) @ (
                np.conj((a3 ** 0.5 / sss) * my).T) + ((a4 ** 0.5 / sss) * mz) @ (np.conj((a4 ** 0.5 / sss) * mz).T) + ((a1 ** 0.5 / sss) * mi) @ (
                          np.conj((a1 ** 0.5 / sss) * mi).T)
            #print(matchek)
            er_gate = U(mat)
            circuit.append([er_gate(q)], strategy=InsertStrategy.INLINE)


def error(circuit, qutrits, ind):
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
            sumchek = 0


            mat = (a1 ** 0.5 * mi + a2 ** 0.5 * mx + a3 ** 0.5 * my + a4 ** 0.5 * mz) / sss
            er_gate = U(mat)
            circuit.append([er_gate(q)], strategy=InsertStrategy.INLINE)


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
Ta = range(0,50,2)
for PMS2 in np.arange(0.99, 0.995, 0.01):
    PMS2 = 1
    print(sps1)
    sps1 = []

    for t in Ta:
        #sps1.append(PMS2)
        sch = 0
        sch_m = 0
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
            #encoding_qubit(circuit1, qutrits1)
            time_error(circuit1, qutrits1, t)
            #circuit1.append([y(qutrits1[2])])
            #decoding_qubit(circuit1, qutrits1)
            '''
            res1 = sim.simulate(circuit1)
            print('88888888')
            print('res1', res1)
            print('88888888')
            '''
            #error_correction(circuit1, qutrits1)
            #povorot_r = np.conj(R(0, alf1, 0, 1) @ R(np.pi / 2, alf2, 0, 1)).T
            povorot_r = R(np.pi / 2, -alf2, 0, 1) @ R(0, -alf1, 0, 1)
            pg_r = U(povorot_r)
            circuit1.append([pg_r(qutrits1[0])], strategy=InsertStrategy.INLINE)
            #print(circuit1)
            circuit1.append([cirq.measure(qutrits1[0])])

            res1 = sim.simulate(circuit1)
            measured_bit = res1.measurements[str(qutrits1[0])][0]

            if measured_bit == 0:
                sch_m = sch_m + 1
            #print(f'Measured bit: {measured_bit}')
            # print(np.dot(res1.final_state_vector + m(z,z,z,z,z), res1.final_state_vector + m(z,z,z,z,z)))

            '''
            vec = circuit1.final_state_vector(initial_state=0, qubit_order=qutrits1)
            ssum = 0
            for o in range(80, 3**5):
                if abs(vec[o]) > 0.00000001:
                    ssum += 1
            if ssum == 0:
                sch += 1
            '''

        sps1.append(sch_m / N)
        print(sch_m / N)
a = [1.0, 1.0, 1.0, 0.75, 0.8, 0.55, 0.55, 0.55, 0.6, 0.6, 0.45, 0.65, 0.5, 0.6, 0.3, 0.4, 0.4, 0.55, 0.65, 0.65, 0.45, 0.8, 0.5, 0.5, 0.4]
'''[1.0, 0.7, 0.75, 0.55, 0.55, 0.55, 0.55, 0.4, 0.45, 0.45, 0.45, 0.6, 0.55, 0.6, 0.5, 0.35, 0.5, 0.45, 0.6, 0.4]
a = [1.0, 0.76, 0.68, 0.54, 0.51, 0.58, 0.56, 0.44, 0.55, 0.48, 0.45, 0.5, 0.49, 0.51, 0.49, 0.49, 0.55, 0.58, 0.51, 0.42]
a = [1.0, 0.88, 0.83, 0.71, 0.6, 0.53, 0.66, 0.57, 0.52, 0.6, 0.53, 0.45, 0.56, 0.48, 0.51, 0.49, 0.42, 0.55, 0.55, 0.49, 0.47, 0.46, 0.5, 0.52, 0.46,0.42, 0.49, 0.42, 0.46, 0.46, 0.48, 0.45, 0.46, 0.54, 0.48, 0.52, 0.49, 0.47, 0.54, 0.48, 0.59, 0.57, 0.5, 0.52, 0.49, 0.44, 0.51, 0.43, 0.52, 0.48]
a = [1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.6, 0.4, 0.7, 0.6, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.7, 0.5, 0.4, 0.4, 0.5, 0.7, 0.3, 0.4]
a = [1.0, 0.99, 0.95, 0.93, 0.91, 0.77, 0.76, 0.68, 0.64, 0.66, 0.5, 0.56, 0.6, 0.6, 0.58, 0.53, 0.6, 0.53, 0.55, 0.42, 0.47, 0.59, 0.49, 0.49, 0.43]
a = [1.0, 0.984, 0.928, 0.878, 0.846, 0.776, 0.728, 0.715, 0.643, 0.623, 0.603, 0.577, 0.561, 0.539, 0.519, 0.506, 0.544, 0.502, 0.515, 0.498, 0.494, 0.494, 0.503, 0.494, 0.538]
'''

sps1 = [1.0, 1.0, 0.9, 0.8, 0.8, 0.6, 0.8, 0.85, 0.65, 0.45, 0.7, 0.45, 0.55, 0.55, 0.6, 0.5, 0.6, 0.4, 0.4, 0.5, 0.6, 0.6, 0.55, 0.5, 0.55]

Ta = np.arange(0,1,0.1)
a = [0.98, 0.98, 0.88, 0.74, 0.78, 0.66, 0.54, 0.62, 0.46, 0.5, 0.44, 0.62, 0.52, 0.34, 0.46, 0.54, 0.48, 0.54, 0.32, 0.52, 0.42, 0.46, 0.52, 0.38, 0.42]
sps1 = [1.0, 0.9, 0.94, 0.88, 0.78, 0.8, 0.62, 0.78, 0.66, 0.64, 0.54, 0.64, 0.66, 0.6, 0.5, 0.54, 0.42, 0.5, 0.4, 0.54, 0.42, 0.48, 0.52, 0.34, 0.5]
a = [0.994, 0.998, 0.996, 0.994, 0.992, 0.993, 0.99, 0.989, 0.984, 0.985]
s = [0.989, 0.983, 0.979, 0.973, 0.983, 0.981, 0.973, 0.969, 0.965, 0.957]
s += np.array([0.992, 0.981, 0.98, 0.978, 0.975, 0.98, 0.979, 0.963, 0.971, 0.958])
s += np.array([0.981, 0.983, 0.99, 0.98, 0.981, 0.98, 0.964, 0.967, 0.975, 0.974])
s += np.array([0.985, 0.986, 0.981, 0.978, 0.983, 0.969, 0.975, 0.969, 0.962, 0.962])
s += np.array([0.988, 0.987, 0.981, 0.977, 0.987, 0.976, 0.976, 0.97, 0.977, 0.969])
s += np.array([0.986, 0.99, 0.976, 0.988, 0.976, 0.976, 0.979, 0.971, 0.967, 0.959])
s += np.array([0.988, 0.98, 0.973, 0.977, 0.977, 0.97, 0.971, 0.977, 0.968, 0.967])
print(s / 7)
a = [0.996, 0.996, 0.991, 0.994, 0.991, 0.989, 0.992, 0.99, 0.991, 0.989, 0.992, 0.981, 0.98, 0.978, 0.975, 0.98, 0.979, 0.963, 0.971, 0.958]
a = [0.996, 0.996, 0.991, 0.994, 0.991, 0.989, 0.992, 0.99, 0.991, 0.989,0.987  ,    0.98428571 ,0.98  ,     0.97871429 ,0.98028571 ,0.976,
 0.97385714, 0.96942857 ,0.96928571 ,0.96371429]
sps1 = [1.0, 0.998, 0.9942, 0.9944, 0.9868, 0.9896, 0.9832, 0.9812, 0.9792, 0.9816, 0.9752, 0.9732, 0.9694, 0.9624, 0.9672, 0.962, 0.9564, 0.9572, 0.9562, 0.9534]

a = [0.994, 0.992, 0.9936, 0.9886, 0.9902, 0.9838, 0.98, 0.9766, 0.9698, 0.9686, 0.9652, 0.9508, 0.9496, 0.94, 0.9362, 0.9282, 0.9244, 0.9086, 0.9118, 0.8954, 0.894, 0.8772, 0.875, 0.8654, 0.863]
sps1 = [1.0, 0.9958, 0.9904, 0.9872, 0.977, 0.9756, 0.973, 0.9634, 0.9498, 0.9532, 0.9496, 0.9468, 0.941, 0.9372, 0.9284, 0.9224, 0.9156, 0.9182, 0.9018, 0.8962, 0.9066, 0.9004, 0.8856, 0.8916, 0.881]

Ta = np.arange(0,5,0.2)
#print(sps1)
fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot()
ax.scatter(Ta, sps1, color='b', s = 5, label = 'без коррекции')
ax.scatter(Ta, a, color='r', s = 5, label = 'c коррекции')

ax.set_xlabel('t, mcs (T0 = 25 mcs)')
ax.set_xlabel('fid')
plt.title('P1 = 1, P2 = 1')
plt.legend()
plt.grid()

plt.show()

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
'''