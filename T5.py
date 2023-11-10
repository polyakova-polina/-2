

import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random
N = 1000
PMS = 0.89


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
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    xx = TwoQuditMSGate3()
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    #error(cirquit, [q1, q2], PMS)



def U1_c(cirquit, q1, q2):
    u1 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    u2 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u6 = U(R(np.pi / 2, np.pi, 0, 2), 'Ry(π)02')
    xx_c = TwoQuditMSGate3_c()
    cirquit.append([xx_c(q1, q2)], strategy=InsertStrategy.INLINE)
    #error(cirquit, [q1, q2], PMS)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)



def CX(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 1), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u5 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    xx = TwoQuditMSGate3()
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    #error(cirquit, [q1, q2], PMS)
    cirquit.append([u3(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u5(q1)], strategy=InsertStrategy.INLINE)



def CCX(cirquit, q1, q2, q3):
    U1(cirquit, q1, q2)
    CX(cirquit, q2, q3)
    U1_c(cirquit, q1, q2)


def CZ(cirquit, q1, q2):
    h = H()
    cirquit.append(h(q2), strategy=InsertStrategy.INLINE)
    CX(cirquit, q1, q2)
    cirquit.append(h(q2), strategy=InsertStrategy.INLINE)


def CCZ(cirquit, q1, q2, q3):
    h = H()
    cirquit.append(h(q3), strategy=InsertStrategy.INLINE)
    CCX(cirquit, q1, q2, q3)
    cirquit.append(h(q3), strategy=InsertStrategy.INLINE)


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
        return R(np.pi/2, np.pi, 0, 1)

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
    gates = [h(q2), h(q3), h(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    CCZ(circuit, q1, q3, q4)
    gates = [x(q3), x(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    CCZ(circuit, q4, q3, q1)
    gates = [x(q3), x(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
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
    gates = [x(q3), x(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    CCZ(circuit, q1, q3, q4)
    gates = [x(q3), x(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    CCZ(circuit, q1, q3, q4)
    gates = [h(q2), h(q3), h(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)


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
    #cirquit.append([cirq.measure(a1)])

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
    U1(cirquit, q1, q2)
    U1(cirquit, q2, q3)
    U1(cirquit, q3, q4)
    CX(cirquit, q4, q5)
    U1_c(cirquit, q3, q4)
    U1_c(cirquit, q2, q3)
    U1_c(cirquit, q1, q2)

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
  get_syndrome(circuit, qutrits)
  #get_syndrome_r(circuit1, qutrits1)


  q0 = qutrits1[0]
  q1 = qutrits1[1]
  q2 = qutrits1[2]
  q3 = qutrits1[3]
  q4 = qutrits1[4]
  a1 = qutrits1[5]
  a2 = qutrits1[6]
  a3 = qutrits1[7]
  a4 = qutrits1[8]

  # Операции для исправления ошибок X
  circuit1.append([x(a1), x(a3), x(a4)], strategy=InsertStrategy.INLINE)
  CCCCX(circuit, a1, a2, a3, a4, q0)
  circuit1.append([x(a1), x(a3), x(a4)], strategy=InsertStrategy.INLINE)

  circuit1.append([x(a3), x(a4)], strategy=InsertStrategy.INLINE)
  CCCCX(circuit, a1, a2, a3, a4, q1)
  circuit1.append([x(a3), x(a4)], strategy=InsertStrategy.INLINE)

  circuit1.append([x(a2), x(a3), x(a4)], strategy=InsertStrategy.INLINE)
  CCCCX(circuit, a1, a2, a3, a4, q2)
  circuit1.append([x(a2), x(a3), x(a4)], strategy=InsertStrategy.INLINE)

  circuit1.append([x(a1), x(a2), x(a4)], strategy=InsertStrategy.INLINE)
  CCCCX(circuit, a1, a2, a3, a4, q3)
  circuit1.append([x(a1), x(a2), x(a4)], strategy=InsertStrategy.INLINE)

  circuit1.append([x(a1), x(a2), x(a3)], strategy=InsertStrategy.INLINE)
  CCCCX(circuit, a1, a2, a3, a4, q4)
  circuit1.append([x(a1), x(a2), x(a3)], strategy=InsertStrategy.INLINE)


  # Операции для исправления ошибок Z
  circuit1.append([x(a2), x(a4)], strategy=InsertStrategy.INLINE)
  CCCCZ(circuit, a1, a2, a3, a4, q0)
  circuit1.append([x(a2), x(a4)], strategy=InsertStrategy.INLINE)

  circuit1.append([x(a1), x(a2)], strategy=InsertStrategy.INLINE)
  CCCCZ(circuit, a1, a2, a3, a4, q1)
  circuit1.append([x(a1), x(a2)], strategy=InsertStrategy.INLINE)

  circuit1.append([x(a3), x(a1)], strategy=InsertStrategy.INLINE)
  CCCCZ(circuit, a1, a2, a3, a4, q2)
  circuit1.append([x(a3), x(a1)], strategy=InsertStrategy.INLINE)

  circuit1.append([x(a2), x(a3)], strategy=InsertStrategy.INLINE)
  CCCCZ(circuit, a1, a2, a3, a4, q3)
  circuit1.append([x(a2), x(a3)], strategy=InsertStrategy.INLINE)

  circuit1.append([x(a1), x(a4)], strategy=InsertStrategy.INLINE)
  CCCCZ(circuit, a1, a2, a3, a4, q4)
  circuit1.append([x(a1), x(a4)], strategy=InsertStrategy.INLINE)

  # Операции для исправления ошибок Y
  circuit1.append([x(a4)], strategy=InsertStrategy.INLINE)
  CCCCY(circuit, a1, a2, a3, a4, q0)
  circuit1.append([x(a4)], strategy=InsertStrategy.INLINE)

  CCCCY(circuit, a1, a2, a3, a4, q1)

  circuit1.append([x(a3)], strategy=InsertStrategy.INLINE)
  CCCCY(circuit, a1, a2, a3, a4, q2)
  circuit1.append([x(a3)], strategy=InsertStrategy.INLINE)

  circuit1.append([x(a2)], strategy=InsertStrategy.INLINE)
  CCCCY(circuit, a1, a2, a3, a4, q3)
  circuit1.append([x(a2)], strategy=InsertStrategy.INLINE)

  circuit1.append([x(a1)], strategy=InsertStrategy.INLINE)
  CCCCY(circuit, a1, a2, a3, a4, q4)
  circuit1.append([x(a1)], strategy=InsertStrategy.INLINE)

def error(circuit, qutrits, p):
    x = X1()
    y = Y1()
    z = Z1()
    rv = random.randint(0, 10000)
    n = len(qutrits)
    if rv < p * 10000:
        return
    else:
        for i in range(n):
            rv = random.randint(0,3)
            if rv == 1:
                circuit.append(x(qutrits[i]), strategy=InsertStrategy.INLINE)
            if rv == 2:
                circuit.append(y(qutrits[i]), strategy=InsertStrategy.INLINE)
            if rv == 3:
                circuit.append(z(qutrits[i]), strategy=InsertStrategy.INLINE)

def X1_l(circuit, lqubits):
  x = X1()
  z = Z1()
  q1, q2, q3, q4, q5 = lqubits[0], lqubits[1], lqubits[2], lqubits[3], lqubits[4]
  gates = [z(q1), z(q4)]
  circuit.append(gates, strategy=InsertStrategy.INLINE)
  gates = [x(q1), x(q2),x(q3),x(q4),x(q5)]
  circuit.append(gates, strategy=InsertStrategy.INLINE)

def X1_l(circuit, lqubits):
  x = X1()
  z = Z1()
  q1, q2, q3, q4, q5 = lqubits[0], lqubits[1], lqubits[2], lqubits[3], lqubits[4]
  gates = [x(q1), x(q4)]
  circuit.append(gates, strategy=InsertStrategy.INLINE)
  gates = [z(q5)]
  circuit.append(gates, strategy=InsertStrategy.INLINE)

#def make_error(p):


#Основная операция
x = X1()
x2 = X2()
z = Z1()
y = Y1()
x_conj = X1_conj()
x2_conj = X2_conj()
h = H()

sim = cirq.Simulator()
circuit1 = cirq.Circuit()
qutrits1 = []



for i in range(10):
    qutrits1.append(cirq.LineQid(i, dimension=3))

#кодируемое состояние
gates1 = [x(qutrits1[0])]
#circuit1.append(gates1)
encoding_qubit(circuit1, qutrits1)

#ошибка
gates1 = [z(qutrits1[4])]
#circuit1.append(gates1)

#error_correction(circuit1, qutrits1)

#error(circuit1, qutrits1, 0.5)
decoding_qubit(circuit1, qutrits1)
#circuit1.append([cirq.measure(qutrits1[1])])
#circuit1.append([cirq.measure(qutrits1[2])])
#circuit1.append([cirq.measure(qutrits1[3])])
#circuit1.append([cirq.measure(qutrits1[4])])
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
measured_bit = res1.measurements[str(qutrits1[4])][0]
print(f'Measured bit: {measured_bit}')
'''
def m(a,b, c, d, e):
   return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)

def make_scheme(t):
    circuit1 = cirq.Circuit()
    qutrits1 = []
    for i in range(5):
        qutrits1.append(cirq.LineQid(i, dimension=3))
    encoding_qubit(circuit1, qutrits1)

    # t-fian

    decoding_qubit(circuit1, qutrits1)
    for i in range(5):
        circuit1.append([cirq.measure(qutrits1[i])])


sch = 0

#print(res1.final_state_vector)
#print(circuit1)
# print(res1)