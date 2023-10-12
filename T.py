import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
def make_CX_qutrits(q):
    return x(q[1]).controlled_by(q[0])

def make_CCX_qutrits(q):
    return (x(q[2]).controlled_by(q[0])).controlled_by(q[1])

def make_CX_qutrits_conj(q):
    return x_conj(q[1]).controlled_by(q[0])

def make_CCX_qutrits_conj(q):
    return (x_conj(q[2]).controlled_by(q[0])).controlled_by(q[1])

def make_CZ_qutrits(q):
    return z(q[1]).controlled_by(q[0])

class H(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.conj(1 / (2 ** 0.5) * np.array([[1, 1, 0],
                                                  [1, -1, 0],
                                                  [0, 0, 2 ** 0.5]]))

    def _circuit_diagram_info_(self, args):
        return '[+1]'

class X1_conj(cirq.Gate):

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.conj(np.array([[0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0, 1]]))

    def _circuit_diagram_info_(self, args):
        return '[+1]'

class X2_conj(cirq.Gate):

    def _qid_shape_(self):

        return (3,)


    def _unitary_(self):

        return np.conj(np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]]))

    def _circuit_diagram_info_(self, args):
        return '[+1]'

class Z1(cirq.Gate):

    def _qid_shape_(self):

        return (3,)


    def _unitary_(self):

        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]])

    def _circuit_diagram_info_(self, args):
        return '[+1]'

class X2(cirq.Gate):

    def _qid_shape_(self):

        return (3,)


    def _unitary_(self):

        return np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]])

    def _circuit_diagram_info_(self, args):
        return '[+1]'

class X1(cirq.Gate):
    """Ворота, которые добавляют единицу в вычислительную основу кутрита.

     Эти ворота действуют на трехуровневые системы. В вычислительной основе
     этой системе он осуществляет преобразование U|x〉 = |x + 1 mod 3〉, или
     другими словами, U|0〉 = |1〉, U|1〉 = |2〉 и U|2> = |0〉.
     """

    def _qid_shape_(self):
        # Реализуя этот метод, эти ворота реализуют
         # протокол cirq.qid_shape и вернет кортеж (3,)
         # когда cirq.qid_shape действует на экземпляр этого класса.
         # Это указывает на то, что вентиль действует на один кутрит.
        return (3,)

    def _unitary_(self):
        # Поскольку ворота действуют на трехуровневые системы, они имеют унитарную
         # эффект, представляющий собой унитарную матрицу размером три на три.
        return np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1]])


    def _circuit_diagram_info_(self, args):
        return '[+1]'

def svertka(circuit, controling_qutrits, k):
    for i in range(1, k // 2):
        i -= 1
        curl_x2 = x2(controling_qutrits[i + 1])
        curl_cx = make_CX_qutrits([controling_qutrits[i], controling_qutrits[i + 1]])
        curl_x = x(controling_qutrits[i + 1])
        i += k // 2
        curr_x2 = x2(controling_qutrits[i + 1])
        curr_cx = make_CX_qutrits([controling_qutrits[i], controling_qutrits[i + 1]])
        curr_x = x(controling_qutrits[i + 1])
        circuit.append([curl_x2, curr_x2], strategy=InsertStrategy.INLINE)
        circuit.append([curl_cx, curr_cx], strategy=InsertStrategy.INLINE)
        circuit.append([curl_x, curr_x], strategy=InsertStrategy.INLINE)

    if k % 2 == 1:
        k -= 1
        curr_x2 = x2(controling_qutrits[k])
        curr_cx = make_CX_qutrits([controling_qutrits[k - 1], controling_qutrits[k]])
        curr_x = x(controling_qutrits[k])
        circuit.append([curr_x2], strategy=InsertStrategy.INLINE)
        circuit.append([curr_cx], strategy=InsertStrategy.INLINE)
        circuit.append([curr_x], strategy=InsertStrategy.INLINE)
        k += 1

def main_operation(circuit, controling_qutrits, target_qutrit, k):
    curl_x2 = x2(target_qutrit)
    curl_cx = make_CX_qutrits([controling_qutrits[k // 2 - 1], target_qutrit])
    curl_x = x(target_qutrit)
    curr_cz = make_CZ_qutrits([controling_qutrits[k - 1], target_qutrit])
    circuit.append([curl_x2], strategy=InsertStrategy.INLINE)
    circuit.append([curl_cx], strategy=InsertStrategy.INLINE)
    circuit.append([curl_x], strategy=InsertStrategy.INLINE)
    circuit.append([curr_cz], strategy=InsertStrategy.INLINE)

def razvertka(circuit, controling_qutrits, target_qutrit, k):
    curl_x2 = x2_conj(target_qutrit)
    curl_cx = make_CX_qutrits_conj([controling_qutrits[k // 2 - 1], target_qutrit])
    curl_x = x_conj(target_qutrit)

    circuit.append([curl_x], strategy=InsertStrategy.INLINE)
    circuit.append([curl_cx], strategy=InsertStrategy.INLINE)
    circuit.append([curl_x2], strategy=InsertStrategy.INLINE)

    if k % 2 == 1:
        k -= 1
        curr_x2 = x2_conj(controling_qutrits[k])
        curr_cx = make_CX_qutrits_conj([controling_qutrits[k - 1], controling_qutrits[k]])
        curr_x = x_conj(controling_qutrits[k])
        circuit.append([curr_x], strategy=InsertStrategy.INLINE)
        circuit.append([curr_cx], strategy=InsertStrategy.INLINE)
        circuit.append([curr_x2], strategy=InsertStrategy.INLINE)

        k += 1

    for i in range(1, k // 2)[::-1]:
        i -= 1
        curl_x2 = x2_conj(controling_qutrits[i + 1])
        curl_cx = make_CX_qutrits_conj([controling_qutrits[i], controling_qutrits[i + 1]])
        curl_x = x_conj(controling_qutrits[i + 1])
        i += k // 2
        curr_x2 = x2_conj(controling_qutrits[i + 1])
        curr_cx = make_CX_qutrits_conj([controling_qutrits[i], controling_qutrits[i + 1]])
        curr_x = x_conj(controling_qutrits[i + 1])
        circuit.append([curl_x, curr_x], strategy=InsertStrategy.INLINE)
        circuit.append([curl_cx, curr_cx], strategy=InsertStrategy.INLINE)
        circuit.append([curl_x2, curr_x2], strategy=InsertStrategy.INLINE)

def CnX(circuit, cq, tq):
    k = len(cq)
    h = H()
    circuit.append([h(tq)])
    svertka(circuit, cq, k)
    main_operation(circuit, cq, tq, k)
    razvertka(circuit, cq, tq, k)
    circuit.append([h(tq)])

def encoding_qubit(circuit, log_qubit):

    x = X1()
    x_conj = X1_conj()

    h = H()
    q1, q2, q3, q4, q5 = log_qubit[0], log_qubit[1], log_qubit[2], log_qubit[3], log_qubit[4]
    gates = [h(q2), h(q3), h(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)


    '''
    svertka(circuit, [q1, q3], 2)
    main_operation(circuit, [q1, q3], q4, 2)
    razvertka(circuit, [q1, q3], q4, 2)
    '''
    cur_ccx = make_CCX_qutrits([q1, q2, q3])
    circuit.append([cur_ccx], strategy=InsertStrategy.INLINE)

    gates = [x(q3), x(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)

    '''
    svertka(circuit, [q4, q3], 2)
    main_operation(circuit, [q4, q3], q1, 2)
    razvertka(circuit, [q4, q3], q1, 2)
    '''
    cur_ccx = make_CCX_qutrits([q4, q3, q1])
    circuit.append([cur_ccx], strategy=InsertStrategy.INLINE)


    gates = [x(q3), x(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)

    cur_cx = make_CX_qutrits([q1, q5])
    circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    cur_cx = make_CX_qutrits([q2, q5])
    circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    cur_cx = make_CX_qutrits([q2, q1])
    circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    cur_cx = make_CX_qutrits([q4, q1])
    circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    cur_cx = make_CX_qutrits([q3, q5])
    circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    cur_cz = make_CZ_qutrits([q4, q5])
    circuit.append([cur_cz], strategy=InsertStrategy.INLINE)

def decoding_qubit(circuit, log_qubit):
    x = X1()
    x_conj = X1_conj()

    h = H()
    q1, q2, q3, q4, q5 = log_qubit[0], log_qubit[1], log_qubit[2], log_qubit[3], log_qubit[4]

    cur_cz = make_CZ_qutrits([q4, q5])
    circuit.append([cur_cz], strategy=InsertStrategy.INLINE)

    cur_cx = make_CX_qutrits([q3, q5])
    circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    cur_cx = make_CX_qutrits([q4, q1])
    circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    cur_cx = make_CX_qutrits([q2, q1])
    circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    cur_cx = make_CX_qutrits([q2, q5])
    circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    cur_cx = make_CX_qutrits([q1, q5])
    circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    gates = [x(q3), x(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    '''
    svertka(circuit, [q1, q3], 2)
    main_operation(circuit, [q1, q3], q4, 2)
    razvertka(circuit, [q1, q3], q4, 2)
    '''

    cur_ccx = make_CCX_qutrits([q1, q3, q4])
    circuit.append([cur_ccx], strategy=InsertStrategy.INLINE)

    gates = [x(q3), x(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    '''
    svertka(circuit, [q1, q3], 2)
    main_operation(circuit, [q1, q3], q4, 2)
    razvertka(circuit, [q1, q3], q4, 2)
    '''
    cur_ccx = make_CCX_qutrits([q1, q3, q4])
    circuit.append([cur_ccx], strategy=InsertStrategy.INLINE)

    gates = [h(q2), h(q3), h(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)

def X1_l(circuit, lqubits):
  x = X1()
  z = Z1()
  q1, q2, q3, q4, q5 = lqubits[0], lqubits[1], lqubits[2], lqubits[3], lqubits[4]
  gates = [z(q1), z(q4)]
  circuit.append(gates, strategy=InsertStrategy.INLINE)
  gates = [x(q1), x(q2),x(q3),x(q4),x(q5)]
  circuit.append(gates, strategy=InsertStrategy.INLINE)

def Z1_l(circuit, lqubits):
  q1, q2, q3, q4, q5 = lqubits[0], lqubits[1], lqubits[2], lqubits[3], lqubits[4]
  gates = [make_CX_qutrits([q1, q2])]
  circuit.append(gates, strategy=InsertStrategy.INLINE)
  gates = [make_CX_qutrits([q4, q3])]
  circuit.append(gates, strategy=InsertStrategy.INLINE)
  gates = [z(q1),z(q2),z(q3),z(q4),z(q5)]
  circuit.append(gates, strategy=InsertStrategy.INLINE)
  gates = [make_CX_qutrits([q1, q2])]
  circuit.append(gates, strategy=InsertStrategy.INLINE)
  gates = [make_CX_qutrits([q4, q3])]
  circuit.append(gates, strategy=InsertStrategy.INLINE)

x = X1()
x2 = X2()
z = Z1()
x_conj = X1_conj()
x2_conj = X2_conj()
h = H()

sim = cirq.Simulator()
circuit1 = cirq.Circuit()
qutrits1 = []

for i in range(5):
    qutrits1.append(cirq.LineQid(i, dimension=3))

gates1 = [x2(qutrits1[0])]
circuit1.append(gates1)

encoding_qubit(circuit1, qutrits1)
X1_l(circuit1, qutrits1)

#svertka(circuit1, qutrits1[0:-1], 2)
#main_operation(circuit1, qutrits1[0:-1], qutrits1[-1], 2)
#razvertka(circuit1, qutrits1[0:-1], qutrits1[-1], 2)
#decoding_qubit(circuit1, qutrits1)

res1 = sim.simulate(circuit1)
print(circuit1)
print(res1)






