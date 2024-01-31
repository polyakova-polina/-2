import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features



def R(fi, hi, i = 0, j = 1):
    I = np.array([[1,0,0],[0,1,0],[0,0,1]])
    x01_for_ms = np.array([[0,1,0],
                         [1,0,0],
                         [0,0,0]])
    y01_for_ms = np.array([[0,complex(0,-1),0],
                         [complex(0,1),0,0],
                         [0,0,0]])
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
    if (i,j) == (0,1):
        x_for_ms = x01_for_ms
        y_for_ms = y01_for_ms
    elif (i,j) == (1,2):
        x_for_ms = x12_for_ms
        y_for_ms = y12_for_ms
    else:
        x_for_ms = x02_for_ms
        y_for_ms = y02_for_ms
    m = np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms

    return linalg.expm(complex(0,-1) * m * hi / 2)

def make_ms_matrix(fi, hi, i = 0, j = 1, k = 0, l = 1):
    x_for_ms = np.array([[0,1,0],
                         [1,0,0],
                         [0,0,0]])
    y_for_ms = np.array([[0,complex(0,-1),0],
                         [complex(0,1),0,0],
                         [0,0,0]])
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
        matrix = make_ms_matrix(0, -np.pi/2)
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
        matrix = make_ms_matrix(0, np.pi/2)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

'''

def make_CXR(circuit, q):
    
    I = np.array([[1,0,0], [0,1,0],[0,0,1]])
    u = np.array([[0,0,0],
                     [0,0,complex(0,-1)],
                     [0,complex(0,-1), 0]])

    circuit.append([u(q[0])], strategy=InsertStrategy.INLINE)

    u = np.array([[0, 1, 0],
                     [-1, 0, 0],
                     [0, 0, 1]])

    my_qft2 = cirq.MatrixGate(np.kron(u, I))
    print(cirq.Circuit(my_qft2(q[0], q[1])))

    circuit.append([u(q[0])], strategy=InsertStrategy.INLINE)

    u =
    
circ = cirq.Circuit(
        this_gate.on(*cirq.LineQubit.range(2))
    )
    my_gate = cirq.DenseMatrixGate(make_ms_matrix(0, np.pi/2))
    circuit.append([my_gate(q[0], q[1])], strategy=InsertStrategy.INLINE)
    
'''

class U(cirq.Gate):
    def __init__(self, mat, diag_i = 'R'):
        self.mat = mat
        self.diag_info = diag_i
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return self.mat


    def _circuit_diagram_info_(self, args):
        return self.diag_info

def make_CX_gate(q):
    x = psevdo_X1()
    return x(q[1]).controlled_by(q[0])

def CX_prot(cirquit, q1, q2):
    U_iii = iii()
    cirquit.append([U_iii(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([make_CX_gate([q1,q2])], strategy=InsertStrategy.INLINE)

def CX(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 1), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u5 = U(R(0, np.pi, 1, 2), 'Rx(π)12')

    xx_c = make_ms_matrix(0, -np.pi / 2)
    x01 = U(R(0, np.pi, 0,1))
    u_iii = iii()
    # print(np.conj(xx))
    # print()
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)

    '''
    x12 = U(R(0, np.pi, 1, 2))
    u_1i1 = eie()
    u_emm = emm()
    cirquit.append([x01(q1), x01(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u_iii(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([x12(q1), x12(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append((u_1i1(q2).controlled_by(q1)))
    cirquit.append([x12(q1), x12(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u_emm(q1), u_emm(q2)])
    '''
    xx = TwoQuditMSGate3()
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)

    cirquit.append([u3(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u5(q1)], strategy=InsertStrategy.INLINE)
    #U2 = np.kron(u1, I) @ np.kron(u2, I) @ xx @ np.kron(u3, u3) @ np.kron(u4, I) @ np.kron(u5, I)

def make_CCX_gate(q):
    x = psevdo_X1()
    return (x(q[2]).controlled_by(q[0])).controlled_by(q[1])

def CCX_prot(cirquit, q1, q2, q3):
    U_iii = iii()
    cirquit.append([U_iii(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([make_CCX_gate([q1, q2, q3])], strategy=InsertStrategy.INLINE)

def U1(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')

    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')

    u6 = U(R(np.pi / 2, -np.pi, 0, 2), 'Ry(-π)02')


    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)

    x01 = U(R(0, np.pi, 0, 1))
    u_iii = iii()
    x12 = U(R(0, np.pi, 1, 2))
    u_1i1 = eie()
    u_emm = emm()

    xx = TwoQuditMSGate3()
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)

    '''
    cirquit.append([x01(q1), x01(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u_iii(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([x12(q1), x12(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append((u_1i1(q2).controlled_by(q1)))
    cirquit.append([x12(q1), x12(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u_emm(q1), u_emm(q2)])
    '''

def U1_c(cirquit, q1, q2):
    u1 = U(R(0, np.pi, 1, 2), 'Rx(π)12')

    u2 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')

    u6 = U(R(np.pi / 2, np.pi, 0, 2), 'Ry(π)02')


    x01 = U(R(0, np.pi, 0, 1))
    u_iii = U(np.conj(np.array([[complex(0,1),0,0],[0,complex(0,1),0], [0,0,complex(0,1)]])))
    x12 = U(R(0, np.pi, 1, 2))
    u_1i1 = U(np.conj(np.array([[1,0,0],[0,complex(0,-1),0], [0,0,1]])))
    u_emm = U(np.conj(np.array([[1,0,0],[0, -1,0], [0,0,-1]])))

    '''
    cirquit.append([u_emm(q1), u_emm(q2)])
    cirquit.append([x12(q1), x12(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append((u_1i1(q2).controlled_by(q1)))
    cirquit.append([x12(q1), x12(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u_iii(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([x01(q1), x01(q2)], strategy=InsertStrategy.INLINE)
    '''
    xx_c = TwoQuditMSGate3_c()
    cirquit.append([xx_c(q1, q2)], strategy=InsertStrategy.INLINE)


    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)

def CCX(cirquit, q1, q2, q3):
    U1(cirquit, q1, q2)
    CX(cirquit, q2, q3)
    U1_c(cirquit, q1, q2)




'''
def make_CX_qutrits_conj(q):
    return x_conj(q[1]).controlled_by(q[0])

def make_CCX_qutrits_conj(q):
    return (x_conj(q[2]).controlled_by(q[0])).controlled_by(q[1])
'''

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

        return R(0,np.pi,0,1) @ R(np.pi / 2, np.pi / 2, 0,1)



    def _circuit_diagram_info_(self, args):
        return 'H'

class iii(cirq.Gate):

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        '''
        return np.conj(np.array([[0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0, 1]]))
        '''
        return np.array([[complex(0,1),0,0],[0,complex(0,1),0], [0,0,complex(0,1)]])

    def _circuit_diagram_info_(self, args):
        return 'iii'

class emm(cirq.Gate):

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        '''
        return np.conj(np.array([[0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0, 1]]))
        '''
        return np.array([[1,0,0],[0, -1,0], [0,0,-1]])

    def _circuit_diagram_info_(self, args):
        return 'emm'

class eie(cirq.Gate):

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        '''
        return np.conj(np.array([[0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0, 1]]))
        '''
        return np.array([[1,0,0],[0,complex(0,-1),0], [0,0,1]])

    def _circuit_diagram_info_(self, args):
        return 'eie'

class X1_conj(cirq.Gate):

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        '''
        return np.conj(np.array([[0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0, 1]]))
        '''
        return np.array([[0,complex(0,-1),0],[complex(0,-1),0,0], [0,0,1]])

    def _circuit_diagram_info_(self, args):
        return 'X1_c'

class X2_conj(cirq.Gate):

    def _qid_shape_(self):

        return (3,)


    def _unitary_(self):
        '''
        return np.conj(np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]]))
        '''
        return np.conj(np.array([[0, 0, complex(0, -1)],
                                 [0, 1, 0],
                                 [complex(0, -1), 0, 0]]))

    def _circuit_diagram_info_(self, args):
        return 'X2_c'

class Z1(cirq.Gate):

    def _qid_shape_(self):

        return (3,)


    def _unitary_(self):
        '''
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]])
        '''
        return R(0, np.pi,0,1) @ R(np.pi / 2, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'Z1'

class X2(cirq.Gate):

    def _qid_shape_(self):

        return (3,)


    def _unitary_(self):
        '''
                return np.conj(np.array([[0, 0, 1],
                                 [0, 1, 0],
                                 [1, 0, 0]]))
                '''
        return R(0, np.pi, 0,2)

    def _circuit_diagram_info_(self, args):
        return 'X2'

class psevdo_X1(cirq.Gate):
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
        return 'x1p'

class X1(cirq.Gate):

    def _qid_shape_(self):
        # Реализуя этот метод, эти ворота реализуют
         # протокол cirq.qid_shape и вернет кортеж (3,)
         # когда cirq.qid_shape действует на экземпляр этого класса.
         # Это указывает на то, что вентиль действует на один кутрит.
        return (3,)

    def _unitary_(self):
        # Поскольку ворота действуют на трехуровневые системы, они имеют унитарную
         # эффект, представляющий собой унитарную матрицу размером три на три.
        return R(0, np.pi, 0,1)


    def _circuit_diagram_info_(self, args):
        return 'X1'

def svertka(circuit, controling_qutrits, k):
    for i in range(1, k // 2):
        i -= 1

        curl_x2 = x2(controling_qutrits[i + 1])
        i += k // 2
        curr_x2 = x2(controling_qutrits[i + 1])
        i -= k // 2
        circuit.append([curl_x2, curr_x2], strategy=InsertStrategy.INLINE)

        CX(circuit, controling_qutrits[i], controling_qutrits[i + 1])
        i += k // 2
        CX(circuit, controling_qutrits[i], controling_qutrits[i + 1])
        i -= k // 2

        curl_x = x(controling_qutrits[i + 1])
        i += k // 2
        curr_x = x(controling_qutrits[i + 1])
        circuit.append([curl_x, curr_x], strategy=InsertStrategy.INLINE)




    if k % 2 == 1:
        k -= 1
        curr_x2 = x2(controling_qutrits[k])
        circuit.append([curr_x2], strategy=InsertStrategy.INLINE)
        CX(circuit, controling_qutrits[k - 1], controling_qutrits[k])
        curr_x = x(controling_qutrits[k])
        circuit.append([curr_x], strategy=InsertStrategy.INLINE)

        k += 1

def main_operation(circuit, controling_qutrits, target_qutrit, k):
    curl_x2 = x2(target_qutrit)
    circuit.append([curl_x2], strategy=InsertStrategy.INLINE)
    CX(circuit, controling_qutrits[k // 2 - 1], target_qutrit)
    curl_x = x(target_qutrit)
    circuit.append([curl_x], strategy=InsertStrategy.INLINE)
    CZ(circuit, controling_qutrits[k - 1], target_qutrit)

def razvertka(circuit, controling_qutrits, target_qutrit, k):
    curl_x2 = x2(target_qutrit)
    curl_x = x(target_qutrit)
    circuit.append([curl_x], strategy=InsertStrategy.INLINE)
    CX(circuit, controling_qutrits[k // 2 - 1], target_qutrit)
    circuit.append([curl_x2], strategy=InsertStrategy.INLINE)


    if k % 2 == 1:
        k -= 1
        curr_x2 = x2(controling_qutrits[k])
        curr_x = x(controling_qutrits[k])
        circuit.append([curr_x], strategy=InsertStrategy.INLINE)
        CX(circuit, controling_qutrits[k - 1], controling_qutrits[k])
        circuit.append([curr_x2], strategy=InsertStrategy.INLINE)

        k += 1

    for i in range(1, k // 2)[::-1]:
        i -= 1
        curl_x2 = x2(controling_qutrits[i + 1])

        curl_x = x(controling_qutrits[i + 1])
        i += k // 2
        curr_x2 = x2(controling_qutrits[i + 1])

        curr_x = x(controling_qutrits[i + 1])
        circuit.append([curl_x, curr_x], strategy=InsertStrategy.INLINE)
        CX(circuit, controling_qutrits[i], controling_qutrits[i + 1])
        i += k // 2
        CX(circuit, controling_qutrits[i], controling_qutrits[i + 1])
        i -= k // 2
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
    #x_conj = X1_conj()

    h = H()
    q1, q2, q3, q4, q5 = log_qubit[0], log_qubit[1], log_qubit[2], log_qubit[3], log_qubit[4]
    gates = [h(q2), h(q3), h(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)

    CCZ(circuit, q1, q3, q4)

    #svertka(circuit, [q1, q3], 2)
    #main_operation(circuit, [q1, q3], q4, 2)
    #razvertka(circuit, [q1, q3], q4, 2)
    '''
    cur_ccx = make_CCX_qutrits([q1, q2, q3])
    circuit.append([cur_ccx], strategy=InsertStrategy.INLINE)
    '''
    gates = [x(q3), x(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)

    CCZ(circuit, q4, q3, q1)

    #svertka(circuit, [q4, q3], 2)
    #main_operation(circuit, [q4, q3], q1, 2)
    #razvertka(circuit, [q4, q3], q1, 2)
    '''
    cur_ccx = make_CCX_qutrits([q4, q3, q1])
    circuit.append([cur_ccx], strategy=InsertStrategy.INLINE)
    '''

    gates = [x(q3), x(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)

    CX(circuit, q1, q5)
    #circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    CX(circuit, q2, q5)
    #circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    CX(circuit, q2, q1)
    #circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    CX(circuit, q4, q1)
    #circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    CX(circuit, q3, q5)
    #circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    CZ(circuit, q4, q5)
    #circuit.append([cur_cz], strategy=InsertStrategy.INLINE)

def decoding_qubit(circuit, log_qubit):
    x = X1()
    x_conj = X1_conj()

    h = H()
    q1, q2, q3, q4, q5 = log_qubit[0], log_qubit[1], log_qubit[2], log_qubit[3], log_qubit[4]

    CZ(circuit, q4, q5)
    #circuit.append([cur_cz], strategy=InsertStrategy.INLINE)

    CX(circuit, q3, q5)
    #circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    CX(circuit, q4, q1)
    #circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    CX(circuit, q2, q1)
    #circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    CX(circuit, q2, q5)
    #circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    CX(circuit, q1, q5)
    #circuit.append([cur_cx], strategy=InsertStrategy.INLINE)

    gates = [x(q3), x(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)

    CCZ(circuit, q1, q3, q4)

    #svertka(circuit, [q1, q3], 2)
    #main_operation(circuit, [q1, q3], q4, 2)
    #razvertka(circuit, [q1, q3], q4, 2)
    '''

    cur_ccx = make_CCX_qutrits([q1, q3, q4])
    circuit.append([cur_ccx], strategy=InsertStrategy.INLINE)
    '''
    gates = [x(q3), x(q4)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)

    CCZ(circuit, q1, q3, q4)
    #svertka(circuit, [q1, q3], 2)
    #main_operation(circuit, [q1, q3], q4, 2)
    #razvertka(circuit, [q1, q3], q4, 2)
    '''
    cur_ccx = make_CCX_qutrits([q1, q3, q4])
    circuit.append([cur_ccx], strategy=InsertStrategy.INLINE)
    '''
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
    z = Z1()
    q1, q2, q3, q4, q5 = lqubits[0], lqubits[1], lqubits[2], lqubits[3], lqubits[4]
    CX(circuit, q1, q2)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CX(circuit, q4, q3)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    gates = [z(q1), z(q2), z(q3), z(q4), z(q5)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    CX(q1, q2)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CX(q4, q3)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)

def ZXXZI(cirquit, qudits, a1):
    CZ(cirquit, qudits[0], a1)
    CX(cirquit, a1, qudits[1])
    CX(cirquit, a1, qudits[2])
    CZ(cirquit, qudits[3], a1)
    cirquit.append([cirq.measure(a1)])

def ZXXZI_r(cirquit, qudits, a1):
    CZ(cirquit, qudits[3], a1)
    CX(cirquit, a1, qudits[2])
    CX(cirquit, a1, qudits[1])
    CZ(cirquit, qudits[0], a1)

    cirquit.append([cirq.measure(a1)])

#def stabilizer_mesurements(cirquit, qudits):


'''
def H_l(circuit, lqubits):
    h = H()
    #x = X1()
    q1, q2, q3, q4, q5 = lqubits[0], lqubits[1], lqubits[2], lqubits[3], lqubits[4]
    CX(circuit, q2, q5)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CX(circuit, q3, q5)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CZ(circuit, q5, q4)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CZ(circuit,q5, q1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)

    CX(circuit, q5, q1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    gates = [make_CX_qutrits([q5, q2])]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    gates = [make_CX_qutrits([q5, q3])]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    gates = [make_CX_qutrits([q5, q4])]
    circuit.append(gates, strategy=InsertStrategy.INLINE)

    gates = [h(q5)]
    circuit.append(gates, strategy=InsertStrategy.INLINE)

    gates = [make_CX_qutrits([q5, q4])]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    gates = [make_CX_qutrits([q5, q3])]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    gates = [make_CX_qutrits([q5, q2])]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    gates = [make_CX_qutrits([q5, q1])]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    ''''''

    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    gates = [make_CZ_qutrits([q5, q4])]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    gates = [make_CZ_qutrits([q5, q1])]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    gates = [make_CX_qutrits([q3, q5])]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
    gates = [make_CX_qutrits([q2, q5])]
    circuit.append(gates, strategy=InsertStrategy.INLINE)
'''

x = X1()
x2 = X2()
z = Z1()

x_conj = X1_conj()
x2_conj = X2_conj()
h = H()

sim = cirq.Simulator()
circuit1 = cirq.Circuit()
qutrits1 = []

for i in range(10):
    qutrits1.append(cirq.LineQid(i, dimension=3))

gates1 = [x2(qutrits1[0])]
#circuit1.append(gates1)
gates1 = [x(qutrits1[5])]
#circuit1.append(gates1)
gates1 = [h(qutrits1[0])]
#circuit1.append(gates1)
#gates1 = [cx(qutrits1[0], qutrits1[1])]
#circuit1.append(cx.on(qutrits1[0], qutrits1[1]))
encoding_qubit(circuit1, qutrits1)

gates1 = [x(qutrits1[3])]
circuit1.append(gates1)

gates1 = [x(qutrits1[5])]
#circuit1.append(gates1)


q1 = qutrits1[0]
q2 = qutrits1[1]
q3 = qutrits1[2]
q4 = qutrits1[3]
q5 = qutrits1[4]

ZXXZI(circuit1, [q1, q2, q3, q4, q5], qutrits1[5])
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[5])][0]
print(f'Measured bit: {measured_bit}')

ZXXZI(circuit1, [q2, q3, q4, q5, q1], qutrits1[5])
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[5])][0]
print(f'Measured bit: {measured_bit}')

ZXXZI(circuit1, [q3, q4, q5, q1, q2], qutrits1[5])
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[5])][0]
print(f'Measured bit: {measured_bit}')

ZXXZI(circuit1, [q4, q5, q1, q2, q3], qutrits1[5])
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[5])][0]
print(f'Measured bit: {measured_bit}')



ZXXZI_r(circuit1, [q1, q2, q3, q4, q5], qutrits1[5])
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[5])][0]
print(f'Measured bit: {measured_bit}')

ZXXZI_r(circuit1, [q2, q3, q4, q5, q1], qutrits1[5])
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[5])][0]
print(f'Measured bit: {measured_bit}')

ZXXZI_r(circuit1, [q3, q4, q5, q1, q2], qutrits1[5])
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[5])][0]
print(f'Measured bit: {measured_bit}')

ZXXZI_r(circuit1, [q4, q5, q1, q2, q3], qutrits1[5])
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[5])][0]
print(f'Measured bit: {measured_bit}')






#H_l(circuit1, qutrits1)
#make_CXR(circuit1, [qutrits1[0], qutrits1[1]])
#svertka(circuit1, qutrits1[0:-1], 2)
#main_operation(circuit1, qutrits1[0:-1], qutrits1[-1], 2)
#razvertka(circuit1, qutrits1[0:-1], qutrits1[-1], 2)
#u_1i1 = eie()

#circuit1.append((u_1i1(qutrits1[1]).controlled_by(qutrits1[0])))
#decoding_qubit(circuit1, qutrits1)0,

#circuit1.append([cirq.measure(qutrits1[0])])


#print(circuit1)
#print(res1)






