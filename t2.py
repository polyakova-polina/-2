import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy

n = 6
target_qutrit = 5
controling_q = [_ for _ in range(0, 5)]
qutrits = []
for i in range(n):
    qutrits.append(cirq.LineQid(i, dimension=3))

controling_qutrits = [qutrits[i] for i in controling_q]
target_qutrit = qutrits[target_qutrit]

def make_CX_qutrits(q):
    return x(q[1]).controlled_by(q[0])

def make_CX_qutrits_conj(q):
    return x_conj(q[1]).controlled_by(q[0])

def make_CZ_qutrits(q):
    return z(q[1]).controlled_by(q[0])


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



def build_spanning_tree(E):
    n = len(E)
    G = nx.Graph()

    for i in range(n):
        G.add_node(i)

    for i in range(n):
        for j in range(i+1, n):
            if E[i][j] == 1:
                G.add_edge(i, j)

    T = nx.minimum_spanning_tree(G)

    T_matrix = np.zeros_like(E)
    for u, v in T.edges():
        T_matrix[u][v] = 1
        T_matrix[v][u] = 1

    return T_matrix

import numpy as np


def find_root_with_min_height(tree_matrix):
    transposed_tree_matrix = tree_matrix.transpose()
    min_height = float('inf')
    potential_root = -1

    for node in range(len(transposed_tree_matrix)):
        current_height = bfs(node, transposed_tree_matrix)
        if current_height < min_height:
            min_height = current_height
            potential_root = node

    return potential_root


def bfs(start_node, tree_matrix):
    stack = [(start_node, 0)]
    max_height = 0

    while len(stack) > 1:

        node, height = stack.pop()
        max_height = max(max_height, height)

        for neighbor in range(len(tree_matrix[node])):
            if tree_matrix[node][neighbor] == 1:
                stack.append((neighbor, height + 1))

    return max_height

class node:
    def __init__(self, i, parent, daughters, main):
        self.m = main
        if not main:
            self.p = parent
        else:
            self.parent = None
        self.d = daughters

E = np.array([[0, 1, 1, 0],
              [1, 0, 1, 0],
              [1, 1, 0, 1],
              [0, 0, 1, 0]])

Tree = build_spanning_tree(E)
root = find_root_with_min_height(Tree)


def make_hanged_tree(Tree, root):
    index = root
    d0 = []
    for i in range(len(Tree)):
        if Tree[root][i] == 1:
            d0.append(i)
    tree = [node(index, None, )]


circuit = cirq.Circuit()
x = X1()
x2 = X2()
z = Z1()

x_conj = X1_conj()
x2_conj = X2_conj()
#z_conj = Z1()
#cx = x.controlled_by(qutrits[0])
k = len(controling_qutrits)

def svertka(controling_qutrits, k):
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

def main_operation(controling_qutrits, k):
    curl_x2 = x2(target_qutrit)
    curl_cx = make_CX_qutrits([controling_qutrits[k // 2 - 1], target_qutrit])
    curl_x = x(target_qutrit)
    curr_cz = make_CZ_qutrits([controling_qutrits[k - 1], target_qutrit])
    circuit.append([curl_x2], strategy=InsertStrategy.INLINE)
    circuit.append([curl_cx], strategy=InsertStrategy.INLINE)
    circuit.append([curl_x], strategy=InsertStrategy.INLINE)
    circuit.append([curr_cz], strategy=InsertStrategy.INLINE)

def razvertka(controling_qutrits, k):
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


#ТЕСТ НАЧАЛО

class H(cirq.Gate):

    def _qid_shape_(self):

        return (3,)


    def _unitary_(self):

        return np.conj(1/(2**0.5)*np.array([[1, 1, 0],
                         [1, -1, 0],
                         [0, 0, 2**0.5]]))

    def _circuit_diagram_info_(self, args):
        return '[+1]'

gates = [x(qutrits[i]) for i in range(n)]
#circuit.append(gates)
h = H()
#gates.append(h(qutrits[len(qutrits)-1]))
#gates = [x2_conj(qutrits[i]) for i in range(1)]


#gates = []

circuit.append(gates)

#sim = cirq.Simulator()

#res1 = sim.simulate(circuit)



#print(res1)
#ТЕСТ КОНЕЦ

svertka(controling_qutrits, k)
main_operation(controling_qutrits, k)
razvertka(controling_qutrits, k)


#print(circuit)

sim = cirq.Simulator()

res1 = sim.simulate(circuit)



print(res1)