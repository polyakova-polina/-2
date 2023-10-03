import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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


