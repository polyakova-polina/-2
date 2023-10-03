import numpy as np

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

# Пример использования:
tree = np.array([[0, 1, 1, 0],
                 [1, 0, 0, 0],
                 [1, 0, 0, 0],
                 [0, 0, 0, 0]])

root = find_root_with_min_height(tree)
print(f"Корень с наименьшей высотой: {root}")
