import numpy as np

def write_matrix(n):
    matrix = []
    for i in range(n):
        matrix.append([])
        for j in range(n):
            print('element', i, j, ':')
            re, im = [float(_) for _ in input().split()]
            elem = complex(re, im)
            matrix[i].append(elem)
    matrix = np.array(matrix)
    return matrix

def is_pos_def(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)

def is_er(matrix):
    er_matrix = np.conj(matrix).T
    return np.all((er_matrix - matrix) == 0)

def is_den_matrix(matrix):
    if is_pos_def(matrix) and is_er(matrix) and np.trace(matrix) == 1:
        return True
    return False


print('n:')
n = int(input())
matrix = write_matrix(n)
print(is_den_matrix(matrix))




