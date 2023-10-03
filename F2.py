import numpy as np

def write_matrix(n, m):
    matrix = []
    for i in range(n):
        matrix.append([])
        for j in range(m):
            print('element', i, j, ':')
            re, im = [float(_) for _ in input().split()]
            elem = complex(re, im)
            matrix[i].append(elem)
    matrix = np.array(matrix)
    return matrix


def gram_schmidt(vectors):
    num_vectors = len(vectors)
    ortho_basis = np.zeros_like(vectors)

    for i in range(num_vectors):
        ortho_basis[i] = vectors[i]
        for j in range(i):
            ortho_basis[i] -= np.dot(vectors[i], ortho_basis[j]) / np.dot(ortho_basis[j], ortho_basis[j]) * ortho_basis[
                j]
        ortho_basis[i] /= np.linalg.norm(ortho_basis[i])

    return ortho_basis

def Px(ONB, vector):
    projection = np.zeros_like(vector)
    projection_old = np.zeros_like(vector)
    for i in range(len(ONB)):
        projection = projection + np.dot(vector, ONB[i]) * ONB[i]
        #projection_old = projection_old +
    return projection

def projector(matrix, vector):
    U, s, V = np.linalg.svd(matrix)
    E = np.diag(s)
    mtrx = np.matmul(U, E)
    basis = mtrx.T[np.nonzero(s)[0]]
    ONB = gram_schmidt(basis)
    onb = list(ONB)
    l = len(ONB[0])
    while(len(onb) < l):
        onb.append(np.zeros_like(ONB[0]))
    onb = np.array(onb)
    return Px(ONB, vector).T, onb @ np.conj(onb.T)

print('n:')
#n = int(input())
n = 4
print('m:')
#m = int(input())
m = 4
#matrix = write_matrix(n, m)
matrix = np.array([[1+0j, 0+1j],
          [0+1j, 1+0j],
          [0+0j, 0+0j]])
matrix = np.array([[1, 0, 0, 0],
          [0, 2, 0, 0],
          [0, 3, 0, 0],
          [0, 0, 0, 1]])
vector = np.array([1, 1, 1, 1])
ans, proj = projector(matrix, vector)
print(ans)
print(proj)