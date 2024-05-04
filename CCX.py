import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
import scipy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random

def printm(m):
    print('np.array([', end='')
    for line in m:
        print('[', end='')
        for i in range(len(line)-1):
            print(line[i],',', end='')
        print(line[i], end='')
        print('],')
    print('])')


def comp_m(m):
    real = (m + np.conj(m)) / 2
    im = (m - np.conj(m)) / 2 * -1j
    for i in range(len(real)):
        for j in range(len(real)):
            real[i][j] = np.round(real[i][j],2)
    for i in range(len(im)):
        for j in range(len(im)):
            im[i][j] = np.round(im[i][j],2)
    return real + 1j * im

def dag(matrix):
    return np.conj(matrix.T)

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


I = np.eye(3)

u1 = np.kron(R(0, -np.pi, 1, 2), I)
u2 = np.kron(R(np.pi / 2, np.pi / 2, 0, 1), I)
u3 = np.kron(R(0, -np.pi, 0, 1), R(0, -np.pi, 0, 1))
u4 = np.kron(R(np.pi / 2, -np.pi / 2, 0, 1), I)
u5 = np.kron(R(0, np.pi, 1, 2), I)
xx01 = make_ms_matrix(3, 0, np.pi / 2,0,1,0,1)
cx = u1 @ u2 @ xx01 @ u3 @ u4 @ u5

u1 = np.kron(R(np.pi/2, np.pi / 2, 0, 1), I)
u11 = np.kron(R(np.pi/2, -np.pi / 2, 0, 1), I)
u3 = np.kron(R(0, -np.pi, 0, 1), R(0, -np.pi, 0, 1))
xx01_ = make_ms_matrix(3, 0, np.pi / 2,0,1,0,1)

c2x = u1 @ xx01_ @ u3 @ u11

uu1 = np.kron(I, R(np.pi/2, np.pi / 2, 0, 2))
uu11 = np.kron(I, R(np.pi/2, -np.pi / 2, 0, 2))
uu3 = np.kron(R(0, -np.pi, 0, 1), R(0, -np.pi, 0, 2))
xxu01_ = make_ms_matrix(3, 0, np.pi / 2,0,1,0,2)

x1c = uu1 @ xxu01_ @ uu3 @ uu11

uuu1 = np.kron(R(np.pi/2, np.pi / 2, 0, 2), I)
uuu11 = np.kron(R(np.pi/2, -np.pi / 2, 0, 2), I)
uuu3 = np.kron(R(0, -np.pi, 0, 2), R(0, -np.pi, 0, 1))
xxuu01_ = make_ms_matrix(3, 0, np.pi / 2,0,2,0,1)

c1x = uuu1 @ xxuu01_ @ uuu3 @ uuu11


u1_ = np.kron(R(0, -np.pi, 1, 2), R(np.pi / 2, -np.pi, 0, 2))
u2_ = np.kron(R(np.pi / 2, np.pi / 2, 0, 1), I)
U1 = u1_ @ u2_ @ xx01

x12 = R(0, np.pi, 1, 2)
x02 = R(0, np.pi, 0, 2)

ccx = np.kron(U1, I) @ np.kron(I, cx) @ np.kron(dag(U1), I)
mod = np.kron(np.kron(R(0,np.pi,0,1) @ R(0,np.pi,1,2),R(0,np.pi,0,1) @ R(0,np.pi,1,2)), I)
''' 
swap12 = np.kron(np.array([
 [1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 1]]), I)
 '''

swap12 = np.kron(x1c@c1x@x1c@np.kron(x12,x12)@x1c@c1x@x1c@np.kron(x12,x12)@np.kron(x02,x02)@x1c@c1x@x1c@np.kron(x02,x02),I)

print(np.trace(abs(comp_m(ccx @ swap12 @ mod @ ccx @ (np.kron(I,c2x)) @ mod @ mod @ swap12))))






