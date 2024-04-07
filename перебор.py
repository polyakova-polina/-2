import numpy as np
import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
import scipy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random
m1 = np.eye(10)
m2 = np.eye(10)
m1[1][1], m1[1][8] =  m1[1][8], m1[1][1]
m1[8][8], m1[8][1] =  m1[8][1], m1[8][8]
m2[2][2], m2[2][7] =  m2[2][7], m2[2][2]
m2[7][7], m2[7][2] =  m2[7][2], m2[7][7]
#print(m1@m2)



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
        return np.eye(9)
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
        return np.eye(9)
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


target = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, -1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, -1j, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 1]]
) * -1j


I = np.eye(3)
def mat(x):
    u1 = np.kron(R(x[0], x[1], 0, 2), I)
    u2 = np.kron(R(x[2], x[3], 0, 2), I)
    #xx = make_ms_matrix(3, x[8], x[9],x[10],x[11],x[12],x[13])
    u3 = np.kron(I, R(x[4], x[5], 0, 2))
    u4 = np.kron(I, R(x[6], x[7], 0, 2))
    mat = u1 @ u2 @ u3 @ u4
    return (abs(mat - target).sum())

def matt(x):
    u1 = np.kron(R(x[0], x[1], 0, 2), I)
    u2 = np.kron(R(x[2], x[3], 1, 2), I)
    #xx = make_ms_matrix(3, x[8], x[9],x[10],x[11],x[12],x[13])
    u3 = np.kron(I, R(x[4], x[5], 0, 2))
    u4 = np.kron(I, R(x[6], x[7], 1, 2))
    mat = u1 @ u2 @ u3 @ u4
    return mat

def comp_m(m, par):
    real = (m + np.conj(m)) / 2
    im = (m - np.conj(m)) / 2 * -1j
    for i in range(len(real)):
        for j in range(len(real)):
            real[i][j] = np.round(real[i][j], par)
    for i in range(len(im)):
        for j in range(len(im)):
            im[i][j] = np.round(im[i][j], par)
    return real + 1j * im


def dag(matrix):
    return np.conj(matrix.T)

for i in range(10 ** 0):
    if i % 1000 == 0:
        print(i)
    a = mat(np.array([0, 3, 1.5, 3, 0, 3, 1.5, 3,]))
#res = scipy.optimize.minimize(mat, np.array([0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0]), method='powell')
'''
print(res.x)
print()
#print(abs(comp_m(matt(res.x), 2)[0]))
print()
print(comp_m(matt(res.x), 2)[1])
print()
print(comp_m(matt(res.x), 2)[2])
print()
ans = matt(res.x)
print((abs(ans @ dag(ans) - np.eye(9))).sum())

print((abs(ans @ dag(ans) - np.eye(9))).sum())
'''
ms1 = make_ms_matrix(3,0, np.pi,0,1,0,2)
ms2 = make_ms_matrix(3,0, np.pi,0,1,0,1)
ms24 = make_ms_matrix(3,0, np.pi,1,2,0,1)
ms4 = make_ms_matrix(3,0, np.pi,0,2,0,1)
xx1 = make_ms_matrix(3,0, np.pi,1,2,0,1)
xx2 = make_ms_matrix(3,np.pi / 2, np.pi,0,2,1,2)
xx2 = np.eye(9)
#print(comp_m(xx1@xx2,2))

r2 = R(np.pi / 2, np.pi, 0,1)
r1 = R(np.pi / 2, -np.pi, 0,1)
pmm = R(0, np.pi / 2, 1,2) @ R(0, np.pi / 2, 1,2) @ R(0, np.pi, 1,2)

a = np.kron(r1, r2)
b = a @ a
MS = ms2 @ ms4
#print()
#print(comp_m(ms1 @ ms24 @ ms1 @ ms4,2))

mss = []
msss = []
for i in range(2):
    for j in range(1,3):
        for k in range(2):
            for l in range(1, 3):
                msss.append(make_ms_matrix(3,0, np.pi,i,j,k,l))

for i in range(2):
    for j in range(1,3):
        for k in range(2):
            for l in range(1, 3):
                msss.append(make_ms_matrix(3,0, -np.pi,i,j,k,l))
for i in range(2):
    for j in range(1,3):
        for k in range(2):
            for l in range(1, 3):
                msss.append(make_ms_matrix(3,np.pi / 2, -np.pi,i,j,k,l))
for i in range(2):
    for j in range(1,3):
        for k in range(2):
            for l in range(1, 3):
                msss.append(make_ms_matrix(3,np.pi / 2, np.pi,i,j,k,l))

msss = np.array(msss)

for g in msss:
    mss.append(np.kron(I, g))
    mss.append(np.kron(g, I))

msss = []

for i in range(2):
    for j in range(1,3):
        vsp = R(np.pi / 2, -np.pi, i, j)
        msss.append(vsp)
        #msss.append(vsp)
for i in range(2):
    for j in range(1,3):
        vsp = R(0, -np.pi, i, j)
        msss.append(vsp)
        #mss.append(np.kron(vsp, I))
msss.append(r1)
msss.append(r2)

for g in msss:
    mss.append(np.kron(np.kron(I,I), g))
    mss.append(np.kron(np.kron(I,g), I))
    mss.append(np.kron(np.kron(g, I), I))

#print(mss[0])
'''
l = len(mss) **3
sch = 0
for h1 in mss:
    for h2 in mss:
        for h3 in mss:

            sch +=1
            if sch % 1000 == 0:
                print(sch / l)
            if abs((h1 @ h2 @ h3 - target)).sum() < 0.001:
                print('WIN')
'''
p1 = make_ms_matrix(3,np.pi, np.pi / 2,0,1,0,2) @ make_ms_matrix(3,np.pi, np.pi / 2,0,1,0,2)
g1 = R(0, np.pi, 0,1) @ R(0, np.pi, 0,1)
#-j j 1
g2 = R(0, np.pi, 0,1) @ R(0, np.pi, 0,1)
#-1 -1 1
g3 = R(0, np.pi, 1,2) @ R(0, np.pi, 1,2)
#1 -1 -1
g3 = R(0, np.pi, 1,2) @ R(0, np.pi, 1,2)


def m3(g1,g2,g3):
    return(np.kron(g1,np.kron(g2,g3)))

def m2(g1, g2):
    return(np.kron(g1,g2))

u1 = R(0, -np.pi, 1, 2)
u2 = R(np.pi / 2, np.pi / 2, 0, 1)
u6 = R(np.pi / 2, -np.pi, 0, 2)

xxu1 = make_ms_matrix(3,np.pi/2, np.pi,0,1,0,1)
U1 = m2(u1, u6) @ m2(u2, I) @ xxu1

def printm(m):
    print('np.array([', end='')
    for line in m:
        print('[', end='')
        for i in range(len(line)-1):
            print(line[i],',', end='')
        print(line[i], end='')
        print('],')
    print('])')
printm(comp_m(U1,1))
print(np.kron(np.array([[1,0,0]]), np.array([[0,1,0]])))