import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
from scipy import linalg

def m(a,b, c, d, e):
   return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)

def printm(m):
    for i in m:
        print(*[round(j,2) for j in i])

def printv(m):
    s = 0
    for i in m:
        s += 1
        print(s, round(i, 2))

def dag(m):
    return (np.conj(m)).T

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


I = np.array([[1,0,0],[0,1,0],[0,0,1]])

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



u1 = R(0, -np.pi, 1, 2)


u2 = R(np.pi/2, np.pi / 2, 0,1)

u3 = R(0, -np.pi, 0,1)

u4 = R(np.pi/2, -np.pi / 2, 0,1)

u5 = R(0, np.pi, 1, 2)
xx = make_ms_matrix(0, np.pi/2)
xx_c = make_ms_matrix(0, -np.pi/2)
#print(np.conj(xx))
#print()
U2 = np.kron(u1, I) @ np.kron(u2, I) @ xx @ np.kron(u3, u3) @ np.kron(u4, I) @ np.kron(u5, I)
e = np.array([0,1,0])
z = np.array([1,0,0])
f = np.array([0,0,1])

psi = np.kron(np.kron(z,e),  z)



U1 = np.kron(u1, R(np.pi / 2, -np.pi, 0, 2)) @ np.kron(u2, I) @ xx
U1_c = xx_c @ np.kron(dag(u2), I) @ np.kron(dag(u1), dag(R(np.pi / 2, -np.pi, 0, 2)))

#printm(R(np.pi/2 * 0, np.pi, 0,1) @ R(np.pi/2, np.pi / 2, 0,1))

#printv(R(0, np.pi, 1, 2) @ f)
#printm(U1 @ U1)
#printv(np.kron(U1, I) @ np.kron(I, U2) @ np.kron(U1_c,I) @ psi)
#print(ans)
ans1 = R(np.pi * 0, np.pi, 1, 2) @ R(np.pi / 2, np.pi,1,2)
ans2 = R(np.pi * 0, np.pi) @ R(np.pi / 2, np.pi)
#print(ans1 @ ans1 @ ans2)





print()
#printm(R(0, np.pi,0,1) @ R(np.pi / 2, np.pi, 0, 1))

#X = R(0, np.pi, 0, 1)
#Z = R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi, 0, 1)
I = np.array([[1,0],
             [0,1]])
X = np.array([[0,1],
             [1,0]])
Z = np.array([[1,0],
             [0,-1]])
S1 = m(I, X, X, X, Z)




z = np.array([[1,0, 0]]).T
e = np.array([[0,1, 0]]).T
f = np.array([0,0,1])
zl = (1/8**0.5) * (m(z,z,z,z,z) - m(e,z,e,e,e) - m(z,e,z, e, e)+ m(e,e,e,z,z) + m(e,z,z,e,z)+ m(z,z,e,z,e) + m(e,e,z,z,e) + m(z,e,e,e,z))
z,e = e,z
el = (1/8**0.5) * (m(z,z,z,z,z) - m(e,z,e,e,e) + m(z,e,z, e, e)- m(e,e,e,z,z) + m(e,z,z,e,z)+ m(z,z,e,z,e) - m(e,e,z,z,e) - m(z,e,e,e,z))
z,e = z,e
#fl = (1/8**0.5) * (m(f,z,z,z,z) - m(f,z,z,e,z) + m(f,z, e,z, e)- m(f,z,e,e,e) + m(f,e,z,z,e)+ m(f, e,z,e,e) + m(f,e,e,z,z) + m(f,e,e,e,z))
psi = zl
#printv(S1 @ zl - zl)
psi = (z + e) / 2**0.5
print(np.trace(psi @ psi.T))

