import numpy as np
'''
a = np.array([[0, 1], [1, 0]])
c = np.array([[1, 0], [0, -1]])
b = np.array([[0,complex(0, -1)],
   [complex(0, 1), 0]])

a1 = np.kron(a, a)
#a2 = np.kron(a1, a)
#a3 = np.kron(a2, a)
print(a)
v1 = np.array([1, 0])
b1 = np.kron(b, b)
c1 = np.kron(c,c)
#b2 = np.kron(b1, b)
#b3 = np.kron(b2, b)
print(np.matmul(b1, c1) - np.matmul(c1, b1))
print()
print(np.linalg.eig(a1))
print(np.linalg.eig(b1))
print(np.linalg.eig(c1))
print(2 ** -0.5)

'''

def m(a,b, c, d, e):
   return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)

z = np.array([1,0])
e = np.array([0,1])

zl = 0.25 * (m(z,z,z,z,z) + m(e,z,z,e,z) + m(z,e,z, z, e)+ m(e,z,e,z,z) + m(z,e,z,e,z)- m(e,e,z,e,e) - m(z,z,e,e,z) - m(e,e,z,z,z) - m(e,e,e,z,e) - m(z,z,z,e,e)- m(e,e,e,e,z) - m(z, e,e,e,e) - m(e,z,z,z,e) - m(z,e,e,z,z) - m(e,z,e,e,e) + m(z,z,e,z,e))
z,e = e,z
el = 0.25 * (m(z,z,z,z,z) + m(e,z,z,e,z) + m(z,e,z, z, e)+ m(e,z,e,z,z)- m(e,e,z,e,e) - m(z,z,e,e,z) - m(e,e,z,z,z) - m(e,e,e,z,e) - m(z,z,z,e,e)- m(e,e,e,e,z) - m(z, e,e,e,e) - m(e,z,z,z,e) - m(z,e,e,z,z) - m(e,z,e,e,e) + m(z,z,e,z,e))
z,e = e,z

X = np.array([[0,1], [1,0]])
Y = np.array([[0,complex(0, -1)],
   [complex(0, 1), 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.array([[1,0], [0,1]])
S1 = m(X,Z,Z,X,I)
S2 = m(I,X,Z,Z,X)
S3 = m(X,I,X,Z,Z)
S4 = m(Z,X,I,X,Z)
psi = zl

error = m(I,I,I,Z,I)

psi_er = np.matmul(error, psi)

bra_pe = np.conj(psi_er)

'''
print(np.matmul(np.matmul(bra_pe, S1), psi_er))
print(np.matmul(np.matmul(bra_pe, S2), psi_er))
print(np.matmul(np.matmul(bra_pe, S3), psi_er))
print(np.matmul(np.matmul(bra_pe, S4), psi_er))
print(zl)
'''

psi = (np.kron(z, e) + np.kron(e, z)) / 2**0.5

u = np.kron(X, I)

print(u@psi)

print(X@X@(np.conj(X)) - X)

