import numpy as np

def m(a,b, c, d, e):
   return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)

def k(a,b):
   return np.kron(a,b)

z = np.array([1,0,0])
e = np.array([0,1,0])
f = np.array([0,0,1])

zl = (1/8**0.5) * (m(z,z,z,z,z) - m(e,z,e,e,e) - m(z,e,z, e, e)+ m(e,e,e,z,z) + m(e,z,z,e,z)+ m(z,z,e,z,e) + m(e,e,z,z,e) + m(z,e,e,e,z))
z,e = e,z
el = (1/8**0.5) * (m(z,z,z,z,z) - m(e,z,e,e,e) + m(z,e,z, e, e)- m(e,e,e,z,z) + m(e,z,z,e,z)+ m(z,z,e,z,e) - m(e,e,z,z,e) - m(z,e,e,e,z))
z,e = z,e
fl = (1/8**0.5) * (m(f,z,z,z,z) - m(f,z,z,e,z) + m(f,z, e,z, e)- m(f,z,e,e,e) + m(f,e,z,z,e)+ m(f, e,z,e,e) + m(f,e,e,z,z) + m(f,e,e,e,z))


X1 = np.array([[0,1,0], [1,0,0], [0,0,1]])
Y1 = np.array([[0,complex(0, -1),0],[complex(0, 1), 0,0], [0,0,1]])
Z1 = np.array([[1, 0,0], [0, -1,0], [0,0,1]])
X2 = np.array([[0,0,1], [0,1,0], [1,0,0]])
Y2 = np.array([[0,0,complex(0, -1)],[0, 1,0], [complex(0, 1),0,0]])
Z2 = np.array([[1, 0,0], [0, 1,0], [0,0,-1]])
I = np.array([[1,0,0 ], [0,1, 0], [0,0,1]])

A = [('x1', X1),('z1', Z1),('i', I) ]
i = 0
for m1 in A:
  for m2 in A:
    for m3 in A:
      for m4 in A:
        for m5 in A:
          s = m(m1[1], m2[1], m3[1], m4[1], m5[1])

          if np.dot((zl - np.matmul(s, zl)), (zl - np.matmul(s, zl))) <= float(0.001) and np.dot((el + np.matmul(s, el)), (el + np.matmul(s, el))) <= float(0.001):
              i += 1
              print(m1[0],m2[0],m3[0],m4[0],m5[0])
              print(i)





S1 = m(X1,X1,I,Z1,X1)
S2 = m(X1,Z1,Z1,X1,I)
S3 = m(Z1,Z1,X1,I,X1)
S4 = m(Z1,I,Z1,Z1,Z1)
arr =[]
for E in [X1, Z1, Y1, X2]:
    ers = [m(E, I, I, I, I),
    m(I, E, I, I, I),
    m(I, I, E, I, I),
    m(I, I, I, E, I),
    m(I, I, I, I, E)]
    for e in ers:

        psi = e @ zl
        si1 = np.conj(psi) @ S1 @ psi
        si2 = np.conj(psi) @ S2 @ psi
        si3 = np.conj(psi) @ S3 @ psi
        si4 = np.conj(psi) @ S4 @ psi
        arr. append(1000 * int(round(si1 + 1)) + 100 * int(round(si2 + 1)) + 10 * int(round(si3 + 1)) + int(round(si4 + 1)))
        #print(int(round(si1 + 1)), int(round(si2 + 1)), int(round(si3 + 1)), int(round(si4 + 1)))

arr = sorted(arr)
print(*arr)

psi =zl

hl = m(I,I,I,I,I)

hl[0][0] = 1 / 2 ** 0.5

hl[0][1] = 1 / 2 ** 0.5
hl[1][0] = 1 / 2 ** 0.5
hl[1][1] = -1 / 2 ** 0.5
error = m(X1,I,I,I,I)

psi_er = np.matmul(error, psi)

bra_pe = np.conj(psi_er)
'''
print(zl - np.matmul(S1, zl))
print(el - np.matmul(S1, el))
print(fl - np.matmul(S1, fl))

print(zl - np.matmul(S2, zl))
print(zl - np.matmul(S3, zl))
print(zl - np.matmul(S4, zl))
'''
#print(Z1 @ X1 + X1 @ Z1)
#print(hl @ psi - (zl + el) / 2**0.5)
#print(zl)'''