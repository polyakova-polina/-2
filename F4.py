import numpy as np

def m(a,b, c, d, e):
   return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)

def k(a,b):
   return np.kron(a,b)

z = np.array([1,0,0])
e = np.array([0,1,0])

zl = (1/8**0.5) * (m(z,z,z,z,z) - m(e,z,e,e,e) - m(z,e,z, e, e)+ m(e,e,e,z,z) + m(e,z,z,e,z)+ m(z,z,e,z,e) + m(e,e,z,z,e) + m(z,e,e,e,z))
z,e = e,z
el = (1/8**0.5) * (m(z,z,z,z,z) - m(e,z,e,e,e) + m(z,e,z, e, e)- m(e,e,e,z,z) + m(e,z,z,e,z)+ m(z,z,e,z,e) - m(e,e,z,z,e) - m(z,e,e,e,z))
z,e = z,e

X1 = np.array([[0,1,0], [1,0,0], [0,0,1]])
Y1 = np.array([[0,complex(0, -1),0],[complex(0, 1), 0,0], [0,0,1]])
Z1 = np.array([[1, 0,0], [0, -1,0], [0,0,1]])

I = np.array([[1,0,0 ], [0,1, 0], [0,0,1]])






A = [('x', X1),('z', Z1),('i', I)]
i = 0

for m1 in A:
  for m2 in A:
    for m3 in A:
      for m4 in A:
        for m5 in A:
          s = m(m1[1], m2[1], m3[1], m4[1], m5[1])

          if np.dot((zl - s @ zl), (zl - s @ zl)) <= 0.001 and np.dot((el - s @ el), (el - s @ el)) <= 0.001:
              i += 1
              print(m1[0],m2[0],m3[0],m4[0],m5[0])
              print(i)





S1 = m(X1,Z1,Z1,X1,I)
S2 = m(I,X1,Z1,Z1,X1)
S3 = m(X1,I,X1,Z1,Z1)
S4 = m(Z1,X1,I,X1,Z1)
psi =zl
'''


hl = m(I,I,I,I,I)

hl[0][0] = 1 / 2 ** 0.5

hl[0][1] = 1 / 2 ** 0.5
hl[1][0] = 1 / 2 ** 0.5
hl[1][1] = -1 / 2 ** 0.5
error = m(X1,I,I,I,I)

psi_er = np.matmul(error, psi)

bra_pe = np.conj(psi_er)
'''
#print(np.dot(zl - S1 @ zl, zl - S1 @ zl))

#print(np.dot(el, fl))
#print(hl @ psi - (zl + el) / 2**0.5)
#print(zl)'''

