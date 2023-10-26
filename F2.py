import numpy as np

def m(a,b, c, d, e):
   return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)

def k(a,b):
   return np.kron(a,b)

def projection(e, v):
    return np.dot(e, v) / np.dot(e, e) * e



def find_eigen_vectors(matrix):
    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    ev = []
    for i in range(len(eigen_vectors)):
        if eigen_values[i] == float(1):
            ev.append(eigen_vectors[i])


    return ev




z = np.array([1,0,0]).T
e = np.array([0,1,0]).T
f = np.array([0,0,1]).T

zl = (1/8**0.5) * (m(z,z,z,z,z) - m(e,z,e,e,e) - m(z,e,z, e, e)+ m(e,e,e,z,z) + m(e,z,z,e,z)+ m(z,z,e,z,e) + m(e,e,z,z,e) + m(z,e,e,e,z))
z,e = e,z
el = (1/8**0.5) * (m(z,z,z,z,z) - m(e,z,e,e,e) + m(z,e,z, e, e)- m(e,e,e,z,z) + m(e,z,z,e,z)+ m(z,z,e,z,e) - m(e,e,z,z,e) - m(z,e,e,e,z))
z,e = z,e
fl = m(f,f,f,f,f)


X1 = np.array([[0,1,0], [1,0,0], [0,0,1]])
Y1 = np.array([[0,complex(0, -1),0],[complex(0, 1), 0,0], [0,0,1]])
Z1 = np.array([[1, 0,0], [0, -1,0], [0,0,1]])
X2 = np.array([[0,0,1], [0,1,0], [1,0,0]])
Y2 = np.array([[0,0,complex(0, -1)],[0, 1,0], [complex(0, 1),0,0]])
Z2 = np.array([[1, 0,0], [0, 1,0], [0,0,-1]])
I = np.array([[1,0,0 ], [0,1, 0], [0,0,1]])


A = [['1', e], ['0', z], ['2', f]]

slog = np.array([1])
B = []
for v1 in A:
    for v2 in A:
        for v3 in A:
            for v4 in A:
                for v5 in A:
                    slog = m(v1[1], v2[1], v3[1], v4[1], v5[1])
                    s = v1[0] +v2[0] +v3[0] +v4[0] +v5[0]
                    if np.dot(slog, zl) == 0 and np.dot(slog, el) == 0:
                        B.append([s, slog])

A = [('x1', X1),('z1', Z1),('i', I), ('y1', Y1)]
i = 0
for m1 in A:
  for m2 in A:
    for m3 in A:
      for m4 in A:
        for m5 in A:
          s = m(m1[1], m2[1], m3[1], m4[1], m5[1])

          if np.dot((el - np.matmul(s, zl)), (el - np.matmul(s, zl))) == float(0) and np.dot((zl - np.matmul(s, el)), (zl - np.matmul(s, el))) == float(0) and np.dot((fl - np.matmul(s, fl)), (fl - np.matmul(s, fl))) == float(0):
              i += 1
              #print(m1[0],m2[0],m3[0],m4[0],m5[0])
              #print(i)





S1 = m(X1,Z1,Z1,X1,I)
S2 = m(I,X1,Z1,Z1,X1)
S3 = m(X1,I,X1,Z1,Z1)
S4 = m(Z1,X1,I,X1,Z1)
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
cx = np.array([[1,0,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0,0],
               [0,0,0,0,1,0,0,0,0],
               [0,0,0,1,0,0,0,0,0],
               [0,0,0,0,0,1,0,0,0],
               [0,0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,0,0,1]])

cx_r = np.array([[1,0,0,0,0,0,0,0,0],
               [0,0,0,0,1,0,0,0,0],
               [0,0,1,0,0,0,0,0,0],
               [0,0,0,1,0,0,0,0,0],
               [0,1,0,0,0,0,0,0,0],
               [0,0,0,0,0,1,0,0,0],
               [0,0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,0,0,1]])





matrix = m(Z1, I, I, Z1, I) @ m(X1, X1, X1, X1, X1) # Пример матрицы 3^5 на 3^5
matrix1 = np.kron(np.kron(np.kron(cx, I), I), I) @ np.kron(np.kron(np.kron(I, I), cx_r), I) @ m(Z1, Z1, Z1, Z1, Z1) @ np.kron(np.kron(np.kron(cx, I), I), I) @ np.kron(np.kron(np.kron(I, I), cx_r), I)
eigen_vectors = find_eigen_vectors(matrix)
#print(zl)

print(len(eigen_vectors[0]))
evs = []
for i in range(len(eigen_vectors)):
    if np.dot(zl, eigen_vectors[i]) == 0 and np.dot(el, eigen_vectors[i]) == 0 and np.dot(matrix1 @ eigen_vectors[i] - eigen_vectors[i], matrix1 @ eigen_vectors[i] - eigen_vectors[i]) == float(0):

        evs.append(eigen_vectors[i])
#print(evs)
for i in range(4):
    v = evs[i]
    for ee in B:
        eee = ee[1]
        es = ee[0]
        pr = projection(eee, v)
        k = (np.dot(pr, pr) / np.dot(eee,eee)) ** 0.5
        if k != float(0):
            print(k,' ', es)
        v = v - pr
    print(np.dot(v,v))
    #print(v)






'''    
for i, eigen_vector in enumerate(eigen_vectors.T):
    print(f"Собственный вектор {i + 1}: {eigen_vector}")
'''
#print(np.dot(zl, fl))



#print(cx_r @ np.kron(z, e) - np.kron(e, e))
#print(hl @ psi - (zl + el) / 2**0.5)
#print(z)