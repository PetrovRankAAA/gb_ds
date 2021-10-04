import numpy as np

np.set_printoptions(precision=2, suppress=True)
matrix = np.array([[1, 2, 0], [0, 0, 5], [3, -4, 2], [1, 6, 5], [0, 1, 0]])
U, S, W = np.linalg.svd(matrix)
# print(U)
# [[ 0.17  0.16 -0.53 -0.8  -0.16]
#  [ 0.39 -0.53  0.61 -0.43  0.03]
#  [-0.14 -0.82 -0.52  0.14  0.07]
#  [ 0.89  0.06 -0.25  0.38 -0.06]
#  [ 0.08  0.11 -0.08 -0.11  0.98]]
# print(S)
#  [8.82 6.14 2.53]
# print(W)
# [[ 0.07  0.72  0.69]
#  [-0.37  0.67 -0.65]
#  [-0.93 -0.21  0.31]]
V = np.transpose(W)
# [[ 0.07 -0.37 -0.93]
#  [ 0.72  0.67 -0.21]
#  [ 0.69 -0.65  0.31]]
D = np.zeros_like(matrix, dtype=float)
D[np.diag_indices(min(matrix.shape))] = S
# print(D)
# [[8.82 0.   0.  ]
#  [0.   6.14 0.  ]
#  [0.   0.   2.53]
#  [0.   0.   0.  ]
#  [0.   0.   0.  ]]
print(np.dot(np.dot(U, D), V.T))
# [[ 1.  2.  0.]
#  [ 0. -0.  5.]
#  [ 3. -4.  2.]
#  [ 1.  6.  5.]
#  [-0.  1. -0.]]

#А) Найти евклидову норму
e_norm = []
for vector in matrix:
    e_norm.append(sum([el**2 for el in vector])**0.5)

x = sum([el**2 for el in e_norm])**0.5
# x = 11.04536

e_norm = np.asarray(e_norm)[np.newaxis]
#[[2.24 5.   5.39 7.87 1.  ]]

matrix_x = np.dot(matrix.T, e_norm.T)
# [[26.27]
#  [31.18]
#  [75.14]]

#евклидова норма равна
matrix_e_norm = max((matrix_x / x))
# [6.8]

# Б)Найти норму Фробениуса

frob_norm = 0
for row in matrix:
    for el in row:
        frob_norm += el**2

# Норма Фробениуса равна:
print(frob_norm**0.5)
# 11.045361017187261

# проверка
print(np.linalg.norm(np.asarray(matrix), ord='fro'))
# 11.045361017187261