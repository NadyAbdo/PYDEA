import numpy as np

X = [
    [2, 4, 3],
    [5, 1, 6],
    [8, 7, 9]
]

Y = [
    [6, 1, 5, 2],
    [4, 7, 2, 9],
    [5, 4, 8, 1]
]

result = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

# Perform matrix multiplication using nested loops
for i in range(len(X)):
    for j in range(len(Y[0])):
        for k in range(len(Y)):
            result[i][j] += X[i][k] * Y[k][j]

for row in result:
    print(row)

result = np.dot(X, Y)

print(result)