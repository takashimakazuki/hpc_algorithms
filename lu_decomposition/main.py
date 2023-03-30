import numpy as np


def lu(X):
    n = X.shape[0]
    L, U = np.zeros([n, n]), np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i == 0:
                if j == 0:
                    L[i, j] = 1
                U[i, j] = X[i, j]
            elif j == 0:
                L[i, j] = X[i, j] / U[0, 0]
            elif i == j:
                L[i, j] = 1
                U[i, j] = X[i,j] - np.dot(L[i,:j],U[:i,j])
            elif i > j:
                L[i, j] = (X[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]
            elif j > i:
                U[i, j] = X[i, j] - np.dot(L[i, :i], U[:i, j])
    return L, U


def main():
    A = np.array([[2., 1.], [1., 2.]])
    L, U = lu(A)
    print(L)
    print(U)

main()
