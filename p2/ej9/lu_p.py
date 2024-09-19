import numpy as np
from numpy.typing import NDArray

from ..ej6.tests import test_lu

Arr = NDArray[np.float64]


def dlup(A: Arr) -> tuple[Arr, Arr, Arr]:
    """
    Performs LU decomposition with partial pivoting on A.
    Overwrites A with U upper triangular and L lower triangular.
    Returns L, U and P such that PA=LU.
    """
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("Invalid shape")

    P = np.eye(n)
    for i in range(n - 1):
        pivot = np.argmax(np.abs(A[i:, i])) + i
        if not np.isclose(A[pivot, i], 0):
            A[[i, pivot]] = A[[pivot, i]]
            P[[i, pivot]] = P[[pivot, i]]
        A[i + 1 :, i] /= A[i, i]
        A[i + 1 :, i + 1 :] -= np.outer(A[i + 1 :, i], A[i, i + 1 :])

    return np.tril(A, -1) + np.eye(n), np.triu(A), P


if __name__ == "__main__":
    t = test_lu(dlup)
    print(t)
