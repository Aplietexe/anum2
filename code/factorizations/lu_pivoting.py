import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def lu(A: Arr) -> tuple[Arr, Arr, Arr, int]:
    """
    Performs LU decomposition with partial pivoting on A.
    Overwrites A with U upper triangular and L lower triangular.
    Returns L, U and P such that PA=LU, and the sign of the permutation.
    O(2n^3/3) complexity.
    """
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("Invalid shape")

    P = np.eye(n)
    p = 1
    for i in range(n - 1):
        pivot = np.argmax(np.abs(A[i:, i])) + i
        if np.isclose(A[pivot, i], 0):
            continue
        if pivot != i:
            A[[i, pivot]] = A[[pivot, i]]
            P[[i, pivot]] = P[[pivot, i]]
            p *= -1
        A[i + 1 :, i] /= A[i, i]
        A[i + 1 :, i + 1 :] -= np.outer(A[i + 1 :, i], A[i, i + 1 :])

    return np.tril(A, -1) + np.eye(n), np.triu(A), P, p


if __name__ == "__main__":
    from tests.factorizations.lu import test_lu

    t = test_lu(lu)
    print(t)
