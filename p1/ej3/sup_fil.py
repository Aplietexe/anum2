import numpy as np
from numpy.typing import NDArray
from tests import test_sup_solve

Arr = NDArray[np.float64]


def sol_trsupfil(A: Arr, b: Arr) -> Arr:
    """
    Solves Ax = b, where A is an upper triangular invertible matrix.
    Does not overwrite A or b.
    """
    n = A.shape[0]

    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")
    if not np.allclose(A, np.triu(A)):
        raise ValueError("Matrix is not upper triangular")
    if np.any(np.isclose(np.diagonal(A), 0)):
        raise ValueError("Matrix is not invertible")

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - x[i + 1 :] @ A[i, i + 1 :]) / A[i, i]
    return x


if __name__ == "__main__":
    t = test_sup_solve(sol_trsupfil)
    print(t)
