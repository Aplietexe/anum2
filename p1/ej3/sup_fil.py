import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def sol_trsupfil(A: Arr, b: Arr) -> Arr:
    """
    Solves Ax = b, where A is an upper triangular invertible matrix.
    Overwrites b with x, does not overwrite A.
    """
    n = A.shape[0]

    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")
    if not np.allclose(A, np.triu(A)):
        raise ValueError("Matrix is not upper triangular")
    if np.any(np.isclose(np.diagonal(A), 0)):
        raise ValueError("Matrix is not invertible")

    i = n - 1
    while np.isclose(b[i], 0) and i >= 0:
        b[i] = 0
        i -= 1
    for i in range(i, -1, -1):
        b[i] -= b[i + 1 :] @ A[i, i + 1 :]
        b[i] /= A[i, i]
    return b


if __name__ == "__main__":
    from .tests import test_sup_solve

    t = test_sup_solve(sol_trsupfil)
    print(t)
