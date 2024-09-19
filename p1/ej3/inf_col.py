import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def sol_trinfcol_rec(A: Arr, b: Arr) -> Arr:
    """
    Solves Ax = b, where A is a lower triangular invertible matrix.
    Uses recursion, therefore it is slower than an iterative algorithm.
    Does not overwrite A or b.
    """
    n = A.shape[0]

    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")
    if not np.allclose(A, np.tril(A)):
        raise ValueError("Matrix is not lower triangular")
    if np.any(np.isclose(np.diagonal(A), 0)):
        raise ValueError("Matrix is not invertible")

    def rec(Ai: Arr, bi: Arr) -> Arr:
        if Ai.shape[0] == 0:
            return np.array([])
        x = np.array([bi[0] / Ai[0, 0]])
        x = np.append(x, rec(Ai[1:, 1:], bi[1:] - Ai[1:, 0] * x[0]))
        return x

    return rec(A, b)


def sol_trinfcol(A: Arr, b: Arr) -> Arr:
    """
    Solves Ax = b, where A is a lower triangular invertible matrix.
    Overwrites b with x, does not overwrite A.
    """
    n = A.shape[0]

    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")
    if not np.allclose(A, np.tril(A)):
        raise ValueError("Matrix is not lower triangular")
    if np.any(np.isclose(np.diagonal(A), 0)):
        raise ValueError("Matrix is not invertible")

    i = 0
    while np.isclose(b[i], 0) and i < n:
        b[i] = 0
        i += 1
    for i in range(i, n):
        b[i] /= A[i, i]
        b[i + 1 :] -= A[i + 1 :, i] * b[i]
    return b


if __name__ == "__main__":
    from .tests import test_inf_solve

    t = test_inf_solve(sol_trinfcol_rec)
    print(t)
    t = test_inf_solve(sol_trinfcol)
    print(t)
