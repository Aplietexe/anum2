import numpy as np
from numpy.typing import NDArray
from tests import test_sup_solve

Arr = NDArray[np.float64]


def sol_trsupcol_rec(A: Arr, b: Arr) -> Arr:
    """
    Solves Ax = b, where A is an upper triangular invertible matrix."
    Uses recursion, therefore it is slower than an iterative algorithm.
    Does not overwrite A or b.
    """
    n = A.shape[0]

    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")
    if not np.allclose(A, np.triu(A)):
        raise ValueError("Matrix is not upper triangular")
    if np.any(np.isclose(np.diagonal(A), 0)):
        raise ValueError("Matrix is not invertible")

    def rec(Ai: Arr, bi: Arr) -> Arr:
        if Ai.shape[0] == 0:
            return np.array([])
        x = np.array([bi[-1] / Ai[-1, -1]])
        x = np.append(rec(Ai[:-1, :-1], bi[:-1] - Ai[:-1, -1] * x[0]), x)
        return x

    return rec(A, b)


def sol_trsupcol(A: Arr, b: Arr) -> Arr:
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

    x = b.copy()
    for i in range(n - 1, -1, -1):
        x[i] /= A[i, i]
        x[:i] -= A[:i, i] * x[i]
    return x


if __name__ == "__main__":
    t = test_sup_solve(sol_trsupcol_rec)
    print(t)
    t = test_sup_solve(sol_trsupcol)
    print(t)
