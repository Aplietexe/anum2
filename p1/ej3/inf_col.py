import numpy as np
from numpy.typing import NDArray
from tests import test_inf_solve

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


if __name__ == "__main__":
    t = test_inf_solve(sol_trinfcol_rec)
    print(t)
