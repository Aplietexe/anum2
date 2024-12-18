import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def solve_upper_triangular(A: Arr, b: Arr) -> Arr:
    """
    Solves Ax = b, where A is an upper triangular invertible matrix."
    Uses recursion, therefore it is slower than an iterative algorithm.
    Does not overwrite A or b.
    O(n^2) complexity.
    """
    n = A.shape[0]

    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")
    if np.any(np.isclose(np.diagonal(A), 0)):
        raise ValueError("Matrix is not invertible")
    if not np.allclose(A, np.triu(A)):
        print("Warning: Matrix is not upper triangular")

    def rec(Ai: Arr, bi: Arr) -> Arr:
        if Ai.shape[0] == 0:
            return np.array([])
        x = np.array([bi[-1] / Ai[-1, -1]])
        x = np.append(rec(Ai[:-1, :-1], bi[:-1] - Ai[:-1, -1] * x[0]), x)
        return x

    return rec(A, b)


if __name__ == "__main__":
    from tests.systems.triangular import test_sup_solve

    t = test_sup_solve(solve_upper_triangular)
    print(t)
