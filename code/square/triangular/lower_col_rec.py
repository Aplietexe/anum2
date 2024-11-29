import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def solve_lower_triangular(A: Arr, b: Arr) -> Arr:
    """
    Solves Ax = b, where A is a lower triangular invertible matrix.
    Uses recursion, therefore it is slower than an iterative algorithm.
    Does not overwrite A or b.
    O(n^2) complexity.
    """
    n = A.shape[0]

    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")
    if np.any(np.isclose(np.diagonal(A), 0)):
        raise ValueError("Matrix is not invertible")
    if not np.allclose(A, np.tril(A)):
        print("Warning: Matrix is not lower triangular")

    def rec(Ai: Arr, bi: Arr) -> Arr:
        if Ai.shape[0] == 0:
            return np.array([])
        x = np.array([bi[0] / Ai[0, 0]])
        x = np.append(x, rec(Ai[1:, 1:], bi[1:] - Ai[1:, 0] * x[0]))
        return x

    return rec(A, b)


if __name__ == "__main__":
    from tests.systems.triangular import test_inf_solve

    t = test_inf_solve(solve_lower_triangular)
    print(t)
