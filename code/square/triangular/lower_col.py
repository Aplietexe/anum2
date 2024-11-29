import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def solve_lower_triangular(A: Arr, b: Arr) -> Arr:
    """
    Solves Ax = b, where A is a lower triangular invertible matrix.
    Overwrites b with x, does not overwrite A.
    O(n^2) complexity.
    """
    n = A.shape[0]

    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")
    if np.any(np.isclose(np.diagonal(A), 0)):
        raise ValueError("Matrix is not invertible")
    if not np.allclose(A, np.tril(A)):
        print("Warning: Matrix is not lower triangular")

    i = 0
    while np.isclose(b[i], 0) and i < n:
        b[i] = 0
        i += 1
    for i in range(i, n):
        b[i] /= A[i, i]
        b[i + 1 :] -= A[i + 1 :, i] * b[i]
    return b


if __name__ == "__main__":
    from tests.systems.triangular import test_inf_solve

    t = test_inf_solve(solve_lower_triangular)
    print(t)
