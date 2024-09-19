import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def sol_trinffil(A: Arr, b: Arr) -> Arr:
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
        b[i] -= b[:i] @ A[i, :i]
        b[i] /= A[i, i]
    return b


if __name__ == "__main__":
    from .tests import test_inf_solve

    t = test_inf_solve(sol_trinffil)
    print(t)
