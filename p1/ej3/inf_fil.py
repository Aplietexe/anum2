import numpy as np
from numpy.typing import NDArray
from tests import test_inf_solve

Arr = NDArray[np.float64]


def sol_trinffil(A: Arr, b: Arr) -> Arr:
    """
    Solves Ax = b, where A is a lower triangular invertible matrix.
    Does not overwrite A or b.
    """
    n = A.shape[0]

    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")
    if not np.allclose(A, np.tril(A)):
        raise ValueError("Matrix is not lower triangular")
    if np.any(np.isclose(np.diagonal(A), 0)):
        raise ValueError("Matrix is not invertible")

    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - x[:i] @ A[i, :i]) / A[i, i]
    return x


if __name__ == "__main__":
    t = test_inf_solve(sol_trinffil)
    print(t)
