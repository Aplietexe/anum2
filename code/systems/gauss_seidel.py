import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def gauss_seidel(A: Arr, b: Arr, x: Arr, eps: float, maxits: int) -> Arr:
    """
    Solves the system Ax = b using the Gauss-Seidel iteration.
    Overwrites x with the solution.
    Returns the solution.
    """
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,) or x.shape != (n,):
        raise ValueError("Invalid shapes")
    if np.any(np.isclose(np.diag(A), 0)):
        raise ValueError("Diagonal elements must be nonzero")

    b /= np.diag(A)
    A = (np.tril(A, -1) + np.triu(A, 1)) / np.diag(A)[:, None]

    x_new = x.copy()
    for _ in range(maxits):
        for i in range(n):
            x_new[i] = b[i] - A[i] @ x_new
        if np.linalg.norm(x_new - x, np.inf) < eps:
            return x_new
        x = x_new.copy()

    print("Warning: Maximum number of iterations reached")

    return x_new


if __name__ == "__main__":
    from tests.systems.diagonally_dominant import test_diagonally_dominant

    t = test_diagonally_dominant(gauss_seidel)
