import numpy as np
from numpy.typing import NDArray

from .factorizations.lu_pivoting import lu
from .square.triangular.lower_row import solve_lower_triangular
from .square.triangular.upper_row import solve_upper_triangular

Arr = NDArray[np.float64]


def inv(A: Arr) -> Arr | None:
    """
    Given a square matrix A, computes its inverse if it exists.
    Overwrites A with the inverse and returns it.
    Returns None if A is not invertible.
    """
    n = A.shape[0]
    L, U, P, _ = lu(A)
    if np.isclose(np.prod(np.diag(U)), 0):
        return None

    for i in range(n):
        y = solve_lower_triangular(L, P[:, i])
        A[:, i] = solve_upper_triangular(U, y)

    return A


if __name__ == "__main__":
    from tests.inv import test_inv

    t = test_inv(inv)
    print(t)
