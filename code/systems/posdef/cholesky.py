import numpy as np
from numpy.typing import NDArray

from ...factorizations.cholesky_interior import cholesky
from ..triangular.lower_row import solve_lower_triangular
from ..triangular.upper_row import solve_upper_triangular

Arr = NDArray[np.float64]


def solve_positive_definite(A: Arr, b: Arr) -> Arr | None:
    """
    Solves Ax = b, where A is a positive definite matrix.
    Overwrites b with x, does not overwrite A.
    Returns None if A is not positive definite.
    O(n^3/3) complexity.
    """
    U = cholesky(A)
    if U is None:
        return None

    y = solve_lower_triangular(U.T, b)
    x = solve_upper_triangular(U, y)
    return x


if __name__ == "__main__":
    from tests.systems.positive_definite import test_posdef_solve

    t = test_posdef_solve(solve_positive_definite)
    print(t)
