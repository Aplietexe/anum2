import numpy as np
from numpy.typing import NDArray

from .gaussian_pivoting import gaussian_elimination
from .triangular.upper_row import solve_upper_triangular

Arr = NDArray[np.float64]


def solve_square(A: Arr, b: Arr) -> Arr:
    """
    Solves the system Ax = b using Gaussian elimination with pivoting.
    Overwrites A with U upper triangular and b with the solution.
    Returns the solution.
    """
    A, b = gaussian_elimination(A, b)
    return solve_upper_triangular(A, b)


if __name__ == "__main__":
    from tests.systems.square import test_square

    t = test_square(solve_square)
    print(t)
