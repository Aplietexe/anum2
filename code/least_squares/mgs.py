import numpy as np
from numpy.typing import NDArray

from ..factorizations.qr_mgs import qr
from ..square.triangular.upper_row import solve_upper_triangular

Arr = NDArray[np.float64]


def least_squares(A: Arr, b: Arr) -> tuple[Arr, np.float64]:
    """
    Solves the least squares problem min ||Ax - b||_2.
    Returns x, ||Ax - b||_2.
    O(2mn^2) complexity.
    """
    Q1, R1 = qr(A.copy())
    r, n = R1.shape
    c = Q1.T @ b

    y = np.zeros(n)
    y[:r] = solve_upper_triangular(R1[:, :r], c)

    return y, np.linalg.norm(A @ y - b)


if __name__ == "__main__":
    from tests.least_squares import test_least_squares

    t = test_least_squares(least_squares, rtol=5e-8, its=10000)
    print(t)
