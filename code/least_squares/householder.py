import numpy as np
from numpy.typing import NDArray

from ..factorizations.qr.qr_householder_pivot import qr
from ..systems.triangular.upper_row import solve_upper_triangular

Arr = NDArray[np.float64]


def least_squares(A: Arr, b: Arr) -> tuple[Arr, np.float64]:
    """
    Solves the least squares problem min ||Ax - b||_2.
    Returns x, ||Ax - b||_2.
    O(2mn^2) complexity.
    """
    m, n = A.shape
    Q, R, P = qr(A)
    q = Q.T @ b
    p = np.where(np.isclose(np.diag(R), 0))[0]
    p = p[0] if p.size else min(m, n)
    y = np.zeros(n)
    y[:p] = solve_upper_triangular(R[:p, :p], q[:p])

    return P @ y, np.linalg.norm(q[p:])


if __name__ == "__main__":
    from tests.least_squares import test_least_squares

    t = test_least_squares(least_squares, rtol=1e-9)
    print(t)
