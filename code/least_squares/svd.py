import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def least_squares(A: Arr, b: Arr, eps: float = 1e-10) -> tuple[Arr, np.float64]:
    """
    Solves the least squares problem min ||Ax - b||_2 using the SVD.
    Returns x, ||Ax - b||_2.
    O(2mn^2) complexity.
    """
    m, n = A.shape
    U, S, Vt = np.linalg.svd(A)
    b_tilde = U.T @ b
    r = np.where(np.abs(S) < eps)[0]
    r = r[0] if r.size else min(m, n)
    y = b_tilde[:r] / S[:r]
    x_tilde = np.zeros(n)
    x_tilde[:r] = y
    x = Vt.T @ x_tilde
    return x, np.linalg.norm(b - A @ x, 2)


if __name__ == "__main__":
    from tests.least_squares import test_least_squares

    t = test_least_squares(least_squares, rtol=1e-8, min_norm=True)
    print(t)
