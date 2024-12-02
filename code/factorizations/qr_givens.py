import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def givens_2d(x: Arr) -> Arr:
    """
    Returns the Givens rotation matrix R that zeroes the second element of a 2D vector.
    """
    if x.shape != (2,):
        raise ValueError("Invalid shape")
    x1, x2 = x
    if np.isclose(x2, 0):
        return np.array([[1, 0], [0, 1]])

    if np.abs(x2) > np.abs(x1):
        t = -x1 / x2
        s = -np.sign(x2) / np.sqrt(1 + t**2)
        c = s * t
    else:
        t = -x2 / x1
        c = np.sign(x1) / np.sqrt(1 + t**2)
        s = c * t

    return np.array([[c, -s], [s, c]])


def qr(A: Arr) -> tuple[Arr, Arr]:
    """
    Performs QR decomposition on A.
    Overwrites the upper triangular part of A with R.
    O(2n^3) complexity.
    """
    m, n = A.shape
    p = min(m - 1, n)
    Q = np.eye(m)

    for j in range(p):
        for i in range(j + 1, m):
            if np.isclose(A[i, j], 0):
                continue
            G = givens_2d(A[[j, i], j])
            A[[j, i], j:] = G @ A[[j, i], j:]
            Q[:, [j, i]] = Q[:, [j, i]] @ G.T

    if m <= n and A[m - 1, m - 1] < 0:
        A[m - 1, m - 1 :] *= -1
        Q[:, m - 1] *= -1

    return Q, A


if __name__ == "__main__":
    from tests.factorizations.qr import test_qr

    t = test_qr(qr, maxn=35, its=30000)
    print(t)
