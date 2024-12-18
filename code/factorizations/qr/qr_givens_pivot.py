import numpy as np
from numpy.typing import NDArray

from .givens_rotation import givens_2d

Arr = NDArray[np.float64]


def qr(A: Arr, eps: float = 1e-8) -> tuple[Arr, Arr, Arr]:
    """
    Performs QR decomposition on A using Givens rotations with pivoting.
    Overwrites the upper triangular part of A with R.
    O(2n^3) complexity.
    """
    m, n = A.shape
    p = min(m - 1, n)
    Q = np.eye(m)
    P = np.eye(n)
    c = np.sum(A**2, axis=0)

    for j in range(p):
        l = np.argmax(c[j:]) + j
        if c[l] <= eps:
            break
        A[:, [j, l]] = A[:, [l, j]]
        P[:, [j, l]] = P[:, [l, j]]
        c[[j, l]] = c[[l, j]]

        for i in range(j + 1, m):
            if np.isclose(A[i, j], 0):
                continue
            G = givens_2d(A[[j, i], j])
            A[[j, i], j:] = G @ A[[j, i], j:]
            Q[:, [j, i]] = Q[:, [j, i]] @ G.T

        c[j:] -= A[j, j:] ** 2

    if m <= n and A[m - 1, m - 1] < 0:
        A[m - 1, m - 1 :] *= -1
        Q[:, m - 1] *= -1

    return Q, A, P


if __name__ == "__main__":
    from tests.factorizations.qr import test_qr

    t = test_qr(qr, maxn=35, its=30000)
    print(t)
