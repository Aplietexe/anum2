from math import isclose

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def qr(A: Arr) -> tuple[Arr, Arr]:
    """
    Performs QR decomposition on A using modified Gram-Schmidt.
    O(2mn^2) complexity.
    """
    m, n = A.shape
    p = min(m, n)

    Q = np.zeros((m, p))
    R = np.zeros((p, n))

    for j in range(p):
        R[j, j] = np.linalg.norm(A[:, j])
        Q[:, j] = A[:, j] / R[j, j]
        R[j, j + 1 :] = Q[:, j].T @ A[:, j + 1 :]
        A[:, j + 1 :] -= np.outer(Q[:, j], R[j, j + 1 :])

    return Q, R


if __name__ == "__main__":
    from tests.factorizations.qr import test_reduced_qr

    t = test_reduced_qr(qr, rtol=1e-9, its=30000, under=False)
    print(t)
