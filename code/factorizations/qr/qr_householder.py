import numpy as np
from numpy.typing import NDArray

from .householder_reflection import householder_reflection

Arr = NDArray[np.float64]


def qr(A: Arr) -> tuple[Arr, Arr]:
    """
    Performs QR decomposition on A.
    Overwrites the upper triangular part of A with R.
    O(2n^3) complexity.
    """
    m, n = A.shape
    p = min(m, n)
    Q = np.eye(m)

    for j in range(p):
        u, rho = householder_reflection(A[j:, j].copy())
        w = rho * u
        A[j:, j:] -= np.outer(w, u.T @ A[j:, j:])
        Q[:, j:] -= np.outer(Q[:, j:] @ w, u)

    return Q, A


if __name__ == "__main__":
    from tests.factorizations.qr import test_qr

    t = test_qr(qr)
    print(t)
