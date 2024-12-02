import numpy as np
from numpy.typing import NDArray

from .householder_reflection import householder_reflection

Arr = NDArray[np.float64]


def qr(A: Arr) -> tuple[Arr, Arr, Arr]:
    """
    Performs QR decomposition on A using Householder reflections and pivoting.
    Overwrites the upper triangular part of A with R.
    Returns Q, R, P
    O(2p^2(m+n) + 4pm^2) complexity.
    """
    m, n = A.shape
    p = min(m, n)
    Q = np.eye(m)
    P = np.eye(n)
    c = np.sum(A**2, axis=0)

    for j in range(p):
        l = np.argmax(c[j:]) + j
        if np.isclose(c[l], 0):
            break
        A[:, [j, l]] = A[:, [l, j]]
        P[:, [j, l]] = P[:, [l, j]]
        c[[j, l]] = c[[l, j]]
        u, rho = householder_reflection(A[j:, j].copy())
        w = rho * u
        A[j:, j:] -= np.outer(w, u.T @ A[j:, j:])
        Q[:, j:] -= np.outer(Q[:, j:] @ w, u)
        c[j:] -= A[j, j:] ** 2

    return Q, A, P


if __name__ == "__main__":
    from tests.factorizations.qr import test_qr

    t = test_qr(qr, rtol=1e-7)
    print(t)
