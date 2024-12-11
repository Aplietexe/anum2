import numpy as np
from numpy.typing import NDArray

from .householder_reflection import householder_reflection

Arr = NDArray[np.float64]


def hessenberg(A: Arr) -> tuple[Arr, Arr]:
    """
    Returns H, Q such that H = Q.T @ A @ Q, H is upper
    Hessenberg and Q is orthogonal.
    Overwrites the upper Hessenberg part of A with H.
    """
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("Invalid shape")

    Q = np.eye(n)
    for j in range(n - 2):
        u, rho = householder_reflection(A[j + 1 :, j].copy())
        w = rho * u
        A[j + 1 :, j:] -= np.outer(w, u.T @ A[j + 1 :, j:])
        A[:, j + 1 :] -= np.outer(A[:, j + 1 :] @ w, u)
        Q[:, j + 1 :] -= np.outer(Q[:, j + 1 :] @ w, u)

    return A, Q


if __name__ == "__main__":
    from tests.factorizations.hessenberg import test_hessenberg

    t = test_hessenberg(hessenberg)
    print(t)
