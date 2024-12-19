import numpy as np
from numpy.typing import NDArray

from .qr.givens_rotation import givens_2d

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
        for i in range(j + 2, n):
            rot = givens_2d(A[[j + 1, i], j])
            A[[j + 1, i], j:] = rot @ A[[j + 1, i], j:]
            A[:, [j + 1, i]] = A[:, [j + 1, i]] @ rot.T
            Q[:, [j + 1, i]] = Q[:, [j + 1, i]] @ rot.T

    return A, Q


if __name__ == "__main__":
    from tests.factorizations.hessenberg import test_hessenberg

    t = test_hessenberg(hessenberg)
    print(t)
