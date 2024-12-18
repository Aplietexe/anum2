import numpy as np
from numpy.typing import NDArray

from .hessenberg import hessenberg
from .qr.givens_rotation import givens_2d

Arr = NDArray[np.float64]


def schur(A: Arr, maxits: int) -> tuple[Arr, Arr]:
    """
    Returns H, Q such that H = Q.T @ A @ Q is the Schur
    decomposition of A.
    """
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("Invalid shape")

    H, Q = hessenberg(A)

    for _ in range(maxits):
        G = np.zeros((n - 1, 2, 2))
        for j in range(n - 1):
            G[j] = givens_2d(H[j : j + 2, j])
            H[j : j + 2, j:] = G[j] @ H[j : j + 2, j:]
        for l in range(n - 1):
            H[:, l : l + 2] = H[:, l : l + 2] @ G[l].T
            Q[:, l : l + 2] = Q[:, l : l + 2] @ G[l].T

    return H, Q


if __name__ == "__main__":
    from tests.factorizations.schur import test_schur

    t = test_schur(schur)
    print(t)
