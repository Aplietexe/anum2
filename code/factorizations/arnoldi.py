import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def arnoldi(A: Arr, v: Arr, m: int, eps: float = 1e-10) -> tuple[Arr, Arr]:
    n = A.shape[0]
    if A.shape != (n, n) or v.shape != (n,):
        raise ValueError("Invalid shapes")
    if m > n:
        raise ValueError("m must be less than or equal to n")

    H = np.zeros((m + 1, m))
    V = np.zeros((n, m))
    sigma = np.linalg.norm(v)

    for j in range(m):
        V[:, j] = v / sigma
        v = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = v @ V[:, i]
            v -= H[i, j] * V[:, i]
        sigma = np.linalg.norm(v)
        if sigma <= eps:
            return H[: j + 2, : j + 1], V[:, : j + 1]
        H[j + 1, j] = sigma

    return H, V


if __name__ == "__main__":
    from tests.factorizations.hessenberg import test_reduced_hessenberg

    t = test_reduced_hessenberg(arnoldi)
    print(t)
