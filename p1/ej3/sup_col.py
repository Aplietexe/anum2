import numpy as np
from numpy.typing import NDArray
from tests import test_sup_solve

Arr = NDArray[np.float64]


def sol_trsupcol(A: Arr, b: Arr) -> Arr | None:
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")
    if np.any(A.diagonal() == 0):
        return None
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - x[i + 1 :] @ A[i, i + 1 :]) / A[i, i]
    return x


if __name__ == "__main__":
    t = test_sup_solve(sol_trsupcol)
    print(t)
