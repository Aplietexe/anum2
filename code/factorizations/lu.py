import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def lu(A: Arr) -> tuple[Arr, Arr]:
    """
    Performs LU decomposition without pivoting on A.
    Overwrites A with U upper triangular and L lower triangular.
    Returns L and U.
    O(2n^3/3) complexity.
    """
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("Invalid shape")

    for i in range(n - 1):
        if np.isclose(A[i, i], 0):
            raise ValueError("Zero pivot")
        A[i + 1 :, i] /= A[i, i]
        A[i + 1 :, i + 1 :] -= np.outer(A[i + 1 :, i], A[i, i + 1 :])

    return np.tril(A, -1) + np.eye(n), np.triu(A)


if __name__ == "__main__":
    from tests.factorizations.lu import test_lu

    t = test_lu(lu, max_cond=1e5, rtol=1e-5)
    print(t)
