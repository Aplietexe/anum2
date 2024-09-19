import numpy as np
from numpy.typing import NDArray

from ..ej5.tests import test_gauss

Arr = NDArray[np.float64]


def egaussp(A: Arr, b: Arr) -> tuple[Arr, Arr]:
    """
    Performs Gaussian elimination with partial pivoting on Ax=b.
    Overwrites A with U upper triangular and b with y such that Ux=y.
    Returns modified A and b.
    """
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")

    for i in range(n - 1):
        pivot = np.argmax(np.abs(A[i:, i])) + i
        if not np.isclose(A[pivot, i], 0):
            A[[i, pivot]] = A[[pivot, i]]
            b[[i, pivot]] = b[[pivot, i]]
        A[i + 1 :, i] /= A[i, i]
        A[i + 1 :, i + 1 :] -= np.outer(A[i + 1 :, i], A[i, i + 1 :])
        b[i + 1 :] -= A[i + 1 :, i] * b[i]

    return A, b


if __name__ == "__main__":
    t = test_gauss(egaussp)
    print(t)
