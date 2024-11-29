import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def gaussian_elimination(A: Arr, b: Arr) -> tuple[Arr, Arr]:
    """
    Performs Gaussian elimination without pivoting on Ax=b.
    Overwrites the upper triangular part of A with U, and b with y such that
    Ux=y is equivalent to Ax=b.
    Returns modified A and b.
    O(2n^3/3) complexity.
    """
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")

    for i in range(n - 1):
        if np.isclose(A[i, i], 0):
            raise ValueError("Zero pivot")
        A[i + 1 :, i] /= A[i, i]
        A[i + 1 :, i + 1 :] -= np.outer(A[i + 1 :, i], A[i, i + 1 :])
        b[i + 1 :] -= A[i + 1 :, i] * b[i]

    return A, b


if __name__ == "__main__":
    from tests.systems.gaussian_elimination import test_gauss

    t = test_gauss(gaussian_elimination)
    print(t)
