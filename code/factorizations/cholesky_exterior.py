import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def cholesky(A: Arr) -> Arr | None:
    """
    Given a symmetric matrix A, computes its Cholesky factor if it exists.
    Writes the Cholesky factor in the upper triangular part of A and returns it.
    Destroys the lower triangular part of A.
    Returns None if A is not positive definite.
    O(n^3/3) complexity.
    """
    n = A.shape[0]

    if A.shape != (n, n):
        raise ValueError("Invalid shape")
    if not np.allclose(A, A.T):
        raise ValueError("Matrix is not symmetric")

    for i in range(n):
        if A[i, i] < 0 or np.isclose(A[i, i], 0):
            return None
        A[i, i:] /= np.sqrt(A[i, i])
        A[i + 1 :, i + 1 :] -= np.outer(A[i, i + 1 :], A[i, i + 1 :])

    return A


if __name__ == "__main__":
    from tests.factorizations.cholesky import test_cholesky

    t = test_cholesky(cholesky)
    print(t)
