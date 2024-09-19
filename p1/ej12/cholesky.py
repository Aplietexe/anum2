import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def cholesky(A: Arr) -> Arr | None:
    """
    Given a symmetric matrix A, computes its Cholesky factor if it exists.
    Overwrites the upper triangular part of A with the Cholesky factor and returns it.
    Returns None if A is not positive definite.
    """
    n = A.shape[0]

    if A.shape != (n, n):
        raise ValueError("Invalid shape")
    if not np.allclose(A, A.T):
        raise ValueError("Matrix is not symmetric")

    for i in range(n):
        A[i, i:] -= A[:i, i] @ A[:i, i:]
        if A[i, i] < 0 or np.isclose(A[i, i], 0):
            return None
        A[i, i:] /= np.sqrt(A[i, i])

    return A


if __name__ == "__main__":
    from .tests import test_cholesky

    t = test_cholesky(cholesky)
    print(t)
