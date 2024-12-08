import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def jacobi(A: Arr) -> Arr:
    """
    Returns the rotation matrix Q that diagonalizes a symmetric 2x2 matrix A.
    Reads upper triangular part of A.
    """
    if A.shape != (2, 2):
        raise ValueError("Invalid shape")
    if not np.isclose(A[0, 1], A[1, 0]):
        print("Warning: Matrix is not symmetric")

    if np.isclose(A[0, 1], 0):
        return np.array([[1, 0], [0, 1]])

    tau = (A[1, 1] - A[0, 0]) / (2 * A[0, 1])
    if tau >= 0:
        t = -1 / (tau + np.sqrt(1 + tau**2))
    else:
        t = 1 / (-tau + np.sqrt(1 + tau**2))
    c = 1 / np.sqrt(1 + t**2)
    s = c * t

    return np.array([[c, -s], [s, c]])


def off(A: Arr) -> tuple[np.float64, int, int]:
    """
    Given A symmetric, returns val, i, j such that
    - val is the Frobenius distance of A to its diagonal part
    - A[i,j] is the off-diagonal element with the largest absolute value.
    """
    n = A.shape[0]
    off = 0
    i = 0
    j = 1
    for r in range(1, n):
        for s in range(r):
            off += A[r, s] ** 2
            if np.abs(A[r, s]) > np.abs(A[i, j]):
                i = r
                j = s
    return np.sqrt(2 * off), i, j


def diagonalize(A: Arr, eps: np.float64, maxits: int) -> tuple[Arr, Arr]:
    """
    Given A n x n symmetric with n >= 2, returns D, Q such that D = Q.T @ A @ Q
    using Jacobi's method.
    Overwrites A with D.
    """
    n = A.shape[0]
    Q = np.eye(n)

    for _ in range(maxits):
        norm, i, j = off(A)
        if norm <= eps:
            break

        J = jacobi(A[np.ix_([i, j], [i, j])])
        A[[i, j], :] = J.T @ A[[i, j], :]
        A[:, [i, j]] = A[:, [i, j]] @ J
        Q[:, [i, j]] = Q[:, [i, j]] @ J

    norm = off(A)[0]
    if norm > eps:
        print(f"Warning: off(A) = {norm} > {eps}")

    return A, Q


if __name__ == "__main__":
    from tests.eigen.symmetric import test_symmetric_diagonalization

    t = test_symmetric_diagonalization(diagonalize)
    print(t)
