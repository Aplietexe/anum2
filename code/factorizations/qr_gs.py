import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def qr(A: Arr) -> tuple[Arr, Arr]:
    """
    Performs QR decomposition on A using Gram-Schmidt orthonormalization.
    O(2mn^2) complexity.
    """
    m, n = A.shape
    R = np.zeros((n, n))
    Q = np.zeros((m, n))
    R[0, 0] = np.linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0, 0]

    for j in range(1, n):
        R[:j, j] = Q[:, :j].T @ A[:, j]
        q = A[:, j] - Q[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(q)
        Q[:, j] = q / R[j, j]

    return Q, R


if __name__ == "__main__":
    from tests.factorizations.qr import test_reduced_qr

    t = test_reduced_qr(qr, under=False)
    print(t)
