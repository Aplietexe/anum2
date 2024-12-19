import numpy as np
from numpy.typing import NDArray

from ..eigen.jacobi import diagonalize
from ..factorizations.qr.qr_householder_pivot import qr

Arr = NDArray[np.float64]


def svd(A: Arr, eps: float = 1e-10, its: int = 1000) -> tuple[Arr, Arr, Arr]:
    """
    Computes the singular value decomposition of a matrix.
    Returns U, S, Vt.
    """
    C = A.T @ A
    D, W = diagonalize(C, eps, its)
    Q, R, P = qr(A @ W)
    U = Q
    V = W @ P
    S = R

    return U, S, V.T


if __name__ == "__main__":
    from tests.factorizations.svd import test_svd

    t = test_svd(svd)
    print(t)
