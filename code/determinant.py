import time

import numpy as np
from numpy.typing import NDArray

from .factorizations.lu_pivoting import lu

Arr = NDArray[np.float64]


def det_lu(A: Arr) -> np.float64:
    """
    Given a square matrix A, computes its determinant using LU decomposition.
    Overwrites A with its LU decomposition.
    Returns the determinant of A.
    O(2n^3/3) complexity.
    """
    _, U, _, p = lu(A)
    return np.prod(np.diag(U)) * p


if __name__ == "__main__":
    from tests.determinant import test_det

    t = test_det(det_lu)
    print(t)
