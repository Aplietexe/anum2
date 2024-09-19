import time

import numpy as np
from numpy.typing import NDArray

from .ej9.lu_p import dlup

Arr = NDArray[np.float64]


def det_lu(A: Arr) -> np.float64:
    """
    Given a square matrix A, computes its determinant using LU decomposition.
    Overwrites A with its LU decomposition.
    Returns the determinant of A.
    """
    _, U, _ = dlup(A)
    return np.prod(np.diag(U))


if __name__ == "__main__":
    mats = np.zeros((1000, 50, 50))
    for i in range(1000):
        mats[i] = np.random.uniform(-10, 10, (50, 50))

    np_dets = np.zeros(1000)
    np_time = -time.perf_counter()
    for i in range(1000):
        np_dets[i] = np.linalg.det(mats[i])
    np_time += time.perf_counter()

    my_dets = np.zeros(1000)
    my_time = -time.perf_counter()
    for i in range(1000):
        my_dets[i] = det_lu(mats[i])
    my_time += time.perf_counter()

    print(f"np.linalg.det: {np_time:.6f}s")
    print(f"det_lu: {my_time:.6f}s")
