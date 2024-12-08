import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


import time
from typing import Callable


def dominant_eigen(A: Arr, q: Arr, eps: np.float64, maxits: int) -> tuple[Arr, float]:
    """
    Computes the dominant eigenvalue and eigenvector of a matrix using the power
    method in the infinity norm.
    """
    prev_rho = np.inf
    for _ in range(maxits):
        j = np.argmax(np.abs(q))
        q /= q[j]
        q = A @ q
        rho = q[j]
        if np.abs(rho - prev_rho) <= eps:
            prev_rho = rho
            break
        prev_rho = rho
    else:
        print("Warning: Maximum number of iterations reached")

    return q / np.linalg.norm(q, np.inf), prev_rho


if __name__ == "__main__":
    from tests.eigen.dominant import test_dominant_eigen

    t = test_dominant_eigen(dominant_eigen, np.inf)
    print(t)
