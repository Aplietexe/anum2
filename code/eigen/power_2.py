import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def dominant_eigen(A: Arr, q0: Arr, eps: np.float64, maxits: int) -> tuple[Arr, float]:
    q = A @ q0
    prev_rho = float(q0 @ q / (q0 @ q0))
    for _ in range(maxits):
        q0 = q / np.linalg.norm(q)
        q = A @ q0
        rho = float(q0 @ q)
        if np.abs(prev_rho - rho) <= eps:
            return q0, rho
        prev_rho = rho

    print("Warning: Maximum number of iterations reached")

    return q0, prev_rho


if __name__ == "__main__":
    from tests.eigen.dominant import test_dominant_eigen

    t = test_dominant_eigen(dominant_eigen, 2, rtol=5e-2)
    print(t)
