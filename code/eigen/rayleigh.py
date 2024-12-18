import numpy as np
from numpy.typing import NDArray

from ..systems.elimination.gaussian_pivoting import gaussian_elimination
from ..systems.triangular.upper_row import solve_upper_triangular

Arr = NDArray[np.float64]


def eigen(A: Arr, q0: Arr, eps: float, maxits: int) -> tuple[Arr, float]:
    """
    Computes the dominant eigenvalue and eigenvector of a matrix using the Rayleigh
    quotient iteration.
    """
    I = np.eye(A.shape[0])
    q0 /= np.linalg.norm(q0)
    q = q0
    rho = q0.dot(A @ q0)
    for _ in range(maxits):
        B, y = gaussian_elimination(A - rho * I, q.copy())
        if np.abs(B[-1, -1]) <= eps:
            return q0, rho
        z = solve_upper_triangular(B, y, eps)
        norm = np.linalg.norm(z)
        q0 = z / norm
        rho += q0.dot(q) / norm
        q = q0

    print("Warning: Maximum number of iterations reached")

    return q0, rho


if __name__ == "__main__":
    from tests.eigen.eigenpair import test_eigenpair

    t = test_eigenpair(eigen)
    print(t)
