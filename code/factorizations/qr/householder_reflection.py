import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def householder_reflection(x: Arr) -> tuple[Arr, np.float64]:
    """
    Returns u, rho such that (I - rho * u @ u.T) @ x = ||x||_2 * e1.
    Overwrites x with u.
    O(3n) complexity.
    """
    n = x.shape[0]
    if x.shape != (n,):
        raise ValueError("Invalid shape")

    if n == 1:
        sigma = 0
    else:
        sigma = x[1:].dot(x[1:])

    if np.isclose(sigma, 0) and x[0] >= 0:
        rho = np.float64(0)
        u = x
        u[0] = 1
    else:
        mu = np.sqrt(x[0] ** 2 + sigma)
        if x[0] <= 0:
            gamma = x[0] - mu
        else:
            gamma = -sigma / (x[0] + mu)
        rho = 2 * gamma**2 / (sigma + gamma**2)
        u = x / gamma
        u[0] = 1

    return u, rho
