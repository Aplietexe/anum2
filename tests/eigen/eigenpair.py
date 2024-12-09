import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_eigenpair(
    eigen: Callable[[Arr, Arr, float, int], tuple[Arr, float]],
    maxn: int = 40,
    maxa: float = 1e2,
    eps: float = 1e-10,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    its: int = 4000,
) -> float:
    """
    Tests eigenvalue and eigenvector computation. Returns the average time.
    """
    t = 0.0
    c = 0
    while c < its:
        n = np.random.randint(2, maxn)
        eigvals = np.random.uniform(-maxa, maxa, n)
        eigvecs = np.random.uniform(-maxa, maxa, (n, n))
        A = eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs)
        q0 = np.random.uniform(-maxa, maxa, n)

        _A = A.copy()
        _q0 = q0.copy()
        t -= time.perf_counter()
        q, rho = eigen(_A, _q0, eps, 2000)
        t += time.perf_counter()

        # Correct shapes
        assert q.shape == (n,)

        # is eigenpair
        tol = max(float(rtol * np.linalg.norm(rho * q)), atol)
        assert np.linalg.norm(A @ q - rho * q) < tol

        # q is normalized
        assert np.abs(np.linalg.norm(q) - 1) < rtol

        c += 1

    return t / c
