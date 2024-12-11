import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_diagonally_dominant(
    solve: Callable[[Arr, Arr, Arr, float, int], Arr],
    maxn: int = 70,
    maxa: float = 1e6,
    eps: float = 1e-10,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    its: int = 5000,
) -> float:
    """
    Tests a square solver on diagonally dominant matrices.
    Returns the average time to solve a system.
    """
    t = 0.0
    c = 0
    for _ in range(its):
        n = np.random.randint(1, maxn)
        A = np.random.uniform(-maxa, maxa, (n, n))
        A += np.diag(np.abs(A).sum(axis=1)) * 1.1
        true_x = np.random.uniform(-maxa, maxa, n)
        x0 = np.random.uniform(-maxa, maxa, n)
        b = A @ true_x

        t -= time.perf_counter()
        x = solve(A, b, x0, eps, 1000)
        t += time.perf_counter()

        tol = max(float(rtol * np.linalg.norm(true_x)), atol)
        assert np.linalg.norm(x - true_x) < tol

        c += 1
        print(c)

    return t / c
