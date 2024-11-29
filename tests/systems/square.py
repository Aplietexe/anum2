import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_square(
    solve: Callable[[Arr, Arr], Arr],
    maxn: int = 70,
    maxa: float = 1e6,
    max_cond: float = 1e7,
    rtol: float = 1e-6,
    its: int = 10000,
) -> float:
    """
    Tests a square solver. Returns the average time to solve a system.
    """
    t = 0.0
    c = 0
    for _ in range(its):
        n = np.random.randint(1, maxn)
        A = np.random.uniform(-maxa, maxa, (n, n))
        if abs(np.linalg.cond(A)) > max_cond:
            continue
        true_x = np.random.uniform(-maxa, maxa, n)
        b = A @ true_x
        t -= time.perf_counter()
        x = solve(A, b)
        t += time.perf_counter()
        np.testing.assert_allclose(x, true_x, rtol=rtol)
        c += 1

    return t / c
