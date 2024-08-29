import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_inf_solve(
    solve: Callable[[Arr, Arr], Arr | None],
    maxn: int = 100,
    maxa: float = 1e6,
    max_cond: float = 1e7,
    rtol: float = 1e-6,
) -> float:
    t = 0.0
    for _ in range(10000):
        n = np.random.randint(1, maxn)
        A = np.random.uniform(-maxa, maxa, (n, n))
        A = np.tril(A)
        true_x = np.random.uniform(-maxa, maxa, n)
        b = A @ true_x
        t -= time.perf_counter()
        x = solve(A, b)
        t += time.perf_counter()
        if abs(np.linalg.cond(A)) < max_cond:
            assert x is not None
            np.testing.assert_allclose(x, true_x, rtol=rtol)

    return t


def test_sup_solve(
    solve: Callable[[Arr, Arr], Arr | None],
    maxn: int = 100,
    maxa: float = 1e6,
    max_cond: float = 1e7,
    rtol: float = 1e-6,
) -> float:
    t = 0.0
    for _ in range(10000):
        n = np.random.randint(1, maxn)
        A = np.random.uniform(-maxa, maxa, (n, n))
        A = np.triu(A)
        true_x = np.random.uniform(-maxa, maxa, n)
        b = A @ true_x
        t -= time.perf_counter()
        x = solve(A, b)
        t += time.perf_counter()
        if abs(np.linalg.cond(A)) < max_cond:
            assert x is not None
            np.testing.assert_allclose(x, true_x, rtol=rtol)

    return t
