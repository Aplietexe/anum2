import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_lu(
    dlu: Callable[[Arr], tuple[Arr, Arr]],
    maxn: int = 70,
    maxa: float = 1e6,
    max_cond: float = 1e5,
    rtol: float = 2e-6,
    its: int = 10000,
) -> float:
    """
    Tests LU factorization. Returns the average time to factorize a matrix.
    """
    t = 0.0
    c = 0
    for _ in range(its):
        n = np.random.randint(1, maxn)
        A = np.random.uniform(-maxa, maxa, (n, n))
        if abs(np.linalg.cond(A)) > max_cond:
            continue
        _A = A.copy()
        t -= time.perf_counter()
        L, U = dlu(_A)
        t += time.perf_counter()
        np.testing.assert_allclose(L @ U, A, rtol=rtol)
        c += 1

    return t / c
