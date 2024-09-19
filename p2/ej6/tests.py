import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_lu(
    dlu: Callable[[Arr], tuple[Arr, Arr]] | Callable[[Arr], tuple[Arr, Arr, Arr]],
    maxn: int = 70,
    maxa: float = 1e6,
    max_cond: float = 1e7,
    rtol: float = 1e-6,
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
        res = dlu(_A)
        t += time.perf_counter()
        L = res[0]
        U = res[1]
        if len(res) == 3:
            A = res[2] @ A
        np.testing.assert_allclose(L @ U, A, rtol=rtol)
        c += 1

    return t / c
