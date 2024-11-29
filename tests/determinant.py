import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_det(
    det: Callable[[Arr], np.float64],
    maxn: int = 40,
    maxa: float = 1e6,
    rtol: float = 1e-6,
    atol: float = 1e-10,
    its: int = 5000,
) -> float:
    """
    Tests a determinant computation. Returns the average time to compute the determinant.
    """
    t = 0.0
    c = 0
    for _ in range(its):
        n = np.random.randint(1, maxn)
        A = np.random.uniform(-maxa, maxa, (n, n))
        if abs(np.linalg.cond(A)) > 1e7:
            continue
        _A = A.copy()
        t -= time.perf_counter()
        d = det(_A)
        t += time.perf_counter()
        np.testing.assert_allclose(np.linalg.det(A), d, rtol=rtol, atol=atol)
        c += 1

    return t / c
