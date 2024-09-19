import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_inv(
    inv: Callable[[Arr], Arr | None],
    maxn: int = 40,
    maxa: float = 1e6,
    rtol: float = 1e-6,
    atol: float = 1e-10,
    its: int = 5000,
) -> float:
    """
    Tests an inverse matrix computation. Returns the average time to compute the inverse.
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
        A_inv = inv(_A)
        t += time.perf_counter()
        assert A_inv is not None
        prod = A @ A_inv
        np.testing.assert_allclose(np.diag(prod), 1, rtol=rtol)
        prod[np.eye(n, dtype=bool)] = 0
        np.testing.assert_allclose(prod, 0, atol=atol)
        c += 1

    return t / c
