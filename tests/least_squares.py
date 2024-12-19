import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_least_squares(
    least_squares: Callable[[Arr, Arr], tuple[Arr, np.float64]],
    maxn: int = 70,
    maxa: float = 1e6,
    rtol: float = 1e-6,
    atol: float = 1e-5,
    its: int = 5000,
    min_norm=False,
) -> float:
    """
    Tests a least squares solver. Returns the average time to compute the determinant.
    """
    t = 0.0
    c = 0
    for _ in range(its):
        m = np.random.randint(1, maxn)
        n = np.random.randint(1, maxn)
        A = np.random.uniform(-maxa, maxa, (m, n))
        if abs(np.linalg.cond(A)) > 1e7:
            continue
        b = np.random.uniform(-maxa, maxa, m)
        _A = A.copy()
        _b = b.copy()
        t -= time.perf_counter()
        x, r = least_squares(_A, _b)
        t += time.perf_counter()
        np.testing.assert_allclose(A.T @ (A @ x), A.T @ b, rtol=rtol)
        np.testing.assert_allclose(r, np.linalg.norm(A @ x - b), rtol=rtol, atol=atol)

        if min_norm:
            np_sol = np.linalg.lstsq(A, b, rcond=-1)[0]
            np.testing.assert_allclose(x, np_sol, rtol=rtol, atol=atol)

        c += 1

    return t / c
