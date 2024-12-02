import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_qr(
    qr: Callable[[Arr], tuple[Arr, Arr]] | Callable[[Arr], tuple[Arr, Arr, Arr]],
    maxn: int = 70,
    maxa: float = 1e6,
    rtol: float = 1e-6,
    atol: float = 1e-14,
    its: int = 10000,
) -> float:
    """
    Tests QR factorization. Returns the average time to factorize a matrix.
    """
    t = 0.0
    c = 0
    for _ in range(its):
        m = np.random.randint(1, maxn)
        n = np.random.randint(1, maxn)
        A = np.random.uniform(-maxa, maxa, (m, n))
        _A = A.copy()
        t -= time.perf_counter()
        res = qr(_A)
        t += time.perf_counter()
        Q = res[0]
        R = np.triu(res[1])
        if len(res) == 3:
            A = A @ res[2]

        # Correct shapes
        assert Q.shape == (m, m)
        assert R.shape == (m, n)
        if len(res) == 3:
            assert res[2].shape == (n, n)

        # Factorizes A
        np.testing.assert_allclose(Q @ R, A, rtol=rtol)

        # Q is orthogonal
        prod = Q.T @ Q
        np.testing.assert_allclose(np.diag(prod), 1, rtol=rtol)
        prod[np.eye(m, dtype=bool)] = 0
        np.testing.assert_allclose(prod, 0, atol=atol)

        # Diagonal of R is nonnegative/positive
        assert np.all(np.diag(R) >= 0)
        if len(res) == 3:
            assert not np.any(np.isclose(np.diag(R), 0))

        c += 1

    return t / c


def test_reduced_qr(
    qr: Callable[[Arr], tuple[Arr, Arr]],
    maxn: int = 70,
    maxa: float = 1e6,
    rtol: float = 1e-6,
    atol: float = 1e-10,
    its: int = 10000,
    under: bool = True,
) -> float:
    """
    Tests reduced QR factorization. Returns the average time to factorize a matrix.
    """
    t = 0.0
    c = 0
    for _ in range(its):
        m = np.random.randint(1, maxn)
        n = np.random.randint(1, maxn)
        if not under and m < n:
            m, n = n, m
        A = np.random.uniform(-maxa, maxa, (m, n))
        _A = A.copy()
        t -= time.perf_counter()
        Q, R = qr(_A)
        t += time.perf_counter()
        R = np.triu(R)

        # Correct shapes
        assert Q.shape == (m, n)
        assert R.shape == (n, n)

        # Factorizes A
        np.testing.assert_allclose(Q @ R, A, rtol=rtol)

        # Q has orthonormal columns
        prod = Q.T @ Q
        np.testing.assert_allclose(np.diag(prod), 1, rtol=rtol)
        prod[np.eye(n, dtype=bool)] = 0
        np.testing.assert_allclose(prod, 0, atol=atol)

        # Diagonal of R is nonnegative
        assert np.all(np.diag(R) >= 0)

        c += 1

    return t / c
