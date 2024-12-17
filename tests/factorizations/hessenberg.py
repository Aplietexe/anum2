import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_hessenberg(
    hessenberg: Callable[[Arr], tuple[Arr, Arr]],
    maxn: int = 70,
    maxa: float = 1e6,
    rtol: float = 1e-6,
    atol: float = 1e-14,
    its: int = 1000,
) -> float:
    """
    Tests Hessenberg factorization. Returns the average time to factorize a matrix.
    """
    t = 0.0
    c = 0
    while c < its:
        n = np.random.randint(1, maxn)
        A = np.random.uniform(-maxa, maxa, (n, n))
        _A = A.copy()
        t -= time.perf_counter()
        H, Q = hessenberg(_A)
        t += time.perf_counter()

        # Correct shapes
        assert H.shape == (n, n)
        assert Q.shape == (n, n)

        # Factorizes A
        targ = Q.T @ A @ Q
        tol = max(float(rtol * np.linalg.norm(targ)), atol)
        assert np.linalg.norm(H - targ) < tol

        # Q is orthogonal
        tol = max(rtol * np.sqrt(n), atol)
        assert np.linalg.norm(Q.T @ Q - np.eye(n)) < tol

        # H is upper Hessenberg
        targ = np.triu(H, -1)
        tol = max(float(rtol * np.linalg.norm(targ)), atol)
        assert np.linalg.norm(H - targ) < tol

        c += 1

    return t / c


def test_reduced_hessenberg(
    hessenberg: Callable[[Arr, Arr, int], tuple[Arr, Arr]],
    maxn: int = 70,
    maxm: int = 20,
    maxa: float = 1e6,
    rtol: float = 1e-6,
    atol: float = 1e-14,
    its: int = 1000,
) -> float:
    """
    Tests Hessenberg factorization. Returns the average time to factorize a matrix.
    """
    t = 0.0
    c = 0
    while c < its:
        n = np.random.randint(1, maxn)
        m = np.random.randint(1, min(n + 1, maxm))
        A = np.random.uniform(-maxa, maxa, (n, n))
        v = np.random.uniform(-maxa, maxa, n)
        _A = A.copy()
        _v = v.copy()
        t -= time.perf_counter()
        H, V = hessenberg(_A, _v, m)
        t += time.perf_counter()

        # Correct shapes
        r = V.shape[1]
        assert H.shape == (r + 1, r)
        assert V.shape == (n, r)

        # H is upper Hessenberg
        targ = np.triu(H, -1)
        tol = max(float(rtol * np.linalg.norm(targ)), atol)
        assert np.linalg.norm(H - targ) < tol

        # V has orthonormal columns
        tol = max(rtol * np.sqrt(r), atol)
        assert np.linalg.norm(V.T @ V - np.eye(r)) < tol

        # V makes A into H_trunc
        H_trunc = H[:r]
        tol = max(float(rtol * np.linalg.norm(H_trunc)), atol)
        assert np.linalg.norm(V.T @ A @ V - H_trunc) < tol

        c += 1

    return t / c
