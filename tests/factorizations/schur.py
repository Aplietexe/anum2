import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_schur(
    schur: Callable[[Arr, int], tuple[Arr, Arr]],
    maxn: int = 10,
    maxa: float = 1e2,
    rtol: float = 1e-4,
    atol: float = 1e-7,
    its: int = 50,
) -> float:
    """
    Tests Schur factorization. Returns the average time to factorize a matrix.
    """
    t = 0.0
    c = 0
    while c < its:
        n = np.random.randint(1, maxn)
        A = np.random.uniform(-maxa, maxa, (n, n))
        _A = A.copy()
        t -= time.perf_counter()
        H, Q = schur(_A, 5000)
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

        # Has Schur structure
        eigenvals = np.linalg.eigvals(H)
        tol = np.maximum(rtol * np.abs(eigenvals), atol)
        # tresh = np.max(np.abs(np.tril(H, -2)))
        tresh = 1e-8
        k = 0
        while k < n:
            if k == n - 1 or np.abs(H[k + 1, k]) <= tresh:  # Real eigenvalue
                assert np.any(
                    np.abs(eigenvals - H[k, k]) < tol
                ), f"{eigenvals} {H[k, k]}"
                k += 1
            else:  # Complex eigenvalue
                curr_eigenvals = np.linalg.eigvals(H[k : k + 2, k : k + 2])
                assert np.any(np.abs(eigenvals - curr_eigenvals[0]) < tol)
                assert np.any(np.abs(eigenvals - curr_eigenvals[1]) < tol)
                k += 2

        c += 1
        print(c)

    return t / c
